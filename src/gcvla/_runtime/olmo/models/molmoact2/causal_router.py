# Modifications by Tommoro: causal GC/history routing and checkpoint loading.
"""Causal GC/history routing with per-stream recurrent memory."""

from pathlib import Path

import torch
from torch import nn

ARCHITECTURE = "anchored_causal_gru_gc32_prior"


class AnchoredRouter(nn.Module):
    def __init__(self, hidden_dim, anchor_hidden_dim):
        hidden, anchor_hidden = hidden_dim, anchor_hidden_dim
        super().__init__()
        self.anchor=nn.Sequential(nn.Linear(64,anchor_hidden),nn.GELU(),nn.Linear(anchor_hidden,anchor_hidden),nn.GELU(),nn.Linear(anchor_hidden,2*anchor_hidden),nn.GELU(),nn.Linear(2*anchor_hidden,1))
        self.gru=nn.GRU(64,hidden,batch_first=True)
        self.correction=nn.Linear(hidden+64,1)
        self.register_buffer('feature_mean',torch.zeros(64))
        self.register_buffer('feature_std',torch.ones(64))
    def forward(self,x,state=None):
        normalized=(x-self.feature_mean)/self.feature_std
        if state is None:
            first=normalized[:,0]
            recurrent=None
            prior=self.anchor(x[:,0])
        else:
            first,recurrent,prior=state
        hidden,recurrent=self.gru(normalized,recurrent)
        firsts=first[:,None].expand(-1,x.shape[1],-1)
        logits=prior[:,None]+self.correction(torch.cat((hidden,firsts),-1))
        return logits.squeeze(-1),(first,recurrent,prior)

def load_router_payload(payload):
    if not isinstance(payload, dict) or set(payload) != {
        "format_version",
        "architecture",
        "config",
        "state_dict",
    }:
        raise ValueError("Invalid router payload fields")
    if (
        type(payload["format_version"]) is not int
        or payload["format_version"] != 1
        or payload["architecture"] != ARCHITECTURE
    ):
        raise ValueError("Unsupported router format")
    config = payload["config"]
    if not isinstance(config, dict) or set(config) != {"hidden_dim", "anchor_hidden_dim"}:
        raise ValueError("Invalid router configuration")
    if any(type(value) is not int or not 1 <= value <= 4096 for value in config.values()):
        raise ValueError("Invalid router dimensions")
    state = payload["state_dict"]
    if not isinstance(state, dict) or not all(
        isinstance(key, str)
        and isinstance(value, torch.Tensor)
        and value.dtype == torch.float32
        and value.device.type == "cpu"
        and torch.isfinite(value).all()
        for key, value in state.items()
    ):
        raise ValueError("Router state must contain finite CPU float32 tensors")
    with torch.random.fork_rng(devices=[]):
        model = AnchoredRouter(**config)
        expected = model.state_dict()
        if set(state) != set(expected) or any(
            state[key].shape != expected[key].shape for key in expected
        ):
            raise ValueError("Router tensor layout does not match configuration")
        model.load_state_dict(state, strict=True)
        if not (model.feature_std > 0).all():
            raise ValueError("Router normalization scale must be positive")
    return CausalRouter(model.eval())


def load_router(path: str | Path):
    return load_router_payload(torch.load(path, map_location="cpu", weights_only=True))


class CausalRouter:
    """CPU inference keeps recurrent routing independent of policy placement."""

    def __init__(self, model):
        self.model = model
        self.states = {}

    def reset(self):
        self.states.clear()

    def decision(self, context, stream_index, replan_index):
        if (
            type(stream_index) is not int
            or stream_index < 0
            or type(replan_index) is not int
            or replan_index < 0
        ):
            raise ValueError("Invalid router stream or replan index")
        if (
            not isinstance(context, torch.Tensor)
            or context.shape != (1, 96)
            or not context.is_floating_point()
            or not torch.isfinite(context).all()
        ):
            raise ValueError("Router requires one finite floating-point 96D context row")
        if replan_index == 0:
            self.states.pop(stream_index, None)
        state, expected = self.states.get(stream_index, (None, 0))
        if replan_index != expected:
            raise ValueError("Missing or reordered router prefix")
        with torch.inference_mode():
            # Context layout: base[32], GC[32], trajectory statistics[32].
            x = (
                context[0, 32:96]
                .detach()
                .to(device="cpu", dtype=torch.float32)
            )
            logits, state = self.model(x.reshape(1, 1, 64), state)
            if not torch.isfinite(logits).all():
                raise ValueError("Nonfinite router decision")
            value = float(logits[0, 0])
        self.states[stream_index] = (state, expected + 1)
        return value >= 0, value
