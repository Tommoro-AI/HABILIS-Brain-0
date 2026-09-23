"""Complete unsharded release weights; no optimizer, trainer, or distributed resume."""

from pathlib import Path

import torch


@torch.no_grad()
def load_model_state(directory, model):
    if torch.distributed.is_initialized():
        raise RuntimeError("Run one inference worker per GPU, not distributed training")
    state = torch.load(Path(directory) / "model.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
