# Modifications by Tommoro: public runtime extraction and adaptation.
# This file differs from its original source; see manifest.json for source hashes.

import dataclasses
import logging
import json
import os
from pathlib import Path
import re
from dataclasses import field
from typing import ClassVar, List, Optional, Sequence, Tuple, Iterator
import torch
import torch.nn as nn
import torch.nn.functional as F
from .causal_router import load_router
from olmo import tokenizer as tok
from olmo.config import D
from olmo.extra_tokens import (
    ACTION_END_TOKEN,
    ACTION_START_TOKEN,
    ACTION_TOKENS,
    DEFAULT_NUM_ACTION_TOKENS,
    DEPTH_OUTPUT_TOKEN,
    DEPTH_TOKENS,
)
from olmo.models.model import OLMoOutput
from olmo.data.dynamic_packer import EXAMPLE_SUBSEGMENT_INCREMENT
from olmo.models.molmo2.molmo2 import Molmo2, Molmo2Config
from olmo.nn.action_expert import ActionExpert, ActionExpertConfig
from olmo.data.robot_processing import RobotProcessorConfig
from olmo.preprocessing.multimodal_collator import MMCollator
from olmo.tokenizer import get_special_token_ids

log = logging.getLogger(__name__)


def _env_flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


def _validate_public_environment():
    allowed = {
        "MOLMOACT2_ACTION_EXPERT_DROP_DEPTH_TOKENS": "1",
        "MOLMOACT2_DEPTH_DECODE_FALLBACK": "1",
        "MOLMOACT2_GCVLA_FIXED_QUERY_DEPTH": "1",
        "MOLMOACT2_PRESERVE_RAW_GRIPPER": "1",
        "MOLMOACT2_GCRF_RESIDUAL_PATH": None,
        "MOLMOACT2_GCRF_ROUTER_PATH": None,
        "MOLMOACT2_GCRF_ROUTER_ENABLED": "1",
        "MOLMOACT2_GCRF_ROUTER_OVERRIDE": "1",
        "MOLMOACT2_GCRF_ONPOLICY": "1",
        "MOLMOACT2_EPISODE_FLOW_RNG": "1",
        "MOLMOACT2_GCRF_PRESERVE_BASE_RNG": "1",
        "MOLMOACT2_GCRF_POLICY_SPLIT_BASE_WIDTH": "14870",
        "MOLMOACT2_GCRF_POLICY_SPLIT_BOUNDARIES": None,
        "MOLMOACT2_GCRF_POLICY_DISABLE_TF32": "1",
        "MOLMOACT2_ACTION_TRACE_PATH": None,
        "MOLMOACT2_GCRF_ONPOLICY_TRACE_PATH": None,
        "MOLMOACT2_GCRF_ROLLOUT_ID": None,
    }
    for name, value in os.environ.items():
        if name.startswith("MOLMOACT2_"):
            if name not in allowed or (allowed[name] is not None and value != allowed[name]):
                raise ValueError("Unsupported environment for the public inference checkpoint")


class GCRFResidualAdapter(nn.Module):
    """External bounded residual branch around a frozen flow sampler."""

    def __init__(
        self,
        *,
        action_dim: int,
        flow_steps: int,
        gc_dim: int = 32,
        history_dim: int = 32,
        hidden_dim: int = 128,
        latent_dim: int = 8,
        residual_budget: float = 0.05,
        on_policy_direct: bool = False,
        gc_only_policy: bool = False,
        normalize_gc_policy: bool = False,
        prototype_policy: bool = False,
        value_option_policy: bool = False,
        option_count: int = 0,
        option_margin: float = 0.25,
        value_continuous_policy: bool = False,
        value_continuous_margin: float = 0.0,
        gate_enabled: bool = False,
        gate_history: int = 3,
        gate_hidden_dim: int = 64,
        gate_duration: int = 2,
        gate_threshold: float = 0.0,
        progressive_recurrent: bool = False,
        recurrent_hidden_dim: int = 128,
        progressive_alpha_bias: float = -2.0,
        progressive_bypass_threshold: float = 0.05,
    ) -> None:
        if (
            not on_policy_direct
            or not value_continuous_policy
            or any(
                (
                    prototype_policy,
                    value_option_policy,
                    progressive_recurrent,
                    gate_enabled,
                    gc_only_policy,
                    normalize_gc_policy,
                )
            )
        ):
            raise ValueError("Unsupported residual architecture in the public checkpoint")
        super().__init__()
        self.action_dim = action_dim
        self.flow_steps = flow_steps
        self.gc_dim = gc_dim
        self.history_dim = history_dim
        self.latent_dim = latent_dim
        self.residual_budget = float(residual_budget)
        self.on_policy_direct = bool(on_policy_direct)
        self.gc_only_policy = bool(gc_only_policy)
        self.normalize_gc_policy = bool(normalize_gc_policy)
        self.prototype_policy = bool(prototype_policy)
        self.value_option_policy = bool(value_option_policy)
        self.option_count = int(option_count)
        self.option_margin = float(option_margin)
        self.value_continuous_policy = bool(value_continuous_policy)
        self.value_continuous_margin = float(value_continuous_margin)
        self.gate_enabled = bool(gate_enabled)
        self.gate_history = max(1, int(gate_history))
        self.gate_duration = max(1, int(gate_duration))
        self.gate_threshold = float(gate_threshold)
        self.progressive_recurrent = bool(progressive_recurrent)
        self.recurrent_hidden_dim = int(recurrent_hidden_dim)
        self.progressive_alpha_bias = float(progressive_alpha_bias)
        self.progressive_bypass_threshold = float(progressive_bypass_threshold)
        context_dim = action_dim + gc_dim + history_dim
        self.policy = nn.Sequential(
            nn.Linear(context_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, 2 * latent_dim)
        )
        self.option_q = None
        self.continuous_q = nn.Sequential(
            nn.Linear(context_dim + latent_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, 1)
        )
        self.residual = nn.Sequential(
            nn.Linear(context_dim + latent_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, flow_steps * action_dim),
        )
        self.gate = None
        self.progressive_gru = None
        self.progressive_latent_head = None
        self.progressive_alpha_head = None
        self._router = None
        self._last_router_active: Optional[bool] = None
        self._last_router_logit: Optional[float] = None
        self._last_router_active_mask: Optional[torch.Tensor] = None
        self._last_router_logits: Optional[torch.Tensor] = None

    def _context(
        self, base_velocity: torch.Tensor, gc_summary: torch.Tensor, history: torch.Tensor
    ) -> torch.Tensor:
        if base_velocity.ndim > 2:
            base_summary = base_velocity.reshape(
                base_velocity.shape[0], -1, base_velocity.shape[-1]
            ).mean(dim=1)
        else:
            base_summary = base_velocity
        return torch.cat([base_summary, gc_summary, history], dim=-1)

    def _distribution_from_context(
        self, context: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        disable_tf32 = context.is_cuda and _env_flag("MOLMOACT2_GCRF_POLICY_DISABLE_TF32")
        previous_allow_tf32 = torch.backends.cuda.matmul.allow_tf32
        if disable_tf32:
            torch.backends.cuda.matmul.allow_tf32 = False
        split_width = int(os.environ.get("MOLMOACT2_GCRF_POLICY_SPLIT_BASE_WIDTH", "0"))
        split_block_width = 0
        split_boundaries = [
            int(value)
            for value in os.environ.get("MOLMOACT2_GCRF_POLICY_SPLIT_BOUNDARIES", "").split(",")
            if value.strip()
        ]
        fp32_from_width = 0
        try:
            if split_width:
                first = self.policy[0]
                last = self.policy[2]
                if not 0 < split_width < first.out_features:
                    raise ValueError("GCRF policy split width must be between zero and hidden_dim")
                base_hidden = F.silu(
                    F.linear(context, first.weight[:split_width], first.bias[:split_width])
                )
                policy_out = F.linear(base_hidden, last.weight[:, :split_width], last.bias)
                if split_boundaries:
                    if split_boundaries[-1] != first.out_features or any(
                        (
                            end <= start
                            for start, end in zip(
                                [split_width] + split_boundaries[:-1], split_boundaries
                            )
                        )
                    ):
                        raise ValueError("invalid GCRF policy split boundaries")
                    block_ranges = zip([split_width] + split_boundaries[:-1], split_boundaries)
                    for start, end in block_ranges:
                        extra_hidden = F.silu(
                            F.linear(context, first.weight[start:end], first.bias[start:end])
                        )
                        policy_out = policy_out + F.linear(
                            extra_hidden, last.weight[:, start:end], None
                        )
                else:
                    extra_hidden = F.silu(
                        F.linear(context, first.weight[split_width:], first.bias[split_width:])
                    )
                    policy_out = policy_out + F.linear(
                        extra_hidden, last.weight[:, split_width:], None
                    )
            else:
                policy_out = self.policy(context)
        finally:
            if disable_tf32:
                torch.backends.cuda.matmul.allow_tf32 = previous_allow_tf32
        mean, _ = policy_out.chunk(2, dim=-1)
        log_std = torch.full_like(mean, -2.0)
        return (mean, log_std)

    def _continuous_q_from_context(self, context: torch.Tensor) -> torch.Tensor:
        if self.continuous_q is None:
            raise RuntimeError("continuous value policy payload is missing Q")
        split_width = int(os.environ.get("MOLMOACT2_GCRF_POLICY_SPLIT_BASE_WIDTH", "0"))
        split_block_width = 0
        split_boundaries = [
            int(value)
            for value in os.environ.get("MOLMOACT2_GCRF_POLICY_SPLIT_BOUNDARIES", "").split(",")
            if value.strip()
        ]
        if not split_width:
            return self.continuous_q(context)
        first = self.continuous_q[0]
        last = self.continuous_q[2]
        if not 0 < split_width < first.out_features:
            raise ValueError("GCRF continuous-Q split width must be between zero and hidden_dim")
        base_hidden = F.silu(
            F.linear(context, first.weight[:split_width], first.bias[:split_width])
        )
        value = F.linear(base_hidden, last.weight[:, :split_width], last.bias)
        if split_boundaries:
            if split_boundaries[-1] != first.out_features or any(
                (
                    end <= start
                    for start, end in zip([split_width] + split_boundaries[:-1], split_boundaries)
                )
            ):
                raise ValueError("invalid GCRF continuous-Q split boundaries")
            block_ranges = zip([split_width] + split_boundaries[:-1], split_boundaries)
            for start, end in block_ranges:
                extra_hidden = F.silu(
                    F.linear(context, first.weight[start:end], first.bias[start:end])
                )
                value = value + F.linear(extra_hidden, last.weight[:, start:end], None)
            return value
        extra_hidden = F.silu(
            F.linear(context, first.weight[split_width:], first.bias[split_width:])
        )
        return value + F.linear(extra_hidden, last.weight[:, split_width:], None)

    def _router_decision(self, context, latent, mean, replan_index):
        """Evaluate causal GC/history state for the current stream."""
        if self._router is None or not _env_flag("MOLMOACT2_GCRF_ROUTER_ENABLED"):
            return None
        active, value = self._router.decision(
            context, getattr(self, "_router_stream_index", 0), replan_index
        )
        result = torch.tensor([active], dtype=torch.bool, device=context.device)
        self._last_router_active = active
        self._last_router_active_mask = result
        self._last_router_logit = value
        self._last_router_logits = torch.tensor([value], dtype=context.dtype, device=context.device)
        return result

    def sample_replan(
        self,
        base_velocity: torch.Tensor,
        gc_summary: torch.Tensor,
        history: torch.Tensor,
        replan_index: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample one latent and return latent, log-prob, context, mean, log_std."""
        context = self._context(base_velocity, gc_summary, history)
        mean, log_std = self._distribution_from_context(context)
        std = log_std.exp()
        sampled_latent = mean + std * torch.randn_like(std)
        zero = torch.zeros_like(mean)
        q_zero = self._continuous_q_from_context(torch.cat([context, zero], dim=-1)).squeeze(-1)
        q_mean = self._continuous_q_from_context(torch.cat([context, mean], dim=-1)).squeeze(-1)
        use_residual = q_mean > q_zero + self.value_continuous_margin
        latent = torch.where(use_residual[:, None], mean, zero)
        log_prob = torch.zeros(context.shape[0], device=context.device)
        return (latent, log_prob, context, mean, log_std)

    def residual_from_latent(self, latent: torch.Tensor) -> torch.Tensor:
        """Use one replan latent as a bounded residual direction."""
        if latent.shape[-1] != self.action_dim:
            raise ValueError(
                f"on-policy direct latent must have action_dim={self.action_dim}, got {latent.shape[-1]}"
            )
        return self.residual_budget * torch.tanh(latent)

    def forward(
        self,
        base_velocity: torch.Tensor,
        gc_summary: torch.Tensor,
        history: torch.Tensor,
        step_index: int,
    ) -> torch.Tensor:
        context = self._context(base_velocity, gc_summary, history)
        mean, _ = self._distribution_from_context(context)
        residual = self.residual(torch.cat([context, mean], dim=-1))
        residual = self.residual_budget * torch.tanh(residual)
        return residual.view(-1, self.flow_steps, self.action_dim)[:, step_index]


@dataclasses.dataclass
class MolmoAct2Config(Molmo2Config):
    """Configuration for the MolmoAct2 model."""

    _model_name: ClassVar[str] = "molmoact2"
    max_action_dim: int = 32
    "Maximum dimensionality of each action vector after right-padding."
    action_horizon: int = 30
    "Number of action steps predicted by the policy."
    n_action_steps: Optional[int] = None
    "Deprecated checkpoint field. Inference defaults now come from per-tag metadata."
    n_obs_steps: int = 1
    "Number of observation steps provided to the policy."
    action_expert: ActionExpertConfig = field(default_factory=ActionExpertConfig)
    "Configuration for the diffusion-style action head."
    add_action_expert: bool = True
    "If True, build the action expert branch. Disable for pure autoregressive pretraining."
    action_expert_detach_vlm: bool = False
    "If True, stop gradients from the action expert branch from flowing back into VLM conditioning features."
    action_expert_depth_gate: bool = False
    "If True, learn a per-example scalar gate that scales depth-token conditioning for the action expert."
    action_expert_depth_gate_per_layer: bool = False
    "If True, learn one depth gate per selected action-expert conditioning layer."
    action_expert_depth_gate_init_bias: float = -4.0
    "Initial depth gate logit. Negative values make the initial policy close to the no-depth path."
    action_format: str = "continuous"
    'Action supervision mode: "continuous", "discrete", or "both".'
    state_format: str = "discrete"
    'State conditioning mode: "continuous", "discrete", or "both".'
    flow_matching_num_steps: int = 10
    "Number of integration steps during flow-matching inference."
    flow_matching_cutoff: float = 1.0
    flow_matching_time_offset: float = 0.001
    flow_matching_time_scale: float = 0.999
    flow_matching_beta_alpha: float = 1.0
    flow_matching_beta_beta: float = 1.5
    num_flow_timesteps: int = 1
    "Number of timesteps/noise vectors to use per batch item during training."
    mask_action_chunk_padding: bool = True
    "Deprecated knob. Time padding from tag horizon to max horizon is always masked in training."
    mask_action_dim_padding: bool = True
    "If True, exclude right-padded action dimensions from flow-matching dynamics and loss."
    enable_depth_reasoning: bool = False
    "If True, activate depth-reasoning training/inference paths when the caller requests them."
    num_depth_codes: int = 100
    "Number of depth code positions emitted per image (for example a 10x10 grid)."
    depth_code_input_noise_rate: float = 0.0
    "Training-only fraction of teacher-forced depth code tokens replaced with random depth codes."
    robot_processor: Optional[RobotProcessorConfig] = None
    "Shared normalization pipeline used to normalize/unnormalize actions and states."

    @classmethod
    def update_legacy_settings(cls, config: D) -> D:
        config = super().update_legacy_settings(config)
        if "action_dim" in config:
            legacy_dim = int(config["action_dim"])
            if "max_action_dim" in config and int(config["max_action_dim"]) != legacy_dim:
                raise ValueError(
                    f"Found conflicting MolmoAct2 action dimensions in config: action_dim={legacy_dim} vs max_action_dim={int(config['max_action_dim'])}."
                )
            config["max_action_dim"] = legacy_dim
            del config["action_dim"]
        if "action_expert" in config and config.action_expert is not None:
            config.action_expert = ActionExpertConfig.update_legacy_settings(config.action_expert)
        if "action_expert_layer_mode" in config:
            value = str(config["action_expert_layer_mode"])
            if value != "per_layer":
                raise ValueError(
                    f"MolmoAct2 action expert only supports per-layer conditioning; found legacy action_expert_layer_mode={value!r}."
                )
            del config["action_expert_layer_mode"]
        if "action_expert_condition_source" in config:
            value = str(config["action_expert_condition_source"])
            if value != "kv_cache":
                raise ValueError(
                    f"MolmoAct2 action expert only supports KV-cache conditioning; found legacy action_expert_condition_source={value!r}."
                )
            del config["action_expert_condition_source"]
        if "robot_processor" not in config:
            if "robot_preprocessor" in config and config.robot_preprocessor is not None:
                config.robot_processor = config.robot_preprocessor
            elif "robot_postprocessor" in config and config.robot_postprocessor is not None:
                config.robot_processor = config.robot_postprocessor
        if "robot_preprocessor" in config:
            del config["robot_preprocessor"]
        if "robot_postprocessor" in config:
            del config["robot_postprocessor"]
        if "robot_processor" in config and config.robot_processor is not None:
            config.robot_processor = RobotProcessorConfig.update_legacy_settings(
                config.robot_processor
            )
        if "progress_token_value_encoding" in config:
            del config["progress_token_value_encoding"]
        if "progress_token_value_encoding_scale" in config:
            del config["progress_token_value_encoding_scale"]
        return config

    def build_model(self, device=None):
        return MolmoAct2(self, device)

    def build_collator(self, output_shapes, pad_mode: str, include_metadata=True) -> MMCollator:
        return MMCollator(
            get_special_token_ids(self.build_tokenizer()),
            output_shapes,
            include_metadata=include_metadata,
            pad=pad_mode,
            cp_enabled=self.cp_enabled,
            packed_action_shape=(self.action_horizon, self.max_action_dim),
        )


class MolmoAct2(Molmo2):
    """MolmoAct2 extends Molmo2 with an action diffusion head."""

    def __init__(self, config: MolmoAct2Config, device=None):
        _validate_public_environment()
        if config.action_expert_depth_gate:
            raise ValueError("Unsupported action-expert gate in public checkpoint")
        super().__init__(config, device)
        valid_action_formats = {"continuous", "discrete", "both"}
        if config.action_format not in valid_action_formats:
            raise ValueError(
                f"Unknown action_format '{config.action_format}'. Expected one of {sorted(valid_action_formats)}."
            )
        valid_state_formats = {"continuous", "discrete", "both"}
        if config.state_format not in valid_state_formats:
            raise ValueError(
                f"Unknown state_format '{config.state_format}'. Expected one of {sorted(valid_state_formats)}."
            )
        if int(config.num_depth_codes) <= 0:
            raise ValueError(f"num_depth_codes must be > 0, got {config.num_depth_codes}.")
        if not 0.0 <= float(config.depth_code_input_noise_rate) <= 1.0:
            raise ValueError(
                f"depth_code_input_noise_rate must be in [0, 1], got {config.depth_code_input_noise_rate}."
            )
        if config.action_expert_depth_gate and (not config.add_action_expert):
            raise ValueError("action_expert_depth_gate requires add_action_expert=True.")
        if config.action_expert_depth_gate_per_layer and (not config.action_expert_depth_gate):
            raise ValueError(
                "action_expert_depth_gate_per_layer requires action_expert_depth_gate=True."
            )
        if config.flow_matching_time_offset > config.flow_matching_cutoff:
            raise ValueError(
                f"flow_matching_time_offset must be <= flow_matching_cutoff (got {config.flow_matching_time_offset} > {config.flow_matching_cutoff})."
            )
        if config.flow_matching_time_scale <= 0:
            raise ValueError(
                f"flow_matching_time_scale must be > 0 (got {config.flow_matching_time_scale})."
            )
        self._action_start_token_id: Optional[int] = None
        self._action_end_token_id: Optional[int] = None
        self._eos_token_id: Optional[int] = None
        self._image_prompt_token_id: Optional[int] = None
        self._depth_gate_token_ids: Tuple[int, ...] = ()
        self._depth_code_token_ids: Tuple[int, ...] = ()
        self._fixed_query_depth_token_id: Optional[int] = None
        self._action_code_token_ids: Tuple[int, ...] = ()
        self._fixed_query_action_token_id: Optional[int] = None
        self._fixed_query_debug_logged = False
        self._fixed_action_query_debug_logged = False
        self._gcrf_residual_adapter = None
        self._gcrf_residual_path = ""
        try:
            tokenizer = self.config.build_tokenizer()
            self._image_prompt_token_id = int(getattr(tokenizer, "image_prompt_token_id"))
        except Exception:
            self._image_prompt_token_id = None
        if config.action_format == "both":
            tokenizer = self.config.build_tokenizer()
            action_start_ids = tokenizer.encode(ACTION_START_TOKEN)
            action_end_ids = tokenizer.encode(ACTION_END_TOKEN)
            if len(action_start_ids) != 1 or len(action_end_ids) != 1:
                raise ValueError(
                    "action_format='both' requires single-token <action_start>/<action_end>. Enable action tokens in the tokenizer before using both-mode supervision."
                )
            self._action_start_token_id = int(action_start_ids[0])
            self._action_end_token_id = int(action_end_ids[0])
            eos_id = getattr(tokenizer, "eos_token_id", None)
            self._eos_token_id = None if eos_id is None else int(eos_id)
            log.info(
                "Masking discrete answer spans for action expert conditioning (action_start=%d, action_end=%d, eos=%s).",
                self._action_start_token_id,
                self._action_end_token_id,
                str(self._eos_token_id),
            )
        if (
            config.action_expert_depth_gate
            or _env_flag("MOLMOACT2_ACTION_EXPERT_DROP_DEPTH_TOKENS")
            or _env_flag("MOLMOACT2_GCVLA_FIXED_QUERY_DEPTH")
        ):
            self._depth_gate_token_ids = self._resolve_depth_gate_token_ids()
        if _env_flag("MOLMOACT2_GCVLA_FIXED_QUERY_DEPTH"):
            self._depth_code_token_ids, self._fixed_query_depth_token_id = (
                self._resolve_fixed_query_depth_token_ids()
            )
        if config.action_expert.max_action_dim != config.max_action_dim:
            config.action_expert.max_action_dim = config.max_action_dim
        if config.action_expert.max_horizon < config.action_horizon:
            config.action_expert.max_horizon = config.action_horizon
        self.action_expert: Optional[ActionExpert]
        if config.add_action_expert:
            if config.action_expert.num_layers != self.config.llm.n_layers:
                raise ValueError(
                    f"Action expert depth ({config.action_expert.num_layers}) must match LLM layers ({self.config.llm.n_layers}) when using per_layer conditioning."
                )
            llm_head_dim = (
                self.config.llm.head_dim
                if self.config.llm.head_dim is not None
                else self.config.llm.d_model // self.config.llm.n_heads
            )
            llm_kv_dim = self.config.llm.effective_n_kv_heads * llm_head_dim
            self.action_expert = config.action_expert.build(
                llm_dim=self.config.llm.d_model,
                llm_kv_dim=llm_kv_dim,
                llm_num_kv_heads=self.config.llm.effective_n_kv_heads,
                llm_num_layers=self.config.llm.n_layers,
                device=device,
            )
        else:
            self.action_expert = None
        self.action_expert_depth_gate: Optional[nn.Module]
        if config.action_expert_depth_gate:
            gate_input_dim = llm_kv_dim
            if config.action_expert_depth_gate_per_layer:
                num_gate_layers = len(self._require_action_expert().blocks)
                self.action_expert_depth_gate = nn.ModuleList(
                    (nn.Linear(gate_input_dim, 1).to(device=device) for _ in range(num_gate_layers))
                )
            else:
                self.action_expert_depth_gate = nn.Linear(gate_input_dim, 1).to(device=device)
            self.reset_action_expert_depth_gate_parameters()
        else:
            self.action_expert_depth_gate = None
        self.action_plan_kv_adapter: Optional[nn.Module]
        self.action_plan_kv_adapter = None

    def _load_gcrf_residual_adapter(self, device: torch.device) -> None:
        path = os.environ.get("MOLMOACT2_GCRF_RESIDUAL_PATH", "").strip()
        if not path:
            return
        if self._gcrf_residual_path == path and self._gcrf_residual_adapter is not None:
            return
        payload = torch.load(path, map_location="cpu", weights_only=True)
        config = payload.get("config", {}) if isinstance(payload, dict) else {}
        aux_device_name = ""
        adapter_device = device
        cpu_rng_state = torch.random.get_rng_state()
        cuda_rng_state = None
        if device.type == "cuda":
            cuda_rng_state = torch.cuda.get_rng_state(device)
        try:
            adapter = GCRFResidualAdapter(
                action_dim=int(config.get("action_dim", self.config.max_action_dim)),
                flow_steps=int(config.get("flow_steps", self.config.flow_matching_num_steps)),
                gc_dim=int(config.get("gc_dim", 32)),
                history_dim=int(config.get("history_dim", 32)),
                hidden_dim=int(config.get("hidden_dim", 128)),
                latent_dim=int(config.get("latent_dim", 8)),
                residual_budget=float(config.get("residual_budget", 0.05)),
                on_policy_direct=bool(config.get("on_policy_direct", False)),
                gc_only_policy=bool(config.get("gc_only_policy", False)),
                normalize_gc_policy=bool(config.get("normalize_gc_policy", False)),
                prototype_policy=bool(config.get("prototype_policy", False)),
                value_option_policy=bool(config.get("value_option_policy", False)),
                option_count=int(config.get("option_count", 0)),
                option_margin=float(config.get("option_margin", 0.25)),
                value_continuous_policy=bool(config.get("value_continuous_policy", False)),
                value_continuous_margin=float(config.get("value_continuous_margin", 0.0)),
                gate_enabled=bool(config.get("gate_enabled", False)),
                gate_history=int(config.get("gate_history", 3)),
                gate_hidden_dim=int(config.get("gate_hidden_dim", 64)),
                gate_duration=int(config.get("gate_duration", 2)),
                gate_threshold=float(config.get("gate_threshold", 0.0)),
                progressive_recurrent=bool(config.get("progressive_recurrent", False)),
                recurrent_hidden_dim=int(config.get("recurrent_hidden_dim", 128)),
                progressive_alpha_bias=float(config.get("progressive_alpha_bias", -2.0)),
                progressive_bypass_threshold=float(config.get("progressive_bypass_threshold", 0.05)),
            )
            state_dict = payload.get("state_dict", payload) if isinstance(payload, dict) else payload
            adapter.load_state_dict(state_dict, strict=True)
            if isinstance(payload, dict) and "prototype_gc" in payload:
                raise ValueError("Unsupported auxiliary tensor payload in public checkpoint")
            if isinstance(payload, dict) and "option_latents" in payload:
                raise ValueError("Unsupported auxiliary tensor payload in public checkpoint")
            adapter = adapter.to(adapter_device).eval()
            router_path = os.environ.get("MOLMOACT2_GCRF_ROUTER_PATH", "").strip()
            if router_path:
                adapter._router = load_router(router_path)
                log.info("Loaded causal GC/history router")
        finally:
            torch.random.set_rng_state(cpu_rng_state)
            if cuda_rng_state is not None:
                torch.cuda.set_rng_state(cuda_rng_state, device)
        self._gcrf_residual_adapter = adapter
        self._gcrf_residual_path = path
        log.info(
            "Loaded external GCRF residual adapter from %s on auxiliary device %s",
            path,
            adapter_device,
        )

    @staticmethod
    def _gcrf_compact_gc(
        layer_kv_states: Sequence[Tuple[torch.Tensor, torch.Tensor]],
        device: Optional[torch.device] = None,
    ) -> Optional[torch.Tensor]:
        values = []
        for _, value in layer_kv_states:
            if device is None:
                value = value.float()
            else:
                value = value.detach().to(device=device, dtype=torch.float32)
            if value.ndim < 2:
                continue
            flattened = value.reshape(value.shape[0], -1)
            pooled = F.adaptive_avg_pool1d(flattened.unsqueeze(1), 32).squeeze(1)
            values.append(pooled)
        if not values:
            return None
        return torch.stack(values[-12:], dim=1).mean(dim=1)

    @staticmethod
    def _gcrf_history(
        trajectory: torch.Tensor, device: Optional[torch.device] = None
    ) -> torch.Tensor:
        if device is None:
            trajectory = trajectory.float()
        else:
            trajectory = trajectory.detach().to(device=device, dtype=torch.float32)
        mean = F.adaptive_avg_pool1d(trajectory.mean(dim=1).unsqueeze(1), 16).squeeze(1)
        std = F.adaptive_avg_pool1d(trajectory.std(dim=1, unbiased=False).unsqueeze(1), 16).squeeze(
            1
        )
        return torch.cat([mean, std], dim=-1)

    def _gcrf_residual(
        self,
        trajectory: torch.Tensor,
        velocity: torch.Tensor,
        gc_summary: Optional[torch.Tensor],
        step_index: int,
    ) -> Optional[torch.Tensor]:
        adapter = self._gcrf_residual_adapter
        if adapter is None or gc_summary is None:
            return None
        adapter_device = next(adapter.parameters()).device
        router_mask = getattr(adapter, "_last_router_active_mask", None)
        if (
            router_mask is not None
            and _env_flag("MOLMOACT2_GCRF_ROUTER_ENABLED")
            and _env_flag("MOLMOACT2_GCRF_ROUTER_OVERRIDE")
            and (not bool(router_mask.to(dtype=torch.bool).any().item()))
        ):
            return None
        onpolicy_latent = getattr(self, "_gcrf_onpolicy_latent", None)
        if not _env_flag("MOLMOACT2_GCRF_ONPOLICY") or onpolicy_latent is None:
            return None
        residual = adapter.residual_from_latent(onpolicy_latent.to(device=adapter_device))
        router_mask = getattr(adapter, "_last_router_active_mask", None)
        if (
            router_mask is not None
            and _env_flag("MOLMOACT2_GCRF_ROUTER_ENABLED")
            and _env_flag("MOLMOACT2_GCRF_ROUTER_OVERRIDE")
        ):
            router_mask = router_mask.to(device=residual.device, dtype=torch.bool).reshape(-1)
            if router_mask.numel() != residual.shape[0]:
                raise RuntimeError("router mask batch dimension mismatch")
            if not bool(router_mask.any().item()):
                return None
            if bool((~router_mask).any().item()):
                residual = residual.masked_fill((~router_mask).view(-1, 1, 1), 0.0)
        return residual.to(device=velocity.device, dtype=velocity.dtype)

    def _gcrf_onpolicy_replan_is_active(self) -> bool:
        stream_index = int(getattr(self, "_gcrf_router_stream_index", 0))
        replan_index = int(getattr(self, "_gcrf_router_replan_counts", {}).get(stream_index, 0))
        return 0 <= replan_index < 2147483647

    def _gcrf_trace_provenance_fields(self, stream_index: int) -> dict:
        """Return fixed-batch provenance for the current batch slot."""
        metadata = getattr(self, "_gcrf_trace_metadata", {}) or {}
        row = {
            "trace_contract": metadata.get("trace_contract", ""),
            "suite": metadata.get("suite", ""),
            "task_id": metadata.get("task_id", ""),
            "batch_ix": metadata.get("batch_ix"),
            "batch_size": metadata.get("batch_size"),
            "n_episodes": metadata.get("n_episodes"),
            "cli_seed": metadata.get("cli_seed"),
            "checkpoint_sha256": "",
            "adapter_sha256": "",
            "config_sha256": "",
            "code_sha256": "",
        }
        init_indices = metadata.get("init_indices") or []
        episode_seeds = metadata.get("episode_seeds") or []
        if 0 <= stream_index < len(init_indices):
            row["init_index"] = int(init_indices[stream_index])
        if 0 <= stream_index < len(episode_seeds):
            row["episode_seed"] = int(episode_seeds[stream_index])
        return row

    def _gcrf_advance_inactive_replan(self, context: Optional[torch.Tensor] = None) -> None:
        self._gcrf_onpolicy_latent = None
        stream_index = int(getattr(self, "_gcrf_router_stream_index", 0))
        replan_index = int(getattr(self, "_gcrf_router_replan_counts", {}).get(stream_index, 0))
        trace_path = os.environ.get("MOLMOACT2_GCRF_ONPOLICY_TRACE_PATH", "").strip()
        if trace_path and context is not None:
            row = {
                "rollout_id": os.environ.get("MOLMOACT2_GCRF_ROLLOUT_ID", ""),
                "stream_index": stream_index,
                "replan_index": replan_index,
                "active": False,
                "context": context.detach().float().cpu().tolist(),
            }
            row.update(self._gcrf_trace_provenance_fields(stream_index))
            path = Path(trace_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a") as handle:
                handle.write(json.dumps(row, separators=(",", ":")) + "\n")
        if not hasattr(self, "_gcrf_router_replan_counts"):
            self._gcrf_router_replan_counts = {}
        self._gcrf_router_replan_counts[stream_index] = replan_index + 1
        self._gcrf_onpolicy_replans = replan_index + 1

    def _gcrf_sample_onpolicy_replan(
        self,
        adapter: GCRFResidualAdapter,
        velocity: torch.Tensor,
        gc_summary: torch.Tensor,
        history: torch.Tensor,
    ) -> None:
        adapter_device = next(adapter.parameters()).device
        velocity = velocity.detach().to(device=adapter_device)
        gc_summary = gc_summary.detach().to(device=adapter_device)
        history = history.detach().to(device=adapter_device)
        stream_index = int(getattr(self, "_gcrf_router_stream_index", 0))
        self._gcrf_router_stream_index = stream_index
        adapter._router_stream_index = stream_index
        router_replan_index = int(
            getattr(self, "_gcrf_router_replan_counts", {}).get(stream_index, 0)
        )
        router_active = None
        latent = None
        log_prob = None
        preserve_base_rng = _env_flag("MOLMOACT2_GCRF_PRESERVE_BASE_RNG")
        cpu_rng_state = torch.random.get_rng_state() if preserve_base_rng else None
        cuda_rng_state = None
        if preserve_base_rng and velocity.is_cuda:
            cuda_rng_state = torch.cuda.get_rng_state(velocity.device)
        try:
            latent, log_prob, context, mean, log_std = adapter.sample_replan(
                velocity, gc_summary, history, replan_index=router_replan_index
            )
        finally:
            if cpu_rng_state is not None:
                torch.random.set_rng_state(cpu_rng_state)
            if cuda_rng_state is not None:
                torch.cuda.set_rng_state(cuda_rng_state, velocity.device)
        stream_index = int(getattr(self, "_gcrf_router_stream_index", 0))
        replan_index = int(getattr(self, "_gcrf_router_replan_counts", {}).get(stream_index, 0))
        router_active = adapter._router_decision(context, latent, mean, replan_index)
        if router_active is not None and _env_flag("MOLMOACT2_GCRF_ROUTER_OVERRIDE"):
            router_mask = router_active
            if not torch.is_tensor(router_mask):
                router_mask = torch.full(
                    (latent.shape[0],), bool(router_mask), device=latent.device
                )
            router_mask = router_mask.to(device=latent.device, dtype=torch.bool).reshape(-1)
            if router_mask.numel() != latent.shape[0]:
                raise RuntimeError("router mask batch dimension mismatch")
            latent = mean.clone()
            if bool((~router_mask).any().item()):
                latent = torch.where(
                    router_mask[:, None],
                    mean,
                    torch.zeros_like(mean),
                )
            log_prob = torch.zeros(mean.shape[0], device=mean.device)
        elif router_active is None and _env_flag("MOLMOACT2_GCRF_ROUTER_OVERRIDE"):
            latent = torch.zeros_like(mean)
            log_prob = torch.zeros(mean.shape[0], device=mean.device)
        self._gcrf_onpolicy_latent = latent
        replan_index = int(getattr(self, "_gcrf_router_replan_counts", {}).get(stream_index, 0))
        trace_path = os.environ.get("MOLMOACT2_GCRF_ONPOLICY_TRACE_PATH", "").strip()
        if trace_path:
            row = {
                "rollout_id": os.environ.get("MOLMOACT2_GCRF_ROLLOUT_ID", ""),
                "stream_index": stream_index,
                "replan_index": replan_index,
                "active": True,
                "context": context.detach().float().cpu().tolist(),
                "latent": latent.detach().float().cpu().tolist(),
                "mean": mean.detach().float().cpu().tolist(),
                "log_std": log_std.detach().float().cpu().tolist(),
                "log_prob": log_prob.detach().float().cpu().tolist(),
            }
            row.update(self._gcrf_trace_provenance_fields(stream_index))
            if adapter._last_router_active is not None:
                row["router_active"] = bool(adapter._last_router_active)
                row["router_logit"] = float(adapter._last_router_logit)
            if adapter._last_router_active_mask is not None:
                row["router_active_mask"] = adapter._last_router_active_mask.detach().cpu().tolist()
            if adapter._last_router_logits is not None:
                row["router_logits"] = adapter._last_router_logits.detach().float().cpu().tolist()
            path = Path(trace_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a") as handle:
                handle.write(json.dumps(row, separators=(",", ":")) + "\n")
        if not hasattr(self, "_gcrf_router_replan_counts"):
            self._gcrf_router_replan_counts = {}
        self._gcrf_router_replan_counts[stream_index] = replan_index + 1
        self._gcrf_onpolicy_replans = replan_index + 1

    def reset(self) -> None:
        """Reset episode-local GCRF state before a new evaluation episode."""
        self._gcrf_onpolicy_latent = None
        self._gcrf_onpolicy_replans = 0
        self._gcrf_router_stream_index = 0
        self._gcrf_router_replan_counts = {}
        adapter = self._gcrf_residual_adapter
        if adapter is not None:
            if adapter._router is not None:
                adapter._router.reset()
            adapter._last_router_active = None
            adapter._last_router_logit = None
            adapter._last_router_active_mask = None
            adapter._last_router_logits = None

    def reset_action_expert_depth_gate_parameters(self) -> None:
        if self.action_expert_depth_gate is None:
            return
        gates = (
            self.action_expert_depth_gate
            if isinstance(self.action_expert_depth_gate, nn.ModuleList)
            else [self.action_expert_depth_gate]
        )
        for gate in gates:
            if not isinstance(gate, nn.Linear):
                raise TypeError(f"Expected depth gate to be nn.Linear, got {type(gate).__name__}.")
            nn.init.zeros_(gate.weight)
            nn.init.constant_(gate.bias, float(self.config.action_expert_depth_gate_init_bias))

    def _resolve_depth_gate_token_ids(self) -> Tuple[int, ...]:
        added_tokens = list(
            self.config.llm.tokenizer.resolve_new_tokens_for_both_input_and_output()
        )
        depth_tokens = [DEPTH_OUTPUT_TOKEN]
        depth_tokens.extend(
            (
                token
                for token in added_tokens
                if token == DEPTH_TOKENS.start_token
                or token == DEPTH_TOKENS.end_token
                or DEPTH_TOKENS.parse_index(token) is not None
            )
        )
        tokenizer = self.config.build_tokenizer()
        token_ids = []
        for token in dict.fromkeys(depth_tokens):
            encoded = tokenizer.encode(token)
            if len(encoded) == 1:
                token_ids.append(int(encoded[0]))
        if not token_ids:
            raise ValueError(
                "action_expert_depth_gate=True requires depth tokens in the tokenizer."
            )
        return tuple(token_ids)

    def _resolve_fixed_query_depth_token_ids(self) -> Tuple[Tuple[int, ...], int]:
        """Resolve depth-code IDs and use depth bin zero as a checkpoint-compatible query."""
        tokenizer = self.config.build_tokenizer()
        code_ids = []
        query_id = None
        for token in self.config.llm.tokenizer.resolve_new_tokens_for_both_input_and_output():
            depth_index = DEPTH_TOKENS.parse_index(token)
            if depth_index is None:
                continue
            encoded = tokenizer.encode(token)
            if len(encoded) != 1:
                continue
            token_id = int(encoded[0])
            code_ids.append(token_id)
            if int(depth_index) == 0:
                query_id = token_id
        if not code_ids or query_id is None:
            raise ValueError(
                "MOLMOACT2_GCVLA_FIXED_QUERY_DEPTH requires single-token depth codes, including depth bin zero."
            )
        return (tuple(code_ids), query_id)

    def _replace_depth_codes_with_fixed_queries(
        self,
        input_ids: Optional[torch.Tensor],
        input_embeddings: Optional[torch.Tensor],
        labels: Optional[torch.Tensor],
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Remove teacher-forced depth values while preserving their sequence slots."""
        if not _env_flag("MOLMOACT2_GCVLA_FIXED_QUERY_DEPTH") or input_ids is None:
            return (input_ids, None)
        if input_embeddings is not None:
            raise ValueError(
                "Fixed GC queries require token-ID embedding lookup; input_embeddings must be None."
            )
        if not self._depth_code_token_ids or self._fixed_query_depth_token_id is None:
            raise RuntimeError("Fixed GC query token IDs were not initialized.")
        code_ids = torch.as_tensor(
            self._depth_code_token_ids, device=input_ids.device, dtype=input_ids.dtype
        )
        query_mask = (input_ids.unsqueeze(-1) == code_ids).any(dim=-1)
        if not self._fixed_query_debug_logged:
            label_code_count = -1
            if labels is not None and labels.shape == input_ids.shape:
                label_code_count = int(
                    (labels.unsqueeze(-1) == code_ids).any(dim=-1).sum().detach().cpu().item()
                )
            gate_ids = torch.as_tensor(
                self._depth_gate_token_ids, device=input_ids.device, dtype=input_ids.dtype
            )
            gate_count = int(
                (input_ids.unsqueeze(-1) == gate_ids).any(dim=-1).sum().detach().cpu().item()
            )
            log.warning(
                "Fixed GC query first-batch audit: input_shape=%s input_code_count=%d label_code_count=%d any_depth_token_count=%d code_vocab=%d query_id=%d.",
                tuple(input_ids.shape),
                int(query_mask.sum().detach().cpu().item()),
                label_code_count,
                gate_count,
                len(self._depth_code_token_ids),
                int(self._fixed_query_depth_token_id),
            )
            self._fixed_query_debug_logged = True
        query_input_ids = input_ids.clone()
        query_input_ids.masked_fill_(query_mask, int(self._fixed_query_depth_token_id))
        return (query_input_ids, query_mask)

    def _require_action_expert(self) -> ActionExpert:
        if self.action_expert is None:
            raise RuntimeError(
                "This MolmoAct2 instance was built with add_action_expert=False, so action expert training/generation is unavailable."
            )
        return self.action_expert

    def forward(
        self,
        input_ids: torch.LongTensor,
        input_embeddings: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        attention_bias: Optional[torch.Tensor] = None,
        response_mask: Optional[torch.Tensor] = None,
        subsegment_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
        loss_masks: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
        image_masks: Optional[torch.Tensor] = None,
        token_pooling: Optional[torch.Tensor] = None,
        low_res_token_pooling: Optional[torch.Tensor] = None,
        num_images: Optional[torch.Tensor] = None,
        multimodal_type: Optional[torch.Tensor] = None,
        num_image_starts: Optional[torch.Tensor] = None,
        response_logits_only: bool = False,
        past_key_values: Optional[Sequence[Tuple[torch.Tensor, torch.Tensor]]] = None,
        use_cache: bool = False,
        last_logits_only: bool = False,
        output_hidden_states: Optional[bool] = None,
        append_last_valid_logits: Optional[torch.Tensor] = None,
        collect_layer_hidden_states: bool = False,
        states: Optional[torch.Tensor] = None,
        actions: Optional[torch.Tensor] = None,
        action_horizon_is_pad: Optional[torch.Tensor] = None,
        action_is_pad: Optional[torch.Tensor] = None,
        action_dim_is_pad: Optional[torch.Tensor] = None,
        packed_batch_idx: Optional[torch.Tensor] = None,
        packed_example_ids: Optional[torch.Tensor] = None,
        packed_action_chunk_is_valid: Optional[torch.Tensor] = None,
        packed_num_chunks: Optional[torch.Tensor] = None,
        packed_action_chunk_cap: Optional[torch.Tensor] = None,
        packed_action_chunk_overflow: Optional[torch.Tensor] = None,
    ) -> OLMoOutput:
        if labels is not None or loss_masks is not None or actions is not None:
            raise RuntimeError("Training targets are not accepted by the inference model")
        "Run the base VLM and (optionally) compute the action loss."
        output_hidden_states = output_hidden_states if output_hidden_states is not None else False
        capture_l36_gc = False
        if action_horizon_is_pad is not None and action_is_pad is not None:
            raise ValueError("Provide only one of action_horizon_is_pad or legacy action_is_pad.")
        resolved_action_horizon_is_pad = (
            action_horizon_is_pad if action_horizon_is_pad is not None else action_is_pad
        )
        if actions is not None:
            raise RuntimeError("Action loss is unavailable in inference release")
        collect_layer_states = collect_layer_hidden_states
        collect_layer_input_states = False
        collect_block_output_states = collect_layer_states
        collect_layer_kv = actions is not None
        backbone_input_ids, fixed_query_mask = self._replace_depth_codes_with_fixed_queries(
            input_ids, input_embeddings, labels
        )
        backbone_input_ids, fixed_action_query_mask = (backbone_input_ids, None)
        parallel_query_mask_active = False
        forward_kwargs = dict(
            input_ids=backbone_input_ids,
            input_embeddings=input_embeddings,
            attention_mask=attention_mask,
            attention_bias=attention_bias,
            response_mask=response_mask,
            subsegment_ids=subsegment_ids,
            position_ids=position_ids,
            labels=labels,
            loss_masks=loss_masks,
            images=images,
            image_masks=image_masks,
            token_pooling=token_pooling,
            low_res_token_pooling=low_res_token_pooling,
            num_images=num_images,
            multimodal_type=multimodal_type,
            num_image_starts=num_image_starts,
            response_logits_only=response_logits_only,
            past_key_values=past_key_values,
            use_cache=use_cache,
            last_logits_only=last_logits_only,
            append_last_valid_logits=append_last_valid_logits,
        )
        base_output, layer_states, layer_kv_states = self._run_backbone(
            collect_layer_hidden_states=collect_block_output_states,
            collect_layer_kv_states=collect_layer_kv,
            collect_layer_input_states=False,
            output_hidden_states=output_hidden_states,
            **forward_kwargs,
        )
        metrics = dict(base_output.metrics or {})
        internal = dict(base_output.internal or {})
        if fixed_query_mask is not None:
            metrics["gcvla_fixed_query_slots"] = (
                fixed_query_mask.sum(dim=-1).float().mean().detach()
            )
        if actions is not None:
            raise RuntimeError("Action loss is unavailable in inference release")
        return base_output._replace(metrics=metrics, internal=internal)

    @torch.no_grad()
    def generate_actions(
        self,
        input_ids: Optional[torch.LongTensor],
        input_embeddings: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        attention_bias: Optional[torch.Tensor] = None,
        response_mask: Optional[torch.Tensor] = None,
        subsegment_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
        loss_masks: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
        image_masks: Optional[torch.Tensor] = None,
        token_pooling: Optional[torch.Tensor] = None,
        low_res_token_pooling: Optional[torch.Tensor] = None,
        num_images: Optional[torch.Tensor] = None,
        multimodal_type: Optional[torch.Tensor] = None,
        num_image_starts: Optional[torch.Tensor] = None,
        response_logits_only: bool = False,
        past_key_values: Optional[Sequence[Tuple[torch.Tensor, torch.Tensor]]] = None,
        use_cache: bool = False,
        last_logits_only: bool = False,
        append_last_valid_logits: Optional[torch.Tensor] = None,
        states: Optional[torch.Tensor] = None,
        action_dim_is_pad: Optional[torch.Tensor] = None,
        num_steps: Optional[int] = None,
        generator: Optional[torch.Generator] = None,
        encoder_kv_states: Optional[Sequence[Tuple[torch.Tensor, torch.Tensor]]] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Generate an action trajectory via flow-matching integration."""
        action_expert = self._require_action_expert()
        if states is None and self.config.state_format in {"continuous", "both"}:
            raise ValueError(
                f"States must be provided for action generation with state_format='{self.config.state_format}'."
            )
        raw_encoder_attention_mask = None
        if input_ids is not None:
            raw_encoder_attention_mask = self._get_encoder_attention_mask(input_ids, attention_mask)
        if encoder_kv_states is not None:
            layer_kv_states = self._select_layer_kv_states(encoder_kv_states)
            layer_states = None
            if encoder_attention_mask is None and input_ids is not None:
                encoder_attention_mask = raw_encoder_attention_mask
        else:
            if input_ids is None:
                raise ValueError(
                    "input_ids must be provided when encoder conditioning is not precomputed."
                )
            encoder_attention_mask = raw_encoder_attention_mask
            forward_kwargs = dict(
                input_ids=input_ids,
                input_embeddings=input_embeddings,
                attention_mask=attention_mask,
                attention_bias=attention_bias,
                response_mask=response_mask,
                subsegment_ids=subsegment_ids,
                position_ids=position_ids,
                labels=labels,
                loss_masks=loss_masks,
                images=images,
                image_masks=image_masks,
                token_pooling=token_pooling,
                low_res_token_pooling=low_res_token_pooling,
                num_images=num_images,
                multimodal_type=multimodal_type,
                num_image_starts=num_image_starts,
                response_logits_only=response_logits_only,
                past_key_values=past_key_values,
                use_cache=use_cache,
                last_logits_only=last_logits_only,
                append_last_valid_logits=append_last_valid_logits,
            )
            collect_layer_input_states = False
            _, layer_states, layer_kv_states = self._run_backbone(
                collect_layer_hidden_states=False,
                collect_layer_kv_states=True,
                collect_layer_input_states=False,
                output_hidden_states=False,
                **forward_kwargs,
            )
            if layer_kv_states is None:
                raise RuntimeError("Failed to capture KV states for action generation.")
            layer_kv_states = self._select_layer_kv_states(layer_kv_states)
            layer_states = None
        depth_gate, depth_mask = self._depth_gate_from_condition(
            input_ids=input_ids,
            encoder_attention_mask=encoder_attention_mask,
            layer_states=layer_states,
            layer_kv_states=layer_kv_states,
        )
        layer_states = self._apply_depth_gate_to_layer_states(layer_states, depth_mask, depth_gate)
        layer_kv_states = self._apply_depth_gate_to_layer_kv_states(
            layer_kv_states, depth_mask, depth_gate
        )
        layer_kv_states = layer_kv_states
        inference_gc_mask = self._get_token_set_mask(input_ids, self._depth_code_token_ids)
        inference_fast_mask = self._get_token_set_mask(input_ids, self._action_code_token_ids)
        spectral_reader_layer_kv_states, encoder_attention_mask, spectral_condition_mask = (
            layer_kv_states,
            encoder_attention_mask,
            None,
        )
        gc_hidden_trace_path = ""
        trace_depth_mask = depth_mask
        gcvla_query_mask = self._get_depth_token_mask(input_ids, raw_encoder_attention_mask)
        steps = num_steps or self.config.flow_matching_num_steps
        sample_source = layer_states[0] if layer_states is not None else layer_kv_states[0][0]
        batch_size = sample_source.shape[0]
        device = sample_source.device
        trajectory = torch.randn(
            (batch_size, self.config.action_horizon, self.config.max_action_dim),
            device=device,
            generator=generator,
        )
        trajectory = self._mask_action_dim_tensor(
            trajectory,
            action_dim_is_pad=action_dim_is_pad,
            enabled=self.config.mask_action_dim_padding,
        )
        dt = 1.0 / steps
        layer_states, layer_kv_states = self._maybe_detach_action_expert_condition(
            layer_states, layer_kv_states
        )
        self._load_gcrf_residual_adapter(device)
        adapter_device = (
            next(self._gcrf_residual_adapter.parameters()).device
            if self._gcrf_residual_adapter is not None
            else device
        )
        gcrf_gc_summary = (
            self._gcrf_compact_gc(layer_kv_states, adapter_device)
            if self._gcrf_residual_adapter is not None
            else None
        )
        self._gcrf_onpolicy_latent = None
        if not hasattr(self, "_gcrf_onpolicy_replans"):
            self._gcrf_onpolicy_replans = 0
        gcrf_onpolicy_replan_handled = False
        for i in range(steps):
            t = torch.full((batch_size,), i / steps, device=device)
            trajectory = self._mask_action_dim_tensor(
                trajectory,
                action_dim_is_pad=action_dim_is_pad,
                enabled=self.config.mask_action_dim_padding,
            )
            action_expert_kwargs = dict(
                encoder_kv_states=layer_kv_states,
                gcvla_reader_encoder_kv_states=None,
                encoder_attention_mask=encoder_attention_mask,
                state_embeddings=states,
                gcvla_query_mask=gcvla_query_mask,
            )
            velocity = action_expert(trajectory, t, **action_expert_kwargs)
            velocity = self._mask_action_dim_tensor(
                velocity,
                action_dim_is_pad=action_dim_is_pad,
                enabled=self.config.mask_action_dim_padding,
            )
            adapter = self._gcrf_residual_adapter
            if (
                adapter is not None
                and _env_flag("MOLMOACT2_GCRF_ONPOLICY")
                and (not gcrf_onpolicy_replan_handled)
                and (gcrf_gc_summary is not None)
            ):
                if self._gcrf_onpolicy_replan_is_active():
                    self._gcrf_sample_onpolicy_replan(
                        adapter,
                        velocity,
                        gcrf_gc_summary,
                        self._gcrf_history(trajectory, adapter_device),
                    )
                else:
                    inactive_context = adapter._context(
                        velocity, gcrf_gc_summary, self._gcrf_history(trajectory, adapter_device)
                    )
                    self._gcrf_advance_inactive_replan(inactive_context)
                gcrf_onpolicy_replan_handled = True
            gcrf_residual = self._gcrf_residual(trajectory, velocity, gcrf_gc_summary, i)
            if gcrf_residual is not None:
                velocity = velocity + gcrf_residual
                velocity = self._mask_action_dim_tensor(
                    velocity,
                    action_dim_is_pad=action_dim_is_pad,
                    enabled=self.config.mask_action_dim_padding,
                )
            trajectory = trajectory + dt * velocity
            trajectory = self._mask_action_dim_tensor(
                trajectory,
                action_dim_is_pad=action_dim_is_pad,
                enabled=self.config.mask_action_dim_padding,
            )
        base_hidden_trace_path = ""
        return trajectory

    def _run_backbone(
        self,
        output_hidden_states: bool,
        collect_layer_hidden_states: bool,
        collect_layer_kv_states: bool,
        collect_layer_input_states: bool = False,
        **forward_kwargs,
    ) -> Tuple[
        OLMoOutput,
        Optional[Sequence[torch.Tensor]],
        Optional[Sequence[Tuple[torch.Tensor, torch.Tensor]]],
    ]:
        kwargs = dict(forward_kwargs)
        kwargs["collect_layer_hidden_states"] = collect_layer_hidden_states
        kwargs["collect_layer_kv_states"] = collect_layer_kv_states
        kwargs["output_hidden_states"] = output_hidden_states or collect_layer_input_states
        original_use_cache = bool(kwargs.get("use_cache", False))
        base_output = super().forward(**kwargs)
        internal = dict(base_output.internal or {})
        layer_states = internal.pop("layer_hidden_states", None)
        if collect_layer_input_states:
            hidden_states = base_output.hidden_states
            if hidden_states is None:
                raise RuntimeError(
                    "Backbone did not return hidden states for input-layer action conditioning."
                )
            num_layers = len(self.transformer.blocks)
            if len(hidden_states) < num_layers:
                raise RuntimeError(
                    f"Backbone returned too few hidden-state tensors for input-layer action conditioning: got {len(hidden_states)}, expected at least {num_layers}."
                )
            layer_states = tuple(hidden_states[: num_layers + 1])
        layer_kv_states = base_output.attn_key_values if collect_layer_kv_states else None
        attn_key_values = base_output.attn_key_values if original_use_cache else None
        if not output_hidden_states:
            base_output = base_output._replace(hidden_states=None)
        base_output = base_output._replace(internal=internal, attn_key_values=attn_key_values)
        return (base_output, layer_states, layer_kv_states)

    def _get_encoder_attention_mask(
        self, input_ids: Optional[torch.Tensor], attention_mask: Optional[torch.Tensor]
    ) -> Optional[torch.Tensor]:
        if attention_mask is not None:
            mask = attention_mask.to(dtype=torch.bool).clone()
        elif input_ids is not None:
            mask = input_ids != -1
        else:
            return None
        if self.config.action_format != "both" or input_ids is None:
            return mask
        eos_id = self._eos_token_id
        if eos_id is not None:
            mask &= input_ids != eos_id
        for batch_idx in range(input_ids.shape[0]):
            row_ids = input_ids[batch_idx]
            row_mask = mask[batch_idx]
            self._mask_discrete_output_span(
                row_ids, row_mask, self._action_start_token_id, self._action_end_token_id
            )
        return mask

    def _get_depth_token_mask(
        self,
        input_ids: Optional[torch.Tensor],
        encoder_attention_mask: Optional[torch.Tensor] = None,
    ) -> Optional[torch.Tensor]:
        if input_ids is None:
            return None
        if not self._depth_gate_token_ids:
            return None
        depth_token_ids = torch.as_tensor(
            self._depth_gate_token_ids, device=input_ids.device, dtype=input_ids.dtype
        )
        depth_mask = (input_ids.unsqueeze(-1) == depth_token_ids).any(dim=-1)
        if encoder_attention_mask is not None:
            depth_mask = depth_mask & encoder_attention_mask.to(
                device=input_ids.device, dtype=torch.bool
            )
        return depth_mask

    @staticmethod
    def _get_token_set_mask(
        input_ids: Optional[torch.Tensor], token_ids: Sequence[int]
    ) -> Optional[torch.Tensor]:
        if input_ids is None or not token_ids:
            return None
        ids = torch.as_tensor(token_ids, device=input_ids.device, dtype=input_ids.dtype)
        return (input_ids.unsqueeze(-1) == ids).any(dim=-1)

    def _get_gc_action_plan_mask(
        self,
        input_ids: Optional[torch.Tensor],
        encoder_attention_mask: Optional[torch.Tensor] = None,
    ) -> Optional[torch.Tensor]:
        """Select the pre-FAST token whose hidden state has consumed all GC tokens."""
        if input_ids is None or self._action_start_token_id is None:
            return None
        plan_mask = input_ids == self._action_start_token_id
        if encoder_attention_mask is not None:
            plan_mask = plan_mask & encoder_attention_mask.to(
                device=input_ids.device, dtype=torch.bool
            )
        return plan_mask

    def _drop_depth_tokens_from_action_expert_mask(
        self, input_ids: Optional[torch.Tensor], encoder_attention_mask: Optional[torch.Tensor]
    ) -> Optional[torch.Tensor]:
        if not _env_flag("MOLMOACT2_ACTION_EXPERT_DROP_DEPTH_TOKENS"):
            return encoder_attention_mask
        depth_mask = self._get_depth_token_mask(input_ids, encoder_attention_mask)
        if depth_mask is None:
            return encoder_attention_mask
        if encoder_attention_mask is None:
            keep_mask = torch.ones_like(depth_mask, dtype=torch.bool, device=depth_mask.device)
        else:
            keep_mask = encoder_attention_mask.to(device=depth_mask.device, dtype=torch.bool)
        return keep_mask & ~depth_mask

    def _depth_gate_from_source(
        self,
        gate_head: nn.Linear,
        *,
        source: torch.Tensor,
        depth_mask: torch.Tensor,
        encoder_attention_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if source.ndim == 4:
            source = source.reshape(source.shape[0], source.shape[1], -1)
        if source.ndim != 3:
            raise ValueError(
                f"Depth gate expected a sequence tensor with 3 dims after flattening, got {tuple(source.shape)}."
            )
        if source.shape[:2] != depth_mask.shape:
            raise ValueError(
                f"Depth gate conditioning shape mismatch: condition={tuple(source.shape)}, depth_mask={tuple(depth_mask.shape)}."
            )
        if encoder_attention_mask is not None:
            valid_mask = encoder_attention_mask.to(device=source.device, dtype=torch.bool)
        else:
            valid_mask = torch.ones(depth_mask.shape, device=source.device, dtype=torch.bool)
        depth_mask = depth_mask.to(device=source.device, dtype=torch.bool)
        pool_mask = valid_mask & ~depth_mask
        has_pool = pool_mask.any(dim=-1, keepdim=True)
        pool_mask = torch.where(has_pool, pool_mask, valid_mask)
        weights = pool_mask.to(dtype=source.dtype).unsqueeze(-1)
        denom = weights.sum(dim=1).clamp_min(1.0)
        pooled = (source * weights).sum(dim=1) / denom
        gate_logits = gate_head(pooled.to(dtype=gate_head.weight.dtype))
        return torch.sigmoid(gate_logits).to(dtype=source.dtype)

    def _depth_gate_from_condition(
        self,
        *,
        input_ids: Optional[torch.Tensor],
        encoder_attention_mask: Optional[torch.Tensor],
        layer_states: Optional[Sequence[torch.Tensor]],
        layer_kv_states: Optional[Sequence[Tuple[torch.Tensor, torch.Tensor]]],
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        gate_head = self.action_expert_depth_gate
        if gate_head is None:
            return (None, None)
        depth_mask = self._get_depth_token_mask(input_ids, encoder_attention_mask)
        if depth_mask is None:
            return (None, depth_mask)
        if layer_states is not None:
            sources = list(layer_states)
        elif layer_kv_states is not None:
            sources = [value for _, value in layer_kv_states]
        else:
            return (None, depth_mask)
        if isinstance(gate_head, nn.ModuleList):
            if len(gate_head) != len(sources):
                raise ValueError(
                    f"Per-layer depth gate count mismatch: gates={len(gate_head)}, condition_layers={len(sources)}."
                )
            gates = [
                self._depth_gate_from_source(
                    gate,
                    source=source,
                    depth_mask=depth_mask,
                    encoder_attention_mask=encoder_attention_mask,
                )
                for gate, source in zip(gate_head, sources)
            ]
            return (gates, depth_mask)
        if not isinstance(gate_head, nn.Linear):
            raise TypeError(
                f"Expected depth gate to be nn.Linear or nn.ModuleList, got {type(gate_head).__name__}."
            )
        gate = self._depth_gate_from_source(
            gate_head,
            source=sources[-1],
            depth_mask=depth_mask,
            encoder_attention_mask=encoder_attention_mask,
        )
        return (gate, depth_mask)

    @staticmethod
    def _depth_gate_for_layer(
        gate: torch.Tensor | Sequence[torch.Tensor], layer_idx: int, *, num_layers: int
    ) -> torch.Tensor:
        if isinstance(gate, torch.Tensor):
            return gate
        if len(gate) != num_layers:
            raise ValueError(
                f"Depth gate layer count mismatch: gates={len(gate)}, layers={num_layers}."
            )
        return gate[layer_idx]

    @staticmethod
    def _mean_depth_gate(gate: torch.Tensor | Sequence[torch.Tensor]) -> torch.Tensor:
        if isinstance(gate, torch.Tensor):
            return gate.mean()
        if not gate:
            raise ValueError("Cannot compute mean for an empty depth gate sequence.")
        return torch.stack([layer_gate.mean() for layer_gate in gate]).mean()

    def _apply_depth_gate_to_layer_states(
        self,
        layer_states: Optional[Sequence[torch.Tensor]],
        depth_mask: Optional[torch.Tensor],
        gate: Optional[torch.Tensor | Sequence[torch.Tensor]],
    ) -> Optional[Sequence[torch.Tensor]]:
        if layer_states is None or depth_mask is None or gate is None:
            return layer_states
        gated_states = []
        for layer_idx, hidden in enumerate(layer_states):
            layer_gate = self._depth_gate_for_layer(gate, layer_idx, num_layers=len(layer_states))
            mask = depth_mask.to(device=hidden.device, dtype=torch.bool)
            scale = torch.ones((*mask.shape, 1), device=hidden.device, dtype=hidden.dtype)
            scale = torch.where(
                mask.unsqueeze(-1),
                layer_gate.to(device=hidden.device, dtype=hidden.dtype).view(-1, 1, 1),
                scale,
            )
            gated_states.append(hidden * scale)
        return gated_states

    def _apply_depth_gate_to_layer_kv_states(
        self,
        layer_kv_states: Optional[Sequence[Tuple[torch.Tensor, torch.Tensor]]],
        depth_mask: Optional[torch.Tensor],
        gate: Optional[torch.Tensor | Sequence[torch.Tensor]],
    ) -> Optional[Sequence[Tuple[torch.Tensor, torch.Tensor]]]:
        if layer_kv_states is None or depth_mask is None or gate is None:
            return layer_kv_states
        gated_kv = []
        for layer_idx, (key, value) in enumerate(layer_kv_states):
            layer_gate = self._depth_gate_for_layer(
                gate, layer_idx, num_layers=len(layer_kv_states)
            )
            mask = depth_mask.to(device=key.device, dtype=torch.bool)
            view_shape = [mask.shape[0], mask.shape[1]] + [1] * (key.ndim - 2)
            scale = torch.ones(view_shape, device=key.device, dtype=key.dtype)
            gate_view = layer_gate.to(device=key.device, dtype=key.dtype).view(
                layer_gate.shape[0], *[1] * (key.ndim - 1)
            )
            scale = torch.where(mask.view(view_shape), gate_view, scale)
            gated_kv.append((key * scale, value * scale))
        return gated_kv

    @staticmethod
    def _mask_discrete_output_span(
        row_ids: torch.Tensor,
        row_mask: torch.Tensor,
        start_id: Optional[int],
        end_id: Optional[int],
    ) -> None:
        if start_id is None or end_id is None:
            return
        start_positions = (row_ids == start_id).nonzero(as_tuple=False).flatten().tolist()
        if not start_positions:
            return
        end_positions = (row_ids == end_id).nonzero(as_tuple=False).flatten().tolist()
        end_ptr = 0
        for start_pos in start_positions:
            while end_ptr < len(end_positions) and end_positions[end_ptr] < start_pos:
                end_ptr += 1
            if end_ptr >= len(end_positions):
                row_mask[start_pos:] = False
                break
            end_pos = end_positions[end_ptr]
            row_mask[start_pos : end_pos + 1] = False
            end_ptr += 1

    def _cache_to_sequence(self, cache: torch.Tensor) -> torch.Tensor:
        if cache.dim() != 4:
            raise ValueError(
                f"Expected KV cache tensor with 4 dims, got shape {tuple(cache.shape)}"
            )
        head_candidates = {self.config.llm.effective_n_kv_heads, self.config.llm.n_heads}
        if cache.shape[1] in head_candidates:
            bsz, n_heads, seq_len, head_dim = cache.shape
            return cache.permute(0, 2, 1, 3).reshape(bsz, seq_len, n_heads * head_dim)
        if cache.shape[2] in head_candidates:
            bsz, seq_len, n_heads, head_dim = cache.shape
            return cache.reshape(bsz, seq_len, n_heads * head_dim)
        if cache.shape[1] <= cache.shape[2]:
            bsz, n_heads, seq_len, head_dim = cache.shape
            return cache.permute(0, 2, 1, 3).reshape(bsz, seq_len, n_heads * head_dim)
        bsz, seq_len, n_heads, head_dim = cache.shape
        return cache.reshape(bsz, seq_len, n_heads * head_dim)

    def _select_layer_kv_states(
        self, layer_kv_states: Sequence[Tuple[torch.Tensor, torch.Tensor]]
    ) -> Sequence[Tuple[torch.Tensor, torch.Tensor]]:
        if not layer_kv_states:
            raise ValueError("No layer KV states provided for action expert conditioning.")
        action_expert = self._require_action_expert()
        kv_seq = [
            (self._cache_to_sequence(k), self._cache_to_sequence(v)) for k, v in layer_kv_states
        ]
        num_target = len(action_expert.blocks)
        num_available = len(kv_seq)
        if num_available != num_target:
            raise ValueError(
                f"Expected {num_target} KV states, received {num_available} with per_layer mode."
            )
        return kv_seq

    def _chunk_attention_mask(
        self,
        encoder_attention_mask: Optional[torch.Tensor],
        subsegment_ids: Optional[torch.Tensor],
        packed_batch_idx: torch.Tensor,
        packed_example_ids: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        if encoder_attention_mask is not None:
            mask = encoder_attention_mask.index_select(0, packed_batch_idx)
        else:
            mask = None
        if subsegment_ids is None:
            return mask
        example_assignments = (
            subsegment_ids.index_select(0, packed_batch_idx) // EXAMPLE_SUBSEGMENT_INCREMENT
        )
        chunk_examples = packed_example_ids.view(-1, 1)
        chunk_mask = example_assignments == chunk_examples
        if mask is None:
            return chunk_mask
        return chunk_mask & mask

    def _maybe_detach_action_expert_condition(
        self,
        layer_states: Optional[Sequence[torch.Tensor]],
        layer_kv_states: Optional[Sequence[Tuple[torch.Tensor, torch.Tensor]]],
    ) -> Tuple[
        Optional[Sequence[torch.Tensor]], Optional[Sequence[Tuple[torch.Tensor, torch.Tensor]]]
    ]:
        if not self.config.action_expert_detach_vlm:
            return (layer_states, layer_kv_states)
        detached_layer_states = None
        detached_layer_kv_states = None
        if layer_states is not None:
            detached_layer_states = [hidden.detach() for hidden in layer_states]
        if layer_kv_states is not None:
            detached_layer_kv_states = [
                (key.detach(), value.detach()) for key, value in layer_kv_states
            ]
        return (detached_layer_states, detached_layer_kv_states)

    @staticmethod
    def _action_dim_valid_mask(
        target: torch.Tensor, action_dim_is_pad: Optional[torch.Tensor]
    ) -> Optional[torch.Tensor]:
        if action_dim_is_pad is None:
            return None
        mask = ~action_dim_is_pad.to(device=target.device, dtype=torch.bool)
        if mask.ndim == 1:
            mask = mask.unsqueeze(0)
        if mask.shape[-1] != target.shape[-1]:
            raise ValueError(
                f"action_dim_is_pad width does not match target width: mask={mask.shape[-1]}, target={target.shape[-1]}."
            )
        if mask.shape[0] == 1 and target.shape[0] != 1:
            mask = mask.expand(target.shape[0], -1)
        if mask.shape[0] != target.shape[0]:
            raise ValueError(
                f"action_dim_is_pad batch size does not match target batch size: mask={mask.shape[0]}, target={target.shape[0]}."
            )
        while mask.ndim < target.ndim:
            mask = mask.unsqueeze(1)
        return mask

    @classmethod
    def _mask_action_dim_tensor(
        cls, tensor: torch.Tensor, *, action_dim_is_pad: Optional[torch.Tensor], enabled: bool
    ) -> torch.Tensor:
        if not enabled:
            return tensor
        valid_mask = cls._action_dim_valid_mask(tensor, action_dim_is_pad)
        if valid_mask is None:
            return tensor
        return tensor.masked_fill(~valid_mask, 0)
