# Modifications by Tommoro: public runtime extraction and adaptation.
# This file differs from its original source; see manifest.json for source hashes.

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.utils.constants import ACTION


@PreTrainedConfig.register_subclass("molmoact2")
@dataclass
class MolmoAct2Config(PreTrainedConfig):
    """
    Lightweight config wrapper for the bundled MolmoAct2 policy runtime.

    Action generation uses the resolved MolmoAct2 checkpoint. Required fields:
      - `checkpoint_path`: path to a MolmoAct2 checkpoint directory (unsharded preferred).

    Optional:
      - `seq_len`: max token length for the Molmo tokenizer/collator.
      - `num_steps`: flow-matching integration steps for action generation.
    """

    checkpoint_path: str = "allenai/MolmoAct2"
    seq_len: Optional[int] = None
    num_steps: Optional[int] = None
    # Inference action mode:
    # - "continuous": flow-matching action expert (generate_actions)
    # - "discrete": autoregressive token generation + discrete action decode
    inference_action_mode: str = "continuous"
    # Required when inference_action_mode="discrete".
    discrete_action_tokenizer: Optional[str] = "allenai/MolmoAct2-FAST-Tokenizer"
    enable_depth_reasoning: bool = False
    enable_inference_cuda_graph: bool = True
    # Controls checkpoint load dtype for inference. Use bfloat16/float16 only
    # when VRAM pressure matters; float32 is fastest on the current local path.
    inference_dtype: str = "float32"
    num_depth_tokens_per_image: Optional[int] = None
    verbose: bool = False
    norm_tag: str = ""

    # Provide minimal feature metadata to satisfy the policy factory. These will be
    # overridden at runtime by `make_policy` if dataset/env features are available.
    input_features: dict[str, PolicyFeature] = field(default_factory=dict)
    output_features: dict[str, PolicyFeature] = field(
        default_factory=lambda: {ACTION: PolicyFeature(type=FeatureType.ACTION, shape=[7])}
    )

    def __post_init__(self) -> None:
        super().__post_init__()
        self.inference_action_mode = str(self.inference_action_mode or "continuous").strip().lower()
        if self.inference_action_mode not in {"continuous", "discrete"}:
            raise ValueError(
                f"Unsupported inference_action_mode={self.inference_action_mode!r}. "
                "Expected one of {'continuous', 'discrete'}."
            )
        if self.seq_len is not None and self.seq_len < 1:
            raise ValueError(f"seq_len must be >= 1 or None, got {self.seq_len}.")
        self.inference_dtype = str(self.inference_dtype or "auto").strip().lower()
        valid_dtypes = {"auto", "bfloat16", "bf16", "float16", "fp16", "half", "float32", "fp32", "full"}
        if self.inference_dtype not in valid_dtypes:
            raise ValueError(
                f"Unsupported inference_dtype={self.inference_dtype!r}. "
                f"Expected one of {sorted(valid_dtypes)}."
            )

    @property
    def observation_delta_indices(self):
        return None

    @property
    def action_delta_indices(self):
        return None

    @property
    def reward_delta_indices(self):
        return None



    def validate_features(self) -> None:
        # The bundled wrapper has no additional configuration checks.
        return

    @property
    def checkpoint_dir(self) -> Path:
        return Path(self.checkpoint_path)
