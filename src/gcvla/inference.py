"""Inference on observations using the bundled runtime."""

import os
import sys

from gcvla.checkpoint import prepare_checkpoint
from gcvla.contracts import assert_no_future_gc_at_inference
from gcvla.runner import environment
from gcvla.runtime_compat import register_public_model_name


def load_policy(config):
    if "torch" in sys.modules:
        raise RuntimeError("Configure the inference process before importing torch")
    prepare_checkpoint(config.checkpoint, config.cache)
    env = environment(config, config.output_dir)
    for key in list(os.environ):
        if key.startswith(("MOLMOACT2_", "GCVLA_", "LIBERO_", "LEROBOT_", "REUSE")):
            os.environ.pop(key)
    os.environ.update(env)
    sys.path[:0] = env["PYTHONPATH"].split(os.pathsep)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    register_public_model_name()
    from lerobot.policies.molmoact2.configuration_molmoact2 import MolmoAct2Config
    from lerobot.policies.molmoact2.modeling_molmoact2 import MolmoAct2Policy

    policy = MolmoAct2Policy(
        MolmoAct2Config(
            checkpoint_path=str(config.cache / "gc-vla"),
            device="cuda",
            norm_tag="libero",
            num_steps=10,
            inference_dtype="float32",
            enable_depth_reasoning=True,
            num_depth_tokens_per_image=200,
            enable_inference_cuda_graph=False,
        )
    )
    return ObservationPolicy(policy.eval().requires_grad_(False))


class ObservationPolicy:
    def __init__(self, policy):
        self._policy = policy

    def reset(self, *, episode_seeds):
        seeds = list(episode_seeds)
        if not seeds or any(type(seed) is not int or seed < 0 for seed in seeds):
            raise ValueError("Provide one nonnegative integer episode seed per batch slot")
        self._policy.reset()
        self._policy._gcrf_trace_metadata = {"episode_seeds": seeds}

    def select_action(self, observation):
        assert_no_future_gc_at_inference(observation)
        import torch

        with torch.inference_mode():
            return self._policy.select_action(observation)
