# Modifications by Tommoro: public runtime extraction and adaptation.
# This file differs from its original source; see manifest.json for source hashes.

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import gymnasium as gym
from lerobot.envs.libero import create_libero_envs
from lerobot.processor.env_processor import LiberoProcessorStep
from lerobot.processor.pipeline import PolicyProcessorPipeline

def make_env_pre_post_processors(env_cfg, policy_cfg):
    return (PolicyProcessorPipeline(steps=[LiberoProcessorStep()]),
            PolicyProcessorPipeline(steps=[]))

def make_env(cfg, n_envs=1, use_async_envs=False, **kwargs):
    if cfg.type != "libero" or cfg.task is None or n_envs < 1:
        raise ValueError("Unsupported environment configuration")
    return create_libero_envs(
        task=cfg.task, n_envs=n_envs, camera_name=cfg.camera_name,
        init_states=cfg.init_states, gym_kwargs=cfg.gym_kwargs,
        env_cls=gym.vector.AsyncVectorEnv if use_async_envs else gym.vector.SyncVectorEnv,
        control_mode=cfg.control_mode, episode_length=cfg.episode_length)
