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

from lerobot.configs.types import FeatureType
from lerobot.envs.utils import env_to_policy_features
from lerobot.policies.molmoact2.modeling_molmoact2 import MolmoAct2Policy
from lerobot.policies.molmoact2.processor_molmoact2 import make_molmoact2_pre_post_processors
from lerobot.policies.utils import validate_visual_features_consistency

def make_policy(cfg, ds_meta=None, env_cfg=None, rename_map=None):
    if cfg.type != "molmoact2" or ds_meta is not None or env_cfg is None:
        raise ValueError("Unsupported policy configuration")
    if cfg.pretrained_path or cfg.use_peft:
        raise ValueError("Unsupported policy initialization")
    features = env_to_policy_features(env_cfg)
    cfg.output_features = {k: v for k, v in features.items() if v.type is FeatureType.ACTION}
    if not cfg.input_features:
        cfg.input_features = {k: v for k, v in features.items() if k not in cfg.output_features}
    policy = MolmoAct2Policy(config=cfg)
    policy.to(cfg.device)
    if not rename_map:
        validate_visual_features_consistency(cfg, features)
    return policy

def make_pre_post_processors(policy_cfg, pretrained_path=None, **kwargs):
    if pretrained_path:
        raise ValueError("Unsupported processor initialization")
    return make_molmoact2_pre_post_processors(policy_cfg)
