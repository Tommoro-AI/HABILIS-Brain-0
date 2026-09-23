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

from torch import nn
from lerobot.configs.policies import PreTrainedConfig

class PreTrainedPolicy(nn.Module):
    def __init__(self, config, *inputs, **kwargs):
        super().__init__()
        if not isinstance(config, PreTrainedConfig):
            raise TypeError("Expected PreTrainedConfig")
        self.config = config

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if not getattr(cls, "config_class", None) or not getattr(cls, "name", None):
            raise TypeError("Policy must declare config_class and name")
