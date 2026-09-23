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

from __future__ import annotations
from lerobot.configs.types import FeatureType


def raise_feature_mismatch_error(
    provided_features: set[str],
    expected_features: set[str],
) -> None:
    """
    Raises a standardized ValueError for feature mismatches between dataset/environment and policy config.
    """
    missing = expected_features - provided_features
    extra = provided_features - expected_features
    # TODO (jadechoghari): provide a dynamic rename map suggestion to the user.
    raise ValueError(
        f"Feature mismatch between dataset/environment and policy config.\n"
        f"- Missing features: {sorted(missing) if missing else 'None'}\n"
        f"- Extra features: {sorted(extra) if extra else 'None'}\n\n"
        f"Please ensure your dataset and policy use consistent feature names.\n"
        f"If your dataset uses different observation keys (e.g., cameras named differently), "
        f"use the `--rename_map` argument, for example:\n"
        f'  --rename_map=\'{{"observation.images.left": "observation.images.camera1", '
        f'"observation.images.top": "observation.images.camera2"}}\''
    )


def validate_visual_features_consistency(
    cfg: PreTrainedConfig,
    features: dict[str, PolicyFeature],
) -> None:
    """
    Validates visual feature consistency between a policy config and provided dataset/environment features.

    Validation passes if EITHER:
    - Policy's expected visuals are a subset of dataset (policy uses some cameras, dataset has more)
    - Dataset's provided visuals are a subset of policy (policy declares extras for flexibility)

    Args:
        cfg (PreTrainedConfig): The model or policy configuration containing input_features and type.
        features (Dict[str, PolicyFeature]): A mapping of feature names to PolicyFeature objects.
    """
    expected_visuals = {k for k, v in cfg.input_features.items() if v.type == FeatureType.VISUAL}
    provided_visuals = {k for k, v in features.items() if v.type == FeatureType.VISUAL}

    # Accept if either direction is a subset
    policy_subset_of_dataset = expected_visuals.issubset(provided_visuals)
    dataset_subset_of_policy = provided_visuals.issubset(expected_visuals)

    if not (policy_subset_of_dataset or dataset_subset_of_policy):
        raise_feature_mismatch_error(provided_visuals, expected_visuals)

