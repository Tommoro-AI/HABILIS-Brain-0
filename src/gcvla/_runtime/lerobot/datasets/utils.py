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



def flatten_dict(d: dict, parent_key: str = "", sep: str = "/") -> dict:
    """Flatten a nested dictionary by joining keys with a separator.

    Example:
        >>> dct = {"a": {"b": 1, "c": {"d": 2}}, "e": 3}
        >>> print(flatten_dict(dct))
        {'a/b': 1, 'a/c/d': 2, 'e': 3}

    Args:
        d (dict): The dictionary to flatten.
        parent_key (str): The base key to prepend to the keys in this level.
        sep (str): The separator to use between keys.

    Returns:
        dict: A flattened dictionary.
    """
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def unflatten_dict(d: dict, sep: str = "/") -> dict:
    """Unflatten a dictionary with delimited keys into a nested dictionary.

    Example:
        >>> flat_dct = {"a/b": 1, "a/c/d": 2, "e": 3}
        >>> print(unflatten_dict(flat_dct))
        {'a': {'b': 1, 'c': {'d': 2}}, 'e': 3}

    Args:
        d (dict): A dictionary with flattened keys.
        sep (str): The separator used in the keys.

    Returns:
        dict: A nested dictionary.
    """
    outdict = {}
    for key, value in d.items():
        parts = key.split(sep)
        d = outdict
        for part in parts[:-1]:
            if part not in d:
                d[part] = {}
            d = d[part]
        d[parts[-1]] = value
    return outdict

