"""Configuration loading for MLTrail. All tuneable paths/params live in config.yaml."""
from pathlib import Path

import yaml

DEFAULTS = {
    "registry_path": "data/registry.json",
    "training_sets_dir": "data/training_sets",
    "date_format": "%Y%m%d_%H%M%S",
    "featurizers": {
        "path": "/home/gtamo/Scripts",
        "module": "Rdkit_tools",
        "map": {"MF_2048": "get_MF_bits_from_df", "H236": "compute_H236_features"},
    },
}


def load_config(path=None):
    """Load a MLTrail config, layering the YAML file (if given) over built-in defaults.

    Input: path to a config.yaml (or None to use defaults only).
    Output: a dict of config values with every default key guaranteed present.
    """
    config = dict(DEFAULTS)
    if path is not None:
        loaded = yaml.safe_load(Path(path).read_text()) or {}
        config.update(loaded)
    return config
