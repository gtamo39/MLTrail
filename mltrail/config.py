"""Configuration loading for MLTrail. All tuneable paths/params live in config.yaml."""
from pathlib import Path

import yaml

DEFAULTS = {
    "registry_path": "data/registry.json",
    "trained_models_dir": "data/models",
    "training_sets_dir": "data/training_sets",
    "date_format": "%Y%m%d_%H%M%S",
    # Empty by default -> MLTrail's built-in featurizers are used (no external module).
    # Configure a featurizers.module + map to override with an external featurizer.
    "featurizers": {},
    # chemprop models are predicted by shelling out to a chemprop-env CLI (chemprop is not a
    # dependency of MLTrail's own env). Empty -> `chemprop` on PATH, auto accelerator.
    "chemprop": {},
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


# MLTrail's own config.yaml, resolved package-relative so the same vault is found from any CWD.
PACKAGE_CONFIG = Path(__file__).resolve().parent.parent / "config" / "config.yaml"


def default_config():
    """The default config when no explicit path is given (CLI without --config, notebooks).

    Loads MLTrail's own package-relative ``config/config.yaml`` — so the same vault resolves from
    any working directory — falling back to built-in defaults if that file is missing.
    """
    return load_config(str(PACKAGE_CONFIG)) if PACKAGE_CONFIG.exists() else load_config(None)
