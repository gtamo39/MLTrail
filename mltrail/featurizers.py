"""Featurizer resolution: map a features_type string to a callable.

The callable's contract is ``df[compound, smiles] -> df[compound, <feature cols>]``.
Featurizers are resolved from an external module named in the config (e.g. the shared
``Rdkit_tools``) rather than reimplemented, so features match exactly how each model
was trained — this prevents train/predict skew.
"""
import importlib
import sys


def get_featurizer(features_type, config):
    """Return the featurizer callable registered for `features_type`.

    Input: a features_type string and a config dict with a `featurizers` section
    (`path`, `module`, `map`). Raises KeyError for an unregistered features_type
    and ImportError/AttributeError if the module or function cannot be resolved.
    """
    fcfg = config.get("featurizers", {})
    mapping = fcfg.get("map", {})
    if features_type not in mapping:
        raise KeyError(
            f"no featurizer registered for features_type {features_type!r}; "
            f"known: {sorted(mapping)}"
        )
    path = fcfg.get("path")
    if path and path not in sys.path:
        sys.path.insert(0, path)
    module = importlib.import_module(fcfg["module"])
    return getattr(module, mapping[features_type])
