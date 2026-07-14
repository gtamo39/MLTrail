"""Model backends: load/save an artifact and expose its features, dispatched by framework.

sklearn is fully supported, including the ``{'model', 'feature_cols', ...}`` dict format
used by the H236 production models. chemprop is a documented stub (and lives in a separate
conda env). Generative models never reach a backend — they are blocked in `predict`.
"""
import shutil
from pathlib import Path

import joblib

# Frameworks whose artifacts are joblib-serialized estimators (optionally wrapped in a dict).
JOBLIB_FRAMEWORKS = {"sklearn", "xgboost", "lightgbm"}


def save_model(model, dest_dir, model_id, version):
    """Materialize a model artifact into the vault as ``<id>_v<version>`` and return its path.

    `model` is either a filesystem path (an existing artifact copied verbatim — a file keeps
    its extension, a directory is copytree'd) or an in-memory object (joblib.dump'ed to
    ``<id>_v<version>.joblib``). `dest_dir` is created if absent.
    """
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{model_id}_v{version}"
    if isinstance(model, (str, Path)):
        src = Path(model)
        if not src.exists():
            raise FileNotFoundError(f"model artifact not found: {src}")
        if src.is_dir():
            dest = dest_dir / stem
            if dest.exists():
                shutil.rmtree(dest)
            shutil.copytree(src, dest)
        else:
            dest = dest_dir / f"{stem}{src.suffix}"
            shutil.copy2(src, dest)
    else:
        dest = dest_dir / f"{stem}.joblib"
        joblib.dump(model, dest)
    return str(dest)


def load_model(framework, model_path):
    """Load a model artifact. Returns (estimator, feature_cols) where feature_cols is None
    for a bare estimator, or the trained column order for a `{'model', 'feature_cols'}` dict.
    """
    if framework in JOBLIB_FRAMEWORKS:
        obj = joblib.load(model_path)
        if isinstance(obj, dict) and "model" in obj:
            return obj["model"], obj.get("feature_cols")
        return obj, None
    if framework == "chemprop":
        raise NotImplementedError(
            "chemprop prediction is not implemented in v1 (and requires the `reinvent` "
            "conda env — chemprop is not installed in `ML`)."
        )
    raise ValueError(f"unknown framework {framework!r}")


def align_features(feat_df, feature_cols):
    """Drop 'compound' and, if feature_cols is given, select/reorder to the trained columns.

    Missing columns are filled with 0 (absent fingerprint bits); output is float32 to match
    how the production models were trained.
    """
    X = feat_df.drop(columns=["compound"], errors="ignore")
    if feature_cols is not None:
        X = X.reindex(columns=feature_cols, fill_value=0)
    return X.astype("float32")
