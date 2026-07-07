"""Model backends: load an artifact and expose its features, dispatched by framework.

sklearn is fully supported, including the ``{'model', 'feature_cols', ...}`` dict format
used by the H236 production models. chemprop is a documented stub (and lives in a separate
conda env). Generative models never reach a backend — they are blocked in `predict`.
"""
import joblib

# Frameworks whose artifacts are joblib-serialized estimators (optionally wrapped in a dict).
JOBLIB_FRAMEWORKS = {"sklearn", "xgboost", "lightgbm"}


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
