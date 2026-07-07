"""Prediction orchestration: read dataset -> featurize -> load model -> align -> predict.

Output columns: smiles, compound, then prediction column(s) (+ probability for
classification). Rows whose SMILES fail to parse get null predictions.
"""
import numpy as np
import pandas as pd

from .backends import align_features, load_model
from .featurizers import get_featurizer
from .readers import read_dataset
from .schema import is_generative


def predict(registry, model_id, dataset, smiles_column=None, compound_id=None,
            pred_output=None, model_path=None, config=None):
    """Predict properties for a dataset using a registered model; returns a DataFrame.

    Reads `dataset`, featurizes per the model's features_type, aligns feature columns to
    those the model was trained on, predicts, and (if `pred_output` given) writes a CSV.
    `model_path` overrides the registered path (for folder-style artifacts). Raises
    ValueError for generative models, which MLTrail stores as metadata only.
    """
    entry = registry.details(model_id)
    if is_generative(entry["model_type"]):
        raise ValueError(
            f"model {model_id} is generative ({entry['model_type']}); MLTrail stores its "
            "metadata only — predict/generate is not possible."
        )

    df = read_dataset(dataset, smiles_column, compound_id).reset_index(drop=True)
    df["_key"] = np.arange(len(df))

    featurize = get_featurizer(entry["features_type"], config or {})
    feats = featurize(pd.DataFrame({"compound": df["_key"].values, "smiles": df["smiles"].values}))

    model, feature_cols = load_model(entry["framework"], model_path or entry["model_path"])
    X = align_features(feats, feature_cols)

    preds = _format_predictions(entry["model_type"], model, X)
    preds.insert(0, "_key", feats["compound"].values)

    out = df.merge(preds, on="_key", how="left").drop(columns="_key")
    prediction_cols = [c for c in out.columns if c not in ("smiles", "compound")]
    out = out[["smiles", "compound", *prediction_cols]]
    if pred_output:
        out.to_csv(pred_output, index=False)
    return out


def _format_predictions(model_type, model, X):
    """Build a DataFrame of prediction column(s) for the valid rows, shaped by model_type."""
    result = pd.DataFrame(index=range(len(X)))
    if model_type == "single_task_classification":
        result["prediction"] = model.predict(X)
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X)
            result["probability"] = proba[:, 1] if proba.shape[1] == 2 else proba.max(axis=1)
    elif model_type in ("multitask_regression", "multitask_classification"):
        preds = np.asarray(model.predict(X))
        preds = preds[:, None] if preds.ndim == 1 else preds
        for j in range(preds.shape[1]):
            result[f"pred_{j}"] = preds[:, j]
    else:  # single_task_regression and any other single-output model
        result["prediction"] = model.predict(X)
    return result.reset_index(drop=True)
