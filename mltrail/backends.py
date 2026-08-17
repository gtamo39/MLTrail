"""Model backends: load/save an artifact and expose its features, dispatched by framework.

sklearn is fully supported, including the ``{'model', 'feature_cols', ...}`` dict format
used by the H236 production models. chemprop models are stored as a checkpoint directory and
predicted by shelling out to a chemprop-env CLI (chemprop is not a dependency of MLTrail's own
env) — see ``chemprop_predict``. Generative models never reach a backend — blocked in `predict`.
"""
import shutil
import subprocess
import tempfile
from pathlib import Path

import joblib
import pandas as pd

# Frameworks whose artifacts are joblib-serialized estimators (optionally wrapped in a dict).
JOBLIB_FRAMEWORKS = {"sklearn", "xgboost", "lightgbm"}

# chemprop checkpoint patterns, most-preferred first (the exported `best.pt` over raw lightning ckpts).
CHEMPROP_CKPT_GLOBS = ("**/best*.pt", "**/*.pt", "**/best*.ckpt", "**/*.ckpt")


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
        # chemprop is not loaded in-process (its Python inference API is version-fragile and it may
        # not be importable in the caller's env). predict.py routes chemprop through chemprop_predict,
        # which shells out to the chemprop CLI — load_model is never called for it.
        raise NotImplementedError(
            "chemprop models are predicted via the CLI (backends.chemprop_predict), not load_model"
        )
    raise ValueError(f"unknown framework {framework!r}")


def resolve_checkpoint(model_path):
    """Resolve a chemprop checkpoint file from a registered model_path.

    Accepts a checkpoint file directly, or a chemprop model directory (as saved by the training
    run) under which the exported ``best*.pt`` is preferred over raw lightning ``*.ckpt``.
    """
    p = Path(model_path)
    if p.is_file():
        return p
    for pattern in CHEMPROP_CKPT_GLOBS:
        hits = sorted(p.glob(pattern), key=lambda f: f.stat().st_mtime)
        if hits:
            return hits[-1]
    raise FileNotFoundError(f"no chemprop checkpoint (.pt/.ckpt) found under {p}")


def chemprop_predict(model_path, smiles, config, target_columns=None):
    """Predict with a chemprop model by shelling out to the chemprop CLI; return a DataFrame.

    Writes the SMILES to a temp CSV, runs ``chemprop predict`` with the resolved checkpoint, and
    reads the predictions back (one column per target). `config['chemprop']` supplies the CLI path
    (``cli``, default ``chemprop`` on PATH) and optional ``accelerator`` (cpu|gpu|auto). Output
    columns are renamed to `target_columns` when their count matches. Row order is preserved.
    """
    cp_cfg = config.get("chemprop") or {}
    cli = cp_cfg.get("cli") or "chemprop"
    checkpoint = resolve_checkpoint(model_path)
    with tempfile.TemporaryDirectory() as tmp:
        test_in, preds_out = Path(tmp) / "in.csv", Path(tmp) / "out.csv"
        pd.DataFrame({"smiles": list(smiles)}).to_csv(test_in, index=False)
        cmd = [cli, "predict", "-i", str(test_in), "-s", "smiles",
               "--model-path", str(checkpoint), "--preds-path", str(preds_out)]
        if cp_cfg.get("accelerator"):
            cmd += ["--accelerator", str(cp_cfg["accelerator"])]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0 or not preds_out.exists():
            raise RuntimeError(f"chemprop predict failed (exit {proc.returncode}):\n{proc.stderr[-2000:]}")
        out = pd.read_csv(preds_out)
    preds = out.drop(columns=[c for c in ("smiles",) if c in out.columns]).reset_index(drop=True)
    if target_columns and len(target_columns) == preds.shape[1]:
        preds.columns = list(target_columns)
    return preds


def align_features(feat_df, feature_cols):
    """Drop 'compound' and, if feature_cols is given, select/reorder to the trained columns.

    Missing columns are filled with 0 (absent fingerprint bits); output is float32 to match
    how the production models were trained.
    """
    X = feat_df.drop(columns=["compound"], errors="ignore")
    if feature_cols is not None:
        X = X.reindex(columns=feature_cols, fill_value=0)
    return X.astype("float32")
