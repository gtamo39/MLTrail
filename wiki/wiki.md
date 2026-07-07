# MLTrail — Project Wiki

Durable memory for this repo. Aggregate facts only (no SMILES / compound IDs / per-compound values).

## Purpose
MLTrail is a local, MLflow-lite **model registry / vault** for trained ML models (sklearn, ChemProp,
REINVENT checkpoints, …). It registers models, versions them, and predicts properties on new datasets.
Usable from the CLI (`mltrail --...`) and imported in notebooks (methods return DataFrames).

## Status (2026-07-06)
- **Registry core: built + smoke-tested** (synthetic metadata). Files: `mltrail/{registry,schema,config}.py`.
- **Predict half: built + smoke-tested** (public SMILES, tiny models). Files: `mltrail/{readers,featurizers,backends,predict}.py`. Verified: MF_2048 + H236 regression (incl. feature_cols reindex), classification (prediction+probability), generative→error, model_path override, invalid-SMILES→null, csv/sdf reading.
- **CLI + packaging: built + tested** — `mltrail/cli.py`, `pyproject.toml` (`mltrail` console entry point). All verbs exercised: add / overwrite / predict / list / details / search / trail / **delete** / **save-trainset**.
- **Training-set delta storage: built + tested** — `mltrail/training_sets.py`. Per-model folder of timestamped parquet chunks; `save_training_set` runs an internal dedup check and writes only rows not already stored (across all versions); `load_training_set` concats chunks → the full set. Needs `pyarrow`.
- **Tests: 32 passing** — stdlib `unittest` (matches Px_interface style): `tests/{helpers,test_schema,test_registry,test_predict,test_training_sets}.py`. Runs directly in the `ML` env — **no pytest needed**. `test_predict` builds models + runs all predictions once (module cache + setUpClass); assertions read the cache.
- **Demo notebook: built + executed clean** — `vignettes/demo.ipynb` (public SMILES + synthetic models, outputs cleared). Mirrors the test suite.
- **v0.1 feature-complete.** Open follow-ups: pin `requirements.txt` fully, real ChemProp backend (needs `reinvent` env), optional applicability-domain/confidence on predictions.

## How to run
- Notebook/CLI need `PYTHONPATH=/home/gtamo/MLTrail` + the `ML` env python, OR `pip install -e . --no-deps` into `ML` (gives the `mltrail` command).
- Tests: from the repo root, `PYTHONPATH=/home/gtamo/MLTrail python -m unittest discover -s tests` (runs in the `ML` env; or `python tests/test_registry.py` for one module).

## Environment
- **Home env = `ML`** (conda, py3.12): has pandas, numpy, pyyaml, sklearn, rdkit, joblib, openpyxl. **Missing `pytest`** (user to install when we add tests).
- `reinvent` env (py3.10) has chemprop + reinvent + pytest (different interpreter).
- **Never install to `base`.** Run MLTrail with `PYTHONPATH=/home/gtamo/MLTrail /home/gtamo/miniconda3/envs/ML/bin/python`.

## Locked design decisions
- Tool name: **`mltrail`** everywhere.
- **`framework`** is a mandatory field (sklearn/chemprop/reinvent/…), separate from `model_type` (the ML task). model_type drives output formatting; framework drives load/predict dispatch.
- **Versioning**: registry keyed by integer `id`; each model holds `versions[]`.
  - `--add` (no id) → new model, auto id, v1.
  - `--add --id` → **new version**; identity fields immutable (error on change); version fields NOT inherited; missing `metrics` → `"N/A"` + warning (fixable later via `--overwrite`).
  - `--overwrite --id` → replace the **latest version in place** (history kept; identity may change).
  - `--delete --id` → remove a model and its **entire version trail**; `next_id` is never reused.
  - `--trail` groups by `id`.
- **Generative models** (`model_type: generative`, e.g. REINVENT) are **metadata-only**: `--predict`/`--generate` raise an error. Never loaded → the `reinvent` env is never needed by MLTrail.
- `metrics` stored as a **dict** `{name: value}` (not parallel lists).
- `df_pred` stored as a **path** (`df_pred_path`), never inline.
- Date format `%Y%m%d_%H%M%S`.

## Registry schema (on disk)
`{"next_id": int, "models": {"<id>": {id, <identity...>, "versions": [<version...>]}}}`
- **Identity (immutable across versions, all mandatory):** experiment_name, experiment_measure, unit, model_type, framework, features_type.
- **Version fields:** model_path (mandatory), + optional dataset_path, comments, df_pred_path, metrics. Plus auto `version`, `date`.

## Featurizers (features_type → callable)
Featurizers are **reused from an external module named in config.yaml** (parity with training, no reimplementation). Contract: `df[compound, smiles] → df[compound, <feature cols>]`.
Source module: **`/home/gtamo/Scripts/Rdkit_tools.py`** (import name `Rdkit_tools`; on notebook path, not installed).
- **`MF_2048`** = `get_MF_bits_from_df(df, nBits=2048, radius=2)` → `compound | F0..F2047` (Morgan/ECFP4, int8).
- **`H236`** = `compute_H236_features(df)` → **4269 features**: `F0..F2047` (Morgan) + `Hba,Hbd,MW,TPSA,LogP,NRB` (6 physchem) + `MACCS_0..166` (167) + `AP_0..AP2047` (hashed AtomPair). The H236 *model* uses a >2%-prevalence subset selected by `_build_feature_columns(mf, 'multi_fp_champion', 0.02)` in `MS_ML/python/compute_R2_for_all_genes.py`; the featurizer returns the full universe and the model's stored `feature_cols` selects the subset.

## Model artifact format (critical for the sklearn backend)
MS_ML production sklearn models are saved as a **dict**, not a bare estimator:
`joblib.dump({'model': est, 'feature_cols': [...], 'features': 'H236', 'sklearn_ver': ...})`.
Backend must: load dict → take `model` + `feature_cols` → featurize to the universe → **reindex to `feature_cols` (order matters, missing bits → 0)** → predict. Bare estimators (no dict) are also supported (no reindex).

## Config (config/config.yaml)
`registry_path`, `training_sets_dir`, `date_format`, and `featurizers: {path, module, map: {features_type: function}}`.
CLI default lookup: `config/config.yaml`, then `config.yaml`, then built-in defaults.

## Training-set delta storage
`training_sets_dir/<model_id>/<timestamp>_<idx>.parquet` — one chunk per save, holding only rows new vs. all prior chunks. Dedup key defaults to the full row; `dedup_on=[cols]` (e.g. `["smiles"]`) dedups by identity instead. Scope is per-model (accumulates across versions). `--save-trainset --id N --dataset f [--dedup_on smiles]`; API: `reg.save_training_set(id, df_or_path)` / `reg.load_training_set(id)`. Model entry gains a `training_set_dir` key (surfaced by `--details`).

## Repo layout
`mltrail/`: `__init__` (exposes Registry, load_config, predict), `config`, `schema`, `registry`, `readers`, `featurizers`, `backends`, `predict`, `training_sets`, `cli`.
`config/config.yaml` · `vignettes/demo.ipynb` · `tests/` · `wiki/` · `data/specs.txt` (vault `data/registry.json` + `data/training_sets/` gitignored).
