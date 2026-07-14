# MLTrail — Project Wiki

Durable memory for this repo. Aggregate facts only (no SMILES / compound IDs / per-compound values).

## Purpose
MLTrail is a local, MLflow-lite **model registry / vault** for trained ML models (sklearn, ChemProp,
REINVENT checkpoints, …). It registers models, versions them, and predicts properties on new datasets.
Usable from the CLI (`mltrail --...`) and imported in notebooks (methods return DataFrames).

## Status (2026-07-14)
- **Standalone built-in featurizers: built + parity-tested.** `MF_2048` + `H236` are now vendored **verbatim** into `mltrail/featurizers.py` (RDKit-only, no `tqdm`), so MLTrail featurizes with **no `/home/gtamo/Scripts` path**. Built-ins are the **default**; an external `featurizers.module`+`map` still overrides. Parity test (`tests/test_featurizers.py`) asserts built-in == `Rdkit_tools` bit-for-bit (skips if unavailable). **43 tests** now. Default `config.yaml`/`DEFAULTS` no longer configure an external module.

## Status (2026-07-10)
- **Managed artifacts (vault-owned files): built + tested.** `registry.add` now **imports** the model into `trained_models_dir` and (optionally) archives the training set — the vault is self-contained, not a bag of pointers. See "Managed-artifact storage" below. Tests: **37 passing** (was 32). CLI add→predict→delete smoke-tested on an isolated temp vault.

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

## Managed-artifact storage (2026-07-10)
`add()` takes ownership of the artifact; `model_path` is **vault-derived, not caller-set**.
- **`add(model=..., training_set=None, smiles_column, compound_id_column, label_column, dedup_on)`.**
  `model` is a **path** (copied verbatim into the vault; file keeps its extension, a dir is `copytree`'d) or an **object** (`joblib.dump`'ed). Stored path = `trained_models_dir/<id>_v<ver><ext>`. Helper: `backends.save_model(model, dest_dir, id, version)`.
- **New version (`--add --id`) requires a fresh `model`** — no artifact ⇒ `ValidationError` (never inherits the prior file). **`--overwrite`** without a `model` keeps the existing artifact.
- **Generative models are the exception**: `model_path` is stored as a **verbatim pointer** (not copied — they live in the `reinvent` env, outside the vault).
- **`training_set`** (DataFrame/path) is sliced to canonical **`[compound_id, smiles, label]`** via the column-name params and fed to the existing delta-chunk archive (default `dedup_on=["smiles"]`). Features are NOT stored — they're reconstructible from `smiles` via `features_type`; the **label is not** (it's the measured value), so it must be kept.
- **`delete()`** now also removes the model artifacts (only those inside `trained_models_dir`) and the `training_sets_dir/<id>` folder; `next_id` still never reused.
- **Validation**: `validate_new_model` checks identity fields only; artifact presence is enforced by the Registry.
- **CLI**: `--model_path` now means *source artifact to import* (add) / *load override* (predict); new `--training_set` + `--label_column` (reuse `--smiles_column` / `--compound_id`). `config.yaml`/`DEFAULTS` gain `trained_models_dir`.

## Registry schema (on disk)
`{"next_id": int, "models": {"<id>": {id, <identity...>, "versions": [<version...>]}}}`
- **Identity (immutable across versions, all mandatory):** experiment_name, experiment_measure, unit, model_type, framework, features_type.
- **Version fields:** model_path (mandatory), + optional dataset_path, comment, df_pred_path, metrics. Plus auto `version`, `date`. (`comment` = free-text note on what the model/version does; CLI `--comment`, shown by `--details`. Renamed from `comments` 2026-07-14.)

## Featurizers (features_type → callable)
**Built-in and standalone by default** — vendored verbatim into `mltrail/featurizers.py` (RDKit only), so no external path is needed. `BUILTIN_FEATURIZERS = {"MF_2048": morgan_2048, "H236": h236}`. `get_featurizer` resolves built-ins unless the config's `featurizers` section names a `module` that maps the type (then that external module **overrides** — the original parity-with-training escape hatch). Contract: `df[compound, smiles] → df[compound, <feature cols>]`.
Vendored from **`/home/gtamo/Scripts/Rdkit_tools.py`** (`get_MF_bits_from_df` / `compute_H236_features`); a parity test (`tests/test_featurizers.py`) asserts built-in == external bit-for-bit and guards RDKit-version drift.
- **`MF_2048`** = `morgan_2048(df, nBits=2048, radius=2)` → `compound | F0..F2047` (Morgan/ECFP4, int8; `GetMorganGenerator(radius=2, fpSize=2048, includeChirality=True)`).
- **`H236`** = `h236(df)` → **4269 features**: `F0..F2047` (Morgan) + `Hba,Hbd,MW,TPSA,LogP,NRB` (6 physchem) + `MACCS_0..166` (167) + `AP_0..AP2047` (hashed AtomPair). The H236 *model* uses a >2%-prevalence subset selected by `_build_feature_columns(mf, 'multi_fp_champion', 0.02)` in `MS_ML/python/compute_R2_for_all_genes.py`; the featurizer returns the full universe and the model's stored `feature_cols` selects the subset.

## Model artifact format (critical for the sklearn backend)
MS_ML production sklearn models are saved as a **dict**, not a bare estimator:
`joblib.dump({'model': est, 'feature_cols': [...], 'features': 'H236', 'sklearn_ver': ...})`.
Backend must: load dict → take `model` + `feature_cols` → featurize to the universe → **reindex to `feature_cols` (order matters, missing bits → 0)** → predict. Bare estimators (no dict) are also supported (no reindex).

## Config (config/config.yaml)
`registry_path`, `trained_models_dir`, `training_sets_dir`, `date_format`, and an **optional** `featurizers: {path, module, map}` (omitted by default → built-in featurizers; present → external override).
CLI config resolution (2026-07-14): `--config <path>` → MLTrail's **package-relative** `config/config.yaml`
(`Path(__file__).parent.parent/config/config.yaml` in `cli._resolve_config`, so the same vault resolves
from any CWD — not the CWD's own `config.yaml`) → built-in defaults.

## Training-set delta storage
`training_sets_dir/<model_id>/<timestamp>_<idx>.parquet` — one chunk per save, holding only rows new vs. all prior chunks. Dedup key defaults to the full row; `dedup_on=[cols]` (e.g. `["smiles"]`) dedups by identity instead. Scope is per-model (accumulates across versions). `--save-trainset --id N --dataset f [--dedup_on smiles]`; API: `reg.save_training_set(id, df_or_path)` / `reg.load_training_set(id)`. Model entry gains a `training_set_dir` key (surfaced by `--details`).

## Repo layout
`mltrail/`: `__init__` (exposes Registry, load_config, predict), `config`, `schema`, `registry`, `readers`, `featurizers`, `backends`, `predict`, `training_sets`, `cli`.
`config/config.yaml` · `vignettes/demo.ipynb` · `tests/` · `wiki/` · `data/specs.txt` (vault `data/registry.json` + `data/training_sets/` gitignored).
