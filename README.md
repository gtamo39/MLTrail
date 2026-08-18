# MLTrail

A local, **MLflow-lite model registry / vault** for trained ML models (sklearn, ChemProp,
REINVENT checkpoints, …). It registers models, versions them, archives their training sets, and
runs predictions on new datasets — all on-machine, no cloud, no tracking server.

Use it two ways:

- **CLI** — `mltrail --add …`, `mltrail --predict …`, etc.
- **Library** — `from mltrail import Registry`; methods return pandas DataFrames for notebooks.

---

## Installation

MLTrail's core dependencies (pandas, numpy, pyyaml, scikit-learn, rdkit, joblib, openpyxl,
pyarrow) are expected to live in a dedicated conda env — **never install to `base`**. In this
setup that env is `ML` (conda, py3.12).

### Option A — get the `mltrail` console command (recommended)

Install the package *without* touching the conda-provided scientific stack:

```bash
conda activate ML
pip install -e . --no-deps      # installs only the `mltrail` entry point
mltrail --list                  # verify
```

### Option B — no install, run from source

Point `PYTHONPATH` at the repo and call the env's python directly:

```bash
PYTHONPATH=/home/gtamo/MLTrail /home/gtamo/miniconda3/envs/ML/bin/python -m mltrail.cli --list
```

### Fresh environment

If you need to build the env from scratch, the pinned versions are in
[`requirements.txt`](requirements.txt):

```bash
conda create -n ML python=3.12
conda activate ML
pip install -r requirements.txt
pip install -e . --no-deps
```

> Optional backends (ChemProp, REINVENT) live in separate envs and are **not** needed for the
> core. Generative models are metadata-only — MLTrail never loads them.

---

## Configuration

All paths and the featurizer wiring live in [`config/config.yaml`](config/config.yaml). This is the
CLI's default (read package-relative, from any working directory), so **edit it in place** to change
the default vault; to run against a *different* config, save a copy elsewhere and pass `--config <path>`:

```yaml
registry_path: data/registry.json        # the JSON vault (created on first --add)
trained_models_dir: data/models           # imported model artifacts, named <id>_v<ver>
training_sets_dir: data/training_sets     # per-model delta parquet chunks
date_format: "%Y%m%d_%H%M%S"

# Featurizers are BUILT IN (MF_2048, H236 — RDKit only), so no wiring is needed by default.
# To override with an external module, add a featurizers section (must be bit-identical):
# featurizers:
#   path: /home/gtamo/Scripts
#   module: Rdkit_tools
#   map:
#     MF_2048: get_MF_bits_from_df
#     H236: compute_H236_features
```

MLTrail ships **standalone featurizers** (`MF_2048`, `H236`) that depend only on RDKit — it
featurizes on its own, no external path. They are vendored bit-for-bit from the shared
`Rdkit_tools` (a parity test enforces this), and an external module still overrides when
configured. The CLI looks up config in this order: `--config <path>` → MLTrail's own
`config/config.yaml` (package-relative, so the same vault resolves from any working directory) →
built-in defaults.

---

## How to use

### Registering a model

Identity fields (immutable across versions, all mandatory): `experiment_name`,
`experiment_measure`, `unit`, `model_type`, `framework`, `features_type`.

MLTrail **imports** the model into the vault: `--model_path` is the *source* artifact to copy in,
and the stored path is vault-derived as `trained_models_dir/<id>_v<ver>` — so the vault is
self-contained (move/back it up as one folder). Passing `--training_set` archives it alongside,
sliced to `compound_id / smiles / label` (features are reconstructible from SMILES via
`features_type`, so only the label is kept).

```bash
# first registration -> new model, auto id, version 1
mltrail --add \
  --experiment_name solubility \
  --experiment_measure logS \
  --unit "log(mol/L)" \
  --model_type single_task_regression \
  --framework sklearn \
  --features_type H236 \
  --model_path /path/to/solubility_rf.joblib \
  --training_set trainset.csv --smiles_column smiles --compound_id id --label_column logS \
  --comment "RF on H236; internal logD set, salts stripped, 5-fold CV" \
  --metrics R2=0.81 mse=0.12
```

`--comment` is a free-text note documenting what the model/version does; it's stored per version
and shown by `--details`.

`model_type` must be one of: `single_task_regression`, `single_task_classification`,
`multitask_regression`, `multitask_classification`, `generative`. Generative models are
metadata-only: `--model_path` is stored as a verbatim pointer (never copied).

### Versioning the same model

```bash
mltrail --add --id 1 --model_path /path/to/solubility_v2.joblib --metrics R2=0.86   # new version (needs a new artifact)
mltrail --overwrite --id 1 --metrics R2=0.87                                         # fix latest in place (keeps the artifact)
mltrail --delete --id 1                                                              # remove model, all versions, and its vault files
```

### Predicting on a new dataset

```bash
# print predictions to stdout
mltrail --predict --id 1 --dataset new_compounds.csv --smiles_column smiles --compound_id id

# write to a CSV
mltrail --predict --id 1 --dataset new_compounds.sdf --pred_output preds.csv
```

Datasets can be csv / tsv / excel / sdf / parquet. Invalid SMILES yield null predictions (reported
in the summary line).

#### chemprop (D-MPNN) models

chemprop models are registered like any other — `--framework chemprop`, the `--model_path` a trained
chemprop model directory (or a `.pt`/`.ckpt` checkpoint), and `--target_columns` naming the (multitask)
endpoints, which become the prediction output columns. No `features_type` featurizer is used — chemprop
builds its graph from SMILES itself (`features_type: smiles`).

Because chemprop is not a dependency of MLTrail's own env, prediction **shells out to a chemprop CLI**,
configured under `chemprop:` in `config.yaml` (`cli:` = the chemprop executable, `accelerator:` =
cpu|gpu|auto). This works whether MLTrail runs from the `ML` env (shells to the chemprop env's binary)
or from the `chemprop` env (`chemprop` on PATH). A registered chemprop model then predicts through the
exact same `--predict` / `registry.predict(...)` path as sklearn models, returning one column per target.

### Inspecting the registry

```bash
mltrail --list                                   # id, date, experiment_name, measure, comment (clipped; full in --details)
mltrail --details --id 1                          # every attribute of a model
mltrail --search --framework sklearn              # models matching ANY given identity field
mltrail --trail --id 1 --metrics R2               # a metric across versions (--output_trail to CSV)
```

### Archiving training sets (delta storage)

Stores only rows new versus everything already saved for that model, as timestamped parquet chunks:

```bash
mltrail --save-trainset --id 1 --dataset trainset_v2.csv --dedup_on smiles
```

### From a notebook / script

```python
from mltrail import Registry, load_config

reg = Registry.from_config(load_config("config/config.yaml"))

# `model` is a path (copied into the vault) or an in-memory estimator (joblib.dump'ed)
reg.add(experiment_name="solubility", experiment_measure="logS", unit="log(mol/L)",
        model_type="single_task_regression", framework="sklearn", features_type="H236",
        model=trained_estimator, metrics={"R2": 0.81},
        training_set=train_df, compound_id_column="id", label_column="logS")

reg.list()                       # -> DataFrame
reg.predict(1, "new_compounds.csv", smiles_column="smiles")   # -> DataFrame (path ...)
reg.predict(1, df, smiles_column="smiles", compound_id="compound")   # ... or an in-memory DataFrame
reg.trail("R2", model_id=1)      # -> DataFrame
reg.load_training_set(1)         # -> DataFrame (compound_id, smiles, label)
```

---

## Testing

Tests use the stdlib `unittest` — no `pytest` required:

```bash
PYTHONPATH=/home/gtamo/MLTrail python -m unittest discover -s tests   # all
python tests/test_registry.py                                         # one module
```

---

## Repo layout

```
mltrail/          config · schema · registry · readers · featurizers · backends · predict · training_sets · cli
config/           config.yaml
vignettes/        demo.ipynb   (public SMILES + synthetic models; mirrors the test suite)
tests/            stdlib unittest suite
wiki/             wiki.md — durable project memory
data/             registry.json + models/ + training_sets/  (gitignored vault)
```
