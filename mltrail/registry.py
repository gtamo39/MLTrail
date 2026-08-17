"""The MLTrail model registry: a versioned JSON vault of trained models.

The on-disk shape is::

    {"next_id": <int>, "models": {"<id>": {<identity...>, "versions": [<version...>]}}}

Identity fields are stored once per model; version fields are stored per version.
All mutating ops persist immediately via an atomic write.
"""
import json
import shutil
import warnings
from datetime import datetime
from pathlib import Path

import pandas as pd

from .backends import save_model
from .config import default_config, load_config
from .schema import (
    IDENTITY_FIELDS,
    VERSION_FIELDS,
    ValidationError,
    is_generative,
    validate_new_model,
)
from .training_sets import load_full, read_full, save_delta

TRAINING_SUBSET_COLUMNS = ["compound_id", "smiles", "label"]

LIST_COLUMNS = ["id", "date", "experiment_name", "experiment_measure"]


class Registry:
    """A versioned JSON registry of trained models, keyed by integer model id."""

    def __init__(self, registry_path, date_format="%Y%m%d_%H%M%S", config=None):
        self.path = Path(registry_path)
        self.date_format = date_format
        self.config = config or {}
        self._data = self._load()

    @classmethod
    def from_config(cls, config):
        """Build a Registry from a config dict or a path to config.yaml."""
        cfg = config if isinstance(config, dict) else load_config(config)
        return cls(cfg["registry_path"], cfg.get("date_format", "%Y%m%d_%H%M%S"), config=cfg)

    @classmethod
    def from_default(cls):
        """Build a Registry from MLTrail's own package-relative config — no path needed.

        The notebook/library counterpart of running the CLI with no ``--config``: resolves the
        same default vault from any working directory.
        """
        return cls.from_config(default_config())

    # ---- persistence -----------------------------------------------------

    def _load(self):
        if self.path.exists():
            return json.loads(self.path.read_text())
        return {"next_id": 1, "models": {}}

    def _save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp.write_text(json.dumps(self._data, indent=2))
        tmp.replace(self.path)

    def _now(self):
        return datetime.now().strftime(self.date_format)

    def _require(self, model_id):
        model = self._data["models"].get(str(model_id))
        if model is None:
            raise KeyError(f"no model with id {model_id}")
        return model

    def _make_version(self, version_num, fields):
        """Assemble a version dict from the provided `fields` only (nothing inherited).

        model_path is mandatory; absent metrics default to the string "N/A".
        """
        version = {f: fields[f] for f in VERSION_FIELDS if f in fields}
        if not version.get("model_path"):
            raise ValidationError("model_path is required")
        if not version.get("metrics"):
            version["metrics"] = "N/A"
        version["version"] = version_num
        version["date"] = self._now()
        return version

    # ---- write ops -------------------------------------------------------

    def add(self, model_id=None, overwrite=False, model=None, training_set=None,
            smiles_column="smiles", compound_id_column="compound_id",
            label_column=None, dedup_on=None, **fields):
        """Register a model, add a version, or reset an entry — importing the artifact.

        No model_id            -> new model (auto id), version 1.
        model_id, overwrite    -> reset the latest version in place (history kept).
        model_id               -> append a new version (identity inherited; needs a `model`).
        `model` is a path (copied into trained_models_dir) or an object (joblib.dump'ed); the
        stored model_path is vault-derived as ``<id>_v<ver>``. An optional `training_set`
        (DataFrame/path) is sliced to compound_id/smiles/label and delta-archived. Returns the id.
        """
        fields = {k: v for k, v in fields.items() if v is not None}
        if model_id is None:
            new_id = self._add_new(fields, model)
        elif overwrite:
            new_id = self._overwrite(model_id, fields, model)
        else:
            new_id = self._add_version(model_id, fields, model)
        if training_set is not None:
            df = training_set if isinstance(training_set, pd.DataFrame) else read_full(training_set)
            subset = self._training_subset(df, compound_id_column, smiles_column, label_column)
            self.save_training_set(new_id, subset, dedup_on=dedup_on or ["smiles"])
        return new_id

    def _models_dir(self):
        return Path(self.config.get("trained_models_dir", "data/models"))

    def _materialize(self, model, model_id, version, model_type):
        """Import a model artifact and return its stored model_path.

        Generative models are stored as a verbatim pointer (never copied — they live outside
        the vault); all others are copied/dumped into trained_models_dir as ``<id>_v<ver>``.
        """
        if is_generative(model_type):
            if not isinstance(model, (str, Path)):
                raise ValidationError(
                    "generative models must be given as a path (stored as a pointer, not imported)"
                )
            return str(model)
        if model is None:
            raise ValidationError("a model artifact (path or object) is required")
        return save_model(model, self._models_dir(), model_id, version)

    @staticmethod
    def _training_subset(df, compound_id_column, smiles_column, label_column):
        """Slice a training set to [compound_id, smiles, label] (canonical names)."""
        if not label_column:
            raise ValidationError("label_column is required to store a training set")
        cols = [compound_id_column, smiles_column, label_column]
        missing = [c for c in cols if c not in df.columns]
        if missing:
            raise ValidationError(f"training_set missing column(s): {missing}")
        subset = df[cols].copy()
        subset.columns = TRAINING_SUBSET_COLUMNS
        return subset

    def _add_new(self, fields, model):
        validate_new_model(fields)
        model_id = self._data["next_id"]
        fields["model_path"] = self._materialize(model, model_id, 1, fields["model_type"])
        identity = {f: fields[f] for f in IDENTITY_FIELDS}
        self._data["models"][str(model_id)] = {
            "id": model_id,
            **identity,
            "versions": [self._make_version(1, fields)],
        }
        self._data["next_id"] = model_id + 1
        self._save()
        return model_id

    def _add_version(self, model_id, fields, model):
        entry = self._require(model_id)
        for f in IDENTITY_FIELDS:
            if f in fields and fields[f] != entry[f]:
                raise ValidationError(
                    f"cannot change identity field {f!r} on --add --id "
                    f"({entry[f]!r} -> {fields[f]!r}); use --overwrite to reset the entry"
                )
        version_num = entry["versions"][-1]["version"] + 1
        fields["model_path"] = self._materialize(model, model_id, version_num, entry["model_type"])
        version = self._make_version(version_num, fields)
        if version["metrics"] == "N/A":
            warnings.warn(
                f"model {model_id} v{version['version']}: no metrics provided; set to 'N/A'. "
                f"Use `--overwrite --id {model_id}` to add them later.",
                stacklevel=2,
            )
        entry["versions"].append(version)
        self._save()
        return model_id

    def _overwrite(self, model_id, fields, model):
        entry = self._require(model_id)
        for f in IDENTITY_FIELDS:
            if f in fields:
                entry[f] = fields[f]
        latest = entry["versions"][-1]
        if model is not None:
            fields["model_path"] = self._materialize(model, model_id, latest["version"], entry["model_type"])
        else:
            fields["model_path"] = latest["model_path"]
        entry["versions"][-1] = self._make_version(latest["version"], fields)
        self._save()
        return model_id

    def delete(self, model_id):
        """Delete a model, its version trail, and its vault files (artifacts + training set).

        Raises KeyError if the id is unknown. Only artifacts inside trained_models_dir are
        removed (generative pointers living elsewhere are left alone). `next_id` is left
        untouched so ids are never reused. Returns the deleted id.
        """
        entry = self._require(model_id)
        models_dir = self._models_dir().resolve()
        for version in entry["versions"]:
            path = Path(version.get("model_path", ""))
            try:
                inside_vault = path.resolve().is_relative_to(models_dir)
            except (OSError, ValueError):
                inside_vault = False
            if inside_vault and path.exists():
                shutil.rmtree(path) if path.is_dir() else path.unlink()
        ts_folder = self._training_folder(model_id)
        if ts_folder.exists():
            shutil.rmtree(ts_folder)
        del self._data["models"][str(model_id)]
        self._save()
        return model_id

    # ---- read ops --------------------------------------------------------

    def details(self, model_id, version=None):
        """Return a flat dict (identity + one version) for a model. Latest version by default."""
        model = self._require(model_id)
        if version is None:
            chosen = model["versions"][-1]
        else:
            chosen = next(v for v in model["versions"] if v["version"] == version)
        extras = {k: v for k, v in model.items() if k not in {"id", "versions", *IDENTITY_FIELDS}}
        return {"id": model["id"], **{f: model[f] for f in IDENTITY_FIELDS}, **extras, **chosen}

    def list(self):
        """Return a DataFrame (id, date, experiment_name, experiment_measure), one row per model.

        Rows reflect each model's latest version, sorted alphabetically by experiment_name.
        """
        rows = [
            {c: self.details(m["id"])[c] for c in LIST_COLUMNS}
            for m in self._data["models"].values()
        ]
        df = pd.DataFrame(rows, columns=LIST_COLUMNS)
        return df.sort_values("experiment_name", key=lambda s: s.str.lower()).reset_index(drop=True)

    def search(self, **filters):
        """Return models whose latest version matches ANY provided field (OR, case-insensitive substring)."""
        filters = {k: v for k, v in filters.items() if v is not None}
        rows = []
        for model in self._data["models"].values():
            record = self.details(model["id"])
            if any(self._match(record.get(k), v) for k, v in filters.items()):
                rows.append({c: record[c] for c in LIST_COLUMNS})
        return pd.DataFrame(rows, columns=LIST_COLUMNS)

    @staticmethod
    def _match(value, query):
        return value is not None and str(query).lower() in str(value).lower()

    def trail(self, metric, model_id=None):
        """Return a DataFrame tracking `metric` across versions, for plotting metric-vs-date.

        Columns: id, version, date, experiment_name, <metric>. Grouped by id, sorted by date.
        Restrict to one model with model_id.
        """
        models = [self._require(model_id)] if model_id is not None else self._data["models"].values()
        rows = [
            {
                "id": m["id"],
                "version": v["version"],
                "date": v["date"],
                "experiment_name": m["experiment_name"],
                metric: v["metrics"][metric],
            }
            for m in models
            for v in m["versions"]
            if isinstance(v.get("metrics"), dict) and metric in v["metrics"]
        ]
        columns = ["id", "version", "date", "experiment_name", metric]
        return pd.DataFrame(rows, columns=columns).sort_values(["id", "date"]).reset_index(drop=True)

    def predict(self, model_id, dataset, smiles_column=None, compound_id=None,
                pred_output=None, model_path=None):
        """Predict properties for `dataset` using a registered model (returns a DataFrame).

        `dataset` is a file path (csv/tsv/excel/sdf) or an in-memory pandas DataFrame — pass a
        frame directly to skip the temp-file round-trip. Featurizes per the model's config, aligns
        to its trained columns, and writes a CSV when `pred_output` is given. Raises ValueError for
        generative models.
        """
        from .predict import predict as _predict
        return _predict(self, model_id, dataset, smiles_column, compound_id,
                        pred_output, model_path, config=self.config)

    # ---- training-set archive (delta-only storage) -----------------------

    def _training_folder(self, model_id):
        return Path(self.config.get("training_sets_dir", "data/training_sets")) / str(model_id)

    def save_training_set(self, model_id, data, dedup_on=None):
        """Archive a model's training set, writing only rows not already stored (delta).

        `data` is a DataFrame or a path (csv/tsv/excel/parquet/sdf). An internal dedup check
        runs across all prior chunks so overlapping data is never re-saved. Returns a summary
        dict {n_new, n_existing, n_total, chunk}.
        """
        model = self._require(model_id)
        df = data if isinstance(data, pd.DataFrame) else read_full(data)
        folder = self._training_folder(model_id)
        summary = save_delta(folder, df, dedup_on=dedup_on, date_format=self.date_format)
        model["training_set_dir"] = str(folder)
        self._save()
        return summary

    def load_training_set(self, model_id):
        """Return the full, deduplicated training set for a model (concat of all delta chunks)."""
        self._require(model_id)
        return load_full(self._training_folder(model_id))
