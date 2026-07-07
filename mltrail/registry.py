"""The MLTrail model registry: a versioned JSON vault of trained models.

The on-disk shape is::

    {"next_id": <int>, "models": {"<id>": {<identity...>, "versions": [<version...>]}}}

Identity fields are stored once per model; version fields are stored per version.
All mutating ops persist immediately via an atomic write.
"""
import json
import warnings
from datetime import datetime
from pathlib import Path

import pandas as pd

from .config import load_config
from .schema import (
    IDENTITY_FIELDS,
    VERSION_FIELDS,
    ValidationError,
    validate_new_model,
)

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

    def add(self, model_id=None, overwrite=False, **fields):
        """Register a model, add a version, or reset an entry.

        No model_id            -> new model (auto id), version 1.
        model_id, overwrite    -> reset the latest version in place (history kept).
        model_id               -> append a new version (inherits identity + prior version).
        Returns the model id.
        """
        fields = {k: v for k, v in fields.items() if v is not None}
        if model_id is None:
            return self._add_new(fields)
        if overwrite:
            return self._overwrite(model_id, fields)
        return self._add_version(model_id, fields)

    def _add_new(self, fields):
        validate_new_model(fields)
        model_id = self._data["next_id"]
        identity = {f: fields[f] for f in IDENTITY_FIELDS}
        self._data["models"][str(model_id)] = {
            "id": model_id,
            **identity,
            "versions": [self._make_version(1, fields)],
        }
        self._data["next_id"] = model_id + 1
        self._save()
        return model_id

    def _add_version(self, model_id, fields):
        model = self._require(model_id)
        for f in IDENTITY_FIELDS:
            if f in fields and fields[f] != model[f]:
                raise ValidationError(
                    f"cannot change identity field {f!r} on --add --id "
                    f"({model[f]!r} -> {fields[f]!r}); use --overwrite to reset the entry"
                )
        version = self._make_version(model["versions"][-1]["version"] + 1, fields)
        if version["metrics"] == "N/A":
            warnings.warn(
                f"model {model_id} v{version['version']}: no metrics provided; set to 'N/A'. "
                f"Use `--overwrite --id {model_id}` to add them later.",
                stacklevel=2,
            )
        model["versions"].append(version)
        self._save()
        return model_id

    def _overwrite(self, model_id, fields):
        model = self._require(model_id)
        for f in IDENTITY_FIELDS:
            if f in fields:
                model[f] = fields[f]
        latest = model["versions"][-1]
        model["versions"][-1] = self._make_version(latest["version"], fields)
        self._save()
        return model_id

    def delete(self, model_id):
        """Delete a model and its entire version trail from the registry.

        Raises KeyError if the id is unknown. `next_id` is left untouched so ids
        are never reused. Returns the deleted id.
        """
        self._require(model_id)
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
        return {"id": model["id"], **{f: model[f] for f in IDENTITY_FIELDS}, **chosen}

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

        Featurizes per the model's config, aligns to its trained columns, and writes a CSV
        when `pred_output` is given. Raises ValueError for generative models.
        """
        from .predict import predict as _predict
        return _predict(self, model_id, dataset, smiles_column, compound_id,
                        pred_output, model_path, config=self.config)
