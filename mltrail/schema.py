"""Model-entry schema for the MLTrail registry: field definitions and validation.

A registry entry is split into two parts:
  * identity fields  — define *which model* this is; immutable across versions.
  * version fields   — change on each retrain; stored per version.
"""

# Identity fields: stored once per model, immutable across versions (all mandatory).
IDENTITY_FIELDS = [
    "experiment_name",
    "experiment_measure",
    "unit",
    "model_type",
    "framework",
    "features_type",
]

# Version fields: stored per version. model_path is mandatory; the rest optional.
# target_columns: ordered target names for multitask models (e.g. chemprop) — names the
# prediction output columns; None/absent for single-output models.
VERSION_MANDATORY = ["model_path"]
VERSION_OPTIONAL = ["dataset_path", "comment", "df_pred_path", "metrics", "target_columns"]
VERSION_FIELDS = VERSION_MANDATORY + VERSION_OPTIONAL

# Allowed values. model_type drives prediction-output formatting, so it is enforced.
MODEL_TYPES = {
    "single_task_classification",
    "single_task_regression",
    "multitask_classification",
    "multitask_regression",
    "generative",
}

# Frameworks drive how a model is loaded/called. Extensible: unknown values warn, not error.
FRAMEWORKS = {"sklearn", "chemprop", "reinvent", "xgboost", "lightgbm"}

# model_types that cannot be used with --predict/--generate (metadata-only in MLTrail).
GENERATIVE_TYPES = {"generative"}


class ValidationError(ValueError):
    """Raised when a model entry is missing mandatory fields or has an invalid value."""


def validate_new_model(fields):
    """Validate the identity fields of a first-time (--add, no id) model registration.

    Input: a dict of field name -> value (None/absent means not provided).
    Raises ValidationError on any missing mandatory field or invalid model_type.
    model_path is not checked here — it is vault-derived from the id/version, not caller-set;
    artifact presence is validated by the Registry before materialization.
    Unknown framework/features_type only warn (returned as a list of messages).
    """
    missing = [f for f in IDENTITY_FIELDS if not fields.get(f)]
    if missing:
        raise ValidationError(f"missing mandatory field(s): {missing}")
    if fields["model_type"] not in MODEL_TYPES:
        raise ValidationError(
            f"unknown model_type {fields['model_type']!r}; allowed: {sorted(MODEL_TYPES)}"
        )
    warnings = []
    if fields["framework"] not in FRAMEWORKS:
        warnings.append(f"unrecognized framework {fields['framework']!r} (allowed: {sorted(FRAMEWORKS)})")
    return warnings


def is_generative(model_type):
    """Return True if this model_type is metadata-only (cannot predict/generate)."""
    return model_type in GENERATIVE_TYPES
