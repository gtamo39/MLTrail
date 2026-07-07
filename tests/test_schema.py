"""Unit tests for mltrail.schema field validation.

Run from the repo root:
    python -m unittest discover -s tests
or:
    python tests/test_schema.py
"""
import os, sys, unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mltrail.schema import ValidationError, is_generative, validate_new_model

VALID = {"experiment_name": "e", "experiment_measure": "m", "unit": "u",
         "model_path": "/x.joblib", "model_type": "single_task_regression",
         "framework": "sklearn", "features_type": "MF_2048"}


class TestSchema(unittest.TestCase):
    """Validation of first-time model registrations."""

    def test_validate_new_model_ok(self):
        """A complete, valid field set passes and returns no warnings."""
        # a fully specified sklearn model validates with an empty warning list
        self.assertEqual(validate_new_model(dict(VALID)), [])

    def test_validate_missing_mandatory_field_raises(self):
        """Dropping a mandatory field raises ValidationError naming the field."""
        fields = dict(VALID)
        del fields["features_type"]
        # the missing mandatory field must be reported
        with self.assertRaisesRegex(ValidationError, "features_type"):
            validate_new_model(fields)

    def test_validate_bad_model_type_raises(self):
        """An unknown model_type is rejected (it drives output formatting)."""
        # invalid model_type must raise
        with self.assertRaisesRegex(ValidationError, "model_type"):
            validate_new_model(dict(VALID, model_type="banana"))

    def test_validate_unknown_framework_warns_not_raises(self):
        """An unrecognized framework is a warning, not an error (frameworks are extensible)."""
        warnings = validate_new_model(dict(VALID, framework="mystery_ml"))
        # unknown framework surfaces as a warning message rather than raising
        self.assertTrue(any("mystery_ml" in w for w in warnings))

    def test_is_generative(self):
        """is_generative is True only for the generative model_type."""
        # generative models are metadata-only
        self.assertTrue(is_generative("generative"))
        # predictive models are not generative
        self.assertFalse(is_generative("single_task_regression"))


if __name__ == "__main__":
    unittest.main()
