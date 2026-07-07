"""Unit tests for the Registry (metadata only — fake model paths, no featurization).

Each test gets a fresh temp registry (registry ops mutate state), which is cheap.

Run from the repo root:
    python -m unittest discover -s tests
or:
    python tests/test_registry.py
"""
import os, sys, tempfile, unittest, warnings
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))   # so `import helpers` resolves

import helpers
from mltrail import Registry
from mltrail.schema import ValidationError


class TestRegistry(unittest.TestCase):
    """Add / version / overwrite / delete / list / search / trail on a fresh registry."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.reg = helpers.new_registry(self.tmp)

    def add(self, **kw):
        return helpers.register(self.reg, **kw)

    def test_add_new_assigns_sequential_ids(self):
        """First-time --add assigns auto-incrementing ids starting at 1."""
        # two fresh registrations get ids 1 then 2
        self.assertEqual((self.add(), self.add()), (1, 2))

    def test_add_version_appends_and_keeps_identity(self):
        """--add --id appends a new version and inherits the model's identity fields."""
        mid = self.add(experiment_name="HLM", metrics={"R2": 0.8})
        self.reg.add(model_id=mid, model_path="/fake/v2.joblib", metrics={"R2": 0.85})
        # a second version is appended under the same id
        self.assertEqual(self.reg.details(mid)["version"], 2)
        # identity (experiment_name) is retained across versions
        self.assertEqual(self.reg.details(mid)["experiment_name"], "HLM")

    def test_add_version_without_metrics_warns_and_sets_na(self):
        """A new version with no metrics warns and stores metrics as 'N/A'."""
        mid = self.add(metrics={"R2": 0.8})
        # omitting metrics on a new version must emit a UserWarning
        with self.assertWarns(UserWarning):
            self.reg.add(model_id=mid, model_path="/fake/v2.joblib")
        # and the stored metrics fall back to the 'N/A' sentinel
        self.assertEqual(self.reg.details(mid)["metrics"], "N/A")

    def test_identity_change_on_add_id_raises(self):
        """Changing an identity field via --add --id raises ValidationError."""
        mid = self.add(experiment_measure="clearance")
        # attempting to change identity on a new version is rejected
        with self.assertRaisesRegex(ValidationError, "identity"):
            self.reg.add(model_id=mid, experiment_measure="solubility", model_path="/fake/v2.joblib")

    def test_overwrite_replaces_latest_in_place(self):
        """--overwrite replaces the latest version without appending a new one."""
        mid = self.add(metrics={"R2": 0.1})
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.reg.add(model_id=mid, model_path="/fake/v2.joblib")   # version 2, metrics N/A
        self.reg.add(model_id=mid, overwrite=True, model_path="/fake/fix.joblib", metrics={"R2": 0.9})
        # overwrite keeps the version count (no new version appended)
        self.assertEqual(self.reg.details(mid)["version"], 2)
        # and the latest version now carries the corrected metrics
        self.assertEqual(self.reg.details(mid)["metrics"], {"R2": 0.9})

    def test_delete_removes_all_versions_and_keeps_next_id(self):
        """delete removes a model entirely; the id is not reused afterward."""
        mid = self.add()
        self.reg.add(model_id=mid, model_path="/fake/v2.joblib", metrics={"R2": 0.5})
        self.reg.delete(mid)
        # deleting an already-absent id raises KeyError
        with self.assertRaises(KeyError):
            self.reg.delete(mid)
        # a subsequent add does NOT reuse the deleted id
        self.assertEqual(self.add(), mid + 1)

    def test_missing_mandatory_field_raises(self):
        """Registering without a mandatory field raises ValidationError."""
        # model_path and features_type are missing here
        with self.assertRaises(ValidationError):
            self.reg.add(experiment_name="e", experiment_measure="m", unit="u",
                         model_type="single_task_regression", framework="sklearn")

    def test_list_sorted_alphabetically(self):
        """--list returns the four spec columns sorted case-insensitively by experiment_name."""
        self.add(experiment_name="Zeta")
        self.add(experiment_name="alpha")
        listing = self.reg.list()
        # columns match the spec
        self.assertEqual(list(listing.columns), ["id", "date", "experiment_name", "experiment_measure"])
        # rows are ordered alphabetically (case-insensitive)
        self.assertEqual(list(listing["experiment_name"]), ["alpha", "Zeta"])

    def test_search_or_substring(self):
        """--search matches ANY provided field by case-insensitive substring."""
        self.add(experiment_measure="clearance")
        self.add(experiment_measure="solubility")
        result = self.reg.search(experiment_measure="CLEAR")
        # case-insensitive substring on measure returns exactly the clearance model
        self.assertEqual(len(result), 1)
        self.assertEqual(result.iloc[0]["experiment_measure"], "clearance")

    def test_trail_skips_na_metrics(self):
        """trail returns one row per version that has the metric, skipping 'N/A' versions."""
        mid = self.add(metrics={"R2": 0.81})
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.reg.add(model_id=mid, model_path="/fake/v2.joblib")   # metrics N/A -> skipped
        self.reg.add(model_id=mid, model_path="/fake/v3.joblib", metrics={"R2": 0.85})
        # only the two versions carrying R2 appear (the N/A version is skipped)
        self.assertEqual(list(self.reg.trail("R2", model_id=mid)["R2"]), [0.81, 0.85])

    def test_persistence_reload(self):
        """A registry reloaded from disk sees previously added models."""
        self.add()
        # reopening the same registry file preserves the entry
        self.assertEqual(len(Registry.from_config(helpers.make_config(self.tmp)).list()), 1)


if __name__ == "__main__":
    unittest.main()
