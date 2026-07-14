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

import pandas as pd

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
        self.reg.add(model_id=mid, model={"dummy": True}, metrics={"R2": 0.85})
        # a second version is appended under the same id
        self.assertEqual(self.reg.details(mid)["version"], 2)
        # identity (experiment_name) is retained across versions
        self.assertEqual(self.reg.details(mid)["experiment_name"], "HLM")

    def test_add_version_without_metrics_warns_and_sets_na(self):
        """A new version with no metrics warns and stores metrics as 'N/A'."""
        mid = self.add(metrics={"R2": 0.8})
        # omitting metrics on a new version must emit a UserWarning
        with self.assertWarns(UserWarning):
            self.reg.add(model_id=mid, model={"dummy": True})
        # and the stored metrics fall back to the 'N/A' sentinel
        self.assertEqual(self.reg.details(mid)["metrics"], "N/A")

    def test_identity_change_on_add_id_raises(self):
        """Changing an identity field via --add --id raises ValidationError."""
        mid = self.add(experiment_measure="clearance")
        # attempting to change identity on a new version is rejected
        with self.assertRaisesRegex(ValidationError, "identity"):
            self.reg.add(model_id=mid, experiment_measure="solubility", model={"dummy": True})

    def test_overwrite_replaces_latest_in_place(self):
        """--overwrite replaces the latest version without appending a new one."""
        mid = self.add(metrics={"R2": 0.1})
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.reg.add(model_id=mid, model={"dummy": True})   # version 2, metrics N/A
        self.reg.add(model_id=mid, overwrite=True, model={"dummy": True}, metrics={"R2": 0.9})
        # overwrite keeps the version count (no new version appended)
        self.assertEqual(self.reg.details(mid)["version"], 2)
        # and the latest version now carries the corrected metrics
        self.assertEqual(self.reg.details(mid)["metrics"], {"R2": 0.9})

    def test_delete_removes_all_versions_and_keeps_next_id(self):
        """delete removes a model entirely; the id is not reused afterward."""
        mid = self.add()
        self.reg.add(model_id=mid, model={"dummy": True}, metrics={"R2": 0.5})
        self.reg.delete(mid)
        # deleting an already-absent id raises KeyError
        with self.assertRaises(KeyError):
            self.reg.delete(mid)
        # a subsequent add does NOT reuse the deleted id
        self.assertEqual(self.add(), mid + 1)

    def test_missing_mandatory_field_raises(self):
        """Registering without a mandatory identity field raises ValidationError."""
        # features_type is missing here
        with self.assertRaises(ValidationError):
            self.reg.add(model={"dummy": True}, experiment_name="e", experiment_measure="m",
                         unit="u", model_type="single_task_regression", framework="sklearn")

    def test_add_without_model_raises(self):
        """A non-generative registration with no model artifact raises ValidationError."""
        # model is vault-owned now: an artifact (path or object) is mandatory
        with self.assertRaisesRegex(ValidationError, "artifact"):
            self.reg.add(experiment_name="e", experiment_measure="m", unit="u",
                         model_type="single_task_regression", framework="sklearn",
                         features_type="MF_2048")

    def test_model_path_is_vault_derived(self):
        """add() imports the artifact and stores a vault-derived path named <id>_v<ver>."""
        mid = self.add()
        stored = Path(self.reg.details(mid)["model_path"])
        # the stored path lives inside trained_models_dir and is named by id/version
        self.assertEqual(stored.parent, self.tmp / "models")
        self.assertEqual(stored.name, f"{mid}_v1.joblib")
        # the artifact was actually written there
        self.assertTrue(stored.exists())

    def test_add_new_version_requires_model(self):
        """--add --id with no new model artifact raises (each version needs its own model)."""
        mid = self.add()
        # omitting model on a new version is an error, not an inherit
        with self.assertRaisesRegex(ValidationError, "artifact"):
            self.reg.add(model_id=mid, metrics={"R2": 0.9})

    def test_training_set_sliced_and_archived(self):
        """A training_set passed to add() is sliced to compound_id/smiles/label and archived."""
        train = pd.DataFrame({"cid": ["C1", "C2"], "smi": ["CCO", "CCN"], "y": [0.1, 0.2]})
        mid = self.add(training_set=train, smiles_column="smi",
                       compound_id_column="cid", label_column="y")
        loaded = self.reg.load_training_set(mid)
        # the archive keeps only the canonical [compound_id, smiles, label] columns
        self.assertEqual(list(loaded.columns), ["compound_id", "smiles", "label"])
        # both rows are stored
        self.assertEqual(len(loaded), 2)

    def test_delete_removes_vault_files(self):
        """delete() removes the model artifact and training-set folder from disk."""
        mid = self.add(training_set=pd.DataFrame({"compound_id": ["C1"], "smiles": ["CCO"], "y": [1]}),
                       label_column="y")
        artifact = Path(self.reg.details(mid)["model_path"])
        ts_folder = self.tmp / "training_sets" / str(mid)
        # both the artifact file and the training-set folder exist after add
        self.assertTrue(artifact.exists() and ts_folder.exists())
        self.reg.delete(mid)
        # delete cleans up both from disk
        self.assertFalse(artifact.exists() or ts_folder.exists())

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
            self.reg.add(model_id=mid, model={"dummy": True})   # metrics N/A -> skipped
        self.reg.add(model_id=mid, model={"dummy": True}, metrics={"R2": 0.85})
        # only the two versions carrying R2 appear (the N/A version is skipped)
        self.assertEqual(list(self.reg.trail("R2", model_id=mid)["R2"]), [0.81, 0.85])

    def test_persistence_reload(self):
        """A registry reloaded from disk sees previously added models."""
        self.add()
        # reopening the same registry file preserves the entry
        self.assertEqual(len(Registry.from_config(helpers.make_config(self.tmp)).list()), 1)


if __name__ == "__main__":
    unittest.main()
