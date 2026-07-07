"""Unit tests for delta-only training-set storage (parquet chunks, dedup on save).

Run from the repo root:
    python -m unittest discover -s tests
or:
    python tests/test_training_sets.py
"""
import os, sys, tempfile, unittest
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))   # so `import helpers` resolves

import pandas as pd

import helpers


class TestTrainingSets(unittest.TestCase):
    """Saving a training set stores only rows not already archived; load reconstructs the union."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.reg = helpers.new_registry(self.tmp)
        self.mid = helpers.register(self.reg)
        # public molecules + a synthetic binary label
        self.train = pd.DataFrame({"smiles": ["CCO", "c1ccccc1", "CC(=O)O"], "label": [0, 1, 0]})

    def test_first_save_writes_all_rows(self):
        """The first save writes every row and creates a chunk."""
        summary = self.reg.save_training_set(self.mid, self.train)
        # all three rows are new, nothing pre-existing, a chunk was written
        self.assertEqual((summary["n_new"], summary["n_existing"]), (3, 0))
        self.assertIsNotNone(summary["chunk"])
        # the full training set reconstructs to the three rows
        self.assertEqual(len(self.reg.load_training_set(self.mid)), 3)

    def test_identical_resave_writes_nothing(self):
        """Re-saving identical data writes no chunk (the internal dedup check finds no new rows)."""
        self.reg.save_training_set(self.mid, self.train)
        summary = self.reg.save_training_set(self.mid, self.train)
        # nothing new, so no chunk is written
        self.assertEqual(summary["n_new"], 0)
        self.assertIsNone(summary["chunk"])
        # and the stored set is unchanged
        self.assertEqual(len(self.reg.load_training_set(self.mid)), 3)

    def test_delta_saves_only_new_rows(self):
        """A partially-overlapping save stores only the genuinely new rows."""
        self.reg.save_training_set(self.mid, self.train)
        more = pd.DataFrame({"smiles": ["CCO", "CCN"], "label": [0, 1]})   # CCO overlaps, CCN is new
        summary = self.reg.save_training_set(self.mid, more)
        # only the one new row is stored
        self.assertEqual(summary["n_new"], 1)
        # the reconstructed set is the union (4 rows)
        self.assertEqual(len(self.reg.load_training_set(self.mid)), 4)

    def test_dedup_on_subset_ignores_relabels(self):
        """With dedup_on=['smiles'], a compound already present is not re-added even if relabeled."""
        self.reg.save_training_set(self.mid, self.train, dedup_on=["smiles"])
        relabel = pd.DataFrame({"smiles": ["CCO"], "label": [1]})
        summary = self.reg.save_training_set(self.mid, relabel, dedup_on=["smiles"])
        # smiles identity already stored -> nothing new
        self.assertEqual(summary["n_new"], 0)

    def test_full_row_dedup_keeps_relabeled(self):
        """With the default full-row key, a relabeled compound is a new row and is stored."""
        self.reg.save_training_set(self.mid, self.train)
        relabel = pd.DataFrame({"smiles": ["CCO"], "label": [1]})
        summary = self.reg.save_training_set(self.mid, relabel)
        # the (smiles, label) row differs from any stored row -> kept
        self.assertEqual(summary["n_new"], 1)

    def test_save_from_file_path(self):
        """save_training_set accepts a file path (all columns preserved), not just a DataFrame."""
        path = self.tmp / "train.csv"
        self.train.to_csv(path, index=False)
        summary = self.reg.save_training_set(self.mid, str(path))
        # reading from the CSV stores all three rows
        self.assertEqual(summary["n_new"], 3)

    def test_dir_recorded_in_details(self):
        """After a save, --details surfaces the model's training_set_dir."""
        self.reg.save_training_set(self.mid, self.train)
        # the archive folder is recorded on the model entry
        self.assertIn("training_set_dir", self.reg.details(self.mid))

    def test_load_empty_when_never_saved(self):
        """Loading a training set that was never saved returns an empty DataFrame."""
        # no chunks yet -> empty frame, not an error
        self.assertTrue(self.reg.load_training_set(self.mid).empty)


if __name__ == "__main__":
    unittest.main()
