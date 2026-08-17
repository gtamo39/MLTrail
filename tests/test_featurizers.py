"""Tests for MLTrail's built-in standalone featurizers + parity vs external Rdkit_tools.

The parity test guarantees the vendored built-ins produce bit-identical features to the
shared Rdkit_tools (so models trained via Rdkit_tools don't skew at predict time); it skips
cleanly when Rdkit_tools isn't on the path, keeping the suite standalone.

Run from the repo root:
    python -m unittest discover -s tests
or:
    python tests/test_featurizers.py
"""
import os, sys, unittest
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))   # so `import helpers` resolves

import pandas as pd

import helpers
from mltrail.featurizers import get_featurizer


def _input_df():
    """Public molecules as df[compound, smiles] (the featurizer contract's input)."""
    return pd.DataFrame({"compound": [f"C_{i}" for i in range(len(helpers.PUBLIC_SMILES))],
                         "smiles": helpers.PUBLIC_SMILES})


class TestBuiltinFeaturizers(unittest.TestCase):
    """Shapes/behavior of the built-in featurizers, resolved with no external module."""

    def test_mf2048_shape_and_columns(self):
        """MF_2048 yields compound + F0..F2047 (2049 columns), one row per valid molecule."""
        out = get_featurizer("MF_2048", {})(_input_df())
        # compound column plus exactly 2048 fingerprint bit columns
        self.assertEqual(list(out.columns), ["compound"] + [f"F{i}" for i in range(2048)])
        # one row per input molecule (all public SMILES are valid)
        self.assertEqual(len(out), len(helpers.PUBLIC_SMILES))

    def test_h236_full_universe_width(self):
        """H236 yields the full 4269-feature universe: 2048 Morgan + 6 physchem + 167 MACCS + 2048 AP."""
        out = get_featurizer("H236", {})(_input_df())
        # compound + 4269 features
        self.assertEqual(out.shape[1], 1 + 2048 + 6 + 167 + 2048)
        # the four blocks are present and named as the H236 model expects
        for col in ["F0", "MW", "MACCS_0", "AP_0", "AP_2047"]:
            self.assertIn(col, out.columns)

    def test_invalid_smiles_dropped(self):
        """Rows whose SMILES fail to parse are dropped from the built-in output."""
        df = pd.DataFrame({"compound": ["ok", "bad"], "smiles": ["CCO", "not_a_smiles"]})
        out = get_featurizer("MF_2048", {})(df)
        # only the parseable molecule survives, keyed by its compound id
        self.assertEqual(list(out["compound"]), ["ok"])

    def test_h237_is_h236_plus_descriptor_block(self):
        """H237 yields the H236 universe plus ~200 prefixed DS_ descriptor columns."""
        try:
            out = get_featurizer("H237", {})(_input_df())
        except ImportError as e:
            self.skipTest(f"descriptastorus not installed: {e}")
        h236 = get_featurizer("H236", {})(_input_df())
        # every H236 column survives, in order, followed by the DS_ block
        self.assertEqual(list(out.columns)[:h236.shape[1]], list(h236.columns))
        # the descriptor block is prefixed so it cannot collide with H236's TPSA / LogP
        self.assertTrue(all(c.startswith("DS_") for c in out.columns[h236.shape[1]:]))
        self.assertGreater(out.shape[1] - h236.shape[1], 100)

    def test_unknown_features_type_raises(self):
        """Requesting a features_type neither built-in nor externally mapped raises KeyError."""
        # nonexistent featurizer -> KeyError
        with self.assertRaises(KeyError):
            get_featurizer("NOPE", {})


class TestParityWithRdkitTools(unittest.TestCase):
    """The built-ins must match the external Rdkit_tools bit-for-bit."""

    @classmethod
    def setUpClass(cls):
        # the parity check is only meaningful where Rdkit_tools is importable; skip otherwise
        cfg = {"featurizers": helpers.EXTERNAL_FEATURIZERS}
        try:
            get_featurizer("MF_2048", cfg)
        except (ImportError, ModuleNotFoundError, AttributeError) as e:
            raise unittest.SkipTest(f"Rdkit_tools not available: {e}")
        cls.external_cfg = cfg
        cls.df = _input_df()

    def _assert_parity(self, features_type):
        builtin = get_featurizer(features_type, {})(self.df)
        external = get_featurizer(features_type, self.external_cfg)(self.df)
        # built-in output is identical to Rdkit_tools (columns, order, dtypes, values)
        pd.testing.assert_frame_equal(builtin, external)

    def test_mf2048_parity(self):
        """Built-in MF_2048 == Rdkit_tools.get_MF_bits_from_df."""
        self._assert_parity("MF_2048")

    def test_h236_parity(self):
        """Built-in H236 == Rdkit_tools.compute_H236_features."""
        self._assert_parity("H236")

    def test_h237_parity(self):
        """Built-in H237 == Rdkit_tools.compute_H237_features (needs descriptastorus)."""
        try:
            self._assert_parity("H237")
        except ImportError as e:
            self.skipTest(f"descriptastorus not installed: {e}")


if __name__ == "__main__":
    unittest.main()
