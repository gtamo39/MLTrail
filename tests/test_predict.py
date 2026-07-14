"""End-to-end prediction tests (public SMILES + tiny models trained in-fixture).

The models are built and all predictions run ONCE (see _build / _CACHE); each test only
asserts on the cached DataFrames, so assertions can be tweaked without re-running.

Run from the repo root:
    python -m unittest discover -s tests
or:
    python tests/test_predict.py
"""
import os, sys, tempfile, unittest
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))   # so `import helpers` resolves

import helpers
from mltrail import Registry

_CACHE = {}
_H236_SUBSET = [f"F{i}" for i in range(30)] + ["MW", "TPSA"] + [f"AP_{i}" for i in range(10)]


def _build():
    """Register regression / classification / H236 / generative models and run predictions once."""
    if "built" not in _CACHE:
        tmp = Path(tempfile.mkdtemp())
        config = helpers.make_config(tmp)
        reg = Registry.from_config(config)

        mf_reg = helpers.train_artifact(config, tmp, "mf_reg", "regression", "MF_2048", seed=0)
        mf_clf = helpers.train_artifact(config, tmp, "mf_clf", "classification", "MF_2048", seed=1)
        h236 = helpers.train_artifact(config, tmp, "h236_reg", "regression", "H236", _H236_SUBSET, seed=2)
        csv, sdf = helpers.write_csv(tmp), helpers.write_sdf(tmp)

        ids = dict(
            reg=helpers.register(reg, mf_reg),
            clf=helpers.register(reg, mf_clf, model_type="single_task_classification"),
            h236=helpers.register(reg, h236, features_type="H236"),
            gen=helpers.register(reg, "/fake/r.ckpt", model_type="generative",
                                 framework="reinvent", features_type="n/a"),
            bad=helpers.register(reg, helpers.broken_artifact(tmp)),
        )
        _CACHE["built"] = dict(
            registry=reg, csv=csv, sdf=sdf, mf_reg=mf_reg, ids=ids,
            preds_reg=reg.predict(ids["reg"], csv, smiles_column="smi", compound_id="cid"),
            preds_clf=reg.predict(ids["clf"], csv, smiles_column="smi", compound_id="cid"),
            preds_h236=reg.predict(ids["h236"], csv, smiles_column="smi", compound_id="cid"),
            preds_na=reg.predict(ids["reg"], csv, smiles_column="smi", compound_id="n/a"),
            preds_override=reg.predict(ids["bad"], csv, smiles_column="smi", model_path=mf_reg),
            preds_sdf=reg.predict(ids["reg"], sdf, compound_id="cid"),
        )
    return _CACHE["built"]


class TestPredict(unittest.TestCase):
    """Assertions on the cached prediction outputs."""

    @classmethod
    def setUpClass(cls):
        cls.b = _build()

    def test_regression_output_schema(self):
        """A regression model yields exactly smiles, compound, prediction — one row per input."""
        out = self.b["preds_reg"]
        # output columns follow the spec order
        self.assertEqual(list(out.columns), ["smiles", "compound", "prediction"])
        # one prediction row per input row (4)
        self.assertEqual(len(out), 4)

    def test_invalid_smiles_gives_null_prediction(self):
        """Rows whose SMILES fail to parse get a null prediction rather than being dropped."""
        out = self.b["preds_reg"]
        # the deliberately-invalid SMILES row has a null prediction
        self.assertTrue(out.loc[out["smiles"] == "not_a_smiles", "prediction"].isna().all())
        # the three valid rows are predicted
        self.assertEqual(out["prediction"].notna().sum(), 3)

    def test_classification_has_probability(self):
        """A classification model adds a probability column bounded in [0, 1]."""
        out = self.b["preds_clf"]
        # classification output carries a probability column
        self.assertIn("probability", out.columns)
        # probabilities lie within [0, 1]
        self.assertTrue(out["probability"].dropna().between(0, 1).all())

    def test_h236_feature_alignment(self):
        """H236 predict reindexes the 4269-col universe to the model's stored feature_cols subset."""
        # the three valid molecules are scored despite the universe/subset column mismatch
        self.assertEqual(self.b["preds_h236"]["prediction"].notna().sum(), 3)

    def test_generative_predict_raises(self):
        """Predicting with a generative model raises ValueError (metadata-only)."""
        # generative models cannot predict
        with self.assertRaisesRegex(ValueError, "generative"):
            self.b["registry"].predict(self.b["ids"]["gen"], self.b["csv"], smiles_column="smi")

    def test_model_path_override(self):
        """A model_path override is used instead of the registered (nonexistent) path."""
        # prediction succeeds using the overriding path
        self.assertEqual(self.b["preds_override"]["prediction"].notna().sum(), 3)

    def test_compound_id_na_uses_row_index(self):
        """When compound_id is n/a, the compound column falls back to row indices."""
        # compound ids become 0-based row indices
        self.assertEqual(list(self.b["preds_na"]["compound"]), [0, 1, 2, 3])

    def test_sdf_reading(self):
        """An SDF dataset is read via RDKit and scored, using the 'cid' property as compound id."""
        out = self.b["preds_sdf"]
        # both SDF molecules are scored and keyed by their cid property
        self.assertEqual(list(out["compound"]), ["S1", "S2"])
        self.assertTrue(out["prediction"].notna().all())


if __name__ == "__main__":
    unittest.main()
