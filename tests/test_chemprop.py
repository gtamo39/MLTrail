"""chemprop backend + end-to-end predict tests. The chemprop CLI is MOCKED (no GPU / real model /
chemprop env needed): a fake subprocess reads the input SMILES CSV and writes a preds CSV with one
column per target, so the whole register -> predict path is exercised deterministically.

Run from the repo root:
    python -m unittest discover -s tests
or:
    python tests/test_chemprop.py
"""
import os, sys, subprocess, tempfile, unittest
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))   # so `import helpers` resolves

import pandas as pd

import helpers
from mltrail import Registry
from mltrail.backends import chemprop_predict, resolve_checkpoint

TARGETS = ["hlm", "mlm", "rlm"]


def _fake_chemprop_cli(targets=TARGETS):
    """A subprocess.run stand-in for `chemprop predict`: reads `-i` SMILES, writes `--preds-path`
    with one deterministic column per target (value = row index), preserving row order/count."""
    def run(cmd, capture_output=True, text=True):
        args = {cmd[i]: cmd[i + 1] for i in range(len(cmd) - 1)}
        smi = pd.read_csv(args["-i"])["smiles"]
        out = pd.DataFrame({"smiles": smi.values})
        for j, t in enumerate(targets):
            out[t] = [float(i + j) for i in range(len(smi))]      # distinct per target/row
        out.to_csv(args["--preds-path"], index=False)
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")
    return run


def _build():
    tmp = Path(tempfile.mkdtemp())
    config = helpers.make_config(tmp)
    reg = Registry.from_config(config)

    # a stand-in chemprop model dir: resolve_checkpoint only needs a best*.pt to exist (CLI is mocked)
    model_dir = tmp / "clearance"; (model_dir / "model_0" / "checkpoints").mkdir(parents=True)
    (model_dir / "model_0" / "best.pt").write_bytes(b"stub")
    (model_dir / "model_0" / "checkpoints" / "last.ckpt").write_bytes(b"stub")

    mid = helpers.register(reg, str(model_dir), model_type="multitask_regression",
                           framework="chemprop", features_type="smiles",
                           experiment_name="adme_mt_clearance", target_columns=TARGETS)
    csv = helpers.write_csv(tmp)   # P1..P4, one INVALID SMILES
    orig = subprocess.run
    subprocess.run = _fake_chemprop_cli()
    try:
        preds = reg.predict(mid, csv, smiles_column="smi", compound_id="cid")
    finally:
        subprocess.run = orig
    return dict(reg=reg, mid=mid, model_dir=model_dir, preds=preds, config=config)


class TestChempropBackend(unittest.TestCase):
    """resolve_checkpoint + chemprop_predict (mocked CLI)."""

    @classmethod
    def setUpClass(cls):
        cls.b = _build()

    def test_resolve_checkpoint_prefers_pt_in_dir(self):
        """A model directory resolves to its exported best*.pt (not a raw .ckpt)."""
        ckpt = resolve_checkpoint(self.b["model_dir"])
        # the exported .pt is chosen over lightning .ckpt files
        self.assertEqual(ckpt.suffix, ".pt")

    def test_resolve_checkpoint_accepts_file(self):
        """A checkpoint file path is returned verbatim."""
        f = self.b["model_dir"] / "model_0" / "best.pt"
        # an explicit file path passes through unchanged
        self.assertEqual(resolve_checkpoint(f), f)

    def test_resolve_checkpoint_missing_raises(self):
        """A directory with no checkpoint raises FileNotFoundError."""
        empty = Path(tempfile.mkdtemp())
        # no .pt/.ckpt under the dir -> explicit error
        with self.assertRaises(FileNotFoundError):
            resolve_checkpoint(empty)

    def test_chemprop_predict_renames_to_targets(self):
        """chemprop_predict returns one column per target, renamed to target_columns."""
        subprocess.run, orig = _fake_chemprop_cli(), subprocess.run
        try:
            out = chemprop_predict(str(self.b["model_dir"]), ["CCO", "c1ccccc1"],
                                   self.b["config"], target_columns=TARGETS)
        finally:
            subprocess.run = orig
        # columns are the endpoint names, one row per input SMILES
        self.assertEqual(list(out.columns), TARGETS)
        self.assertEqual(len(out), 2)


class TestChempropPredict(unittest.TestCase):
    """End-to-end registry.predict for a chemprop multitask model (mocked CLI)."""

    @classmethod
    def setUpClass(cls):
        cls.b = _build()

    def test_output_schema_multitask(self):
        """Output is smiles, compound, then one column per target endpoint."""
        out = self.b["preds"]
        # spec order: identifiers first, then every target column
        self.assertEqual(list(out.columns), ["smiles", "compound", *TARGETS])
        # one row per input row (4)
        self.assertEqual(len(out), 4)

    def test_invalid_smiles_gives_null_across_targets(self):
        """The unparseable SMILES row is null for every target; the 3 valid rows are predicted."""
        out = self.b["preds"]
        bad = out.loc[out["smiles"] == "not_a_smiles", TARGETS]
        # invalid row is null across all targets
        self.assertTrue(bad.isna().all(axis=None))
        # exactly the 3 valid molecules are scored on every target
        self.assertEqual(int(out[TARGETS].notna().all(axis=1).sum()), 3)

    def test_targets_distinct(self):
        """Each target column is populated independently (not a single value broadcast)."""
        out = self.b["preds"].dropna(subset=TARGETS)
        # the three endpoint columns are not all identical
        self.assertFalse((out["hlm"] == out["rlm"]).all())


if __name__ == "__main__":
    unittest.main()
