"""Shared helpers for the MLTrail unittest suite. PUBLIC SMILES + synthetic labels only.

Imported by the test modules (tests/ is placed on sys.path by each test file). Provides a
config factory, a fresh Registry, and builders for the {'model','feature_cols',...} dict
artifacts MLTrail loads — so tests never touch project data.
"""
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from mltrail import Registry
from mltrail.featurizers import get_featurizer

FEATURIZERS = {"path": "/home/gtamo/Scripts", "module": "Rdkit_tools",
               "map": {"MF_2048": "get_MF_bits_from_df", "H236": "compute_H236_features"}}

# ethanol, benzene, aspirin, triethylamine, diethyl ether
PUBLIC_SMILES = ["CCO", "c1ccccc1", "CC(=O)Oc1ccccc1C(=O)O", "CCN(CC)CC", "CCOCC"]


def make_config(tmp_dir):
    """A MLTrail config dict with an isolated registry file under tmp_dir."""
    return {"registry_path": str(tmp_dir / "registry.json"),
            "training_sets_dir": str(tmp_dir / "training_sets"),
            "date_format": "%Y%m%d_%H%M%S", "featurizers": FEATURIZERS}


def new_registry(tmp_dir):
    """A fresh Registry backed by an isolated temp config."""
    return Registry.from_config(make_config(tmp_dir))


def register(registry, model_path="/fake/model.joblib", model_type="single_task_regression",
             features_type="MF_2048", framework="sklearn", experiment_name="demo",
             experiment_measure="demo", **extra):
    """Register a model with sensible defaults; return its id."""
    return registry.add(experiment_name=experiment_name, experiment_measure=experiment_measure,
                        unit="u", model_path=model_path, model_type=model_type,
                        framework=framework, features_type=features_type, **extra)


def public_train_df():
    """Public training molecules as df[compound, smiles]."""
    return pd.DataFrame({"compound": [f"C_{i}" for i in range(len(PUBLIC_SMILES))],
                         "smiles": PUBLIC_SMILES})


def train_artifact(config, tmp_dir, name, task="regression", features_type="MF_2048",
                   feature_subset=None, seed=0):
    """Train a tiny model on public molecules; dump it as a {'model','feature_cols'} artifact.

    Returns the artifact path. `feature_subset` (used for H236) stores only a subset of the
    feature columns, to exercise the predict-time reindex/alignment against the full universe.
    """
    feats = get_featurizer(features_type, config)(public_train_df()).drop(columns="compound")
    X = feats[feature_subset] if feature_subset is not None else feats
    np.random.seed(seed)
    if task == "classification":
        estimator, y = RandomForestClassifier(n_estimators=8, random_state=0), np.random.randint(0, 2, len(X))
    else:
        estimator, y = RandomForestRegressor(n_estimators=8, random_state=0), np.random.rand(len(X))
    estimator.fit(X, y)
    path = tmp_dir / f"{name}.joblib"
    joblib.dump({"model": estimator, "feature_cols": list(X.columns), "features": features_type}, path)
    return str(path)


def write_csv(tmp_dir):
    """A prediction CSV with public SMILES plus one INVALID SMILES (null-handling test)."""
    path = tmp_dir / "data.csv"
    pd.DataFrame({"cid": ["P1", "P2", "P3", "P4"],
                  "smi": ["CCO", "c1ccccc1", "not_a_smiles", "CC(=O)O"]}).to_csv(path, index=False)
    return str(path)


def write_sdf(tmp_dir):
    """A 2-molecule SDF carrying a 'cid' property for compound ids."""
    from rdkit import Chem
    path = tmp_dir / "mols.sdf"
    writer = Chem.SDWriter(str(path))
    for cid, smi in [("S1", "CCO"), ("S2", "c1ccccc1")]:
        mol = Chem.MolFromSmiles(smi)
        mol.SetProp("cid", cid)
        writer.write(mol)
    writer.close()
    return str(path)
