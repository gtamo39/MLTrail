"""Featurizer resolution + MLTrail's built-in, standalone featurizers.

A featurizer's contract is ``df[compound, smiles] -> df[compound, <feature cols>]``.

MLTrail ships **built-in** featurizers (``MF_2048``, ``H236``) that depend only on RDKit +
numpy/pandas, so it featurizes on its own with no external module or ``sys.path`` wiring.
The built-ins are vendored bit-for-bit from the shared ``Rdkit_tools`` (same generators,
params and column names) — a parity test guards against any drift. An external module named
in the config's ``featurizers`` section still overrides them when configured.
"""
import importlib
import sys

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, MACCSkeys, rdFingerprintGenerator


def morgan_2048(df_, nBits=2048, radius=2):
    """Morgan (ECFP4) fingerprint bits: ``df[compound, smiles] -> df[compound, F0..F{nBits-1}]``.

    Dense int8 matrix; rows whose SMILES fail to parse are dropped. Vendored verbatim from
    ``Rdkit_tools.get_MF_bits_from_df`` (GetMorganGenerator, radius=2, includeChirality=True).
    """
    df = df_[['compound', 'smiles']].reset_index(drop=True)
    n = len(df)
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=nBits, includeChirality=True)

    bits = np.zeros((n, nBits), dtype=np.int8)
    valid = np.zeros(n, dtype=bool)
    for i in range(n):
        mol = Chem.MolFromSmiles(str(df.at[i, 'smiles']))
        if mol is None:
            continue
        DataStructs.ConvertToNumpyArray(gen.GetFingerprint(mol), bits[i])
        valid[i] = True

    out = pd.DataFrame(bits[valid], columns=['F%d' % i for i in range(nBits)])
    out.insert(0, 'compound', df.loc[valid, 'compound'].values)
    return out


def h236(df_, nBits_morgan=2048, nBits_ap=2048, radius=2):
    """The full H236 feature universe in one call.

    ``df[compound, smiles] -> df[compound, F0.., Hba,Hbd,MW,TPSA,LogP,NRB, MACCS_0.., AP_0..]``:
    Morgan (radius=2, chirality) + 6 physchem descriptors + 167 MACCS keys + hashed AtomPair.
    Returns the FULL bit-vectors (the model's stored feature_cols selects its >2%-prevalence
    subset downstream). Rows with unparseable SMILES are dropped. Vendored verbatim from
    ``Rdkit_tools.compute_H236_features``.
    """
    df = df_[['compound', 'smiles']].reset_index(drop=True)
    n = len(df)

    morgan_gen = rdFingerprintGenerator.GetMorganGenerator(
        radius=radius, fpSize=nBits_morgan, includeChirality=True,
    )
    ap_gen = rdFingerprintGenerator.GetAtomPairGenerator(fpSize=nBits_ap)

    physchem_cols = ['Hba', 'Hbd', 'MW', 'TPSA', 'LogP', 'NRB']
    n_maccs = 167   # RDKit MACCS keys are always 167 bits

    morgan_bits = np.zeros((n, nBits_morgan), dtype=np.int8)
    maccs_bits = np.zeros((n, n_maccs), dtype=np.int8)
    ap_bits = np.zeros((n, nBits_ap), dtype=np.int8)
    physchem = np.full((n, len(physchem_cols)), np.nan, dtype=np.float64)
    valid = np.zeros(n, dtype=bool)

    for i in range(n):
        smi = df.at[i, 'smiles']
        if not isinstance(smi, str) or not smi:
            continue
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        DataStructs.ConvertToNumpyArray(morgan_gen.GetFingerprint(mol), morgan_bits[i])
        DataStructs.ConvertToNumpyArray(MACCSkeys.GenMACCSKeys(mol), maccs_bits[i])
        DataStructs.ConvertToNumpyArray(ap_gen.GetFingerprint(mol), ap_bits[i])
        physchem[i] = [
            AllChem.CalcNumHBA(mol),
            AllChem.CalcNumHBD(mol),
            AllChem.CalcExactMolWt(mol),
            AllChem.CalcTPSA(mol),
            AllChem.CalcCrippenDescriptors(mol)[0],
            AllChem.CalcNumRotatableBonds(mol),
        ]
        valid[i] = True

    morgan_cols = [f'F{i}' for i in range(nBits_morgan)]
    maccs_cols = [f'MACCS_{i}' for i in range(n_maccs)]
    ap_cols = [f'AP_{i}' for i in range(nBits_ap)]

    return pd.concat([
        pd.DataFrame({'compound': df.loc[valid, 'compound'].values}),
        pd.DataFrame(morgan_bits[valid], columns=morgan_cols),
        pd.DataFrame(physchem[valid], columns=physchem_cols),
        pd.DataFrame(maccs_bits[valid], columns=maccs_cols),
        pd.DataFrame(ap_bits[valid], columns=ap_cols),
    ], axis=1)


def h237(df_, nBits_morgan=2048, nBits_ap=2048, radius=2, normalized=True, prefix='DS_'):
    """H236 plus the descriptastorus RDKit2D descriptor block.

    ``df[compound, smiles] -> df[compound, <every H236 column>, DS_<descriptor>...]``
    (~200 descriptors). Prefixed because descriptastorus emits names that collide with
    H236's physchem block (``TPSA``, ``MolLogP`` vs ``TPSA``/``LogP``). ``normalized=True``
    uses the CDF-normalised generator ([0, 1] values) — the setting the models are trained
    with. Rows whose SMILES or descriptors fail are dropped, as in H236. Vendored verbatim
    from ``Rdkit_tools.compute_H237_features`` (serial path).

    Unlike the other built-ins this needs the optional ``descriptastorus`` package; H236 and
    MF_2048 stay usable without it.
    """
    try:
        from descriptastorus.descriptors import rdDescriptors, rdNormalizedDescriptors
    except ImportError as e:
        raise ImportError(
            "H237 features need the optional 'descriptastorus' package. Install it into the "
            "project conda env (never base): pip install "
            "git+https://github.com/bp-kelley/descriptastorus"
        ) from e

    base = h236(df_, nBits_morgan=nBits_morgan, nBits_ap=nBits_ap, radius=radius)
    smiles = df_.drop_duplicates('compound').set_index('compound')['smiles']

    gen = rdNormalizedDescriptors.RDKit2DNormalized() if normalized else rdDescriptors.RDKit2D()
    cols = [f'{prefix}{name}' for name, _ in gen.columns]

    values = np.full((len(base), len(cols)), np.nan, dtype=np.float64)
    valid = np.zeros(len(base), dtype=bool)
    for i, cmp in enumerate(base['compound']):
        smi = smiles.get(cmp)
        if not isinstance(smi, str) or not smi:
            continue
        res = gen.process(smi)
        if res and res[0]:              # res[0] is the success flag
            values[i] = res[1:]
            valid[i] = True

    return pd.concat([base.loc[valid].reset_index(drop=True),
                      pd.DataFrame(values[valid], columns=cols)], axis=1)


# features_type -> built-in callable. Extend here to add a new standalone featurizer.
BUILTIN_FEATURIZERS = {"MF_2048": morgan_2048, "H236": h236, "H237": h237}


def get_featurizer(features_type, config):
    """Return the featurizer callable for `features_type`.

    Resolution: an external module wins **only** when the config's `featurizers` section names
    a `module` that maps this features_type (the override); otherwise the built-in is used.
    Raises KeyError if neither provides the requested features_type.
    """
    fcfg = config.get("featurizers", {}) or {}
    mapping = fcfg.get("map", {})
    module = fcfg.get("module")
    if module and features_type in mapping:
        path = fcfg.get("path")
        if path and path not in sys.path:
            sys.path.insert(0, path)
        return getattr(importlib.import_module(module), mapping[features_type])
    if features_type in BUILTIN_FEATURIZERS:
        return BUILTIN_FEATURIZERS[features_type]
    raise KeyError(
        f"no featurizer registered for features_type {features_type!r}; "
        f"built-ins: {sorted(BUILTIN_FEATURIZERS)}, external map: {sorted(mapping)}"
    )
