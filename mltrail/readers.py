"""Dataset readers: load csv / tsv / excel / sdf into a df with 'compound' and 'smiles' columns."""
from pathlib import Path

import pandas as pd


def read_dataset(path, smiles_column=None, compound_id=None):
    """Read a prediction dataset into a DataFrame with 'compound' and 'smiles' columns.

    csv/tsv/excel require `smiles_column`; SDF derives SMILES from the embedded structure.
    `compound_id` names the id column (csv/excel) or property (sdf); when absent or "n/a",
    row indices are used as the compound id.
    """
    ext = Path(path).suffix.lower()
    if ext == ".sdf":
        return _read_sdf(path, compound_id)
    if ext in {".csv", ".txt"}:
        df = pd.read_csv(path)
    elif ext == ".tsv":
        df = pd.read_csv(path, sep="\t")
    elif ext in {".xlsx", ".xls"}:
        df = pd.read_excel(path)
    else:
        raise ValueError(f"unsupported dataset format {ext!r}; use csv/tsv/excel/sdf")
    return _standardize(df, smiles_column, compound_id)


def _standardize(df, smiles_column, compound_id):
    """Project a tabular df to columns 'compound' and 'smiles'."""
    if smiles_column not in df.columns:
        raise KeyError(f"smiles_column {smiles_column!r} not in dataset columns {list(df.columns)}")
    has_id = compound_id not in (None, "n/a", "") and compound_id in df.columns
    compound = df[compound_id].values if has_id else range(len(df))
    return pd.DataFrame({"compound": list(compound), "smiles": df[smiles_column].astype(str).values})


def _read_sdf(path, compound_id):
    """Read an SDF into 'compound'/'smiles', deriving canonical SMILES from each molecule."""
    from rdkit import Chem

    smiles, ids = [], []
    for i, mol in enumerate(Chem.SDMolSupplier(str(path))):
        if mol is None:
            continue
        smiles.append(Chem.MolToSmiles(mol))
        ids.append(mol.GetProp(compound_id) if compound_id and mol.HasProp(compound_id) else i)
    return pd.DataFrame({"compound": ids, "smiles": smiles})
