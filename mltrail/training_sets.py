"""Delta-only training-set storage.

Each model has a folder of timestamped parquet chunks. The full training set is the
concatenation of all chunks; saving runs an internal dedup check so only rows not already
stored (across every prior chunk / version) are written — overlapping data is never
duplicated, which keeps the archive small as a model is retrained on growing datasets.
"""
import glob
import os
from datetime import datetime
from pathlib import Path

import pandas as pd


def _chunk_paths(folder):
    """Timestamped parquet chunks in `folder`, oldest first (names sort chronologically)."""
    return sorted(glob.glob(os.path.join(str(folder), "*.parquet")))


def load_full(folder):
    """Return the full training set (concat of all chunks), or an empty df if none exist."""
    paths = _chunk_paths(folder)
    if not paths:
        return pd.DataFrame()
    return pd.concat([pd.read_parquet(p) for p in paths], ignore_index=True)


def save_delta(folder, incoming, dedup_on=None, date_format="%Y%m%d_%H%M%S", timestamp=None):
    """Write only the rows of `incoming` not already stored in `folder`; return a summary.

    dedup_on: columns identifying a row (default: all incoming columns). Duplicates within
    `incoming` are collapsed too. Returns {n_new, n_existing, n_total, chunk}, where chunk is
    None when nothing new was written.
    """
    folder = Path(folder)
    existing = load_full(folder)
    keys = dedup_on or list(incoming.columns)

    incoming = incoming.drop_duplicates(subset=keys).reset_index(drop=True)
    if existing.empty:
        new_rows = incoming
    else:
        seen = set(map(tuple, existing[keys].itertuples(index=False, name=None)))
        mask = [row not in seen for row in incoming[keys].itertuples(index=False, name=None)]
        new_rows = incoming[mask]

    chunk = None
    if not new_rows.empty:
        folder.mkdir(parents=True, exist_ok=True)
        stamp = timestamp or datetime.now().strftime(date_format)
        # index suffix keeps chunks unique + ordered even when saved within the same second
        chunk = folder / f"{stamp}_{len(_chunk_paths(folder)):03d}.parquet"
        new_rows.to_parquet(chunk, index=False)
    return {"n_new": len(new_rows), "n_existing": len(existing),
            "n_total": len(existing) + len(new_rows), "chunk": str(chunk) if chunk else None}


def read_full(path):
    """Read a training set from a file, preserving ALL columns (csv/tsv/excel/parquet/sdf)."""
    ext = Path(path).suffix.lower()
    if ext in {".csv", ".txt"}:
        return pd.read_csv(path)
    if ext == ".tsv":
        return pd.read_csv(path, sep="\t")
    if ext in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    if ext == ".parquet":
        return pd.read_parquet(path)
    if ext == ".sdf":
        from rdkit import Chem
        rows = []
        for mol in Chem.SDMolSupplier(str(path)):
            if mol is None:
                continue
            row = {prop: mol.GetProp(prop) for prop in mol.GetPropNames()}
            row["smiles"] = Chem.MolToSmiles(mol)
            rows.append(row)
        return pd.DataFrame(rows)
    raise ValueError(f"unsupported training-set format {ext!r}; use csv/tsv/excel/parquet/sdf")
