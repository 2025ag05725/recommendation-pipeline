"""Index mappings and sparse user-item matrices shared by transformation, training and evaluation."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import sparse


def build_index(users: pd.DataFrame, items: pd.DataFrame) -> tuple[dict[int, int], dict[int, int]]:
    """Catalogue-wide, deterministic id -> row/col index (sorted ids)."""
    uidx = {int(u): i for i, u in enumerate(sorted(users["user_id"].unique()))}
    iidx = {int(i): j for j, i in enumerate(sorted(items["item_id"].unique()))}
    return uidx, iidx


def build_matrix(df: pd.DataFrame, uidx: dict, iidx: dict, value: str | None = "rating") -> sparse.csr_matrix:
    """Sparse (n_users x n_items) matrix. value=None -> binary indicator of interaction."""
    r = df["user_id"].map(uidx).to_numpy()
    c = df["item_id"].map(iidx).to_numpy()
    keep = ~(pd.isna(r) | pd.isna(c))
    data = np.ones(keep.sum()) if value is None else df[value].to_numpy(dtype=float)[keep]
    return sparse.csr_matrix((data, (r[keep].astype(int), c[keep].astype(int))), shape=(len(uidx), len(iidx)))
