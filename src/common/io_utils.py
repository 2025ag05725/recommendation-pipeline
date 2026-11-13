"""Small IO helpers: JSON writing and raw CSV reading with traceable row ids."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


class _Enc(json.JSONEncoder):
    def default(self, o):  # numpy / pandas scalars
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return None if np.isnan(o) else float(o)
        if isinstance(o, (pd.Timestamp,)):
            return o.isoformat()
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, Path):
            return str(o)
        return super().default(o)


def write_json(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        json.dump(obj, fh, indent=2, cls=_Enc)


def read_json(path: Path):
    with open(path) as fh:
        return json.load(fh)


def read_raw_csv(path: Path) -> pd.DataFrame:
    """Read a raw CSV with every column as string (so type problems stay visible).

    `_row_id` is the 1-based line number in the source file (header = line 1), which
    lets every rejected record be traced back to the original file.
    """
    df = pd.read_csv(path, dtype=str, keep_default_na=True)
    df.insert(0, "_row_id", np.arange(2, len(df) + 2))
    return df


def frame_path(base: Path, fmt: str) -> Path:
    """`base` has no suffix; the suffix follows the storage format."""
    return base.parent / f"{base.name}.{'parquet' if fmt == 'parquet' else 'csv'}"


def write_frame(df: pd.DataFrame, base: Path, fmt: str = "parquet", index: bool = False) -> Path:
    path = frame_path(base, fmt)
    path.parent.mkdir(parents=True, exist_ok=True)
    if fmt == "parquet":
        df.to_parquet(path, engine="pyarrow", index=index)
    else:
        df.to_csv(path, index=index)
    return path


def read_frame(base: Path, fmt: str = "parquet", parse_dates: list[str] | None = None, dtype=None, **kw) -> pd.DataFrame:
    """Parquet keeps dtypes; CSV needs parse_dates/dtype hints (ignored for parquet)."""
    path = frame_path(base, fmt)
    if fmt == "parquet":
        return pd.read_parquet(path, engine="pyarrow")
    return pd.read_csv(path, parse_dates=parse_dates, dtype=dtype, **kw)


def file_md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()
