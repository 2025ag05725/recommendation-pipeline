"""Shared parsing helpers used by validation AND preparation (single source of truth)."""
from __future__ import annotations

import pandas as pd

TS_FORMATS = ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d")
INT_RE = r"\d+"


def parse_timestamps(s: pd.Series) -> pd.Series:
    """Parse mixed-format timestamps (ISO variants, epoch seconds). Unparseable -> NaT."""
    s = s.astype("string").str.strip()
    out = pd.Series(pd.NaT, index=s.index, dtype="datetime64[ns]")
    is_epoch = s.str.fullmatch(r"\d{9,11}").fillna(False).astype(bool)
    if is_epoch.any():
        ep = pd.to_datetime(pd.to_numeric(s[is_epoch]), unit="s", errors="coerce").astype("datetime64[ns]")
        out = out.combine_first(ep)
    rest = s.notna() & ~is_epoch
    for fmt in TS_FORMATS:
        todo = rest & out.isna()
        if todo.any():
            piece = pd.to_datetime(s[todo], format=fmt, errors="coerce").astype("datetime64[ns]")
            out = out.combine_first(piece)
    return out


def is_int_string(s: pd.Series) -> pd.Series:
    """True where the (string) value is a non-negative integer literal."""
    return s.astype("string").str.strip().str.fullmatch(INT_RE).fillna(False).astype(bool)
