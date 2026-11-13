"""Configuration loading. Paths in the YAML are resolved against the project root."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "config.yaml"


class Config(dict):
    """A dict with helpers for resolved paths."""

    root: Path = PROJECT_ROOT

    @property
    def fmt(self) -> str:
        return self.get("storage", {}).get("format", "parquet")

    def write_table(self, df, key: str, name: str, index: bool = False) -> Path:
        from src.common.io_utils import write_frame
        return write_frame(df, self.path(key, name), self.fmt, index=index)

    def read_table(self, key: str, name: str, parse_dates=None, dtype=None, **kw):
        from src.common.io_utils import read_frame
        return read_frame(self.path(key, name), self.fmt, parse_dates=parse_dates, dtype=dtype, **kw)

    def path(self, key: str, *parts: str) -> Path:
        p = self.root / self["paths"][key]
        p.mkdir(parents=True, exist_ok=True)
        return p.joinpath(*parts)


def _deep_merge(base: dict, new: dict) -> dict:
    out = dict(base)
    for k, v in new.items():
        out[k] = _deep_merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def _read_yaml(path: Path) -> dict:
    with open(path) as fh:
        data = yaml.safe_load(fh)
    parent = data.pop("extends", None)          # `extends: config.yaml` -> inherit everything, override only what is listed
    return _deep_merge(_read_yaml(path.parent / parent), data) if parent else data


def load_config(path: str | Path | None = None, overrides: dict[str, Any] | None = None) -> Config:
    path = Path(path) if path else DEFAULT_CONFIG
    cfg = Config(_read_yaml(path))
    for dotted, value in (overrides or {}).items():
        node = cfg
        *parents, leaf = dotted.split(".")
        for part in parents:
            node = node[part]
        node[leaf] = value
    return cfg
