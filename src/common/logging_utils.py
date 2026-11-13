"""Logging setup: console + optional file, one shared 'src' logger tree."""
from __future__ import annotations

import logging
import sys
from pathlib import Path

_FMT = "%(asctime)s | %(levelname)-7s | %(threadName)-16s | %(name)s | %(message)s"


def setup_logging(log_file: Path | None = None, level: int = logging.INFO) -> logging.Logger:
    root = logging.getLogger("src")
    root.setLevel(level)
    root.propagate = False
    if not any(isinstance(h, logging.StreamHandler) and getattr(h, "_pipeline", False) for h in root.handlers):
        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(logging.Formatter(_FMT))
        sh._pipeline = True  # type: ignore[attr-defined]
        root.addHandler(sh)
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file)
        fh.setFormatter(logging.Formatter(_FMT))
        fh._pipeline_file = True  # type: ignore[attr-defined]
        root.addHandler(fh)
    return root


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name if name.startswith("src") else f"src.{name}")
