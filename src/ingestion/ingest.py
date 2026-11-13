"""Data ingestion: obtain raw users/items/interactions CSVs and write an integrity manifest.

Source modes (configs/config.yaml -> ingestion.source):
  * synthetic : generate documented synthetic data into data/raw (skipped if files exist unless regenerate=true)
  * csv       : use users.csv / items.csv / interactions.csv already present in data/raw
                (drop a MovieLens export there, renamed to the same columns)

The manifest (row counts, columns, md5) lets later stages verify they validate exactly what was ingested.
"""
from __future__ import annotations

from src.common.io_utils import file_md5, read_raw_csv, write_json
from src.common.logging_utils import get_logger
from src.ingestion.generate_synthetic import generate

log = get_logger("ingestion")
TABLES = {
    "users": ["user_id", "age", "gender", "country", "signup_date"],
    "items": ["item_id", "title", "genres", "release_year"],
    "interactions": ["user_id", "item_id", "rating", "timestamp"],
}


def run(cfg) -> dict:
    raw = cfg.path("raw")
    icfg = cfg["ingestion"]
    present = all((raw / f"{t}.csv").exists() for t in TABLES)
    if icfg["source"] == "synthetic" and (icfg.get("regenerate") or not present):
        generate(cfg, raw)
    elif icfg["source"] == "csv" and not present:
        raise FileNotFoundError(f"ingestion.source=csv but one of {[f'{t}.csv' for t in TABLES]} is missing in {raw}")

    manifest = {}
    for table, expected_cols in TABLES.items():
        path = raw / f"{table}.csv"
        df = read_raw_csv(path)
        cols = [c for c in df.columns if c != "_row_id"]
        missing = [c for c in expected_cols if c not in cols]
        if missing:
            raise ValueError(f"{path.name}: missing required columns {missing}")
        manifest[table] = {"file": str(path.relative_to(cfg.root)), "rows": len(df), "columns": cols, "md5": file_md5(path)}
        log.info("Ingested %-13s rows=%-6d cols=%s", table, len(df), cols)
    write_json(manifest, raw / "manifest.json")
    return {t: m["rows"] for t, m in manifest.items()}
