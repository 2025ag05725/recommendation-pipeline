"""Data preparation: turn validated-but-raw (string) tables into clean, typed, standardised tables.

Repairs applied (all counted in data/processed/preparation_summary.json):
  users        invalid/missing age -> median ; missing gender -> 'U' ; missing country -> 'UNK'
               gender codes standardised to M/F/O/U ; country upper-cased
  items        missing title -> 'Unknown title' ; missing genres -> 'Unknown' ; genres title-cased, de-duplicated,
               sorted, '|'-joined ; invalid release_year -> median
  interactions types cast ; mixed timestamp formats -> single datetime ; duplicate (user,item) -> latest kept
"""
from __future__ import annotations

import pandas as pd

from src.common.io_utils import write_json
from src.common.logging_utils import get_logger
from src.common.parsing import parse_timestamps
from src.ingestion.generate_synthetic import GENRES

log = get_logger("preparation")
DEFAULT_AGE, DEFAULT_YEAR = 30, 2000          # only used when a column has no valid value at all
GENDER_MAP = {"m": "M", "male": "M", "f": "F", "female": "F", "other": "O", "o": "O", "u": "U"}


def _standardise_genres(s: pd.Series, vocabulary) -> pd.Series:
    canon = {g.lower(): g for g in vocabulary}

    def fix(v):
        if pd.isna(v):
            return "Unknown"
        toks = {canon.get(t.strip().lower()) for t in str(v).split("|")} - {None}
        return "|".join(sorted(toks)) if toks else "Unknown"
    return s.map(fix)


def prepare_users(df: pd.DataFrame, cfg: dict, stats: dict) -> pd.DataFrame:
    v = cfg["validation"]
    out = pd.DataFrame({"user_id": df["user_id"].astype(int)})
    age = pd.to_numeric(df["age"], errors="coerce")
    age = age.where((age >= v["age_min"]) & (age <= v["age_max"]))
    stats["users_age_imputed"] = int(age.isna().sum())
    median_age = int(age.median()) if age.notna().any() else DEFAULT_AGE   # e.g. ml-latest-small has no age at all
    out["age"] = age.fillna(median_age).astype(int)
    g = df["gender"].astype("string").str.strip().str.lower().map(GENDER_MAP)
    stats["users_gender_filled"] = int(g.isna().sum())
    out["gender"] = g.fillna("U").astype(str)
    c = df["country"].astype("string").str.strip().str.upper()
    stats["users_country_filled"] = int(c.isna().sum())
    out["country"] = c.fillna("UNK").astype(str)
    out["signup_date"] = parse_timestamps(df["signup_date"]).dt.normalize()
    return out.sort_values("user_id").reset_index(drop=True)


def prepare_items(df: pd.DataFrame, cfg: dict, stats: dict) -> pd.DataFrame:
    ref_year = pd.Timestamp(cfg["reference_date"]).year
    out = pd.DataFrame({"item_id": df["item_id"].astype(int)})
    stats["items_title_filled"] = int(df["title"].isna().sum())
    out["title"] = df["title"].astype("string").str.strip().fillna("Unknown title").astype(str)
    stats["items_genres_unknown"] = int(df["genres"].isna().sum())
    out["genres"] = _standardise_genres(df["genres"], cfg.get("genre_vocabulary", GENRES))
    yr = pd.to_numeric(df["release_year"], errors="coerce")
    yr = yr.where((yr >= cfg["validation"]["release_year_min"]) & (yr <= ref_year))
    stats["items_year_imputed"] = int(yr.isna().sum())
    out["release_year"] = yr.fillna(yr.median() if yr.notna().any() else DEFAULT_YEAR).astype(int)
    return out.sort_values("item_id").reset_index(drop=True)


def prepare_interactions(df: pd.DataFrame, stats: dict) -> pd.DataFrame:
    out = pd.DataFrame({
        "user_id": df["user_id"].astype(int), "item_id": df["item_id"].astype(int),
        "rating": pd.to_numeric(df["rating"]).astype(float), "timestamp": parse_timestamps(df["timestamp"]),
    })
    before = len(out)
    out = out.sort_values(["timestamp", "user_id", "item_id"]).drop_duplicates(["user_id", "item_id"], keep="last")
    stats["interactions_duplicate_pairs_dropped"] = before - len(out)
    return out.sort_values(["timestamp", "user_id", "item_id"]).reset_index(drop=True)


def run(cfg) -> dict:
    proc = cfg.path("processed")
    stats: dict = {}
    users = prepare_users(cfg.read_table("processed", "valid_users", dtype=str), cfg, stats)
    items = prepare_items(cfg.read_table("processed", "valid_items", dtype=str), cfg, stats)
    inter = prepare_interactions(cfg.read_table("processed", "valid_interactions", dtype=str), stats)
    # Final guard: ensure integrity survived (validation already enforced it)
    assert inter["user_id"].isin(users["user_id"]).all() and inter["item_id"].isin(items["item_id"]).all()
    cfg.write_table(users, "processed", "users_clean")
    cfg.write_table(items, "processed", "items_clean")
    cfg.write_table(inter, "processed", "interactions_clean")
    stats.update(users=len(users), items=len(items), interactions=len(inter))
    write_json(stats, proc / "preparation_summary.json")
    log.info("Prepared: %s", stats)
    return stats
