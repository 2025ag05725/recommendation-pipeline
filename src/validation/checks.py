"""Declarative, vectorised data-quality checks (a lightweight stand-in for Great Expectations / Pandera).

Each check evaluates a boolean "failure mask" over a table and records:
  * how many rows failed (also when zero, so the report lists *passed* checks too)
  * one issue row per failing record (table, row_id, check, column, severity, value)

Severity
  error   -> record is quarantined (removed from the valid dataset, written to data/errors/)
  warning -> record is kept; the preparation stage repairs it (impute / standardise / dedupe)
"""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from src.common.parsing import is_int_string, parse_timestamps

GENDER_OK = {"m", "f", "male", "female", "other", "o", "u"}


@dataclass
class IssueLog:
    results: list[dict] = field(default_factory=list)       # one entry per check (pass or fail)
    issues: list[pd.DataFrame] = field(default_factory=list)

    def check(self, table: str, df: pd.DataFrame, name: str, column: str, severity: str,
              description: str, fail_mask: pd.Series) -> pd.Series:
        fail_mask = fail_mask.fillna(False).astype(bool)
        n_fail = int(fail_mask.sum())
        self.results.append({"table": table, "check": name, "column": column, "severity": severity,
                             "description": description, "rows_checked": int(len(df)),
                             "rows_failed": n_fail, "status": "PASS" if n_fail == 0 else ("FAIL" if severity == "error" else "WARN")})
        if n_fail:
            bad = df.loc[fail_mask]
            self.issues.append(pd.DataFrame({
                "table": table, "row_id": bad["_row_id"].to_numpy(), "check": name, "column": column,
                "severity": severity, "value": bad[column].astype("string").to_numpy() if column in bad else None}))
        return fail_mask

    def issues_df(self) -> pd.DataFrame:
        cols = ["table", "row_id", "check", "column", "severity", "value"]
        return pd.concat(self.issues, ignore_index=True)[cols] if self.issues else pd.DataFrame(columns=cols)


def validate_users(df: pd.DataFrame, cfg: dict, log: IssueLog) -> pd.DataFrame:
    t, v, ref = "users", cfg["validation"], pd.Timestamp(cfg["reference_date"])
    err = pd.Series(False, index=df.index)
    err |= log.check(t, df, "missing_user_id", "user_id", "error", "user_id must not be null", df["user_id"].isna())
    err |= log.check(t, df, "invalid_user_id_type", "user_id", "error", "user_id must be an integer",
                     df["user_id"].notna() & ~is_int_string(df["user_id"]))
    err |= log.check(t, df, "duplicate_user_id", "user_id", "error", "user_id must be unique (first occurrence kept)",
                     df["user_id"].notna() & df.duplicated(subset="user_id", keep="first"))
    ts = parse_timestamps(df["signup_date"])
    err |= log.check(t, df, "invalid_signup_date", "signup_date", "error", "signup_date must be a parseable date",
                     ts.isna())
    err |= log.check(t, df, "future_signup_date", "signup_date", "error", "signup_date must not be after reference date",
                     ts > ref)
    age = pd.to_numeric(df["age"], errors="coerce")
    log.check(t, df, "missing_age", "age", "warning", "age should not be null (imputed in preparation)", df["age"].isna())
    log.check(t, df, "invalid_age", "age", "warning", f"age must be numeric in [{v['age_min']}, {v['age_max']}]",
              df["age"].notna() & (age.isna() | (age < v["age_min"]) | (age > v["age_max"])))
    log.check(t, df, "missing_gender", "gender", "warning", "gender should not be null", df["gender"].isna())
    log.check(t, df, "invalid_gender", "gender", "warning", "gender must be a known code",
              df["gender"].notna() & ~df["gender"].astype("string").str.strip().str.lower().isin(GENDER_OK))
    log.check(t, df, "missing_country", "country", "warning", "country should not be null", df["country"].isna())
    return df.loc[~err]


def validate_items(df: pd.DataFrame, cfg: dict, log: IssueLog, known_genres: set[str]) -> pd.DataFrame:
    t, v, ref = "items", cfg["validation"], pd.Timestamp(cfg["reference_date"])
    err = pd.Series(False, index=df.index)
    err |= log.check(t, df, "missing_item_id", "item_id", "error", "item_id must not be null", df["item_id"].isna())
    err |= log.check(t, df, "invalid_item_id_type", "item_id", "error", "item_id must be an integer",
                     df["item_id"].notna() & ~is_int_string(df["item_id"]))
    err |= log.check(t, df, "duplicate_item_id", "item_id", "error", "item_id must be unique (first occurrence kept)",
                     df["item_id"].notna() & df.duplicated(subset="item_id", keep="first"))
    log.check(t, df, "missing_title", "title", "warning", "title should not be null", df["title"].isna())
    log.check(t, df, "missing_genres", "genres", "warning", "genres should not be null", df["genres"].isna())
    tokens = df["genres"].dropna().astype("string").str.split("|").explode().str.strip().str.lower()
    bad_idx = tokens.index[~tokens.isin({g.lower() for g in known_genres})].unique()
    log.check(t, df, "unknown_genre", "genres", "warning", "every genre token must be in the genre vocabulary",
              pd.Series(df.index.isin(bad_idx), index=df.index))
    yr = pd.to_numeric(df["release_year"], errors="coerce")
    log.check(t, df, "invalid_release_year", "release_year", "warning",
              f"release_year must be an integer in [{v['release_year_min']}, {ref.year}]",
              df["release_year"].notna() & (yr.isna() | (yr < v["release_year_min"]) | (yr > ref.year)))
    return df.loc[~err]


def validate_interactions(df: pd.DataFrame, cfg: dict, log: IssueLog,
                          valid_users: pd.DataFrame, valid_items: pd.DataFrame) -> pd.DataFrame:
    t, v, ref = "interactions", cfg["validation"], pd.Timestamp(cfg["reference_date"])
    err = pd.Series(False, index=df.index)
    data_cols = ["user_id", "item_id", "rating", "timestamp"]

    # --- completeness & types
    err |= log.check(t, df, "missing_user_id", "user_id", "error", "user_id must not be null", df["user_id"].isna())
    err |= log.check(t, df, "missing_item_id", "item_id", "error", "item_id must not be null", df["item_id"].isna())
    err |= log.check(t, df, "missing_rating", "rating", "error", "rating must not be null", df["rating"].isna())
    err |= log.check(t, df, "missing_timestamp", "timestamp", "error", "timestamp must not be null", df["timestamp"].isna())
    err |= log.check(t, df, "invalid_user_id_type", "user_id", "error", "user_id must be an integer",
                     df["user_id"].notna() & ~is_int_string(df["user_id"]))
    err |= log.check(t, df, "invalid_item_id_type", "item_id", "error", "item_id must be an integer",
                     df["item_id"].notna() & ~is_int_string(df["item_id"]))
    rating = pd.to_numeric(df["rating"], errors="coerce")
    err |= log.check(t, df, "invalid_rating_type", "rating", "error", "rating must be numeric",
                     df["rating"].notna() & rating.isna())

    # --- value domain
    lo, hi, step = v["rating_min"], v["rating_max"], v["rating_step"]
    err |= log.check(t, df, "rating_out_of_range", "rating", "error", f"rating must be within [{lo}, {hi}]",
                     rating.notna() & ((rating < lo) | (rating > hi)))
    err |= log.check(t, df, "rating_off_scale", "rating", "error", f"rating must be a multiple of {step}",
                     rating.notna() & (rating >= lo) & (rating <= hi) & (((rating - lo) / step) % 1 != 0))
    ts = parse_timestamps(df["timestamp"])
    err |= log.check(t, df, "invalid_timestamp_format", "timestamp", "error", "timestamp must be parseable",
                     df["timestamp"].notna() & ts.isna())
    err |= log.check(t, df, "timestamp_in_future", "timestamp", "error", "timestamp must not exceed reference date",
                     ts > ref)
    err |= log.check(t, df, "timestamp_too_old", "timestamp", "error", f"timestamp must be >= {v['min_valid_date']}",
                     ts < pd.Timestamp(v["min_valid_date"]))

    # --- referential integrity (against *valid* parent rows)
    uid_ok = set(valid_users["user_id"].astype(str).str.strip())
    iid_ok = set(valid_items["item_id"].astype(str).str.strip())
    err |= log.check(t, df, "orphan_user_id", "user_id", "error", "user_id must exist in valid users",
                     df["user_id"].notna() & is_int_string(df["user_id"]) & ~df["user_id"].astype("string").str.strip().isin(uid_ok))
    err |= log.check(t, df, "orphan_item_id", "item_id", "error", "item_id must exist in valid items",
                     df["item_id"].notna() & is_int_string(df["item_id"]) & ~df["item_id"].astype("string").str.strip().isin(iid_ok))

    # --- interaction must not predate the user's signup
    signup = parse_timestamps(valid_users.set_index(valid_users["user_id"].astype(str).str.strip())["signup_date"])
    user_signup = df["user_id"].astype("string").str.strip().map(signup)
    err |= log.check(t, df, "interaction_before_signup", "timestamp", "error", "timestamp must be >= the user's signup_date",
                     ts.notna() & user_signup.notna() & (ts.dt.normalize() < pd.to_datetime(user_signup)))

    # --- duplicates
    err |= log.check(t, df, "exact_duplicate_row", "user_id", "error", "identical (user,item,rating,timestamp) rows are duplicates (first kept)",
                     df.duplicated(subset=data_cols, keep="first") & df[data_cols].notna().all(axis=1))
    log.check(t, df, "duplicate_user_item_pair", "item_id", "warning",
              "same user rated same item more than once (latest kept in preparation)",
              (~err) & df.duplicated(subset=["user_id", "item_id"], keep=False) & df["user_id"].notna())
    return df.loc[~err]
