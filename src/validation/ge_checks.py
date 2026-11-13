"""Data validation with Great Expectations (GX Core 1.x).

Every check below is a real GX *Expectation* evaluated on a pandas batch. GX reports which rows failed
(`unexpected_index_list`); because the DataFrame index is set to the raw file line number (`_row_id`), failed
rows map straight back to the source file. The shared `IssueLog` then turns results into the error dataset,
the quarantine split and the validation report - identical outputs to the lightweight `custom` engine.

Where a rule is not a plain column expectation (parsing a date, comparing with another table) a *derived
column* carries the verdict ("ok"/"bad") or a numeric value, and a standard GX expectation is applied to it.
Rows where the source value is null are never flagged by those derived columns (null handling has its own check).

Severity (error = quarantine, warning = repair later) is our own policy layer on top of GX results.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

from src.common.parsing import is_int_string, parse_timestamps
from src.validation.checks import GENDER_OK, IssueLog

os.environ.setdefault("GX_ANALYTICS_ENABLED", "false")

EPOCH = pd.Timestamp("1970-01-01")
_DAY = pd.Timedelta(days=1)


def _days(ts: pd.Series) -> pd.Series:
    return (ts - EPOCH) / _DAY


def _verdict(raw_null: pd.Series, ok: pd.Series) -> pd.Series:
    """'ok'/'bad' for non-null raw values, None for null raw values (nulls are checked separately)."""
    out = pd.Series(np.where(ok, "ok", "bad"), index=raw_null.index, dtype=object)
    out[raw_null.to_numpy()] = None
    return out


class GXRunner:
    """Thin wrapper: one ephemeral GX context, one batch per table, expectation -> boolean failure mask."""

    def __init__(self, log: IssueLog):
        import great_expectations as gx
        self.gx, self.E, self.log, self._n = gx, gx.expectations, log, 0
        self.context = gx.get_context(mode="ephemeral")
        self.source = self.context.data_sources.add_pandas(name="raw_tables")

    def table(self, name: str, df: pd.DataFrame, helpers: dict[str, pd.Series] | None = None) -> "GXTable":
        ge_df = df.copy()
        for col, values in (helpers or {}).items():
            ge_df[col] = values.to_numpy()
        ge_df.index = pd.Index(df["_row_id"].to_numpy())          # failed rows are reported by this index
        self._n += 1
        asset = self.source.add_dataframe_asset(name=f"{name}_{self._n}")
        batch = asset.add_batch_definition_whole_dataframe("whole").get_batch(batch_parameters={"dataframe": ge_df})
        return GXTable(self, name, ge_df, batch)


class GXTable:
    def __init__(self, runner: GXRunner, name: str, df: pd.DataFrame, batch):
        self.r, self.name, self.df, self.batch = runner, name, df, batch

    def failed(self, *expectations) -> pd.Series:
        """Union of failing rows over one or more expectations."""
        bad: set = set()
        for exp in expectations:
            res = self.batch.validate(exp, result_format="COMPLETE")
            bad |= set(res.result.get("unexpected_index_list") or [])
        return pd.Series(self.df.index.isin(list(bad)), index=self.df.index)

    def check(self, name, column, severity, description, *expectations, post=None) -> pd.Series:
        mask = self.failed(*expectations)
        if post is not None:
            mask = post(mask)
        return self.r.log.check(self.name, self.df, name, column, severity, description, mask)


def _later_duplicates(df: pd.DataFrame, cols: list[str]):
    """GX flags *every* member of a duplicate group; we keep the first occurrence and quarantine the rest."""
    later = df.duplicated(subset=cols, keep="first") & df[cols].notna().all(axis=1)
    return lambda mask: mask & later


def validate_users(df: pd.DataFrame, cfg: dict, runner: GXRunner) -> pd.DataFrame:
    v, ref = cfg["validation"], pd.Timestamp(cfg["reference_date"])
    E = runner.E
    ts = parse_timestamps(df["signup_date"])
    age = pd.to_numeric(df["age"], errors="coerce")
    helpers = {
        "signup_parse": _verdict(df["signup_date"].isna(), ts.notna()),
        "signup_days": _days(ts),
        "age_num": age,
        "gender_norm": df["gender"].astype("string").str.strip().str.lower().astype(object),
    }
    t = runner.table("users", df, helpers)
    err = pd.Series(False, index=t.df.index)
    err |= t.check("missing_user_id", "user_id", "error", "user_id must not be null", E.ExpectColumnValuesToNotBeNull(column="user_id"))
    err |= t.check("invalid_user_id_type", "user_id", "error", "user_id must be an integer",
                   E.ExpectColumnValuesToMatchRegex(column="user_id", regex=r"^\d+$"))
    err |= t.check("duplicate_user_id", "user_id", "error", "user_id must be unique (first occurrence kept)",
                   E.ExpectColumnValuesToBeUnique(column="user_id"), post=_later_duplicates(t.df, ["user_id"]))
    err |= t.check("invalid_signup_date", "signup_date", "error", "signup_date must be a parseable date",
                   E.ExpectColumnValuesToBeInSet(column="signup_parse", value_set=["ok"]))
    err |= t.check("future_signup_date", "signup_date", "error", "signup_date must not be after reference date",
                   E.ExpectColumnValuesToBeBetween(column="signup_days", max_value=float(_days(pd.Series([ref])).iloc[0])))
    t.check("missing_age", "age", "warning", "age should not be null (imputed in preparation)", E.ExpectColumnValuesToNotBeNull(column="age"))
    t.check("invalid_age", "age", "warning", f"age must be numeric in [{v['age_min']}, {v['age_max']}]",
            E.ExpectColumnValuesToMatchRegex(column="age", regex=r"^-?\d+(\.\d+)?$"),
            E.ExpectColumnValuesToBeBetween(column="age_num", min_value=v["age_min"], max_value=v["age_max"]))
    t.check("missing_gender", "gender", "warning", "gender should not be null", E.ExpectColumnValuesToNotBeNull(column="gender"))
    t.check("invalid_gender", "gender", "warning", "gender must be a known code",
            E.ExpectColumnValuesToBeInSet(column="gender_norm", value_set=sorted(GENDER_OK)))
    t.check("missing_country", "country", "warning", "country should not be null", E.ExpectColumnValuesToNotBeNull(column="country"))
    return df.loc[~err.to_numpy()]


def validate_items(df: pd.DataFrame, cfg: dict, runner: GXRunner, known_genres: set[str]) -> pd.DataFrame:
    v, ref = cfg["validation"], pd.Timestamp(cfg["reference_date"])
    E = runner.E
    vocab = {g.lower() for g in known_genres}
    toks = df["genres"].dropna().astype("string").str.split("|").explode().str.strip().str.lower()
    bad_rows = set(toks.index[~toks.isin(vocab)])
    genres_ok = pd.Series(np.where(df.index.isin(list(bad_rows)), "bad", "ok"), index=df.index, dtype=object)
    genres_ok[df["genres"].isna().to_numpy()] = None
    year = pd.to_numeric(df["release_year"], errors="coerce")
    t = runner.table("items", df, {"genres_vocab": genres_ok, "year_num": year})
    err = pd.Series(False, index=t.df.index)
    err |= t.check("missing_item_id", "item_id", "error", "item_id must not be null", E.ExpectColumnValuesToNotBeNull(column="item_id"))
    err |= t.check("invalid_item_id_type", "item_id", "error", "item_id must be an integer",
                   E.ExpectColumnValuesToMatchRegex(column="item_id", regex=r"^\d+$"))
    err |= t.check("duplicate_item_id", "item_id", "error", "item_id must be unique (first occurrence kept)",
                   E.ExpectColumnValuesToBeUnique(column="item_id"), post=_later_duplicates(t.df, ["item_id"]))
    t.check("missing_title", "title", "warning", "title should not be null", E.ExpectColumnValuesToNotBeNull(column="title"))
    t.check("missing_genres", "genres", "warning", "genres should not be null", E.ExpectColumnValuesToNotBeNull(column="genres"))
    t.check("unknown_genre", "genres", "warning", "every genre token must be in the genre vocabulary",
            E.ExpectColumnValuesToBeInSet(column="genres_vocab", value_set=["ok"]))
    t.check("invalid_release_year", "release_year", "warning",
            f"release_year must be an integer in [{v['release_year_min']}, {ref.year}]",
            E.ExpectColumnValuesToMatchRegex(column="release_year", regex=r"^\d+$"),
            E.ExpectColumnValuesToBeBetween(column="year_num", min_value=v["release_year_min"], max_value=ref.year))
    return df.loc[~err.to_numpy()]


def validate_interactions(df: pd.DataFrame, cfg: dict, runner: GXRunner, valid_users: pd.DataFrame, valid_items: pd.DataFrame) -> pd.DataFrame:
    v, ref = cfg["validation"], pd.Timestamp(cfg["reference_date"])
    E = runner.E
    lo, hi, step = v["rating_min"], v["rating_max"], v["rating_step"]
    scale = [float(x) for x in np.arange(lo, hi + step / 2, step)]
    rating = pd.to_numeric(df["rating"], errors="coerce")
    ts = parse_timestamps(df["timestamp"])
    signup = parse_timestamps(valid_users.set_index(valid_users["user_id"].astype(str).str.strip())["signup_date"])
    user_signup = pd.to_datetime(df["user_id"].astype("string").str.strip().map(signup))
    since_signup = (ts.dt.normalize() - user_signup.dt.normalize()) / _DAY
    uid_ok = sorted(valid_users["user_id"].astype(str).str.strip())
    iid_ok = sorted(valid_items["item_id"].astype(str).str.strip())
    data_cols = ["user_id", "item_id", "rating", "timestamp"]

    t = runner.table("interactions", df, {
        "rating_num": rating, "ts_parse": _verdict(df["timestamp"].isna(), ts.notna()),
        "ts_days": _days(ts), "days_since_signup": since_signup})
    err = pd.Series(False, index=t.df.index)
    for col in data_cols:
        err |= t.check(f"missing_{col}", col, "error",
                       f"{col} must not be null", E.ExpectColumnValuesToNotBeNull(column=col))
    for col in ("user_id", "item_id"):
        err |= t.check(f"invalid_{col}_type", col, "error", f"{col} must be an integer", E.ExpectColumnValuesToMatchRegex(column=col, regex=r"^\d+$"))
    err |= t.check("invalid_rating_type", "rating", "error", "rating must be numeric",
                   E.ExpectColumnValuesToMatchRegex(column="rating", regex=r"^-?\d+(\.\d+)?$"))
    out_of_range = t.check("rating_out_of_range", "rating", "error", f"rating must be within [{lo}, {hi}]",
                           E.ExpectColumnValuesToBeBetween(column="rating_num", min_value=lo, max_value=hi))
    err |= out_of_range
    err |= t.check("rating_off_scale", "rating", "error", f"rating must be a multiple of {step}",
                   E.ExpectColumnValuesToBeInSet(column="rating_num", value_set=scale), post=lambda m: m & ~out_of_range)
    err |= t.check("invalid_timestamp_format", "timestamp", "error", "timestamp must be parseable",
                   E.ExpectColumnValuesToBeInSet(column="ts_parse", value_set=["ok"]))
    err |= t.check("timestamp_in_future", "timestamp", "error", "timestamp must not exceed reference date",
                   E.ExpectColumnValuesToBeBetween(column="ts_days", max_value=float(_days(pd.Series([ref])).iloc[0])))
    err |= t.check("timestamp_too_old", "timestamp", "error", f"timestamp must be >= {v['min_valid_date']}",
                   E.ExpectColumnValuesToBeBetween(column="ts_days", min_value=float(_days(pd.Series([pd.Timestamp(v['min_valid_date'])])).iloc[0])))
    # referential integrity = value must be a member of the (valid) parent key set
    err |= t.check("orphan_user_id", "user_id", "error", "user_id must exist in valid users",
                   E.ExpectColumnValuesToBeInSet(column="user_id", value_set=uid_ok), post=lambda m: m & is_int_string(t.df["user_id"]).to_numpy())
    err |= t.check("orphan_item_id", "item_id", "error", "item_id must exist in valid items",
                   E.ExpectColumnValuesToBeInSet(column="item_id", value_set=iid_ok), post=lambda m: m & is_int_string(t.df["item_id"]).to_numpy())
    err |= t.check("interaction_before_signup", "timestamp", "error", "timestamp must be >= the user's signup_date",
                   E.ExpectColumnValuesToBeBetween(column="days_since_signup", min_value=0))
    err |= t.check("exact_duplicate_row", "user_id", "error", "identical (user,item,rating,timestamp) rows are duplicates (first kept)",
                   E.ExpectCompoundColumnsToBeUnique(column_list=data_cols), post=_later_duplicates(t.df, data_cols))
    t.check("duplicate_user_item_pair", "item_id", "warning", "same user rated same item more than once (latest kept in preparation)",
            E.ExpectCompoundColumnsToBeUnique(column_list=["user_id", "item_id"]), post=lambda m: m & ~err)
    return df.loc[~err.to_numpy()]
