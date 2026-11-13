"""Synthetic MovieLens-style dataset generator (users, items, interactions).

Why synthetic: the build environment has no internet access, and the assignment allows
"clearly documented synthetic data". The generator is seeded and deterministic.

Ground truth process
--------------------
* Each item has 1-3 genres, a heavy-tailed popularity weight and a quality bias.
* Each user has a Dirichlet genre preference plus latent taste factors.
* A user's affinity for an item mixes latent-factor and genre match.
* Which items a user watches ~ popularity * exp(taste_strength * affinity)
  (so popularity AND personal taste both matter -> personalised models have signal).
* Rating = clip(round_to_half(3.3 + user_bias + item_bias + 0.7*affinity + noise), 0.5, 5).
* Timestamps are uniform between a user's signup date and the end date.

Dirty-data injection
--------------------
After the clean tables are built they are serialised as *strings* and corrupted at the rates
set in `synthetic_data.dirty` (missing values, invalid ratings, bad/future/ancient timestamps,
orphan foreign keys, duplicates, mixed formats). This gives the validation stage real work.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.common.io_utils import write_json
from src.common.logging_utils import get_logger

log = get_logger("ingestion.generate")

GENRES = ["Action", "Adventure", "Animation", "Biography", "Comedy", "Crime", "Documentary", "Drama",
          "Family", "Fantasy", "Horror", "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller",
          "War", "Western"]
COUNTRIES = ["IN", "US", "GB", "DE", "FR", "BR", "CA", "AU", "JP", "SG"]
GENDER_RAW = ["M", "F", "m", "f", "Male", "Female", "male", "female", "Other"]  # mixed formats on purpose


def _pick(rng: np.random.Generator, n: int, frac: float) -> np.ndarray:
    k = max(1, int(round(n * frac))) if frac > 0 else 0
    return rng.choice(n, size=k, replace=False) if k else np.array([], dtype=int)


def _round_half(x: np.ndarray) -> np.ndarray:
    return np.round(x * 2) / 2


def _clean_tables(cfg: dict, rng: np.random.Generator):
    d = cfg["synthetic_data"]
    nU, nI, D, G = d["n_users"], d["n_items"], d["latent_dim"], len(GENRES)
    start, end = pd.Timestamp(d["start_date"]), pd.Timestamp(d["end_date"])

    # ---- items
    item_g = np.zeros((nI, G))
    for i in range(nI):
        n_g = rng.choice([1, 2, 3], p=[0.45, 0.40, 0.15])
        item_g[i, rng.choice(G, size=n_g, replace=False, p=np.linspace(2, 1, G) / np.linspace(2, 1, G).sum())] = 1
    Wg = rng.normal(0, 1, (G, D))
    q = item_g @ Wg / 1.5 + 0.3 * rng.normal(size=(nI, D))
    item_pop = rng.lognormal(0, 1.0, nI)
    item_bias = rng.normal(0, 0.4, nI)
    release_year = rng.integers(1980, 2025, nI)

    # ---- users
    pref = rng.dirichlet(np.full(G, 0.25), nU)
    p = pref @ Wg * 2.0 + 0.3 * rng.normal(size=(nU, D))
    user_bias = rng.normal(0, 0.4, nU)
    signup = start + pd.to_timedelta(rng.integers(0, (end - start).days - 90, nU), unit="D")
    z = lambda a: (a - a.mean()) / a.std()  # noqa: E731
    affinity = 0.6 * z(p @ q.T) + 0.4 * z(pref @ (item_g / item_g.sum(1, keepdims=True)).T)

    users = pd.DataFrame({
        "user_id": np.arange(1, nU + 1),
        "age": np.clip(rng.normal(34, 11, nU), 14, 80).astype(int),
        "gender": rng.choice(GENDER_RAW, nU, p=[.2, .2, .08, .08, .1, .1, .1, .1, .04]),
        "country": rng.choice(COUNTRIES, nU, p=[.3, .2, .1, .08, .07, .07, .06, .05, .04, .03]),
        "signup_date": signup.strftime("%Y-%m-%d"),
    })
    case_noise = rng.random(nI)
    genre_str = []
    for i in range(nI):
        names = [GENRES[j] for j in np.flatnonzero(item_g[i])]
        s = "|".join(names)
        genre_str.append(s.lower() if case_noise[i] < 0.08 else s.upper() if case_noise[i] < 0.12 else s)
    items = pd.DataFrame({
        "item_id": np.arange(1, nI + 1),
        "title": [f"Movie {i:04d}" for i in range(1, nI + 1)],
        "genres": genre_str,
        "release_year": release_year,
    })

    # ---- interactions
    scale = d["target_interactions"] / (nU * 1.0)
    n_per_user = np.clip(np.round(rng.lognormal(np.log(scale * 0.75), 0.8, nU)), 8, 300).astype(int)
    rows = []
    for u in range(nU):
        w = item_pop * np.exp(d["taste_strength"] * affinity[u])
        items_u = rng.choice(nI, size=min(n_per_user[u], nI), replace=False, p=w / w.sum())
        span = (end - signup[u]).total_seconds()
        ts = np.sort(rng.random(len(items_u)) * span)
        r = 3.3 + user_bias[u] + item_bias[items_u] + 0.7 * affinity[u, items_u] + rng.normal(0, 0.6, len(items_u))
        rows.append(pd.DataFrame({
            "user_id": u + 1, "item_id": items_u + 1, "rating": np.clip(_round_half(r), 0.5, 5.0),
            "timestamp": signup[u] + pd.to_timedelta(ts, unit="s"),
        }))
    inter = pd.concat(rows, ignore_index=True)
    inter = inter.sample(frac=1.0, random_state=cfg["seed"]).reset_index(drop=True)  # raw files are unordered
    return users, items, inter, signup


def _inject_dirt(cfg, rng, users, items, inter, signup):
    dirty = cfg["synthetic_data"]["dirty"]
    nU, nI = len(users), len(items)
    log_counts: dict[str, dict[str, int]] = {"users": {}, "items": {}, "interactions": {}}

    # ---- users (strings)
    u = users.astype(str).copy()
    c = dirty["users"]
    ix = _pick(rng, len(u), c["missing_age"]); u.loc[ix, "age"] = np.nan; log_counts["users"]["missing_age"] = len(ix)
    ix = _pick(rng, len(u), c["invalid_age"]); u.loc[ix, "age"] = rng.choice(["-3", "250", "abc", "0"], len(ix)); log_counts["users"]["invalid_age"] = len(ix)
    ix = _pick(rng, len(u), c["missing_country"]); u.loc[ix, "country"] = np.nan; log_counts["users"]["missing_country"] = len(ix)
    ix = _pick(rng, len(u), c["missing_gender"]); u.loc[ix, "gender"] = np.nan; log_counts["users"]["missing_gender"] = len(ix)
    ix = _pick(rng, len(u), c["bad_signup_date"]); u.loc[ix, "signup_date"] = rng.choice(["2023-13-45", "yesterday", "31/02/2023"], len(ix)); log_counts["users"]["bad_signup_date"] = len(ix)
    ix = _pick(rng, len(u), c["duplicate_rows"]); u = pd.concat([u, u.loc[ix]], ignore_index=True); log_counts["users"]["duplicate_rows"] = len(ix)

    # ---- items
    it = items.astype(str).copy()
    c = dirty["items"]
    ix = _pick(rng, len(it), c["missing_title"]); it.loc[ix, "title"] = np.nan; log_counts["items"]["missing_title"] = len(ix)
    ix = _pick(rng, len(it), c["missing_genres"]); it.loc[ix, "genres"] = np.nan; log_counts["items"]["missing_genres"] = len(ix)
    ix = _pick(rng, len(it), c["invalid_year"]); it.loc[ix, "release_year"] = rng.choice(["unknown", "1850", "2090"], len(ix)); log_counts["items"]["invalid_year"] = len(ix)
    ix = _pick(rng, len(it), c["duplicate_rows"]); it = pd.concat([it, it.loc[ix]], ignore_index=True); log_counts["items"]["duplicate_rows"] = len(ix)

    # ---- interactions: first build a string timestamp column with mixed formats
    x = inter.copy()
    n = len(x)
    c = dirty["interactions"]
    uid_signup = signup[x["user_id"].to_numpy() - 1]
    ix = _pick(rng, n, c["before_signup"])
    x.loc[ix, "timestamp"] = pd.DatetimeIndex(uid_signup[ix]) - pd.to_timedelta(rng.integers(5, 200, len(ix)), unit="D")
    log_counts["interactions"]["before_signup"] = len(ix)

    ts = x["timestamp"]
    fmt_draw = rng.random(n)
    ts_str = ts.dt.strftime("%Y-%m-%d %H:%M:%S")
    iso_t = ts.dt.strftime("%Y-%m-%dT%H:%M:%S")
    epoch = (ts.astype("int64") // 1_000_000_000).astype(str) if ts.dt.unit == "ns" else (ts.astype("datetime64[ns]").astype("int64") // 1_000_000_000).astype(str)
    ts_out = ts_str.where(fmt_draw < 0.85, iso_t.where(fmt_draw < 0.95, epoch))
    x = x.astype({"user_id": str, "item_id": str})
    x["rating"] = x["rating"].map(lambda v: f"{v:g}")
    x["timestamp"] = ts_out

    def setc(col, frac, values, key):
        ix_ = _pick(rng, n, frac)
        x.loc[ix_, col] = values(len(ix_)) if callable(values) else values
        log_counts["interactions"][key] = len(ix_)

    setc("rating", c["missing_rating"], np.nan, "missing_rating")
    setc("rating", c["invalid_rating"], lambda m: rng.choice(["0", "6", "-1", "10", "five", "4.3"], m), "invalid_rating")
    setc("timestamp", c["bad_timestamp"], lambda m: rng.choice(["not_a_date", "2024-99-99 00:00:00", "N/A?", "12/31"], m), "bad_timestamp")
    setc("timestamp", c["future_timestamp"], lambda m: rng.choice(["2026-06-15 10:00:00", "2027-01-01 00:00:00"], m), "future_timestamp")
    setc("timestamp", c["ancient_timestamp"], "1970-01-01 00:00:00", "ancient_timestamp")
    setc("user_id", c["orphan_user"], lambda m: (nU + rng.integers(1, 500, m)).astype(str), "orphan_user")
    setc("item_id", c["orphan_item"], lambda m: (nI + rng.integers(1, 500, m)).astype(str), "orphan_item")
    setc("user_id", c["missing_user"], np.nan, "missing_user")

    ix = _pick(rng, n, c["exact_duplicates"]); exact = x.loc[ix]
    ix = _pick(rng, n, c["conflicting_duplicates"]); conflict = x.loc[ix].copy()
    conflict["rating"] = rng.choice(["1", "2.5", "3", "4.5", "5"], len(conflict))
    conflict["timestamp"] = pd.to_datetime(conflict["timestamp"], errors="coerce").dt.strftime("%Y-%m-%d %H:%M:%S")
    log_counts["interactions"]["exact_duplicates"] = len(exact)
    log_counts["interactions"]["conflicting_duplicates"] = len(conflict)
    x = pd.concat([x, exact, conflict], ignore_index=True).sample(frac=1.0, random_state=cfg["seed"] + 1).reset_index(drop=True)
    return u.sample(frac=1.0, random_state=1).reset_index(drop=True), it.sample(frac=1.0, random_state=2).reset_index(drop=True), x, log_counts


def generate(cfg, out_dir: Path) -> dict:
    rng = np.random.default_rng(cfg["seed"])
    users, items, inter, signup = _clean_tables(cfg, rng)
    log.info("Clean synthetic tables: users=%d items=%d interactions=%d", len(users), len(items), len(inter))
    u, it, x, injected = _inject_dirt(cfg, rng, users, items, inter, signup)
    out_dir.mkdir(parents=True, exist_ok=True)
    u.to_csv(out_dir / "users.csv", index=False)
    it.to_csv(out_dir / "items.csv", index=False)
    x.to_csv(out_dir / "interactions.csv", index=False)
    write_json({"seed": cfg["seed"], "injected_defects": injected,
                "rows_written": {"users": len(u), "items": len(it), "interactions": len(x)}},
               out_dir / "synthetic_generation_log.json")
    log.info("Wrote raw CSVs to %s (users=%d items=%d interactions=%d)", out_dir, len(u), len(it), len(x))
    return injected
