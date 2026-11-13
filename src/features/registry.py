"""Feature registry: the single declaration of every feature (name, entity, type, logic, consumers).

The human-readable feature catalog (reports/feature_catalog.md/.csv) is GENERATED from this list, and
`check_coverage` asserts that every column the FeatureStore materialises is documented here (and vice
versa), so documentation cannot drift from code.  Names ending in `*` are families (e.g. one column per genre).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from fnmatch import fnmatch

import pandas as pd


@dataclass(frozen=True)
class Feature:
    name: str
    entity: str          # user | item | user_item
    dtype: str
    description: str
    logic: str
    used_by: str


FEATURES: list[Feature] = [
    # ---------------- user ----------------
    Feature("user_activity_count", "user", "int", "Number of training interactions by the user.", "count(interactions) per user_id", "popularity fallback, cold-start flag"),
    Feature("user_avg_rating", "user", "float", "Mean rating given by the user (global mean if no history).", "mean(rating) per user_id", "rating baseline, interaction feature"),
    Feature("user_rating_std", "user", "float", "Std-dev of the user's ratings (0 if <2 ratings).", "std(rating) per user_id", "analysis"),
    Feature("user_like_ratio", "user", "float", "Share of the user's ratings >= like_threshold.", "mean(rating >= 4.0)", "analysis"),
    Feature("user_days_since_last_interaction", "user", "float", "Days between the user's latest interaction and the as-of time (days since signup if none).", "(as_of - max(timestamp)) / 1 day", "recency / churn analysis"),
    Feature("user_tenure_days", "user", "float", "Days between signup and the as-of time.", "(as_of - signup_date) / 1 day", "analysis"),
    Feature("user_preferred_genre", "user", "category", "Genre with the highest affinity share ('Unknown' if no liked items).", "argmax(user_genre_affinity_*)", "interaction feature (preferred-genre match)"),
    Feature("user_age", "user", "int", "Age in years (imputed with median in preparation).", "passthrough of users_clean.age", "demographic context"),
    Feature("user_age_group", "user", "category", "Age bucket: <25, 25-34, 35-44, 45-54, 55+.", "pd.cut(age)", "demographic context"),
    Feature("user_gender", "user", "category", "Standardised gender code M/F/O/U.", "passthrough of users_clean.gender", "demographic context"),
    Feature("user_country", "user", "category", "Upper-case country code or UNK.", "passthrough of users_clean.country", "demographic context"),
    Feature("user_genre_affinity_*", "user", "float (one per genre)", "Share of the user's LIKED interactions that fall in each genre (rows sum to 1, all 0 if no likes).", "liked(user x item) @ genre(item x genre), row-normalised", "hybrid model (content signal)"),
    # ---------------- item ----------------
    Feature("item_popularity_count", "item", "int", "Number of training interactions with the item.", "count(interactions) per item_id", "popularity baseline"),
    Feature("item_avg_rating", "item", "float", "Mean rating of the item (global mean if unseen).", "mean(rating) per item_id", "interaction feature"),
    Feature("item_bayes_rating", "item", "float", "Shrunk mean rating: (n*mean + m*global_mean)/(n+m), m=bayes_prior_strength.", "Bayesian average", "popularity tie-break, rating baseline"),
    Feature("item_like_ratio", "item", "float", "Share of the item's ratings >= like_threshold.", "mean(rating >= 4.0)", "analysis"),
    Feature("item_days_since_last_interaction", "item", "float", "Days since anyone last interacted with the item (days since release if never).", "(as_of - max(timestamp)) / 1 day", "freshness analysis"),
    Feature("item_age_years", "item", "int", "Years between release_year and the as-of year.", "as_of.year - release_year", "analysis"),
    Feature("item_num_genres", "item", "int", "Number of genres tagged on the item.", "count(genres)", "content normalisation"),
    Feature("item_genre_*", "item", "int 0/1 (one per genre)", "Multi-hot genre indicator.", "genre in item.genres", "hybrid model, diversity metric"),
    # ---------------- user x item ----------------
    Feature("ui_genre_affinity", "user_item", "float", "Mean of the user's genre affinity over the item's genres.", "sum_g(affinity_ug * item_g) / num_genres", "hybrid model ranking score"),
    Feature("ui_preferred_genre_match", "user_item", "int 0/1", "1 if the item carries the user's preferred genre.", "item_genre[user_preferred_genre]", "analysis / re-ranking"),
    Feature("ui_item_minus_user_avg", "user_item", "float", "item_avg_rating - user_avg_rating (does the item beat what this user usually gives?).", "difference of two entity features", "rating prediction analysis"),
]


def catalog_frame() -> pd.DataFrame:
    return pd.DataFrame([asdict(f) for f in FEATURES]).rename(columns={
        "name": "Feature", "entity": "Entity", "dtype": "Type", "description": "Description", "logic": "Computation", "used_by": "Used by"})


def _covered(col: str, entity: str) -> bool:
    return any(f.entity == entity and fnmatch(col, f.name) for f in FEATURES)


def check_coverage(user_cols, item_cols, ui_cols) -> None:
    missing = [(e, c) for e, cols in (("user", user_cols), ("item", item_cols), ("user_item", ui_cols)) for c in cols if not _covered(c, e)]
    if missing:
        raise AssertionError(f"Materialised features missing from the registry/catalog: {missing}")
    produced = {"user": list(user_cols), "item": list(item_cols), "user_item": list(ui_cols)}
    stale = [f.name for f in FEATURES if not any(fnmatch(c, f.name) for c in produced[f.entity])]
    if stale:
        raise AssertionError(f"Registry lists features that are never materialised: {stale}")


def write_catalog(path_md, path_csv) -> pd.DataFrame:
    df = catalog_frame()
    df.to_csv(path_csv, index=False)
    lines = ["# Feature Catalog", "",
             "Generated from `src/features/registry.py` (validated against the materialised feature tables on every run).", "",
             "| " + " | ".join(df.columns) + " |", "|" + "|".join("---" for _ in df.columns) + "|"]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(f"`{v}`" if i == 0 else str(v).replace("|", "\\|") for i, v in enumerate(r)) + " |")
    path_md.write_text("\n".join(lines) + "\n")
    return df
