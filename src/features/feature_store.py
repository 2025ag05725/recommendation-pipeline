"""A small, file-backed feature layer (custom alternative to Feast).

Offline/online parity: the SAME `FeatureStore` code path serves
  * training  (`fit(train_interactions, ..., as_of)` -> tables)  and
  * prediction (`load()` -> `get_user_features` / `get_item_features` / `get_interaction_features`
                 / `genre_affinity_matrix`), including safe defaults for cold-start users and items.

Leakage control: features are computed ONLY from the interactions passed to `fit` (the training period)
and "recency" features are measured relative to `as_of` (the split cutoff), never the wall clock.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

from src.common.io_utils import read_frame, read_json, write_frame, write_json
from src.transformation.matrices import build_index, build_matrix

AGE_BINS, AGE_LABELS = [0, 24, 34, 44, 54, 200], ["<25", "25-34", "35-44", "45-54", "55+"]


def item_genre_matrix(items: pd.DataFrame, genres: list[str]) -> pd.DataFrame:
    toks = items["genres"].str.split("|").tolist()
    # .to_numpy()/list inputs (not Series) so values are positional and can never mis-align with the id index
    return pd.DataFrame({f"item_genre_{g}": [int(g in t) for t in toks] for g in genres}, index=items["item_id"].to_numpy())


class FeatureStore:
    def __init__(self, like_threshold: float = 4.0, prior_strength: float = 10.0):
        self.like_threshold, self.prior_strength = like_threshold, prior_strength
        self.user_features: pd.DataFrame | None = None
        self.item_features: pd.DataFrame | None = None
        self.genres: list[str] = []
        self.defaults: dict = {}
        self.as_of: pd.Timestamp | None = None

    # ------------------------------------------------------------------ build (offline)
    def fit(self, interactions: pd.DataFrame, users: pd.DataFrame, items: pd.DataFrame, as_of: pd.Timestamp) -> "FeatureStore":
        thr, m = self.like_threshold, self.prior_strength
        self.as_of = pd.Timestamp(as_of)
        self.genres = sorted({g for s in items["genres"] for g in s.split("|")})
        gmean = float(interactions["rating"].mean())
        like = interactions["rating"] >= thr
        glike = float(like.mean())
        days = lambda ts: (self.as_of - ts).dt.total_seconds() / 86400  # noqa: E731

        # ---- user features
        g = interactions.groupby("user_id")
        uf = pd.DataFrame({
            "user_activity_count": g.size(), "user_avg_rating": g["rating"].mean(),
            "user_rating_std": g["rating"].std().fillna(0.0), "user_like_ratio": like.groupby(interactions["user_id"]).mean(),
            "_last": g["timestamp"].max()})
        uf = users.set_index("user_id")[["age", "gender", "country", "signup_date"]].join(uf, how="left")
        uf["user_tenure_days"] = days(uf["signup_date"])
        uf["user_days_since_last_interaction"] = days(uf["_last"]).fillna(uf["user_tenure_days"])
        uf["user_activity_count"] = uf["user_activity_count"].fillna(0).astype(int)
        uf["user_avg_rating"] = uf["user_avg_rating"].fillna(gmean)
        uf["user_rating_std"] = uf["user_rating_std"].fillna(0.0)
        uf["user_like_ratio"] = uf["user_like_ratio"].fillna(glike)
        uf["user_age"] = uf["age"]
        uf["user_age_group"] = pd.cut(uf["age"], AGE_BINS, labels=AGE_LABELS).astype(str)
        uf["user_gender"], uf["user_country"] = uf["gender"], uf["country"]

        ig = item_genre_matrix(items, self.genres)
        uidx, iidx = build_index(users, items)
        liked = interactions[like]
        L = build_matrix(liked, uidx, iidx, value=None)                       # users x items (binary liked)
        IG = sparse.csr_matrix(ig.loc[sorted(iidx, key=iidx.get)].to_numpy(dtype=float))  # items x genres
        counts = np.asarray((L @ IG).todense())
        tot = counts.sum(1, keepdims=True)
        aff = np.divide(counts, tot, out=np.zeros_like(counts), where=tot > 0)
        aff_df = pd.DataFrame(aff, index=sorted(uidx, key=uidx.get), columns=[f"user_genre_affinity_{x}" for x in self.genres])
        pref = np.where(tot[:, 0] > 0, np.array(self.genres, dtype=object)[aff.argmax(1)], "Unknown")
        uf = uf.join(aff_df)
        uf["user_preferred_genre"] = pd.Series(pref, index=aff_df.index)
        user_cols = [c for c in uf.columns if c.startswith("user_")]
        self.user_features = uf[user_cols].copy()

        # ---- item features
        gi = interactions.groupby("item_id")
        itf = items.set_index("item_id")[["release_year"]].join(pd.DataFrame({
            "item_popularity_count": gi.size(), "item_avg_rating": gi["rating"].mean(),
            "item_like_ratio": like.groupby(interactions["item_id"]).mean(), "_last": gi["timestamp"].max()}), how="left")
        n = itf["item_popularity_count"].fillna(0)
        itf["item_popularity_count"] = n.astype(int)
        itf["item_avg_rating"] = itf["item_avg_rating"].fillna(gmean)
        itf["item_bayes_rating"] = (n * itf["item_avg_rating"] + m * gmean) / (n + m)
        itf["item_like_ratio"] = itf["item_like_ratio"].fillna(glike)
        release = pd.to_datetime(itf["release_year"].astype(str) + "-01-01")
        itf["item_days_since_last_interaction"] = days(itf["_last"]).fillna(days(release))
        itf["item_age_years"] = self.as_of.year - itf["release_year"]
        itf["item_num_genres"] = ig.sum(axis=1)
        itf = itf.join(ig)
        self.item_features = itf[[c for c in itf.columns if c.startswith("item_")]].copy()

        self.defaults = {"global_mean_rating": gmean, "global_like_ratio": glike}
        return self

    # ------------------------------------------------------------------ persistence
    def save(self, directory: Path, fmt: str = "parquet") -> None:
        directory.mkdir(parents=True, exist_ok=True)
        write_frame(self.user_features.rename_axis("user_id").reset_index(), directory / "user_features", fmt)
        write_frame(self.item_features.rename_axis("item_id").reset_index(), directory / "item_features", fmt)
        write_json({"as_of": self.as_of, "genres": self.genres, "defaults": self.defaults, "like_threshold": self.like_threshold,
                    "prior_strength": self.prior_strength, "n_users": len(self.user_features), "n_items": len(self.item_features),
                    "user_feature_columns": list(self.user_features.columns), "item_feature_columns": list(self.item_features.columns)},
                   directory / "feature_meta.json")

    @classmethod
    def load(cls, directory: Path, fmt: str = "parquet") -> "FeatureStore":
        meta = read_json(directory / "feature_meta.json")
        kw = {"keep_default_na": False, "na_values": [""]} if fmt == "csv" else {}
        return cls.from_frames(read_frame(directory / "user_features", fmt, **kw).set_index("user_id"),
                               read_frame(directory / "item_features", fmt).set_index("item_id"), meta)

    @classmethod
    def from_frames(cls, user_features: pd.DataFrame, item_features: pd.DataFrame, meta: dict) -> "FeatureStore":
        """Rebuild a store from feature tables (e.g. retrieved from Feast) + the saved metadata."""
        fs = cls(meta["like_threshold"], meta["prior_strength"])
        fs.user_features = user_features[meta["user_feature_columns"]]
        fs.item_features = item_features[meta["item_feature_columns"]]
        fs.genres, fs.defaults, fs.as_of = meta["genres"], meta["defaults"], pd.Timestamp(meta["as_of"])
        return fs

    # ------------------------------------------------------------------ serving (online / batch)
    def _user_defaults(self) -> dict:
        d = {c: 0.0 for c in self.user_features.columns}
        d.update(user_activity_count=0, user_avg_rating=self.defaults["global_mean_rating"], user_like_ratio=self.defaults["global_like_ratio"],
                 user_preferred_genre="Unknown", user_gender="U", user_country="UNK", user_age_group="unknown")
        return d

    def get_user_features(self, user_ids) -> pd.DataFrame:
        """Rows for the requested ids; unknown users receive cold-start defaults."""
        ids = pd.Index(np.asarray(user_ids))
        out = self.user_features.reindex(ids)
        unknown = ~ids.isin(self.user_features.index)
        if unknown.any():
            for c, v in self._user_defaults().items():
                out.loc[unknown, c] = v
        return out

    def get_item_features(self, item_ids) -> pd.DataFrame:
        ids = pd.Index(np.asarray(item_ids))
        out = self.item_features.reindex(ids)
        unknown = ~ids.isin(self.item_features.index)
        if unknown.any():
            d = {c: 0.0 for c in self.item_features.columns}
            d.update(item_avg_rating=self.defaults["global_mean_rating"], item_bayes_rating=self.defaults["global_mean_rating"],
                     item_like_ratio=self.defaults["global_like_ratio"])
            for c, v in d.items():
                out.loc[unknown, c] = v
        return out

    @property
    def genre_cols(self) -> list[str]:
        return [f"item_genre_{g}" for g in self.genres]

    @property
    def affinity_cols(self) -> list[str]:
        return [f"user_genre_affinity_{g}" for g in self.genres]

    def genre_affinity_matrix(self, user_ids, item_ids) -> np.ndarray:
        """(n_users x n_items) matrix of ui_genre_affinity — same maths as the pairwise feature."""
        ua = self.get_user_features(user_ids)[self.affinity_cols].to_numpy(dtype=float)
        it = self.get_item_features(item_ids)
        ig = it[self.genre_cols].to_numpy(dtype=float) / np.maximum(it["item_num_genres"].to_numpy(dtype=float), 1)[:, None]
        return ua @ ig.T

    def get_interaction_features(self, user_ids, item_ids) -> pd.DataFrame:
        """Pairwise features for aligned (user_id, item_id) arrays."""
        uf, itf = self.get_user_features(user_ids), self.get_item_features(item_ids)
        ua = uf[self.affinity_cols].to_numpy(dtype=float)
        ig = itf[self.genre_cols].to_numpy(dtype=float)
        n = np.maximum(itf["item_num_genres"].to_numpy(dtype=float), 1)
        pref = uf["user_preferred_genre"].astype(str).to_numpy()
        match = np.array([ig[i, self.genres.index(p)] if p in self.genres else 0 for i, p in enumerate(pref)])
        return pd.DataFrame({
            "ui_genre_affinity": (ua * ig).sum(axis=1) / n, "ui_preferred_genre_match": match.astype(int),
            "ui_item_minus_user_avg": itf["item_avg_rating"].to_numpy() - uf["user_avg_rating"].to_numpy()})
