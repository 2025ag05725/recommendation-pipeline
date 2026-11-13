"""Feast integration: feature registry, offline (point-in-time) retrieval and online serving.

Division of labour
  * `FeatureStore.fit` (pandas) COMPUTES the features from the training period.
  * Feast REGISTERS them (entities + feature views), serves them OFFLINE for training / batch scoring with
    point-in-time-correct joins (`get_historical_features`) and ONLINE for low-latency prediction
    (`get_online_features`, SQLite online store after `materialize`).
  * `check_online_offline_parity` proves both paths return the same values.

Feast file sources must be Parquet, so this layer always writes Parquet regardless of `storage.format`.
Repo config: feature_repo/feature_store.yaml (created on first use). Targets Feast 0.40+ (see requirements.txt).
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from src.common.io_utils import read_json
from src.common.logging_utils import get_logger
from src.features.feature_store import FeatureStore

log = get_logger("features.feast")

YAML = """project: recsys_pipeline
provider: local
registry: data/registry.db
online_store:
    type: sqlite
    path: data/online_store.db
offline_store:
    type: file
entity_key_serialization_version: 3
"""


def _utc(ts) -> pd.Timestamp:
    ts = pd.Timestamp(ts)
    return ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")


class FeastFeatureLayer:
    def __init__(self, cfg):
        from feast import FeatureStore as FeastStore
        self.cfg = cfg
        self.repo = cfg.root / cfg["features"].get("feast_repo", "feature_repo")
        (self.repo / "data").mkdir(parents=True, exist_ok=True)
        yaml = self.repo / "feature_store.yaml"
        if not yaml.exists():
            yaml.write_text(YAML)
        self.store = FeastStore(repo_path=str(self.repo))
        self.feat_dir = cfg.path("features")
        self.meta_path = self.feat_dir / "feature_meta.json"

    # ------------------------------------------------------------------ definitions
    @staticmethod
    def _dtype(series: pd.Series):
        from feast.types import Float64, Int64, String
        if pd.api.types.is_integer_dtype(series):
            return Int64
        if pd.api.types.is_float_dtype(series):
            return Float64
        return String

    def _definitions(self, user_df: pd.DataFrame, item_df: pd.DataFrame):
        from feast import Entity, FeatureView, Field, FileSource
        user = Entity(name="user", join_keys=["user_id"], description="platform user")
        item = Entity(name="item", join_keys=["item_id"], description="movie")
        views = []
        for name, entity, key, df in (("user_features", user, "user_id", user_df), ("item_features", item, "item_id", item_df)):
            source = FileSource(name=f"{name}_source", path=str(self.feat_dir / f"feast_{name}.parquet"), timestamp_field="event_timestamp")
            schema = [Field(name=c, dtype=self._dtype(df[c])) for c in df.columns if c not in (key, "event_timestamp")]
            views.append(FeatureView(name=name, entities=[entity], ttl=timedelta(days=3650), schema=schema, source=source, online=True))
        return [user, item, *views]

    # ------------------------------------------------------------------ publish
    def publish(self, fs: FeatureStore) -> dict:
        """Write the Feast Parquet sources, `feast apply`, then materialise to the online store."""
        as_of = _utc(fs.as_of)
        frames = {}
        for name, df, key in (("user_features", fs.user_features, "user_id"), ("item_features", fs.item_features, "item_id")):
            out = df.rename_axis(key).reset_index()
            out["event_timestamp"] = as_of
            out.to_parquet(self.feat_dir / f"feast_{name}.parquet", engine="pyarrow", index=False)
            frames[name] = out
        self.store.apply(self._definitions(frames["user_features"], frames["item_features"]))
        self.store.materialize(start_date=datetime(2000, 1, 1, tzinfo=timezone.utc), end_date=(as_of + pd.Timedelta(days=1)).to_pydatetime())
        log.info("Feast: applied entities/feature views and materialised to online store (as_of=%s)", as_of)
        return {"feature_views": ["user_features", "item_features"], "materialized_as_of": str(as_of)}

    # ------------------------------------------------------------------ offline (training / batch scoring)
    def historical(self, view: str, key: str, ids, columns, as_of) -> pd.DataFrame:
        ids = np.asarray(ids)
        entity_df = pd.DataFrame({key: ids, "event_timestamp": pd.to_datetime([_utc(as_of)] * len(ids), utc=True)})
        feats = [f"{view}:{c}" for c in columns]
        out = self.store.get_historical_features(entity_df=entity_df, features=feats).to_df()
        return out.drop(columns=["event_timestamp"]).drop_duplicates(subset=key).set_index(key)[list(columns)]

    def load_feature_store(self) -> FeatureStore:
        """Training path: rebuild the FeatureStore from Feast point-in-time retrieval at the split cutoff."""
        meta = read_json(self.meta_path)
        as_of = pd.Timestamp(meta["as_of"])
        uids = pd.read_parquet(self.feat_dir / "feast_user_features.parquet", columns=["user_id"])["user_id"].to_numpy()
        iids = pd.read_parquet(self.feat_dir / "feast_item_features.parquet", columns=["item_id"])["item_id"].to_numpy()
        users = self.historical("user_features", "user_id", uids, meta["user_feature_columns"], as_of)
        items = self.historical("item_features", "item_id", iids, meta["item_feature_columns"], as_of)
        return FeatureStore.from_frames(users, items, meta)

    # ------------------------------------------------------------------ online (prediction)
    def online(self, view: str, key: str, ids, columns) -> pd.DataFrame:
        rows = [{key: int(i)} for i in ids]
        resp = self.store.get_online_features(features=[f"{view}:{c}" for c in columns], entity_rows=rows).to_df()
        return resp.set_index(key)[list(columns)]

    def check_online_offline_parity(self, fs: FeatureStore, n: int = 50, seed: int = 0) -> dict:
        """Online store values must equal the offline tables (guards against training/serving skew)."""
        rng = np.random.default_rng(seed)
        out = {}
        for view, key, table in (("user_features", "user_id", fs.user_features), ("item_features", "item_id", fs.item_features)):
            ids = rng.choice(table.index.to_numpy(), size=min(n, len(table)), replace=False)
            on = self.online(view, key, ids, list(table.columns)).reindex(ids)
            off = table.loc[ids]
            for c in table.columns:
                if pd.api.types.is_numeric_dtype(off[c]):
                    ok = np.allclose(on[c].to_numpy(dtype=float), off[c].to_numpy(dtype=float), rtol=1e-6, atol=1e-6, equal_nan=True)
                else:
                    ok = (on[c].astype(str).to_numpy() == off[c].astype(str).to_numpy()).all()
                if not ok:
                    raise AssertionError(f"Feast online/offline mismatch in {view}.{c}")
            out[view] = f"{len(ids)} entities x {len(table.columns)} features identical"
        log.info("Feast parity check passed: %s", out)
        return out
