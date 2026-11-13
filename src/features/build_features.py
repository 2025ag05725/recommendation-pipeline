"""Feature-management stage: materialise the feature store from TRAIN and publish the feature catalog."""
from __future__ import annotations

import pandas as pd

from src.common.io_utils import read_json
from src.common.logging_utils import get_logger
from src.features.feature_store import FeatureStore
from src.features.registry import check_coverage, write_catalog

log = get_logger("features")


def run(cfg) -> dict:
    proc, feat, rep = cfg.path("processed"), cfg.path("features"), cfg.path("reports")
    users = cfg.read_table("processed", "users_clean", parse_dates=["signup_date"])
    items = cfg.read_table("processed", "items_clean")
    train = cfg.read_table("processed", "train", parse_dates=["timestamp"])
    as_of = pd.Timestamp(read_json(proc / "split_info.json")["test_cutoff"])

    fs = FeatureStore(cfg["features"]["like_threshold"], cfg["features"]["bayes_prior_strength"]).fit(train, users, items, as_of)
    # exercise the pairwise path on a sample so documentation covers all three entity levels
    sample = train.sample(500, random_state=cfg["seed"])
    ui = fs.get_interaction_features(sample["user_id"].to_numpy(), sample["item_id"].to_numpy())
    check_coverage(fs.user_features.columns, fs.item_features.columns, ui.columns)
    fs.save(feat, cfg.fmt)
    summary_extra = {}
    if cfg["features"].get("engine", "feast") == "feast":
        from src.features.feast_layer import FeastFeatureLayer
        layer = FeastFeatureLayer(cfg)
        summary_extra = {**layer.publish(fs), "parity": layer.check_online_offline_parity(fs)}
    cat = write_catalog(rep / "feature_catalog.md", rep / "feature_catalog.csv")
    log.info("Feature store: %d user rows x %d cols, %d item rows x %d cols; catalog entries=%d",
             *fs.user_features.shape, *fs.item_features.shape, len(cat))
    return {**summary_extra, "user_feature_cols": fs.user_features.shape[1], "item_feature_cols": fs.item_features.shape[1], "catalog_entries": len(cat)}
