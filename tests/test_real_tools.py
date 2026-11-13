"""Tests that exercise the real third-party tools. Each is skipped automatically when its library is missing,
so `pytest` is green in a minimal environment and meaningful in the full one (run `pytest -rs` to see skips)."""
import numpy as np
import pandas as pd
import pytest

from src.common.config import Config, load_config


def _small_raw(tmp_path):
    from src.ingestion.generate_synthetic import generate
    cfg = Config(load_config())
    cfg.root = tmp_path
    cfg["synthetic_data"].update(n_users=250, n_items=120, target_interactions=5000)
    cfg["validation"]["max_error_rate"] = 0.5
    generate(cfg, tmp_path / "data" / "raw")
    return cfg


def test_parquet_roundtrip_preserves_dtypes(tmp_path):
    pytest.importorskip("pyarrow")
    from src.common.io_utils import read_frame, write_frame
    df = pd.DataFrame({"user_id": [1, 2], "ts": pd.to_datetime(["2024-01-01", "2024-02-01"]), "r": [4.5, 3.0], "s": ["a", None]})
    write_frame(df, tmp_path / "t", "parquet")
    out = read_frame(tmp_path / "t", "parquet")
    pd.testing.assert_frame_equal(df, out, check_dtype=False)
    assert (tmp_path / "t.parquet").exists()


def test_great_expectations_engine_matches_custom_engine(tmp_path):
    pytest.importorskip("great_expectations")
    from src.common.io_utils import read_raw_csv
    from src.ingestion.generate_synthetic import GENRES
    from src.validation import checks as C, ge_checks as G
    cfg = _small_raw(tmp_path)
    raw = {t: read_raw_csv(tmp_path / "data" / "raw" / f"{t}.csv") for t in ("users", "items", "interactions")}

    cl = C.IssueLog()
    cu = C.validate_users(raw["users"], cfg, cl); ci = C.validate_items(raw["items"], cfg, cl, set(GENRES))
    cx = C.validate_interactions(raw["interactions"], cfg, cl, cu, ci)

    gl = C.IssueLog()
    runner = G.GXRunner(gl)
    gu = G.validate_users(raw["users"], cfg, runner); gi = G.validate_items(raw["items"], cfg, runner, set(GENRES))
    gx_ = G.validate_interactions(raw["interactions"], cfg, runner, gu, gi)

    for a, b in ((cu, gu), (ci, gi), (cx, gx_)):
        assert set(a["_row_id"]) == set(b["_row_id"])            # same rows survive
    fails = lambda log: {(r["table"], r["check"]): r["rows_failed"] for r in log.results}  # noqa: E731
    assert fails(cl) == fails(gl)                                 # same per-check failure counts


def test_surprise_svd_model_trains_and_predicts():
    pytest.importorskip("surprise")
    from src.features.feature_store import FeatureStore
    from src.training.models import SurpriseSVDRecommender
    rng = np.random.default_rng(0)
    users = pd.DataFrame({"user_id": range(1, 31), "age": 30, "gender": "M", "country": "IN", "signup_date": pd.Timestamp("2024-01-01")})
    items = pd.DataFrame({"item_id": range(1, 21), "title": "t", "genres": ["Drama"] * 10 + ["Comedy"] * 10, "release_year": 2000})
    inter = pd.DataFrame({"user_id": rng.integers(1, 31, 400), "item_id": rng.integers(1, 21, 400), "rating": rng.choice([2.0, 3.5, 4.5], 400),
                          "timestamp": pd.Timestamp("2024-03-01")}).drop_duplicates(["user_id", "item_id"])
    store = FeatureStore().fit(inter, users, items, pd.Timestamp("2024-04-01"))
    m = SurpriseSVDRecommender(n_factors=5, n_epochs=5).fit(inter, store, users, items)
    assert m.recommend([1, 999], 5).shape == (2, 5)
    p = m.predict_rating([1, 2], [3, 999])
    assert ((p >= 0.5) & (p <= 5)).all()


def test_feast_online_matches_offline(tmp_path):
    pytest.importorskip("feast")
    pytest.importorskip("pyarrow")
    from src.features.feast_layer import FeastFeatureLayer
    from src.features.feature_store import FeatureStore
    users = pd.DataFrame({"user_id": [1, 2, 3], "age": [20, 30, 40], "gender": "M", "country": "IN", "signup_date": pd.Timestamp("2024-01-01")})
    items = pd.DataFrame({"item_id": [1, 2, 3], "title": "t", "genres": ["Drama", "Comedy|Drama", "Comedy"], "release_year": 2000})
    inter = pd.DataFrame({"user_id": [1, 1, 2], "item_id": [1, 2, 3], "rating": [5.0, 4.0, 2.0],
                          "timestamp": pd.to_datetime(["2024-02-01", "2024-03-01", "2024-02-15"])})
    cfg = Config(load_config())
    cfg.root = tmp_path
    fs = FeatureStore().fit(inter, users, items, pd.Timestamp("2024-04-01"))
    fs.save(tmp_path / "data" / "features", "parquet")
    layer = FeastFeatureLayer(cfg)
    layer.publish(fs)
    layer.check_online_offline_parity(fs, n=3)
    back = layer.load_feature_store()
    pd.testing.assert_frame_equal(back.item_features.sort_index(), fs.item_features.sort_index(), check_dtype=False)
