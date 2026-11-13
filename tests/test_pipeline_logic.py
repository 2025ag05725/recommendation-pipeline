import importlib.util

import numpy as np
import pandas as pd
import pytest

FMT = "parquet" if importlib.util.find_spec("pyarrow") else "csv"   # parquet is the default storage; csv only if pyarrow is absent

from src.common.parsing import parse_timestamps
from src.features.feature_store import FeatureStore, item_genre_matrix
from src.features.registry import FEATURES, check_coverage
from src.orchestration.runner import downstream_of, topological_levels
from src.orchestration.tasks import TASKS, TaskSpec
from src.transformation.transform import time_split


def test_timestamp_parsing_mixed_formats():
    s = pd.Series(["2024-01-02 03:04:05", "2024-01-02T03:04:05", "1704164645", "nope", None])
    out = parse_timestamps(s)
    assert out.iloc[0] == pd.Timestamp("2024-01-02 03:04:05") and out.iloc[1] == out.iloc[0]
    assert out.iloc[2] == pd.Timestamp("2024-01-02 03:04:05")
    assert out.iloc[3:].isna().all()


def test_time_split_is_chronological_no_leakage():
    ts = pd.date_range("2024-01-01", periods=1000, freq="h")
    df = pd.DataFrame({"user_id": 1, "item_id": range(1000), "rating": 3.0, "timestamp": ts}).sample(frac=1, random_state=0)
    core, val, test, vc, tc = time_split(df, 0.2, 0.1)
    assert core["timestamp"].max() < val["timestamp"].min() <= val["timestamp"].max() < test["timestamp"].min()
    assert len(core) + len(val) + len(test) == 1000 and len(test) == 200


def _mini():
    users = pd.DataFrame({"user_id": [1, 2, 3], "age": [20, 30, 40], "gender": "M", "country": "IN",
                          "signup_date": pd.Timestamp("2024-01-01")})
    items = pd.DataFrame({"item_id": [1, 2, 3], "title": "t", "genres": ["Drama", "Comedy|Drama", "Comedy"], "release_year": 2000})
    inter = pd.DataFrame({"user_id": [1, 1, 2], "item_id": [1, 2, 3], "rating": [5.0, 4.0, 2.0],
                          "timestamp": pd.to_datetime(["2024-02-01", "2024-03-01", "2024-02-15"])})
    return users, items, inter


def test_genre_matrix_is_positionally_aligned():
    _, items, _ = _mini()
    ig = item_genre_matrix(items.assign(item_id=[5, 6, 7]), ["Comedy", "Drama"])   # ids deliberately not 0-based
    assert ig.loc[5].tolist() == [0, 1] and ig.loc[6].tolist() == [1, 1] and ig.loc[7].tolist() == [1, 0]


def test_feature_store_values_and_cold_start(tmp_path):
    users, items, inter = _mini()
    fs = FeatureStore(4.0, 10).fit(inter, users, items, pd.Timestamp("2024-04-01"))
    uf = fs.get_user_features([1, 3, 999])
    assert uf.loc[1, "user_activity_count"] == 2 and uf.loc[1, "user_avg_rating"] == pytest.approx(4.5)
    assert uf.loc[1, "user_days_since_last_interaction"] == pytest.approx(31.0)
    assert uf.loc[3, "user_activity_count"] == 0 and uf.loc[999, "user_activity_count"] == 0     # no history / unknown user
    assert uf.loc[999, "user_avg_rating"] == pytest.approx(inter["rating"].mean())
    assert fs.get_item_features([3]).loc[3, "item_popularity_count"] == 1
    fs.save(tmp_path, FMT)
    fs2 = FeatureStore.load(tmp_path, FMT)                                                              # offline == online
    pd.testing.assert_frame_equal(fs.get_item_features([1, 2, 3]), fs2.get_item_features([1, 2, 3]), check_dtype=False)
    ui = fs2.get_interaction_features([1], [2])
    assert ui["ui_genre_affinity"].iloc[0] == pytest.approx(fs2.genre_affinity_matrix([1], [2])[0, 0])
    check_coverage(fs.user_features.columns, fs.item_features.columns, ui.columns)


def test_dag_is_valid_and_acyclic():
    levels = topological_levels(TASKS)
    flat = [n for lv in levels for n in lv]
    assert flat[0] == "ingest" and flat[-1] == "evaluate" and len(flat) == len(TASKS)
    pos = {n: i for i, n in enumerate(flat)}
    assert all(pos[d] < pos[t.name] for t in TASKS for d in t.deps)
    assert downstream_of("build_features", TASKS) >= {"train_popularity", "train_item_knn", "train_hybrid_svd", "evaluate"}
    with pytest.raises(ValueError):
        topological_levels([TaskSpec("a", "x:y", ("b",)), TaskSpec("b", "x:y", ("a",))])
    with pytest.raises(ValueError):
        topological_levels([TaskSpec("a", "x:y", ("zzz",))])


def test_registry_has_all_three_entities():
    assert {f.entity for f in FEATURES} == {"user", "item", "user_item"}
