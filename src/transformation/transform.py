"""Transformation: time-based split + derived datasets.

Split (chronological, no shuffling, so no future information leaks into training):
    |---------- train_core ----------|--- val ---|---- test ----|
                                      ^val_cutoff ^test_cutoff
  train = train_core + val  (everything strictly before test_cutoff)
  val is used ONLY for hyper-parameter selection (models are tuned on train_core -> val,
  then refit on train and scored once on test).

Derived datasets are computed from TRAIN ONLY:
  * user_activity_summary.csv   * item_popularity.csv   * user_item_matrix.npz (+ index maps)
"""
from __future__ import annotations

import pandas as pd
from scipy import sparse

from src.common.io_utils import write_json
from src.common.logging_utils import get_logger
from src.transformation.matrices import build_index, build_matrix

log = get_logger("transformation")


def time_split(inter: pd.DataFrame, test_fraction: float, val_fraction: float):
    ts = inter["timestamp"].sort_values().reset_index(drop=True)
    n = len(ts)
    test_cut = ts.iloc[int(n * (1 - test_fraction))]
    val_cut = ts.iloc[int(n * (1 - test_fraction - val_fraction))]
    train_core = inter[inter["timestamp"] < val_cut]
    val = inter[(inter["timestamp"] >= val_cut) & (inter["timestamp"] < test_cut)]
    test = inter[inter["timestamp"] >= test_cut]
    return train_core, val, test, val_cut, test_cut


def user_activity_summary(train: pd.DataFrame, as_of: pd.Timestamp) -> pd.DataFrame:
    g = train.groupby("user_id")
    out = pd.DataFrame({
        "n_interactions": g.size(), "n_distinct_items": g["item_id"].nunique(),
        "avg_rating": g["rating"].mean().round(4), "first_interaction": g["timestamp"].min(),
        "last_interaction": g["timestamp"].max(),
    })
    out["active_days"] = (out["last_interaction"].dt.normalize() - out["first_interaction"].dt.normalize()).dt.days + 1
    out["days_since_last_interaction"] = ((as_of - out["last_interaction"]).dt.total_seconds() / 86400).round(2)
    return out.reset_index()


def item_popularity(train: pd.DataFrame, items: pd.DataFrame) -> pd.DataFrame:
    g = train.groupby("item_id")
    out = pd.DataFrame({"n_interactions": g.size(), "n_users": g["user_id"].nunique(), "avg_rating": g["rating"].mean().round(4)})
    out = items[["item_id"]].merge(out.reset_index(), on="item_id", how="left")
    out[["n_interactions", "n_users"]] = out[["n_interactions", "n_users"]].fillna(0).astype(int)
    out["popularity_rank"] = out["n_interactions"].rank(method="min", ascending=False).astype(int)
    return out.sort_values("popularity_rank").reset_index(drop=True)


def run(cfg) -> dict:
    proc = cfg.path("processed")
    users = cfg.read_table("processed", "users_clean", parse_dates=["signup_date"])
    items = cfg.read_table("processed", "items_clean")
    inter = cfg.read_table("processed", "interactions_clean", parse_dates=["timestamp"])

    sc = cfg["split"]
    train_core, val, test, val_cut, test_cut = time_split(inter, sc["test_fraction"], sc["val_fraction"])
    train = pd.concat([train_core, val])
    assert train["timestamp"].max() < test["timestamp"].min() or train["timestamp"].max() < test_cut <= test["timestamp"].min()
    for name, df in (("train_core", train_core), ("val", val), ("train", train), ("test", test)):
        cfg.write_table(df, "processed", name)

    cfg.write_table(user_activity_summary(train, test_cut), "processed", "user_activity_summary")
    cfg.write_table(item_popularity(train, items), "processed", "item_popularity")
    uidx, iidx = build_index(users, items)
    sparse.save_npz(proc / "user_item_matrix.npz", build_matrix(train, uidx, iidx, "rating"))
    write_json({"user_index": {str(k): v for k, v in uidx.items()}, "item_index": {str(k): v for k, v in iidx.items()}},
               proc / "matrix_index.json")

    test_users, train_users = set(test["user_id"]), set(train["user_id"])
    info = {
        "val_cutoff": val_cut, "test_cutoff": test_cut,
        "rows": {"train_core": len(train_core), "val": len(val), "train": len(train), "test": len(test)},
        "train_time_range": [train["timestamp"].min(), train["timestamp"].max()],
        "test_time_range": [test["timestamp"].min(), test["timestamp"].max()],
        "test_users": len(test_users), "test_users_unseen_in_train": len(test_users - train_users),
        "test_items_unseen_in_train": len(set(test["item_id"]) - set(train["item_id"])),
        "matrix_shape": [len(uidx), len(iidx)],
        "matrix_density": round(len(train) / (len(uidx) * len(iidx)), 5),
    }
    write_json(info, proc / "split_info.json")
    log.info("Split: %s", info["rows"])
    return {"train": len(train), "val": len(val), "test": len(test), "matrix_density": info["matrix_density"]}
