"""Serving demo: top-K recommendations for a user using a persisted model + the feature store (prediction path).

    python -m src.training.recommend --user 17 --k 10 --model hybrid_svd
Unknown / cold-start users fall back to popularity.
"""
import argparse
import pickle

import pandas as pd

from src.common.config import load_config


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--user", type=int, required=True)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--config", default=None, help="e.g. configs/config.movielens.yaml")
    ap.add_argument("--model", default="hybrid_svd", choices=["popularity", "item_knn", "hybrid_svd"])
    a = ap.parse_args()
    cfg = load_config(a.config)
    with open(cfg.path("models", f"{a.model}.pkl"), "rb") as fh:
        model = pickle.load(fh)
    items = cfg.read_table("processed", "items_clean").set_index("item_id")
    uf = model.store.get_user_features([a.user]).iloc[0]
    print(f"user {a.user}: history={int(uf['user_activity_count'])} preferred_genre={uf['user_preferred_genre']} model={a.model}")
    for rank, iid in enumerate(model.recommend([a.user], a.k)[0], 1):
        print(f"{rank:2d}. item {iid:<4d} {items.loc[iid, 'title']:<12s} {items.loc[iid, 'genres']}")


if __name__ == "__main__":
    main()
