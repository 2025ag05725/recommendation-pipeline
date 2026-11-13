"""Training stage: tune on (train_core -> val), refit on full TRAIN, persist model, log to the tracker.

One function `train_model(cfg, name)` per model; the orchestrator runs the three as independent tasks.
Tuning picks hyper-parameters by validation NDCG@K (hybrid also picks the explicit-head rank by validation RMSE).
Test data is never touched here.
"""
from __future__ import annotations

import itertools
import pickle
import time

import pandas as pd

from src.common.io_utils import read_json, write_json
from src.common.logging_utils import get_logger
from src.evaluation.evaluate import evaluate_model
from src.features.feature_store import FeatureStore
from src.tracking.tracker import Tracker
from src.training.models import MODEL_CLASSES

log = get_logger("training")


def _load(cfg):
    proc = cfg.path("processed")
    users = cfg.read_table("processed", "users_clean", parse_dates=["signup_date"])
    items = cfg.read_table("processed", "items_clean")
    rd = lambda n: cfg.read_table("processed", n, parse_dates=["timestamp"])  # noqa: E731
    return users, items, rd("train_core"), rd("val"), rd("train"), read_json(proc / "split_info.json")


def _fixed_params(cfg, name: str) -> dict:
    """Non-tuned constructor arguments from the config (everything in the model block except grid/flags)."""
    skip = {"grid", "enabled", "tune_metric"}
    p = {k: v for k, v in cfg["models"][name].items() if k not in skip}
    if name == "surprise_svd":
        p["seed"] = cfg["seed"]
    return p


def _grid(spec: dict):
    keys = list(spec)
    for vals in itertools.product(*(spec[k] for k in keys)):
        yield dict(zip(keys, vals))


def tune(cfg, name: str, users, items, train_core, val, split_info):
    """Grid search on the validation slice. Features are re-fitted on train_core (as-of val cutoff) to avoid leakage."""
    grid = cfg["models"][name].get("grid")
    if not grid:
        return {}, pd.DataFrame()
    fc = cfg["features"]
    store = FeatureStore(fc["like_threshold"], fc["bayes_prior_strength"]).fit(train_core, users, items, pd.Timestamp(split_info["val_cutoff"]))
    k, thr, like = cfg["evaluation"]["k"], cfg["evaluation"]["relevance_threshold"], fc["like_threshold"]
    extra = _fixed_params(cfg, name)
    trials = []
    for params in _grid(grid):
        model = MODEL_CLASSES[name](like_threshold=like, **params, **extra).fit(train_core, store, users, items)
        res = evaluate_model(model, val, k, thr)
        trials.append({**params, f"val_ndcg@{k}": res[f"ndcg@{k}"], f"val_recall@{k}": res[f"recall@{k}"], "val_rmse": res.get("rmse")})
    df = pd.DataFrame(trials)
    metric = f"val_ndcg@{k}"
    if cfg["models"][name].get("tune_metric") == "val_rmse":     # explicit-rating models: minimise validation RMSE
        best = df.loc[df["val_rmse"].idxmin()]
    else:
        best = df.loc[df[metric].idxmax()]
    chosen = {c: type(grid[c][0])(best[c]) for c in grid}   # restore int/float types lost in the DataFrame row
    if name == "hybrid_svd" and "k_explicit" in grid:  # explicit head is judged on RMSE, independent of ranking params
        by_k = df.groupby("k_explicit")["val_rmse"].mean()
        chosen["k_explicit"] = int(by_k.idxmin())
    return chosen, df


def train_model(cfg, name: str) -> dict:
    t0 = time.time()
    users, items, train_core, val, train, split_info = _load(cfg)
    fc = cfg["features"]
    tracker = Tracker(cfg)
    run_id = tracker.start_run(name, {"stage": "train", "model": name})
    try:
        best, trials = tune(cfg, name, users, items, train_core, val, split_info)
        extra = _fixed_params(cfg, name)
        if cfg["features"].get("engine", "feast") == "feast":   # final features: Feast point-in-time retrieval at the split cutoff
            from src.features.feast_layer import FeastFeatureLayer
            store = FeastFeatureLayer(cfg).load_feature_store()
        else:
            store = FeatureStore.load(cfg.path("features"), cfg.fmt)
        model = MODEL_CLASSES[name](like_threshold=fc["like_threshold"], **best, **extra).fit(train, store, users, items)

        mdir = cfg.path("models")
        with open(mdir / f"{name}.pkl", "wb") as fh:
            pickle.dump(model, fh)
        write_json({"run_id": run_id, "model": name, "params": model.params()}, mdir / f"{name}_run.json")
        tracker.log_params(run_id, {**model.params(), "seed": cfg["seed"], "train_rows": len(train), "tuned_on": "train_core->val",
                                    "feature_as_of": store.as_of, "n_tuning_trials": len(trials)})
        tracker.log_metrics(run_id, {"train_seconds": round(time.time() - t0, 3)})
        tracker.log_artifact(run_id, mdir / f"{name}.pkl")
        if len(trials):
            tp = cfg.path("reports", f"tuning_{name}.csv")
            by_rmse = cfg["models"][name].get("tune_metric") == "val_rmse"
            trials.sort_values("val_rmse" if by_rmse else f"val_ndcg@{cfg['evaluation']['k']}", ascending=by_rmse).to_csv(tp, index=False)
            tracker.log_artifact(run_id, tp)
            tracker.log_metrics(run_id, {f"best_val_ndcg@{cfg['evaluation']['k']}": float(trials[f"val_ndcg@{cfg['evaluation']['k']}"].max())})
    except Exception:
        tracker.end_run(run_id, "FAILED")   # do not leave a zombie RUNNING run in MLflow
        raise
    log.info("Trained %s with %s in %.1fs (tracker run %s)", name, model.params(), time.time() - t0, run_id)
    return {"model": name, "run_id": run_id, "params": model.params()}
