"""Evaluation stage.

Protocol (leave-future-out, per user):
  * For every user who has >=1 *relevant* test interaction (rating >= relevance_threshold) AND some training history:
    rank the whole catalogue minus the user's training items, take top-K, compare with the relevant test items.
  * Metrics: Precision@K, Recall@K, NDCG@K (averaged over evaluated users), Coverage (catalogue share recommended),
    Diversity (intra-list genre diversity). Models with a rating head also get RMSE / MAE over ALL test ratings.
  * Users with no training history are excluded from ranking metrics (and counted); every model would fall back to
    popularity for them, so they carry no signal about personalisation.
"""
from __future__ import annotations

import pickle

import pandas as pd

from src.common.io_utils import read_json, write_json
from src.common.logging_utils import get_logger
from src.evaluation import metrics as M
from src.tracking.tracker import Tracker

log = get_logger("evaluation")


def evaluate_model(model, test: pd.DataFrame, k: int, rel_thr: float) -> dict:
    rel = test[test["rating"] >= rel_thr].groupby("user_id")["item_id"].agg(set)
    has_history = [u for u in rel.index if model.uidx.get(int(u), -1) >= 0 and model.seen[model.uidx[int(u)]].nnz > 0]
    users = pd.Index(has_history)
    recs = model.recommend(users.to_numpy(), k)
    lists = [list(r) for r in recs]
    gv = model.store.get_item_features(model.item_ids)[model.store.genre_cols].to_numpy()
    gvec = {int(i): gv[j] for j, i in enumerate(model.item_ids)}
    out = {
        f"precision@{k}": sum(M.precision_at_k(l, rel[u], k) for l, u in zip(lists, users)) / len(users),
        f"recall@{k}": sum(M.recall_at_k(l, rel[u], k) for l, u in zip(lists, users)) / len(users),
        f"ndcg@{k}": sum(M.ndcg_at_k(l, rel[u], k) for l, u in zip(lists, users)) / len(users),
        "coverage": M.catalog_coverage(lists, len(model.item_ids)),
        "diversity": sum(M.intra_list_diversity(l, gvec) for l in lists) / len(lists),
        "n_eval_users": len(users), "n_users_excluded_no_history": int(len(rel) - len(users)),
    }
    pred = model.predict_rating(test["user_id"].to_numpy(), test["item_id"].to_numpy())
    if pred is not None:
        out["rmse"], out["mae"] = M.rmse(test["rating"], pred), M.mae(test["rating"], pred)
    return out


def run(cfg) -> dict:
    proc, mdir, rep = cfg.path("processed"), cfg.path("models"), cfg.path("reports")
    test = cfg.read_table("processed", "test", parse_dates=["timestamp"])
    k, thr = cfg["evaluation"]["k"], cfg["evaluation"]["relevance_threshold"]
    tracker = Tracker(cfg)
    rows = []
    for name in [m for m, c in cfg["models"].items() if c.get("enabled", True)]:
        with open(mdir / f"{name}.pkl", "rb") as fh:
            model = pickle.load(fh)
        run_id = read_json(mdir / f"{name}_run.json")["run_id"]
        res = evaluate_model(model, test, k, thr)
        tracker.log_metrics(run_id, {f"test_{m}": v for m, v in res.items()})
        write_json(res, rep / f"metrics_{name}.json")
        tracker.log_artifact(run_id, rep / f"metrics_{name}.json")
        tracker.end_run(run_id)
        rows.append({"model": name, **res})
        log.info("%-11s %s", name, {m: round(v, 4) for m, v in res.items()})
    table = pd.DataFrame(rows).set_index("model")
    table.to_csv(rep / "model_evaluation.csv")
    write_json({"k": k, "relevance_threshold": thr, "results": table.reset_index().to_dict("records")}, rep / "model_evaluation.json")
    _write_md(table, k, thr, rep / "model_evaluation.md")
    _plot(table, k, rep / "figures" / "model_comparison.png")
    _write_runs_csv(tracker, rep / "experiment_runs.csv")
    return {"best_ndcg_model": table[f"ndcg@{k}"].idxmax(), **{f"ndcg@{k}_{m}": round(v, 4) for m, v in table[f"ndcg@{k}"].items()}}


def _write_md(t: pd.DataFrame, k: int, thr: float, path) -> None:
    cols = [f"precision@{k}", f"recall@{k}", f"ndcg@{k}", "coverage", "diversity", "rmse", "mae"]
    best = {c: (t[c].idxmin() if c in ("rmse", "mae") else t[c].idxmax()) for c in cols if c in t and t[c].notna().any()}
    L = [f"# Model Evaluation (K={k}, relevant = test rating >= {thr})", "",
         "| Model | " + " | ".join(cols) + " | eval users |", "|---|" + "---:|" * (len(cols) + 1)]
    for m, r in t.iterrows():
        cells = [("n/a" if pd.isna(r.get(c)) else (f"**{r[c]:.4f}**" if best.get(c) == m else f"{r[c]:.4f}")) for c in cols]
        L.append(f"| {m} | " + " | ".join(cells) + f" | {int(r['n_eval_users'])} |")
    L += ["", "Bold = best in column (lower is better for RMSE/MAE). `n/a` = model has no rating-prediction head.",
          "Test period = newest 20% of interactions; models were tuned on a validation slice and refit on all pre-test data."]
    path.write_text("\n".join(L) + "\n")


def _plot(t: pd.DataFrame, k: int, path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cols = [f"precision@{k}", f"recall@{k}", f"ndcg@{k}", "coverage", "diversity"]
    fig, axes = plt.subplots(1, len(cols), figsize=(14, 3.2))
    colors = ["#9aa5b1", "#4c8dc9", "#e07b39"]
    for ax, c in zip(axes, cols):
        ax.bar(t.index, t[c], color=colors[: len(t)])
        ax.set_title(c, fontsize=10)
        ax.tick_params(axis="x", labelrotation=25, labelsize=8)
        for i, v in enumerate(t[c]):
            ax.text(i, v, f"{v:.3f}", ha="center", va="bottom", fontsize=8)
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _write_runs_csv(tracker: Tracker, path) -> None:
    rows = []
    for r in tracker.list_runs():
        rows.append({"run_id": r["run_id"], "run_name": r["run_name"], "status": r["status"],
                     **{f"param.{k}": v for k, v in r["params"].items()}, **{f"metric.{k}": v for k, v in r["metrics"].items()}})
    pd.DataFrame(rows).to_csv(path, index=False)
