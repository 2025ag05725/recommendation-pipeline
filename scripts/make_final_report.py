"""Build reports/final_report.pdf (2-3 pages) from the actual pipeline outputs."""
import json
from pathlib import Path

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import KeepTogether, Image, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

import sys

R = Path(__file__).resolve().parents[1]
# usage: make_final_report.py [reports_dir] [processed_dir]   e.g.  reports/movielens data/movielens/processed
rep = Path(sys.argv[1]) if len(sys.argv) > 1 else R / "reports"
proc = Path(sys.argv[2]) if len(sys.argv) > 2 else R / "data/processed"
val = json.loads((rep / "validation_report.json").read_text())
ev = pd.read_csv(rep / "model_evaluation.csv").set_index("model")
split = json.loads((proc / "split_info.json").read_text())
prep = json.loads((proc / "preparation_summary.json").read_text())
run = json.loads((rep / "latest_run_summary.json").read_text())
eng = run.get("engines", {})
real = {"validation": eng.get("validation") == "great_expectations", "features": eng.get("features") == "feast", "storage": eng.get("storage") == "parquet",
        "tracking": eng.get("tracking") == "mlflow", "surprise": eng.get("surprise_model", False)}
airflow_shot = (rep / "figures" / "airflow_successful_run.png").exists()
airflow_note = "; Airflow run: <i>reports/figures/airflow_successful_run.png</i>" if airflow_shot else ""
n_runs = len(pd.read_csv(rep / "experiment_runs.csv")) if (rep / "experiment_runs.csv").exists() else 0
n_checks = len(val["checks"])

ss = getSampleStyleSheet()
B = ParagraphStyle("b", parent=ss["BodyText"], fontSize=8.8, leading=11.4, spaceAfter=3)
H = ParagraphStyle("h", parent=ss["Heading2"], fontSize=11.5, spaceBefore=7, spaceAfter=3, textColor=colors.HexColor("#1f4e79"))
T = ParagraphStyle("t", parent=ss["Title"], fontSize=16, spaceAfter=2)
S = ParagraphStyle("s", parent=B, fontSize=7.6, leading=9.4)
bullet = lambda t: Paragraph("• " + t, ParagraphStyle("bl", parent=B, leftIndent=10, firstLineIndent=-8))  # noqa: E731

def table(data, widths, size=7.6):
    t = Table(data, colWidths=widths, repeatRows=1)
    t.setStyle(TableStyle([("FONTSIZE", (0, 0), (-1, -1), size), ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#dbe7f3")),
                           ("GRID", (0, 0), (-1, -1), 0.3, colors.grey), ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                           ("TOPPADDING", (0, 0), (-1, -1), 2), ("BOTTOMPADDING", (0, 0), (-1, -1), 2)]))
    return t

k = 10
rows = [["Model", f"P@{k}", f"R@{k}", f"NDCG@{k}", "Coverage", "Diversity", "RMSE", "MAE"]]
for m, r in ev.iterrows():
    f = lambda c: "n/a" if pd.isna(r.get(c)) else f"{r[c]:.4f}"  # noqa: E731
    rows.append([m, f(f"precision@{k}"), f(f"recall@{k}"), f(f"ndcg@{k}"), f("coverage"), f("diversity"), f("rmse"), f("mae")])
tv = val["tables"]
vrows = [["Table", "Rows in", "Rejected", "Reject rate", "Rows w/ warnings"]] + [
    [t, s["rows_in"], s["rows_rejected"], f"{s['error_rate']:.2%}", s["warning_rows"]] for t, s in tv.items()]
pop, knn, hyb = ev.loc["popularity"], ev.loc["item_knn"], ev.loc["hybrid_svd"]
lift = hyb[f"ndcg@{k}"] / pop[f"ndcg@{k}"]

# ---------------- data-driven wording (so the text always matches the numbers of THIS run) ----------------
from reportlab.lib.utils import ImageReader

real_data = eng.get("data_source") == "csv"
users_n, items_n, inter_n = tv["users"]["rows_in"], tv["items"]["rows_in"], tv["interactions"]["rows_in"]
has_demo = prep.get("users_gender_filled", users_n) < 0.5 * users_n
nd = ev[f"ndcg@{k}"]
fmt_n = lambda n: f"{int(n):,}"  # noqa: E731

if real_data:
    demo_txt = ("gender and an age bracket are available (age = bracket midpoint) while country is absent and imputed as UNK" if has_demo else
                "no user demographics are available, so age, gender and country are imputed")
    data_txt = (f"<b>Data.</b> Real <b>MovieLens</b> ratings (GroupLens; {fmt_n(users_n)} users, {fmt_n(items_n)} movies, {fmt_n(inter_n)} ratings) converted to the pipeline schema by "
                f"<i>scripts/prepare_movielens.py</i>; {demo_txt}; signup date is the user's first rating.")
    determinism = "All stochastic steps (tuning, factorisation, SGD) use fixed seeds, and each stage reads only the files written by the previous one."
else:
    demo_txt = ""
    data_txt = ("<b>Data.</b> A seeded MovieLens-style synthetic dataset (2,000 users, 900 movies, ~60k ratings, 2023-2025) in which both popularity and personal taste drive viewing and rating, "
                "with realistic injected defects (missing values, invalid ratings, bad/future/ancient timestamps, orphan keys, duplicates, mixed formats). "
                "Results are therefore <i>not</i> comparable with published MovieLens numbers; the pipeline also runs on real MovieLens (see README).")
    determinism = "The whole run is deterministic: two consecutive runs with regenerated data produced byte-identical datasets and identical metrics."

# --- results paragraph
pop_nd = nd["popularity"]
pers = [m for m in ("item_knn", "hybrid_svd") if m in ev.index]
lifts = " and ".join(f"{m} {nd[m] / pop_nd:.2f}×" for m in pers)
res = [(f"Both personalised models beat the popularity baseline on NDCG@{k} ({lifts} the baseline's)." if all(nd[m] > pop_nd for m in pers)
        else f"NDCG@{k} relative to the popularity baseline: {lifts}.")]
order = nd.sort_values(ascending=False)
b1, b2 = order.index[0], order.index[1]
if order.iloc[0] - order.iloc[1] < 0.01:
    res.append(f"{b1} and {b2} are effectively tied on NDCG@{k} ({order.iloc[0]:.4f} vs {order.iloc[1]:.4f}); a gap this small is within what a single split can resolve, so neither should be called best on ranking accuracy.")
else:
    res.append(f"{b1} is best on NDCG@{k} ({order.iloc[0]:.4f}), ahead of {b2} ({order.iloc[1]:.4f}).")
rb, cb = ev[f"recall@{k}"].idxmax(), ev["coverage"].idxmax()
res.append(f"{rb} has the best recall@{k} ({ev.loc[rb, f'recall@{k}']:.4f}) and {cb} the widest catalogue coverage ({ev.loc[cb, 'coverage']:.1%}; popularity {pop['coverage']:.1%}).")
rm = ev["rmse"].dropna()
rmb = rm.idxmin()
res.append(f"Rating error is lowest for {rmb} (RMSE {rm.min():.3f}, MAE {ev.loc[rmb, 'mae']:.3f}) against {pop['rmse']:.3f} for popularity" +
           ("; scikit-surprise SVD predicts ratings well but ranks poorly (NDCG@%d %.3f), because it is trained for rating error, not top-N ranking." % (k, nd["surprise_svd"])
            if "surprise_svd" in ev.index and nd["surprise_svd"] < pop_nd else "."))
dv_h, dv_p = hyb["diversity"], pop["diversity"]
res.append(f"Genre diversity of the lists: hybrid {dv_h:.2f} vs popularity {dv_p:.2f}" +
           (" - similar, so personalisation costs little diversity." if abs(dv_h - dv_p) < 0.02 else
            (" - personalisation costs some diversity." if dv_h < dv_p else " - personalised lists are not less diverse than popularity here.")))
_tp = rep / "tuning_hybrid_svd.csv"
if _tp.exists():
    _td = pd.read_csv(_tp)
    if "alpha" in _td.columns and float(_td.iloc[0]["alpha"]) == 0.0:
        res.append("Validation selected alpha=0 for the hybrid, i.e. the genre-affinity content blend did not improve validation NDCG on this dataset, so its ranking comes from the collaborative SVD component alone.")
results_txt = " ".join(res) + " All comparisons come from one chronological split without confidence intervals, so they are indicative."

# --- validation paragraph
fails = sorted([c for c in val["checks"] if c["rows_failed"] > 0], key=lambda c: -c["rows_failed"])
errs = [c for c in fails if c["severity"] == "error"]
warns = [c for c in fails if c["severity"] == "warning"]
rej_total = sum(sv["rows_rejected"] for sv in tv.values())
wl = "; ".join(f"{c['table']}.{c['check']} ({fmt_n(c['rows_failed'])} rows)" for c in warns[:4])
if rej_total == 0:
    val_txt = ("No record violated an error-severity rule, so nothing was quarantined" + (f"; warnings (kept, repaired in preparation): {wl}" if wl else "") + ". ")
    synth_rep = R / "reports" / "validation_report.json"
    if real_data and synth_rep.exists() and synth_rep.resolve() != (rep / "validation_report.json").resolve():
        sv = json.loads(synth_rep.read_text())
        srej = sum(t["rows_rejected"] for t in sv["tables"].values())
        val_txt += (f"The companion run on the synthetic dataset, with deliberately injected defects, quarantined {fmt_n(srej)} records and so exercises the rejection path "
                    "(<i>reports/validation_report.md</i> in the project root).")
else:
    el = "; ".join(f"{c['table']}.{c['check']} ({fmt_n(c['rows_failed'])})" for c in errs[:5])
    val_txt = (f"{fmt_n(rej_total)} records were quarantined; most frequent error checks: {el}. Rejected users also remove their interactions through the referential check"
               + (", so the interaction reject rate exceeds the injected rate. " if not real_data else ". ")
               + (f"Warnings kept and repaired: {wl}. " if wl else ""))
run_txt = (f"Built-in runner: {sum(t['status'] == 'success' for t in run['tasks'].values())}/{len(run['tasks'])} tasks succeeded in {run['seconds']} s "
           f"(<i>reports/figures/successful_run.png</i>){airflow_note}.")

# --- limitations helpers
data_lim = ("a historical movie-rating snapshot: ratings are explicit and sparse, and " + ("demographics are limited to gender and an age bracket (no country)." if has_demo else "there are no user demographics.")
            if real_data else "synthetic data: absolute scores say little about real users and the generator shapes how easy personalisation is; the real-MovieLens path is documented in the README.")
tune_note = ""
tp = rep / ("tuning_hybrid_svd.csv")
if tp.exists():
    td = pd.read_csv(tp)
    prm = [c for c in td.columns if not c.startswith("val_")]
    best = td.iloc[0]
    edge = [f"{c}={best[c]:g}" for c in prm if td[c].nunique() > 2 and best[c] in (td[c].min(), td[c].max()) and not (c == "alpha" and best[c] == 0)]
    tune_note = (f"the selected hybrid setting has {', '.join(edge)} on the edge of the tuning grid (a wider grid might do slightly better); " if edge
                 else "the selected hybrid setting lies inside the tuning grid; ")
n_head = int(ev["rmse"].notna().sum())
no_head = ", ".join(ev.index[ev["rmse"].isna()])
eval_lim = (f"one chronological split, no confidence intervals, no online test; {tune_note}cold-start users get a popularity fallback and are excluded from ranking metrics; "
            f"the rating head is evaluated for {n_head} of {len(ev)} models ({no_head or 'all'} has none).")
scale_lim = (f"item-item similarity is computed in row chunks and kept as a sparse top-N matrix (it handled {fmt_n(items_n)} movies here), but scoring builds dense users × items blocks in memory; "
             "catalogues beyond a few hundred thousand items would need approximate neighbour search and sparse top-N retrieval.")

ar_w, ar_h = ImageReader(str(R / "reports" / "architecture.png")).getSize()
air_flow = []
if airflow_shot:
    aw, ah = ImageReader(str(rep / "figures" / "airflow_successful_run.png")).getSize()
    w = 15.5 * cm
    h = w * ah / aw
    if h > 7.2 * cm:
        w, h = w * 7.2 * cm / h, 7.2 * cm
    air_flow = [Spacer(1, 3), Image(str(rep / "figures" / "airflow_successful_run.png"), width=w, height=h),
                Paragraph("Figure: Apache Airflow graph view of the DAG run - all tasks succeeded.", S)]


doc = SimpleDocTemplate(str(rep / "final_report.pdf"), pagesize=A4, leftMargin=1.7 * cm, rightMargin=1.7 * cm, topMargin=1.4 * cm, bottomMargin=1.3 * cm,
                        title="Recommendation Pipeline - Final Report", author="DMML Assignment 2")
E = [Paragraph("End-to-End Data Management Pipeline for a Recommendation System", T),
     Paragraph("Data Management for Machine Learning – Assignment 2 · final report", ParagraphStyle("c", parent=B, alignment=1, textColor=colors.grey)),
     Paragraph("1. Overview and design decisions", H),
     Paragraph(f"The pipeline recommends movies and runs as {len(run['tasks'])} dependency-ordered tasks: ingestion, validation, preparation, transformation, "
               "feature management, training of the models, evaluation, with orchestration over all of them. Every stage reads and writes files "
               f"({'Parquet' if real['storage'] else 'CSV'}/JSON/NPZ/pickle) under <i>data/</i>, <i>models/</i> and <i>reports/</i>, so each stage can be re-run, inspected and tested on its own. "
               + determinism, B),
     Image(str(R / "reports" / "architecture.png"), width=9.6 * cm, height=9.6 * cm * ar_h / ar_w),
     bullet(data_txt),
     bullet(f"<b>Validation before cleaning.</b> {n_checks} checks ({val.get('engine', 'custom')}) run on the raw string data. <i>Errors</i> quarantine a record into <i>data/errors/</i> with the reason and original line number; "
            f"<i>warnings</i> (e.g. a missing value) keep the record and are repaired in preparation. A quality gate (reject rate &gt; {val['quality_gate_max_error_rate']:.0%}) fails the run, "
            "and downstream tasks are marked <i>upstream_failed</i>; I verified this behaviour on a copy of the project with a stricter threshold."),
     bullet(f"<b>Preparation / transformation.</b> Imputation (median age: {prep['users_age_imputed']} rows), standardised gender, country and genres, and one rating per (user, item) (latest kept; {prep['interactions_duplicate_pairs_dropped']} older duplicates dropped). "
            f"A chronological split avoids leakage: train {split['rows']['train']:,} / test {split['rows']['test']:,} rows, with a validation slice used only for tuning. "
            "Derived datasets (user activity summary, item popularity, sparse user-item matrix) come from train only."),
     bullet("<b>Feature management.</b> 12 user, 8 item and 3 user-item feature definitions live in one registry; the catalog is generated from it and checked against the materialised tables on every run. "
            + ("Features are published to <b>Feast</b> (user and item feature views), retrieved for training with point-in-time joins at the split date and served online from the SQLite online store; an online-vs-offline parity check guards against training/serving skew. " if real["features"] else
               "Features are served from local tables (Feast disabled in this run). ")
            + "Features are computed on train only and recency is measured against the split date. While building it I found and fixed an index-alignment bug that shifted genres between items; a unit test now guards it."),
     Paragraph("2. Models, tools and orchestration", H),
     bullet("<b>Popularity</b> (baseline): rank by training interaction count, Bayesian rating as tie-break and rating predictor."),
     bullet("<b>Item-kNN CF</b>: cosine item-item similarity on the 'liked' (rating ≥ 4) matrix, top-N neighbours, shrinkage."),
     bullet("<b>Hybrid SVD</b>: PureSVD latent score on liked interactions blended with a content signal (user genre affinity from the feature store), "
            "plus a biased-SVD rating head giving RMSE/MAE. Hyper-parameters are chosen by validation NDCG@10 (explicit rank by validation RMSE) and the model is refit on all pre-test data."),
     bullet("<b>Surprise SVD</b>: scikit-surprise's biased SVD on explicit ratings; ranks by predicted rating, so it is strong on RMSE/MAE but typically weaker for top-N." if real["surprise"] else "<b>Surprise SVD</b>: not run in this build (disabled in config)."),
     bullet("<b>Tools.</b> Python, pandas, NumPy, SciPy; " + "; ".join(x for x in [
            "Great Expectations (validation)" if real["validation"] else "lightweight custom validation", "Feast (features)" if real["features"] else "local feature tables",
            "Parquet storage" if real["storage"] else "CSV storage", "MLflow (tracking)" if real["tracking"] else "local JSON tracker", "scikit-surprise" if real["surprise"] else ""] if x)
            + f"; Graphviz/matplotlib for figures; pytest. {n_runs} tracked runs. Orchestration: Apache Airflow DAG generated from the shared task list" + (" (successful run screenshot included)" if airflow_shot else "")
            + "; a built-in DAG runner (parallel training tasks, retries, failure propagation, HTML run report) is also provided."),
     Paragraph("3. Results", H),
     Paragraph(f"Test protocol: newest 20% of interactions; for each of {int(hyb['n_eval_users']):,} users with history and ≥1 relevant test item (rating ≥ 4), the whole catalogue minus training items is ranked and the top 10 compared with relevant test items. "
               "RMSE/MAE cover all test ratings for models with a rating head.", B),
     KeepTogether([table(rows, [3.3 * cm] + [1.95 * cm] * 7), Spacer(1, 4)]),
     Paragraph(results_txt, B),
     Paragraph("<b>Data validation summary</b>", B), table(vrows, [3.3 * cm, 2.5 * cm, 2.5 * cm, 2.5 * cm, 3.5 * cm]),
     Paragraph(val_txt + " " + run_txt, B),
     *air_flow,
     Paragraph("4. Limitations", H),
     bullet("<b>Data:</b> " + data_lim),
     *([bullet("<b>Fallback engines used in this build:</b> " + ", ".join(k for k, v in real.items() if not v) + " ran with lightweight substitutes because the real tools were unavailable in the build environment; re-run with the default config for the full tool stack.")] if not all(real.values()) else []),
     *([bullet("<b>Airflow:</b> the DAG was not executed for this report (no Airflow screenshot found); the built-in runner's report is shown instead.")] if not airflow_shot else []),
     bullet("<b>Evaluation:</b> " + eval_lim),
     bullet("<b>Scale:</b> " + scale_lim),
     ]
doc.build(E)
print("ok")
