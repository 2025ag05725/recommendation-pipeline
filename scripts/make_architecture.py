"""Render reports/architecture.png/.svg (Graphviz) - the pipeline architecture diagram."""
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOT = r'''
digraph A {
  rankdir=TB; bgcolor="white"; nodesep=0.3; ranksep=0.45; newrank=true; fontname="Helvetica";
  node [fontname="Helvetica", fontsize=11, shape=box, style="rounded,filled", color="#52606d", fillcolor="#eef2f7"];
  edge [color="#52606d", fontname="Helvetica", fontsize=9];

  subgraph cluster_src { label="Sources"; style="rounded,dashed"; color="#9aa5b1"; fontsize=11;
    syn  [label="Data source\n(real MovieLens CSVs  or  seeded\nsynthetic generator with injected defects)", fillcolor="#fff4d6"];
    csv  [label="Raw CSVs\nusers · items · interactions", shape=cylinder, fillcolor="#fff4d6"]; }

  subgraph cluster_orch { label="Orchestration  (built-in DAG runner  |  Airflow DAG — same TASKS list)"; style="rounded"; color="#4c8dc9"; fontsize=11;
    ing  [label="1 Ingestion\nmanifest + md5", fillcolor="#dbeafe"];
    val  [label="2 Validation\n34 checks · quality gate", fillcolor="#dbeafe"];
    prep [label="3 Preparation\nimpute · standardise · dedupe", fillcolor="#dbeafe"];
    trn  [label="4 Transformation\ntime split · matrix · summaries", fillcolor="#dbeafe"];
    feat [label="5 Feature management\nregistry · store · catalog", fillcolor="#dbeafe"];
    m1 [label="6a Popularity\n(baseline)", fillcolor="#fde2cf"];
    m2 [label="6b Item-kNN CF", fillcolor="#fde2cf"];
    m3 [label="6c Hybrid SVD\n(CF + genre affinity)", fillcolor="#fde2cf"];
    ev [label="7 Evaluation\nP/R/NDCG@10 · coverage\ndiversity · RMSE/MAE", fillcolor="#dbeafe"]; }

  err  [label="data/errors/\nrejected rows + issue log", shape=cylinder, fillcolor="#fde0e0"];
  proc [label="data/processed/\nclean tables · train/val/test", shape=cylinder, fillcolor="#e2f3e8"];
  fst  [label="data/features/\nuser & item feature tables", shape=cylinder, fillcolor="#e2f3e8"];
  mdl  [label="models/*.pkl", shape=cylinder, fillcolor="#e2f3e8"];
  trk  [label="Experiment tracker\nMLflow (if installed) | local JSON\nparams · metrics · artifacts", fillcolor="#ece3f7"];
  rep  [label="reports/\nvalidation · catalog · evaluation\nrun report", shape=note, fillcolor="#e2f3e8"];
  srv  [label="Serving\nrecommend CLI\n(same feature store)", fillcolor="#ece3f7"];

  syn -> csv; csv -> ing -> val -> prep -> trn -> feat;
  val -> err [label="invalid rows", style=dashed, color="#d64545"];
  val -> proc [label="valid rows"]; prep -> proc; trn -> proc;
  feat -> fst; feat -> m1; feat -> m2; feat -> m3;
  fst -> m1 [style=dotted]; fst -> m3 [style=dotted]; proc -> m2 [style=dotted];
  m1 -> ev; m2 -> ev; m3 -> ev;
  m1 -> mdl; m2 -> mdl; m3 -> mdl;
  m1 -> trk [style=dashed]; m2 -> trk [style=dashed]; m3 -> trk [style=dashed]; ev -> trk [style=dashed];
  ev -> rep; val -> rep; feat -> rep;
  mdl -> srv; fst -> srv;
}
'''
for fmt, extra in (("png", ["-Gdpi=170"]), ("svg", [])):
    subprocess.run(["dot", f"-T{fmt}", *extra, "-o", str(ROOT / "reports" / f"architecture.{fmt}")], input=DOT.encode(), check=True)
print("written reports/architecture.png/.svg")
