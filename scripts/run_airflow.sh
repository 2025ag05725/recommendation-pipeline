#!/usr/bin/env bash
# Run the DAG once under real Apache Airflow (CLI test run). Needs .venv-airflow (SETUP_GUIDE.md section 11.1).
#   bash scripts/run_airflow.sh            # uses the MovieLens-1M config
#   RECSYS_CONFIG=configs/config.yaml bash scripts/run_airflow.sh   # synthetic instead
set -uo pipefail
cd "$(dirname "$0")/.."
[ -x .venv-airflow/bin/airflow ] || { echo "Airflow venv missing: follow SETUP_GUIDE.md section 11.1 first."; exit 1; }
[ -x .venv/bin/python ] || { echo ".venv (project env) missing."; exit 1; }
export RECSYS_HOME="$(pwd)" RECSYS_PYTHON="$(pwd)/.venv/bin/python"
CFG="${RECSYS_CONFIG:-configs/config.movielens1m.yaml}"
case "$CFG" in /*) ;; *) CFG="$(pwd)/$CFG" ;; esac
export RECSYS_CONFIG="$CFG"
export AIRFLOW_HOME="$(pwd)/.airflow" AIRFLOW__CORE__DAGS_FOLDER="$(pwd)/dags" AIRFLOW__CORE__LOAD_EXAMPLES=False
AF=.venv-airflow/bin/airflow; mkdir -p logs
$AF db migrate >/dev/null 2>&1 || { echo "airflow db migrate failed"; exit 1; }
echo "Import errors (expect 'No data found'):"; $AF dags list-import-errors 2>&1 | grep -v UserWarning
RUN_DATE="$(date -u +%Y-%m-%d)"
echo "Running DAG recommendation_pipeline for $RUN_DATE with $RECSYS_CONFIG (a few minutes) ..."
$AF dags test recommendation_pipeline "$RUN_DATE" 2>&1 | tee logs/airflow_dags_test.log | grep -E "DagRun Finished|ERROR|Traceback" | cut -c1-200
if grep -q "state=success" logs/airflow_dags_test.log; then
  echo; echo "AIRFLOW RUN: SUCCESS"
  echo "Now (same terminal):  .venv-airflow/bin/airflow standalone   -> http://localhost:8080 (admin / password printed)"
  echo "Open DAG recommendation_pipeline > Graph, click the newest run, screenshot, save as reports/movielens1m/figures/airflow_successful_run.png"
  echo "Then: bash scripts/build_reports.sh"
else
  echo; echo "AIRFLOW RUN: FAILED - send me logs/airflow_dags_test.log (or the output above)"; exit 1
fi
