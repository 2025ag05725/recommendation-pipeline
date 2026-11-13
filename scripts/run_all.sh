#!/usr/bin/env bash
# Full clean run: smoke check -> wipe outputs -> synthetic pipeline -> MovieLens-1M pipeline -> diagram/screenshots/reports.
# Run inside the PROJECT venv:   source .venv/bin/activate && bash scripts/run_all.sh
# Optional env: PYTHON=python  SYNTH_CONFIG=configs/config.yaml  ML_CONFIG=configs/config.movielens1m.yaml  SKIP_SMOKE=1  SKIP_ML=1
set -uo pipefail
cd "$(dirname "$0")/.."
PY="${PYTHON:-python}"; SYNTH="${SYNTH_CONFIG:-configs/config.yaml}"; ML="${ML_CONFIG:-configs/config.movielens1m.yaml}"
WARN=()
step() { echo; echo "=================== $* ==================="; STEP_T=$SECONDS; }
done_() { echo "--- done in $((SECONDS-STEP_T))s"; }
must() { "$@" || { echo; echo "!!! FAILED: $*"; echo "Send me the full output above this line."; exit 1; }; }
soft() { "$@" || { echo "!!! WARNING (non-fatal): $*"; WARN+=("$*"); }; }

step "1/7 Environment check";   [ "${SKIP_SMOKE:-0}" = 1 ] && echo skipped || must $PY scripts/smoke_check.py; done_
step "2/7 Clean old outputs";   must bash scripts/clean_outputs.sh; done_
step "3/7 Synthetic pipeline (dirty data -> shows validation catching bad records)"
must $PY run_pipeline.py --config "$SYNTH"; done_
if [ "${SKIP_ML:-0}" = 1 ]; then echo "MovieLens skipped"; else
  step "4/7 Download + convert MovieLens 1M"; must $PY scripts/prepare_movielens.py --dataset ml-1m --out data/movielens1m/raw; done_
  step "5/7 MovieLens-1M pipeline";          must $PY run_pipeline.py --config "$ML"; done_
fi
step "6/7 Architecture diagram + run screenshots"
soft $PY scripts/make_architecture.py
soft $PY scripts/make_run_screenshot.py reports
[ "${SKIP_ML:-0}" = 1 ] || soft $PY scripts/make_run_screenshot.py reports/movielens1m
done_
step "7/7 Final reports (rebuild again after the Airflow screenshot: bash scripts/build_reports.sh)"
bash scripts/build_reports.sh || WARN+=("build_reports.sh"); done_

echo; echo "=================== FINISHED ==================="
echo "Synthetic results : reports/model_evaluation.md , reports/validation_report.md"
[ "${SKIP_ML:-0}" = 1 ] || echo "MovieLens-1M      : reports/movielens1m/model_evaluation.md , reports/movielens1m/validation_report.md"
[ ${#WARN[@]} -eq 0 ] && echo "No warnings." || { echo "Warnings (non-fatal):"; printf '  - %s\n' "${WARN[@]}"; }
echo "Next: bash scripts/run_airflow.sh   (separate terminal/venv; see QUICKSTART.md)"
