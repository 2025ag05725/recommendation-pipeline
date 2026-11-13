#!/usr/bin/env bash
# (Re)build the 2-3 page final report(s). Run again after saving the Airflow screenshot.
set -uo pipefail
cd "$(dirname "$0")/.."
PY="${PYTHON:-python}"
[ -f reports/validation_report.md ] && $PY scripts/make_final_report.py reports data/processed
if [ "${SKIP_ML:-0}" != 1 ] && [ -f reports/movielens1m/validation_report.md ]; then
  $PY scripts/make_final_report.py reports/movielens1m data/movielens1m/processed
fi
