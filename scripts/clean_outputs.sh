#!/usr/bin/env bash
# Delete everything the pipeline GENERATES so the next run starts from scratch.
# Keeps: source code, configs, .venv, .venv-airflow (no re-install needed), reports/architecture.*
#   bash scripts/clean_outputs.sh            # clean generated data, models, reports, MLflow runs, Feast stores, logs
#   bash scripts/clean_outputs.sh --airflow  # additionally reset the Airflow database (then run: airflow db migrate)
set -euo pipefail
cd "$(dirname "$0")/.."
rm -rf data models mlruns mlruns_local logs feature_repo feature_repo_* .pytest_cache
if [ -d reports ]; then
  find reports -mindepth 1 -maxdepth 1 ! -name 'architecture.png' ! -name 'architecture.svg' -exec rm -rf {} +
fi
find . -name __pycache__ -not -path './.venv*' -prune -exec rm -rf {} + 2>/dev/null || true
if [ "${1:-}" = "--airflow" ]; then rm -rf .airflow; echo "Airflow state removed - run 'airflow db migrate' again."; fi
echo "Clean. Generated outputs removed (code, configs and virtual environments kept)."
