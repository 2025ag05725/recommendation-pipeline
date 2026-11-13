# Quickstart - clean run from scratch (synthetic + real MovieLens 1M)

One-time machine setup (WSL2, Python 3.11, `.venv`, `.venv-airflow`) is in `SETUP_GUIDE.md`. **Keep both `.venv` folders** - never delete them.

> **Order matters.** `run_all.sh` (and `clean_outputs.sh`) **delete everything in `reports/`**, including screenshots you saved earlier. Run `run_all.sh` once, *then* Airflow, *then* save the screenshots, *then* `build_reports.sh`. Do not run `run_all.sh` again afterwards.

## The whole thing in 3 commands (about 15-20 minutes)
```bash
cd ~/recommendation-pipeline && source .venv/bin/activate

# 1. smoke check -> wipe old outputs -> synthetic pipeline -> download MovieLens 1M -> ML-1M pipeline -> diagram, screenshots, reports
bash scripts/run_all.sh

# 2. Apache Airflow: runs the DAG once on ML-1M (works from any terminal; uses .venv-airflow automatically)
bash scripts/run_airflow.sh

# 3. After saving the Airflow screenshot (see below), rebuild the final reports
bash scripts/build_reports.sh
```
Each step prints what it is doing and stops with a clear "FAILED" line if something breaks; the output above that line is what you need to diagnose it. Non-fatal problems (e.g. Graphviz missing) are listed at the end.

### Airflow screenshot (the only manual part)
After `run_airflow.sh` prints `AIRFLOW RUN: SUCCESS`:
```bash
.venv-airflow/bin/airflow standalone     # needs the same exports as run_airflow.sh, see below
```
Open http://localhost:8080 (user `admin`, password printed), open DAG `recommendation_pipeline` > **Graph**, click the **newest run** in the grid (so the boxes turn green), screenshot (Win+Shift+S copies to the clipboard only: click the pop-up > Save, or paste into Paint and save as PNG), then copy it. Put it in `reports/figures/` as well if you want the synthetic report to embed it:
```bash
cp "/mnt/c/Users/WINUSER/Pictures/Screenshots/<file>.png" ~/recommendation-pipeline/reports/movielens1m/figures/airflow_successful_run.png
```
Exports for `airflow standalone` (paste first, same terminal):
```bash
cd ~/recommendation-pipeline
export RECSYS_HOME="$(pwd)" RECSYS_PYTHON="$(pwd)/.venv/bin/python" RECSYS_CONFIG="$(pwd)/configs/config.movielens1m.yaml"
export AIRFLOW_HOME="$(pwd)/.airflow" AIRFLOW__CORE__DAGS_FOLDER="$(pwd)/dags" AIRFLOW__CORE__LOAD_EXAMPLES=False
```

### MLflow UI (optional screenshot)
```bash
mlflow ui --backend-store-uri ./mlruns --host 127.0.0.1 --port 5000      # http://localhost:5000
```
Experiments: `recsys-pipeline` (synthetic) and `recsys-movielens-1m`. Tick the four runs > **Compare**, take the screenshot and save it in `reports/movielens1m/figures/` (e.g. `mlflow_runs_comparison.png`).

## Where the results are
| | Synthetic | MovieLens 1M |
|---|---|---|
| Evaluation | `reports/model_evaluation.md` | `reports/movielens1m/model_evaluation.md` |
| Validation | `reports/validation_report.md` | `reports/movielens1m/validation_report.md` |
| Feature catalog | `reports/feature_catalog.md` | `reports/movielens1m/feature_catalog.md` |
| Final report | `reports/final_report.pdf` | `reports/movielens1m/final_report.pdf` |
Architecture diagram: `reports/architecture.png`. DAG: `dags/recommendation_dag.py`.

## Package the submission
```bash
cd ~ && rm -f recommendation-pipeline-final.zip
zip -qr recommendation-pipeline-final.zip recommendation-pipeline \
  -x "recommendation-pipeline/.git/*" "recommendation-pipeline/.venv/*" "recommendation-pipeline/.venv-airflow/*" "recommendation-pipeline/.airflow/*" "*/__pycache__/*"
cp recommendation-pipeline-final.zip /mnt/c/Users/WINUSER/Desktop/        # needs: sudo apt install -y zip
ls -lh recommendation-pipeline-final.zip                                  # includes data and models: roughly 100-250 MB
# smaller variant (about 50 MB): also exclude  "recommendation-pipeline/models/*" "recommendation-pipeline/data/*/raw/*" "recommendation-pipeline/data/*/_download/*" "recommendation-pipeline/mlruns/*/*/artifacts/*"
unzip -l recommendation-pipeline-final.zip | grep -E "final_report|airflow_successful_run"
```
Copy the PDF on its own with `cp reports/movielens1m/final_report.pdf /mnt/c/Users/WINUSER/Desktop/` (if Windows says "Permission denied", an older file of that name is open - close it or pick a new name).

## Publish to GitHub (private repo, data included)
Create an empty **private** repository on github.com first, then:
```bash
cd ~/recommendation-pipeline
printf '%s\n' ".venv/" ".venv-airflow/" ".airflow/" "__pycache__/" "*.pyc" ".pytest_cache/" > .gitignore
git init -b main                      # skip if .git already exists
git config user.name "<your name>" && git config user.email "<your email>"
git add -A && git commit -m "Recommendation pipeline - final version"
git remote add origin https://github.com/<user>/<repo>.git
git push -u origin main               # username + personal access token as password (never paste the token anywhere else)
```
GitHub rejects single files over 100 MB (the largest here, a Feast `online_store.db`, is about 58 MB). Keep the repo private: the MovieLens licence forbids redistribution.

## Repeat run
`bash scripts/clean_outputs.sh` (add `--airflow` to reset the Airflow database too) wipes generated files only. `run_all.sh` does this for you.

---
## Manual steps (what the scripts do)
Assumes the one-time setup in `SETUP_GUIDE.md` sections 2-7 is done (WSL2, Python 3.11, `.venv`, `scripts/smoke_check.py` all OK).
Everything below runs in the **Ubuntu terminal**. Synthetic data and the fallback engines are optional extras and are **not needed**.

```bash
cd ~/recommendation-pipeline && source .venv/bin/activate

# 0. (only for a repeat run) wipe generated outputs - keeps code and virtual environments
bash scripts/clean_outputs.sh

# 1. Download + convert MovieLens 1M
python scripts/prepare_movielens.py --dataset ml-1m --out data/movielens1m/raw

# 2. Run the whole pipeline (ingest > validate > prepare > transform > features > 4 models > evaluate)
python run_pipeline.py --config configs/config.movielens1m.yaml

# 3. Results
cat reports/movielens1m/model_evaluation.md
head -22 reports/movielens1m/validation_report.md

# 4. Diagram + run screenshot
python scripts/make_architecture.py
python scripts/make_run_screenshot.py reports/movielens1m

# 5. MLflow UI (http://localhost:5000, experiment recsys-movielens-1m) - Ctrl+C to stop
mlflow ui --backend-store-uri ./mlruns --host 127.0.0.1 --port 5000
```

## Airflow run + screenshot (own terminal, Airflow venv; see SETUP_GUIDE.md section 11 for the one-time install)
```bash
cd ~/recommendation-pipeline && source .venv-airflow/bin/activate
export RECSYS_HOME="$(pwd)" RECSYS_PYTHON="$(pwd)/.venv/bin/python" RECSYS_CONFIG="$(pwd)/configs/config.movielens1m.yaml"
export AIRFLOW_HOME="$(pwd)/.airflow" AIRFLOW__CORE__DAGS_FOLDER="$(pwd)/dags" AIRFLOW__CORE__LOAD_EXAMPLES=False
airflow db migrate
airflow dags test recommendation_pipeline 2026-01-01        # runs all tasks, ends with state=success
airflow standalone                                          # http://localhost:8080 - trigger the DAG, open Graph, screenshot
```
Save the screenshot as `reports/movielens1m/figures/airflow_successful_run.png`.

## Final report
```bash
source .venv/bin/activate
python scripts/make_final_report.py reports/movielens1m data/movielens1m/processed
```
Deliverables are under `reports/movielens1m/` (plus `reports/architecture.png`, `dags/`, `src/`, `README.md`).
