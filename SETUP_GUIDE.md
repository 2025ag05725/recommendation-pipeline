# Step-by-step Setup Guide (Windows + WSL2, with macOS/Linux notes)

This guide takes you from a fresh Windows PC to: the pipeline running on **synthetic data**, then on **real MovieLens**,
with **MLflow**, **Great Expectations**, **Feast**, **Surprise**, **Parquet**, and **Apache Airflow** (including the Airflow screenshot).
Follow the parts in order. Commands in `code blocks` are typed exactly as shown. Lines starting with `#` are comments (don't type them).

> **Status:** every step below has been run end to end on Windows 11 + WSL2 (Ubuntu 22.04, Python 3.11) with the real tools - Great Expectations 1.24.1,
> Feast, scikit-surprise, MLflow 2.22.5, Parquet and Apache Airflow 2.10.5. If a step still fails for you, copy the *complete* error text (and `python --version`)
> before trying anything else; library versions are the usual cause. **Step 7 (smoke check) exists to catch them early.**
> **Shortcut:** once sections 2-7 are done, `QUICKSTART.md` runs everything (synthetic + MovieLens 1M + Airflow + reports) with three scripts.

Time needed: ~60–90 minutes the first time (most of it downloads/installs). Disk: ~5 GB. RAM: 8 GB minimum.

Contents
1. [Which terminal is which](#1-which-terminal-is-which)
2. [Install WSL2 + Ubuntu](#2-install-wsl2--ubuntu-windows-only)
3. [Prepare Ubuntu (packages, Python 3.11)](#3-prepare-ubuntu)
4. [Get the project into Linux](#4-get-the-project-into-linux)
5. [Create the project environment](#5-create-the-project-environment)
6. [Install the browser used for screenshots](#6-optional-install-the-screenshot-browser)
7. [Smoke check — verify every tool](#7-smoke-check)
8. [Run the pipeline on synthetic data](#8-run-the-pipeline-synthetic-data)
9. [Download and run real MovieLens](#9-download-and-run-real-movielens)
10. [MLflow UI](#10-view-experiments-in-the-mlflow-ui)
11. [Airflow: install, run the DAG, take the screenshot](#11-apache-airflow)
12. [Regenerate the final report](#12-regenerate-the-final-report-pdf-and-diagrams)
13. [Collect your deliverables](#13-collect-your-deliverables)
14. [Troubleshooting](#14-troubleshooting)
15. [macOS / native Linux differences](#15-macos--native-linux)
16. [Command cheat-sheet](#16-command-cheat-sheet)

---

## 1. Which terminal is which

| Name used in this guide | How to open it | Prompt looks like |
|---|---|---|
| **PowerShell (Admin)** | Start menu → type `PowerShell` → right-click → **Run as administrator** | `PS C:\Windows\system32>` |
| **Ubuntu terminal** | Start menu → **Ubuntu** (after step 2) | `yourname@PCNAME:~$` |

Almost everything happens in the **Ubuntu terminal**. You can open several Ubuntu terminals at once (needed for Airflow: one keeps running the server).
**Every new Ubuntu terminal starts without the Python environment active** — activate it again (see the cheat-sheet in section 16).

---

## 2. Install WSL2 + Ubuntu (Windows only)

Requirements: Windows 10 version 2004+ (build 19041+) or Windows 11; virtualization enabled in BIOS/UEFI (usually on by default).

1. Open **PowerShell (Admin)**.
2. Install WSL with Ubuntu 22.04:
   ```powershell
   wsl --install -d Ubuntu-22.04
   ```
   - If it prints a help text instead ("WSL is already installed"), run `wsl --list --online` and then `wsl --install -d Ubuntu-22.04` again.
3. **Restart the PC** when asked.
4. After the restart an **Ubuntu** window opens by itself (if not: Start menu → Ubuntu). Wait for "Installing, this may take a few minutes".
5. It asks you to **create a UNIX username and password**. Choose a short lowercase username (e.g. `nishant`). The password is invisible while typing — that's normal. Remember it; `sudo` asks for it.
6. Check that it is WSL **version 2**. In PowerShell:
   ```powershell
   wsl -l -v
   ```
   You should see `Ubuntu-22.04   Running   2`. If VERSION shows `1`: `wsl --set-version Ubuntu-22.04 2`.
7. If you get *"WSL 2 requires an update to its kernel component"*: run `wsl --update` in PowerShell (Admin).
8. If you get *"Virtualization is disabled"*: restart, enter BIOS/UEFI (key shown at boot: F2/Del/F10), enable **Intel VT-x / AMD-V / SVM**, save, boot.

(Optional, helps with memory-hungry installs) Give WSL more RAM: create `C:\Users\<YourWindowsName>\.wslconfig` with
```
[wsl2]
memory=6GB
processors=4
```
then run `wsl --shutdown` in PowerShell and reopen Ubuntu.

---

## 3. Prepare Ubuntu

In the **Ubuntu terminal**:

```bash
sudo apt update && sudo apt upgrade -y
sudo apt install -y build-essential git unzip zip curl wget software-properties-common graphviz poppler-utils
```
(`build-essential` is the C compiler that scikit-surprise needs; `graphviz` draws the architecture diagram; `zip` and `poppler-utils` are used when packaging and checking the PDF.)

Install **Python 3.11** (Ubuntu 22.04 ships 3.10; we need 3.11 for the libraries):
```bash
sudo add-apt-repository -y ppa:deadsnakes/ppa
sudo apt update
sudo apt install -y python3.11 python3.11-venv python3.11-dev
python3.11 --version          # must print: Python 3.11.x
```

---

## 4. Get the project into Linux

**Important:** keep the project in the Linux home folder (`~`), **not** under `/mnt/c/...` — the Windows drive is very slow from Linux and causes permission problems.

1. Download `recommendation-pipeline.zip` (the file I sent you) to your normal Windows **Downloads** folder.
2. Find your Windows username: in Ubuntu run `ls /mnt/c/Users/` (ignore `Public`, `Default`, `All Users`). Below, replace `WINUSER` with it.
3. Copy and unzip:
   ```bash
   cp /mnt/c/Users/WINUSER/Downloads/recommendation-pipeline.zip ~/
   cd ~
   unzip recommendation-pipeline.zip
   cd ~/recommendation-pipeline
   ls
   ```
   You should see `README.md  SETUP_GUIDE.md  configs  dags  data  requirements.txt  run_pipeline.py  scripts  src  tests ...`.
   (If `cp` says "No such file", the zip has another name/location — run `ls /mnt/c/Users/WINUSER/Downloads/` to look.)

Tips
* Open the project folder in Windows Explorer from Ubuntu: `explorer.exe .` (note the dot).
* Edit files with VS Code + the **WSL** extension (`code .` inside the project folder). **Don't edit with Windows Notepad** (it adds Windows line endings that can break scripts).

---

## 5. Create the project environment

Inside `~/recommendation-pipeline`:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
```
The prompt now starts with `(.venv)`. If it doesn't, the environment is not active.

Install scikit-surprise first, with this order (it avoids the most common build failure with NumPy 2):
```bash
pip install "numpy<2" "cython<3"
pip install --no-build-isolation scikit-surprise
```
Then everything else:
```bash
pip install -r requirements.txt
```
This takes 5–15 minutes (Feast, MLflow and Great Expectations are large). Warnings in yellow are normal; red **ERROR** lines are not.
If it ends with `Successfully installed ...` you're done. If you see a dependency-conflict message ("ResolutionImpossible"), keep the full message: usually one version range in `requirements.txt` needs adjusting.

Check the versions that got installed (useful to include when reporting a problem):
```bash
pip list 2>/dev/null | grep -i -E "^(numpy|pandas|pyarrow|great.expectations|feast|scikit-surprise|mlflow|scipy) "
```

---

## 6. (Optional) Install the screenshot browser

Only needed for `scripts/make_run_screenshot.py` (the built-in runner's screenshot). Skip if you only want the Airflow screenshot.
```bash
playwright install --with-deps chromium
```
(asks for your sudo password; installs system libraries.)

---

## 7. Smoke check

Still in `~/recommendation-pipeline` with `(.venv)` active:
```bash
python scripts/smoke_check.py
```
Expected (versions will differ):
```
OK    parquet (pyarrow)                                     pyarrow 17.x
OK    Great Expectations                                    great_expectations 1.x
OK    Feast                                                 feast 0.4x
OK    scikit-surprise                                       scikit-surprise 1.1.4
OK    MLflow                                                mlflow 2.x
FAIL  Airflow (optional here - lives in its own env)        ModuleNotFoundError: No module named 'airflow'
OK    project unit tests (pytest)                           NN passed
```
The **Airflow FAIL is expected** (Airflow gets its own environment in step 11).
For any other FAIL run `VERBOSE=1 python scripts/smoke_check.py` and keep the full output for diagnosis. **Don't continue until these are OK** — it saves a lot of confusion.

Also run the tests with skipped ones listed (none should be skipped now):
```bash
python -m pytest -q -rs tests
```

---

## 8. Run the pipeline on synthetic data

```bash
python run_pipeline.py
```
What you'll see: log lines like `START ingest`, `SUCCESS validate in 4.2s` … ending with
`Run run_YYYYMMDD_HHMMSS finished: SUCCESS`. Takes about 1–3 minutes. The command's exit status is 0 on success.
If a task says `FAILED`, the error and traceback are printed right below it and saved in `logs/<run_id>/<task>.log`.

Look at the outputs:
```bash
cat reports/model_evaluation.md          # comparison table of the 4 models
cat reports/validation_report.md | head -30     # first line "Engine: great_expectations x.y.z" confirms real GX was used
ls data/processed data/features data/errors     # *.parquet files (Parquet storage)
cat reports/feature_catalog.md | head -20
ls mlruns                                 # MLflow file store
python -m src.training.recommend --user 17 --k 10 --model hybrid_svd    # recommendations using Feast online features
```
Sanity checks that the *real tools* really ran:
* `reports/validation_report.md` says `great_expectations`.
* `ls feature_repo/data` shows `registry.db` and `online_store.db` (Feast).
* `ls models` shows `surprise_svd.pkl`; `reports/model_evaluation.md` has a `surprise_svd` row.
* `ls data/processed` shows `.parquet` (not `.csv`) files.
* `reports/latest_run_summary.json` contains `"engines": {"storage": "parquet", "validation": "great_expectations", "features": "feast", "tracking": "mlflow", ...}`.

---

## 9. Download and run real MovieLens

The script downloads the official GroupLens zip and converts it to the pipeline's format. Inside `~/recommendation-pipeline` with `(.venv)` active:

```bash
python scripts/prepare_movielens.py --dataset ml-latest-small
```
Expected last lines: `Wrote .../data/movielens/raw: users=... items=... interactions=...` (about 610 users, 9.7k movies, 100k ratings; the numbers can differ slightly between dataset releases) and `Next: python run_pipeline.py --config configs/config.movielens.yaml`.

**If the download fails** (firewall/proxy), do it manually:
1. In your Windows browser open <https://grouplens.org/datasets/movielens/latest/> and download **ml-latest-small.zip** ("small": 100,000 ratings).
2. Copy it into Linux and convert:
   ```bash
   cp /mnt/c/Users/WINUSER/Downloads/ml-latest-small.zip ~/
   python scripts/prepare_movielens.py --zip ~/ml-latest-small.zip
   ```
   (Or with curl directly: `curl -L -o ~/ml-latest-small.zip https://files.grouplens.org/datasets/movielens/ml-latest-small.zip`.)

Bigger alternative: `ml-1m` (1 million ratings, has gender + age): `python scripts/prepare_movielens.py --dataset ml-1m`. It is much slower (Great Expectations over 1M rows can take many minutes). Start with `ml-latest-small`.

Run the pipeline on it:
```bash
python run_pipeline.py --config configs/config.movielens.yaml
```
Takes a few minutes (item-kNN and the tuning grids are the slow part). Results live in separate folders so nothing from the synthetic run is overwritten:
```bash
cat reports/movielens/model_evaluation.md
cat reports/movielens/validation_report.md | head -30
ls data/movielens/processed models/movielens
python -m src.training.recommend --config configs/config.movielens.yaml --user 1 --k 10 --model hybrid_svd
```
Expected for `ml-latest-small`: validation reports **0 rejected rows** but warnings for all users (no age/gender/country in this dataset — they are imputed), and warnings for the ~34 movies without genres.
(The MovieLens dataset licence: for research/education, don't redistribute it. Cite Harper & Konstan, ACM TiiS 2015.)

### 9b. MovieLens 1M (larger, recommended for the headline results)
`ml-latest-small` has only ~28 users in the time-based test period, so its metrics are noisy. `ml-1m` (1M ratings, ~6,000 users) gives a much larger evaluation set while keeping the same clean time-based split. It writes to its own folders, so nothing from the runs above is overwritten.
```bash
python scripts/prepare_movielens.py --dataset ml-1m --out data/movielens1m/raw
python run_pipeline.py --config configs/config.movielens1m.yaml        # a few minutes up to ~10, depending on your machine
cat reports/movielens1m/model_evaluation.md
python -m src.training.recommend --config configs/config.movielens1m.yaml --user 1 --k 10 --model hybrid_svd
python scripts/make_run_screenshot.py reports/movielens1m
python scripts/make_final_report.py reports/movielens1m data/movielens1m/processed
```
MLflow shows it as the experiment `recsys-movielens-1m`. Manual download: <https://grouplens.org/datasets/movielens/1m/> then `--zip ~/ml-1m.zip --out data/movielens1m/raw`. For Airflow use `export RECSYS_CONFIG="$(pwd)/configs/config.movielens1m.yaml"`.


---

## 10. View experiments in the MLflow UI

In the Ubuntu terminal (project folder, `(.venv)` active):
```bash
mlflow ui --backend-store-uri ./mlruns --host 127.0.0.1 --port 5000
```
Open **http://localhost:5000** in your normal Windows browser. You'll see experiments `recsys-pipeline` (synthetic) and `recsys-movielens`; click one → one run per model with parameters, metrics (`test_ndcg@10`, `test_rmse`, …) and artifacts (model file, tuning table).
Take a screenshot for your report if useful. Stop the server with **Ctrl+C**. (If localhost doesn't open, see troubleshooting.)

---

## 11. Apache Airflow

Airflow needs its **own** Python environment (its pinned versions conflict with Feast/Great Expectations). It will call the project environment to run the tasks.

### 11.1 Install Airflow
Open the Ubuntu terminal, go to the project and **leave** the project venv if active:
```bash
cd ~/recommendation-pipeline
deactivate 2>/dev/null
python3.11 -m venv .venv-airflow
source .venv-airflow/bin/activate
python -m pip install --upgrade pip
AIRFLOW_VERSION=2.10.5
PYTHON_VERSION=3.11
pip install -r requirements-airflow.txt \
  --constraint "https://raw.githubusercontent.com/apache/airflow/constraints-${AIRFLOW_VERSION}/constraints-${PYTHON_VERSION}.txt"
airflow version        # prints 2.10.5
```
(Takes 3–10 minutes. The `--constraint` URL must be reachable; it is on GitHub.)

### 11.2 Tell Airflow where everything is
These variables must be set **in every terminal that runs an `airflow` command** (the quickest way: paste the block each time, or save it as `airflow_env.sh` and run `source airflow_env.sh`):
```bash
cd ~/recommendation-pipeline
export RECSYS_HOME="$(pwd)"
export RECSYS_PYTHON="$(pwd)/.venv/bin/python"          # PROJECT venv python (not the airflow one)
export AIRFLOW_HOME="$(pwd)/.airflow"
export AIRFLOW__CORE__DAGS_FOLDER="$(pwd)/dags"
export AIRFLOW__CORE__LOAD_EXAMPLES=False
# Only for MovieLens instead of synthetic data:
# export RECSYS_CONFIG="$(pwd)/configs/config.movielens.yaml"
```
Create the Airflow database and check the DAG loads:
```bash
airflow db migrate
airflow dags list-import-errors        # must print "No data found" / nothing
airflow dags list | grep recommendation_pipeline
```

### 11.3 Quick proof in the terminal (no web UI)
```bash
airflow dags test recommendation_pipeline 2026-01-01
```
It runs all tasks once in order and ends with `DagRun Finished: ... state=success`. Each task's log is printed. If a task fails, the log shows the Python error.

### 11.4 Run it in the Airflow web UI and take the screenshot
1. In this terminal start Airflow and **leave it running**:
   ```bash
   airflow standalone
   ```
   It prints, near the end, `Login with username: admin  password: <something>`. Copy the password
   (also stored in `.airflow/standalone_admin_password.txt`).
2. Open **http://localhost:8080** in your Windows browser, log in as `admin` with that password.
3. In the DAG list find **recommendation_pipeline**. Turn on the toggle (un-pause) at its left.
4. Click the ▶ **Trigger DAG** button (right side) → confirm.
5. Click the DAG name → **Graph** tab. Watch the boxes turn green one after another (about 2–5 minutes; with the default SQLite executor tasks run one at a time — that's fine).
6. When **all boxes are dark green** and the run state says **success**, take a screenshot of the Graph view with **Win + Shift + S** (Snipping Tool) → it goes to the clipboard/Pictures\Screenshots. Save it as `airflow_successful_run.png`.
7. Copy it into the project (in a *second* Ubuntu terminal):
   ```bash
   cp "/mnt/c/Users/WINUSER/Pictures/Screenshots/airflow_successful_run.png" ~/recommendation-pipeline/reports/movielens1m/figures/airflow_successful_run.png
   ```
   (If you ran the synthetic report too, also copy it to `reports/figures/` so that report embeds it. Win+Shift+S only copies the image to the clipboard: click the pop-up and Save, or paste into Paint and save as PNG.)
   (Adjust the path to wherever Windows saved it; run `ls /mnt/c/Users/WINUSER/Pictures/Screenshots/` to see.)
8. Stop Airflow in the first terminal with **Ctrl+C**.

If a task box turns **red**, click it → **Logs**. The most common causes: `RECSYS_PYTHON` points to the wrong venv, or the variables from 11.2 weren't set in the terminal that started `airflow standalone`.

---

## 12. Regenerate the final report PDF and diagrams

Use the project venv:
```bash
cd ~/recommendation-pipeline
source .venv/bin/activate
python scripts/make_architecture.py            # needs Graphviz: if "dot not found" -> sudo apt install -y graphviz
python scripts/make_run_screenshot.py          # (needs step 6) built-in runner screenshot -> reports/figures/successful_run.png
python scripts/make_final_report.py            # -> reports/final_report.pdf (2-3 pages), built from YOUR run's results
```
For the **MovieLens** report instead:
```bash
python scripts/make_run_screenshot.py reports/movielens
python scripts/make_final_report.py reports/movielens data/movielens/processed
```
The report detects automatically which tools were used and whether `airflow_successful_run.png` exists in its own `figures/` folder, and words itself accordingly. **Do not re-run `scripts/run_all.sh` after saving screenshots - it wipes `reports/`.** Rebuild only with `bash scripts/build_reports.sh`.
Install Graphviz first if needed: `sudo apt install -y graphviz`.

---

## 13. Collect your deliverables

| Deliverable | Location (inside `~/recommendation-pipeline`) |
|---|---|
| Source code in modules | `src/`, `run_pipeline.py`, `scripts/`, `dags/` |
| README with setup & execution | `README.md`, `SETUP_GUIDE.md` |
| requirements.txt | `requirements.txt` (+ `requirements-airflow.txt`) |
| Architecture diagram | `reports/architecture.png` (`.svg`) |
| Data validation report/logs | `reports/validation_report.md`, `data/errors/` (rejected rows), `logs/` |
| Feature catalog (table) | `reports/feature_catalog.md` |
| Model evaluation, both/all models | `reports/model_evaluation.md` (+ `figures/model_comparison.png`, MLflow UI) |
| Airflow DAG + screenshot of successful run | `dags/recommendation_dag.py`, `reports/movielens1m/figures/airflow_successful_run.png` |
| Final report (2–3 pages) | `reports/movielens1m/final_report.pdf` (MovieLens 1M, main) and `reports/final_report.pdf` (synthetic) |

For the MovieLens 1M run replace `reports/` by `reports/movielens1m/` in the paths above (validation report, feature catalog, model evaluation), and `data/errors/` by `data/movielens1m/errors/`. MLflow screenshots (optional): save them in `reports/movielens1m/figures/`.

Copy everything to Windows (e.g. as one zip on your Desktop):
```bash
cd ~
zip -qr recommendation-pipeline-final.zip recommendation-pipeline \
  -x "recommendation-pipeline/.git/*" "recommendation-pipeline/.venv/*" "recommendation-pipeline/.venv-airflow/*" "recommendation-pipeline/.airflow/*" "*/__pycache__/*"
cp recommendation-pipeline-final.zip /mnt/c/Users/WINUSER/Desktop/
```
(The excludes keep out `.git` and the virtual environments, which are huge. The zip then contains data and models (roughly 100-250 MB); for about 50 MB also exclude `recommendation-pipeline/models/*`, `recommendation-pipeline/data/*/raw/*`, `recommendation-pipeline/data/*/_download/*` and `recommendation-pipeline/mlruns/*/*/artifacts/*`. If Windows says "Permission denied" when copying, an older file with that name is open in a viewer - close it or use a new file name.)

---

## 14. Troubleshooting

| Problem | Fix |
|---|---|
| `python3.11: command not found` | Redo step 3 (deadsnakes PPA). |
| `The virtual environment was not created successfully ... ensurepip` | `sudo apt install -y python3.11-venv`, delete `.venv` (`rm -rf .venv`) and recreate. |
| `(.venv)` missing from prompt / `ModuleNotFoundError: pandas` | Environment not active: `cd ~/recommendation-pipeline && source .venv/bin/activate`. |
| scikit-surprise build error (`numpy`, `Cython`, `gcc`) | Make sure `build-essential` is installed (step 3) and use exactly the two-command order in step 5 (`numpy<2`, `cython<3`, `--no-build-isolation`). Last resort: in `configs/config.yaml` set `surprise_svd: enabled: false` (the pipeline and DAG then skip that model). |
| `ResolutionImpossible` during `pip install -r requirements.txt` | Keep the full message; usually one version range in `requirements.txt` needs adjusting (try a fresh Python 3.11 venv first). |
| `smoke_check` FAIL for Great Expectations or Feast | `VERBOSE=1 python scripts/smoke_check.py` and send the traceback (library APIs differ slightly between versions). |
| Pipeline: `validate` FAILED with `DataQualityError` | The quality gate stopped the run: too many rows rejected (`max_error_rate`). Check `reports/validation_report.md`. On MovieLens make sure you used `--config configs/config.movielens.yaml`. |
| Pipeline killed / `MemoryError` | Give WSL more RAM (`.wslconfig`, step 2), or use `ml-latest-small` (not `ml-1m`). |
| `http://localhost:5000` / `:8080` doesn't open | Make sure the server is still running in its terminal. Try `wsl --shutdown` in PowerShell, reopen Ubuntu, start again. Or use the WSL IP: `hostname -I` and browse to `http://<that-ip>:5000`. |
| `airflow: command not found` | The Airflow venv isn't active: `source ~/recommendation-pipeline/.venv-airflow/bin/activate`. |
| `airflow dags list-import-errors` shows an error | Variables from 11.2 not exported in this terminal, or the Airflow venv was not used. |
| Airflow task red, log says `No module named pandas` (or similar) | `RECSYS_PYTHON` points to the Airflow python. It must be `…/recommendation-pipeline/.venv/bin/python`. Re-export and re-trigger (Clear task). |
| Airflow: DAG doesn't appear | Wait ~30 s (DAG folder is rescanned) and check `AIRFLOW__CORE__DAGS_FOLDER` was exported before `airflow standalone`. |
| `zip` or `pdftotext`: command not found | `sudo apt install -y zip poppler-utils` |
| `run_airflow.sh` says the config file is not found | Update the script (older versions mis-handled a full `RECSYS_CONFIG` path), or `unset RECSYS_CONFIG` and re-run. |
| Screenshots or reports vanished after a re-run | `scripts/run_all.sh` and `clean_outputs.sh` delete `reports/`. Re-save the screenshots, then run only `bash scripts/build_reports.sh`. |
| MLflow lists the same experiment several times | Stop the UI, `rm -rf mlruns`, re-run the pipeline (the tracker now creates the experiment once, with a lock). |
| Files downloaded from the chat are not in the project | Chat downloads land in the Windows Downloads folder, not in Ubuntu: copy them with `cp /mnt/c/Users/WINUSER/Downloads/<file> ~/recommendation-pipeline/<folder>/`. |
| Everything is very slow | The project is under `/mnt/c/...`. Move it to `~` (step 4). |
| `Permission denied` on scripts | Don't use `sudo` with pip/python; if you must, `chmod +x` the file, or run with `python file.py`. |
| Files show strange `^M` errors | Windows line endings from editing in Notepad; run `sudo apt install -y dos2unix && dos2unix <file>`. |
| Want to start the data over | `rm -rf data/processed data/features data/errors models mlruns feature_repo/data` then `python run_pipeline.py` (raw synthetic CSVs are kept; add `--regenerate-data` to recreate). |

When asking me for help always include: **the command you ran, the full output, and the output of** `python --version`, and for pip problems `pip list | grep -i -E "numpy|pandas|feast|great|mlflow|surprise"`.

---

## 15. macOS / native Linux

* **macOS:** `xcode-select --install`; install Homebrew (<https://brew.sh>), then `brew install python@3.11 graphviz`. Use `python3.11 -m venv .venv` exactly as in step 5; Airflow steps 11.x are identical (use `~/` paths and a Terminal window instead of Ubuntu). Browser screenshots: Cmd+Shift+4.
* **Ubuntu/Debian Linux:** steps 3–16 as written (skip step 2, `/mnt/c` copying, `explorer.exe`).
* **Windows without WSL:** everything except Airflow can run in PowerShell, but scikit-surprise needs Microsoft C++ Build Tools and paths/activation differ (`.venv\Scripts\Activate.ps1`). Not recommended — use WSL2 for one consistent environment.

---

## 16. Command cheat-sheet

```bash
# --- every new Ubuntu terminal: go to project + activate the project environment
cd ~/recommendation-pipeline && source .venv/bin/activate

# --- pipeline
python scripts/smoke_check.py
python run_pipeline.py                                              # synthetic
python scripts/prepare_movielens.py --dataset ml-latest-small       # one-time download+convert
python run_pipeline.py --config configs/config.movielens.yaml       # MovieLens small
python scripts/prepare_movielens.py --dataset ml-1m --out data/movielens1m/raw
python run_pipeline.py --config configs/config.movielens1m.yaml     # MovieLens 1M (main results)
mlflow ui --backend-store-uri ./mlruns --host 127.0.0.1 --port 5000 # http://localhost:5000

# --- Airflow (separate environment; set the 11.2 exports first)
source .venv-airflow/bin/activate
airflow db migrate && airflow dags test recommendation_pipeline 2026-01-01
airflow standalone                                                   # http://localhost:8080, user admin

# --- report
source .venv/bin/activate
python scripts/make_architecture.py && bash scripts/build_reports.sh

# --- shortcuts (see QUICKSTART.md)
bash scripts/run_all.sh        # everything: smoke check, clean, synthetic, ML-1M, diagram, screenshots, reports
bash scripts/run_airflow.sh    # DAG once under Airflow (uses .venv-airflow by itself)
bash scripts/clean_outputs.sh  # wipe generated outputs (keeps code and .venv folders)
```
