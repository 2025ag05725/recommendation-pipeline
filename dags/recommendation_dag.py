"""Apache Airflow DAG for the recommendation pipeline.

The DAG is generated from `src.orchestration.tasks.TASKS` (the same task list the built-in runner uses).

Environment isolation: Airflow and the ML stack (Feast, Great Expectations, Surprise ...) have conflicting
dependency pins, so each Airflow task shells out to the PROJECT virtualenv:

    export RECSYS_HOME=/path/to/recommendation-pipeline
    export RECSYS_PYTHON=/path/to/recommendation-pipeline/.venv/bin/python     # python of the project venv
    export RECSYS_CONFIG=/path/to/recommendation-pipeline/configs/config.movielens.yaml   # optional; default config.yaml

The Airflow environment itself only needs `apache-airflow` (this file imports nothing heavy).
Works with Airflow 2.x and 3.x.
"""
from __future__ import annotations

import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

PROJECT_ROOT = Path(os.environ.get("RECSYS_HOME", Path(__file__).resolve().parents[1]))
PROJECT_PYTHON = os.environ.get("RECSYS_PYTHON", "python")
CONFIG_ARG = f' --config "{os.environ["RECSYS_CONFIG"]}"' if os.environ.get("RECSYS_CONFIG") else ""   # e.g. configs/config.movielens.yaml
sys.path.insert(0, str(PROJECT_ROOT))

from airflow import DAG  # noqa: E402

try:  # Airflow 3
    from airflow.providers.standard.operators.bash import BashOperator  # noqa: E402
except ImportError:  # Airflow 2
    from airflow.operators.bash import BashOperator  # noqa: E402

from src.orchestration.tasks import TASKS  # noqa: E402  (stdlib-only module)

with DAG(
    dag_id="recommendation_pipeline",
    description="ingest -> validate -> prepare -> transform -> features -> train (x4) -> evaluate",
    start_date=datetime(2026, 1, 1),
    schedule="@daily",
    catchup=False,
    max_active_runs=1,
    default_args={"owner": "data-team", "retries": 1, "retry_delay": timedelta(seconds=30)},
    tags=["recsys", "dmml"],
    doc_md=__doc__,
) as dag:
    operators = {
        t.name: BashOperator(
            task_id=t.name,
            bash_command=f'cd "{PROJECT_ROOT}" && "{PROJECT_PYTHON}" -m src.orchestration.cli run-task {t.name}{CONFIG_ARG}',
            doc_md=t.description,
        )
        for t in TASKS
    }
    for t in TASKS:
        for dep in t.deps:
            operators[dep] >> operators[t.name]
