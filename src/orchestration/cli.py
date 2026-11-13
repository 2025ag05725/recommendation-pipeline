"""Run a single pipeline task from the command line (this is what Airflow's BashOperator calls).

    python -m src.orchestration.cli run-task validate
    python -m src.orchestration.cli list
"""
import argparse
import json
import sys

from src.common.config import load_config
from src.common.logging_utils import setup_logging
from src.orchestration.tasks import BY_NAME, TASKS


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    rt = sub.add_parser("run-task")
    rt.add_argument("name", choices=list(BY_NAME))
    rt.add_argument("--config", default=None)
    sub.add_parser("list")
    a = ap.parse_args()
    if a.cmd == "list":
        for t in TASKS:
            print(f"{t.name:20s} <- {', '.join(t.deps) or '-'}")
        return 0
    setup_logging()
    cfg = load_config(a.config)
    spec = BY_NAME[a.name]
    model = spec.kwargs.get("name")
    if model and not cfg["models"].get(model, {}).get("enabled", True):
        print(f"task {a.name}: model '{model}' disabled in config - skipping")
        return 0
    result = spec.resolve()(cfg, **spec.kwargs)
    print("RESULT " + json.dumps(result, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
