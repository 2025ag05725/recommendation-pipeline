#!/usr/bin/env python3
"""Run the whole recommendation pipeline (or part of it).

  python run_pipeline.py                       # full run
  python run_pipeline.py --start-from prepare  # re-run prepare and everything downstream
  python run_pipeline.py --only evaluate       # a single task (inputs must already exist)
  python run_pipeline.py --list                # print the DAG
"""
import argparse
import sys

from src.common.config import load_config
from src.orchestration.runner import PipelineRunner, topological_levels
from src.orchestration.tasks import TASKS


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--start-from", choices=[t.name for t in TASKS])
    ap.add_argument("--only", choices=[t.name for t in TASKS])
    ap.add_argument("--regenerate-data", action="store_true", help="regenerate synthetic raw data")
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()
    if args.list:
        for i, lvl in enumerate(topological_levels(TASKS)):
            print(f"level {i}: {', '.join(lvl)}")
        return 0
    cfg = load_config(args.config, {"ingestion.regenerate": True} if args.regenerate_data else None)
    summary = PipelineRunner(cfg).run(start_from=args.start_from, only=args.only)
    return 0 if summary["success"] else 1


if __name__ == "__main__":
    sys.exit(main())
