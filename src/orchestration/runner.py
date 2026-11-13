"""Minimal, dependency-free DAG runner (the "clearly structured Python workflow" orchestrator).

Features: topological ordering with cycle/unknown-dependency detection, parallel execution of independent tasks,
per-task retries, failure propagation (downstream tasks become `upstream_failed`), partial runs
(`start_from`, `only`), per-task log files, machine-readable run summary + HTML run report.
"""
from __future__ import annotations

import logging
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

from src.common.io_utils import write_json
from src.common.logging_utils import get_logger, setup_logging
from src.orchestration.tasks import BY_NAME, TASKS, TaskSpec

log = get_logger("orchestration")


def topological_levels(specs: list[TaskSpec]) -> list[list[str]]:
    """Kahn's algorithm grouped into levels; tasks within a level are independent of each other."""
    names = {t.name for t in specs}
    for t in specs:
        unknown = set(t.deps) - names
        if unknown:
            raise ValueError(f"Task {t.name!r} depends on unknown task(s) {sorted(unknown)}")
    remaining, done, levels = {t.name: set(t.deps) for t in specs}, set(), []
    while remaining:
        ready = sorted(n for n, d in remaining.items() if d <= done)
        if not ready:
            raise ValueError(f"Cycle detected among tasks: {sorted(remaining)}")
        levels.append(ready)
        done |= set(ready)
        for n in ready:
            del remaining[n]
    return levels


def downstream_of(name: str, specs: list[TaskSpec]) -> set[str]:
    out, frontier = {name}, [name]
    while frontier:
        cur = frontier.pop()
        for t in specs:
            if cur in t.deps and t.name not in out:
                out.add(t.name)
                frontier.append(t.name)
    return out


class PipelineRunner:
    def __init__(self, cfg, run_id: str | None = None):
        self.cfg = cfg
        self.run_id = run_id or datetime.now().strftime("run_%Y%m%d_%H%M%S")
        self.log_dir = cfg.path("logs", self.run_id)
        self.status: dict[str, dict] = {t.name: {"status": "pending"} for t in TASKS}

    def _run_task(self, spec: TaskSpec) -> None:
        ocfg = self.cfg["orchestration"]
        fh = logging.FileHandler(self.log_dir / f"{spec.name}.log")
        fh.setFormatter(logging.Formatter("%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"))
        tname = f"task:{spec.name}"
        fh.addFilter(lambda rec, n=tname: rec.threadName == n)
        logging.getLogger("src").addHandler(fh)
        import threading
        threading.current_thread().name = tname
        st = self.status[spec.name]
        st.update(status="running", start=datetime.now().isoformat(timespec="seconds"), attempts=0)
        t0 = time.time()
        try:
            for attempt in range(1, ocfg["retries"] + 2):
                st["attempts"] = attempt
                try:
                    log.info("START %s (attempt %d/%d)", spec.name, attempt, ocfg["retries"] + 1)
                    st["result"] = spec.resolve()(self.cfg, **spec.kwargs)
                    st["status"] = "success"
                    log.info("SUCCESS %s in %.1fs", spec.name, time.time() - t0)
                    break
                except Exception as exc:  # noqa: BLE001
                    st["error"] = f"{type(exc).__name__}: {exc}"
                    log.error("FAILED %s attempt %d: %s\n%s", spec.name, attempt, exc, traceback.format_exc())
                    if attempt > ocfg["retries"] or type(exc).__name__ == "DataQualityError":  # never retry a data-quality gate
                        st["status"] = "failed"
                        break
                    time.sleep(ocfg["retry_delay_seconds"])
        finally:
            st["seconds"] = round(time.time() - t0, 2)
            st["end"] = datetime.now().isoformat(timespec="seconds")
            logging.getLogger("src").removeHandler(fh)
            fh.close()

    def _engines(self) -> dict:
        c = self.cfg
        return {"storage": c.fmt, "validation": c["validation"].get("engine", "great_expectations"),
                "features": c["features"].get("engine", "feast"), "tracking": c["tracking"]["backend"],
                "data_source": c["ingestion"]["source"], "surprise_model": bool(c["models"].get("surprise_svd", {}).get("enabled", False))}

    def run(self, start_from: str | None = None, only: str | None = None) -> dict:
        setup_logging(self.log_dir / "pipeline.log")
        specs = TASKS
        levels = topological_levels(specs)
        selected = {t.name for t in specs}
        if only:
            selected = {only}
        elif start_from:
            selected = downstream_of(start_from, specs)
        log.info("Run %s | DAG levels: %s | selected: %s", self.run_id, levels, sorted(selected))
        t0 = time.time()
        try:   # create the tracking experiment once, up-front, before parallel training tasks start
            from src.tracking.tracker import Tracker
            Tracker(self.cfg)
        except Exception as exc:  # noqa: BLE001
            log.warning("Could not pre-create tracking experiment: %s", exc)
        for level in levels:
            runnable = []
            for n in level:
                bad = [d for d in BY_NAME[n].deps if self.status[d]["status"] in ("failed", "upstream_failed")]
                mname = BY_NAME[n].kwargs.get("name")
                if n not in selected:
                    self.status[n]["status"] = "skipped"
                elif mname and not self.cfg["models"].get(mname, {}).get("enabled", True):
                    self.status[n].update(status="skipped", error="model disabled in config")
                elif bad:
                    self.status[n].update(status="upstream_failed", error=f"upstream failed: {bad}")
                    log.error("SKIP %s: upstream failed %s", n, bad)
                else:
                    runnable.append(BY_NAME[n])
            if runnable:
                with ThreadPoolExecutor(max_workers=self.cfg["orchestration"]["max_workers"], thread_name_prefix="task") as ex:
                    list(ex.map(self._run_task, runnable))
        ok = all(s["status"] in ("success", "skipped") for s in self.status.values())
        summary = {"run_id": self.run_id, "success": ok, "seconds": round(time.time() - t0, 2), "levels": levels,
                   "started": datetime.fromtimestamp(t0).isoformat(timespec="seconds"), "engines": self._engines(),
                   "tasks": {t.name: {**self.status[t.name], "deps": list(t.deps), "stage": t.stage, "description": t.description} for t in specs}}
        rep = self.cfg.path("reports")
        write_json(summary, rep / "runs" / self.run_id / "run_summary.json")
        write_json(summary, rep / "latest_run_summary.json")
        from src.orchestration.report import render_run_report
        render_run_report(summary, self.log_dir / "pipeline.log", rep)
        log.info("Run %s finished: %s in %.1fs", self.run_id, "SUCCESS" if ok else "FAILED", summary["seconds"])
        return summary
