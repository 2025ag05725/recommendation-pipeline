"""Experiment tracking with a single interface and two backends.

* MLflow  (used when `mlflow` is importable and tracking.backend is auto|mlflow)
* Local   (zero-dependency JSON file tracker, used otherwise)

The API is run-id based (not "active run" based) so that a run opened by a training task can be resumed
by the evaluation task, even from another thread/process.

Local layout:  mlruns/<experiment>/<run_id>/{meta.json, params.json, metrics.json, artifacts/...}
"""
from __future__ import annotations

import json
import shutil
import threading
import uuid
from datetime import datetime
from pathlib import Path

from src.common.logging_utils import get_logger

log = get_logger("tracking")
_lock = threading.Lock()


class Tracker:
    def __init__(self, cfg):
        t = cfg["tracking"]
        self.experiment = t["experiment"]
        self.root = cfg.root
        self.backend = "local"
        if t["backend"] in ("auto", "mlflow"):
            try:
                import mlflow  # noqa: F401
                from mlflow.tracking import MlflowClient
                uri = t.get("mlflow_uri") or (cfg.root / 'mlruns').as_uri()
                self._client = MlflowClient(tracking_uri=uri)
                self._exp_id = self._get_or_create_experiment(cfg.root / "mlruns")
                self.backend = "mlflow"
            except ImportError:
                if t["backend"] == "mlflow":
                    raise
        if self.backend == "local":
            self._dir = cfg.root / t["local_dir"] / self.experiment
            self._dir.mkdir(parents=True, exist_ok=True)
        log.info("Experiment tracker backend: %s (experiment=%s)", self.backend, self.experiment)

    def _get_or_create_experiment(self, store_dir: Path) -> str:
        """Race-safe get-or-create. Parallel tasks (threads in the built-in runner, processes in Airflow) would each
        create their own experiment of the same name; a thread lock + a cross-process file lock prevents that, and
        if duplicates already exist the oldest one is always used."""
        import fcntl
        store_dir.mkdir(parents=True, exist_ok=True)
        with _lock, open(store_dir / ".experiment.lock", "w") as fh:
            fcntl.flock(fh, fcntl.LOCK_EX)
            try:
                found = [e for e in self._client.search_experiments(filter_string=f"name = '{self.experiment}'")]
                if not found:
                    return self._client.create_experiment(self.experiment)
                return min(found, key=lambda e: (e.creation_time or 0, e.experiment_id)).experiment_id
            finally:
                fcntl.flock(fh, fcntl.LOCK_UN)

    # ---- helpers for the local backend
    def _rd(self, run_id: str) -> Path:
        return self._dir / run_id

    def _read(self, run_id: str, name: str) -> dict:
        p = self._rd(run_id) / name
        return json.loads(p.read_text()) if p.exists() else {}

    def _write(self, run_id: str, name: str, obj: dict) -> None:
        (self._rd(run_id) / name).write_text(json.dumps(obj, indent=2, default=str))

    # ---- API
    def start_run(self, run_name: str, tags: dict | None = None) -> str:
        if self.backend == "mlflow":
            run = self._client.create_run(self._exp_id, tags={"mlflow.runName": run_name, **(tags or {})})
            return run.info.run_id
        run_id = uuid.uuid4().hex[:16]
        (self._rd(run_id) / "artifacts").mkdir(parents=True)
        self._write(run_id, "meta.json", {"run_id": run_id, "run_name": run_name, "tags": tags or {}, "status": "RUNNING",
                                          "start_time": datetime.now().isoformat(timespec="seconds"), "end_time": None})
        return run_id

    def log_params(self, run_id: str, params: dict) -> None:
        if self.backend == "mlflow":
            for k, v in params.items():
                self._client.log_param(run_id, k, str(v)[:500])
            return
        with _lock:
            cur = self._read(run_id, "params.json"); cur.update({k: v for k, v in params.items()}); self._write(run_id, "params.json", cur)

    def log_metrics(self, run_id: str, metrics: dict) -> None:
        # MLflow metric names allow only [A-Za-z0-9_-. :/] - '@' is rejected, so 'ndcg@10' is stored as 'ndcg_at_10' (all backends, for consistency)
        clean = {k.replace("@", "_at_"): float(v) for k, v in metrics.items() if v is not None}
        if self.backend == "mlflow":
            for k, v in clean.items():
                self._client.log_metric(run_id, k, v)
            return
        with _lock:
            cur = self._read(run_id, "metrics.json"); cur.update(clean); self._write(run_id, "metrics.json", cur)

    def log_artifact(self, run_id: str, path: Path) -> None:
        if self.backend == "mlflow":
            self._client.log_artifact(run_id, str(path))
            return
        shutil.copy(path, self._rd(run_id) / "artifacts" / Path(path).name)

    def end_run(self, run_id: str, status: str = "FINISHED") -> None:
        if self.backend == "mlflow":
            self._client.set_terminated(run_id, status)
            return
        with _lock:
            meta = self._read(run_id, "meta.json"); meta.update(status=status, end_time=datetime.now().isoformat(timespec="seconds"))
            self._write(run_id, "meta.json", meta)

    def list_runs(self) -> list[dict]:
        """Flat list of runs (id, name, status, params, metrics) for the experiment-comparison report."""
        out = []
        if self.backend == "mlflow":
            for r in self._client.search_runs([self._exp_id]):
                out.append({"run_id": r.info.run_id, "run_name": r.data.tags.get("mlflow.runName"), "status": r.info.status,
                            "params": r.data.params, "metrics": r.data.metrics})
            return out
        for d in sorted(self._dir.iterdir()):
            if (d / "meta.json").exists():
                m = self._read(d.name, "meta.json")
                out.append({"run_id": d.name, "run_name": m["run_name"], "status": m["status"], "start_time": m["start_time"],
                            "params": self._read(d.name, "params.json"), "metrics": self._read(d.name, "metrics.json")})
        return out
