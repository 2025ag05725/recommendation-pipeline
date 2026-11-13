"""Runner behaviour on a toy DAG: ordering, retries, failure propagation (no pipeline data needed)."""
import sys
import types

from src.common.config import Config, load_config
from src.orchestration import runner as R
from src.orchestration.tasks import TaskSpec

CALLS: list[str] = []
FLAKY = {"n": 0}


def ok(cfg, name):
    CALLS.append(name)
    return {"name": name}


def flaky(cfg):
    FLAKY["n"] += 1
    if FLAKY["n"] == 1:
        raise RuntimeError("transient")
    CALLS.append("flaky")
    return {}


def boom(cfg):
    raise ValueError("always fails")


def _cfg(tmp_path):
    cfg = Config(load_config())
    cfg.root = tmp_path
    cfg["orchestration"] = {"max_workers": 2, "retries": 1, "retry_delay_seconds": 0}
    return cfg


def _install(monkeypatch, specs):
    monkeypatch.setattr(R, "TASKS", specs)
    monkeypatch.setattr(R, "BY_NAME", {t.name: t for t in specs})
    monkeypatch.setattr(R, "TASKS", specs)
    monkeypatch.setattr("src.orchestration.report.shutil.which", lambda _: None)


def test_order_retry_and_failure_propagation(tmp_path, monkeypatch):
    mod = types.ModuleType("toy")
    mod.ok, mod.flaky, mod.boom = ok, flaky, boom
    monkeypatch.setitem(sys.modules, "toy", mod)
    specs = [TaskSpec("a", "toy:ok", (), {"name": "a"}), TaskSpec("b", "toy:flaky", ("a",)), TaskSpec("c", "toy:boom", ("a",)),
             TaskSpec("d", "toy:ok", ("b", "c"), {"name": "d"}), TaskSpec("e", "toy:ok", ("b",), {"name": "e"})]
    _install(monkeypatch, specs)
    CALLS.clear(); FLAKY["n"] = 0
    summary = R.PipelineRunner(_cfg(tmp_path), "t1").run()
    st = {k: v["status"] for k, v in summary["tasks"].items()}
    assert st == {"a": "success", "b": "success", "c": "failed", "d": "upstream_failed", "e": "success"}
    assert summary["tasks"]["b"]["attempts"] == 2 and summary["tasks"]["c"]["attempts"] == 2
    assert CALLS.index("a") < CALLS.index("flaky") and "d" not in CALLS
    assert summary["success"] is False
    assert (tmp_path / "reports" / "latest_run_summary.json").exists() and (tmp_path / "reports" / "run_report.html").exists()
