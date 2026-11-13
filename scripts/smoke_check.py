"""Verify that every real tool works in YOUR environment before running the pipeline.

    python scripts/smoke_check.py

Each component runs a tiny end-to-end operation and prints OK / FAIL with the error, so version problems
show up in seconds instead of mid-pipeline. Exit code 1 if anything fails.
"""
import os
import sys
import tempfile
import traceback
from pathlib import Path

os.environ.setdefault("GX_ANALYTICS_ENABLED", "false")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

results: list[tuple[str, bool, str]] = []


def check(name):
    def deco(fn):
        try:
            detail = fn() or ""
            results.append((name, True, detail))
        except Exception as exc:  # noqa: BLE001
            results.append((name, False, f"{type(exc).__name__}: {exc}"))
            if os.environ.get("VERBOSE"):
                traceback.print_exc()
        return fn
    return deco


@check("parquet (pyarrow)")
def _():
    import pyarrow
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "t.parquet"
        pd.DataFrame({"a": [1, 2], "t": pd.to_datetime(["2024-01-01", "2024-01-02"])}).to_parquet(p, engine="pyarrow")
        assert len(pd.read_parquet(p)) == 2
    return f"pyarrow {pyarrow.__version__}"


@check("Great Expectations")
def _():
    import great_expectations as gx
    ctx = gx.get_context(mode="ephemeral")
    src = ctx.data_sources.add_pandas("smoke")
    df = pd.DataFrame({"x": [1, 2, None, 2], "y": [1, 2, 3, 2]}, index=[10, 11, 12, 13])
    batch = src.add_dataframe_asset("t").add_batch_definition_whole_dataframe("w").get_batch(batch_parameters={"dataframe": df})
    res = batch.validate(gx.expectations.ExpectColumnValuesToNotBeNull(column="x"), result_format="COMPLETE")
    assert not res.success and list(res.result["unexpected_index_list"]) == [12], res.result
    res2 = batch.validate(gx.expectations.ExpectCompoundColumnsToBeUnique(column_list=["x", "y"]), result_format="COMPLETE")
    assert "unexpected_index_list" in res2.result
    return f"great_expectations {gx.__version__}"


@check("Feast")
def _():
    import feast
    from src.common.config import Config, load_config
    from src.features.feast_layer import FeastFeatureLayer
    from src.features.feature_store import FeatureStore
    users = pd.DataFrame({"user_id": [1, 2], "age": [20, 30], "gender": "M", "country": "IN", "signup_date": pd.Timestamp("2024-01-01")})
    items = pd.DataFrame({"item_id": [1, 2], "title": "t", "genres": ["Drama", "Comedy"], "release_year": 2000})
    inter = pd.DataFrame({"user_id": [1, 2], "item_id": [1, 2], "rating": [5.0, 3.0], "timestamp": pd.to_datetime(["2024-02-01", "2024-02-02"])})
    with tempfile.TemporaryDirectory() as d:
        cfg = Config(load_config())
        cfg.root = Path(d)
        fs = FeatureStore().fit(inter, users, items, pd.Timestamp("2024-03-01"))
        fs.save(Path(d) / "data" / "features", "parquet")
        layer = FeastFeatureLayer(cfg)
        layer.publish(fs)
        layer.check_online_offline_parity(fs, n=2)
        layer.load_feature_store()
    return f"feast {feast.__version__}"


@check("scikit-surprise")
def _():
    import surprise
    from surprise import SVD, Dataset, Reader
    df = pd.DataFrame({"u": [1, 1, 2, 2, 3], "i": [1, 2, 1, 3, 2], "r": [5, 3, 4, 2, 1]})
    ts = Dataset.load_from_df(df, Reader(rating_scale=(1, 5))).build_full_trainset()
    algo = SVD(n_factors=2, n_epochs=3, random_state=0)
    algo.fit(ts)
    assert algo.pu.shape[1] == 2 and algo.qi.shape[1] == 2 and len(algo.bu) == ts.n_users
    return f"scikit-surprise {surprise.__version__}"


@check("MLflow")
def _():
    import mlflow
    from mlflow.tracking import MlflowClient
    with tempfile.TemporaryDirectory() as d:
        c = MlflowClient(tracking_uri=Path(d).as_uri())
        eid = c.create_experiment("smoke")
        run = c.create_run(eid, tags={"mlflow.runName": "r"})
        c.log_param(run.info.run_id, "k", "v")
        c.log_metric(run.info.run_id, "m", 1.0)
        f = Path(d) / "a.txt"
        f.write_text("x")
        c.log_artifact(run.info.run_id, str(f))
        c.set_terminated(run.info.run_id, "FINISHED")
        assert c.search_runs([eid])[0].data.metrics["m"] == 1.0
    return f"mlflow {mlflow.__version__}"


@check("Airflow (optional here - lives in its own env)")
def _():
    import airflow
    return f"apache-airflow {airflow.__version__}"


@check("project unit tests (pytest)")
def _():
    import subprocess
    r = subprocess.run([sys.executable, "-m", "pytest", "-q", "-x", str(Path(__file__).resolve().parents[1] / "tests")], capture_output=True, text=True)
    assert r.returncode == 0, r.stdout[-600:] + r.stderr[-300:]
    return r.stdout.strip().splitlines()[-1]


w = max(len(n) for n, _, _ in results)
for n, ok, d in results:
    print(f"{'OK  ' if ok else 'FAIL'}  {n:<{w}}  {d}")
sys.exit(0 if all(ok for _, ok, _ in results[:5] + results[6:]) else 1)
