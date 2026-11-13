"""Single source of truth for the pipeline DAG.

Both the built-in runner (`run_pipeline.py`) and the Airflow DAG (`dags/recommendation_dag.py`) are generated
from `TASKS`, so the two orchestration back-ends can never disagree about order or dependencies.

    ingest -> validate -> prepare -> transform -> build_features -+-> train_popularity -+
                                                                  +-> train_item_knn   -+-> evaluate
                                                                  +-> train_hybrid_svd -+
                                                                  +-> train_surprise_svd-+
"""
from __future__ import annotations

import importlib
from dataclasses import dataclass, field


@dataclass(frozen=True)
class TaskSpec:
    name: str
    target: str                       # "module:function"; function signature fn(cfg, **kwargs) -> dict
    deps: tuple[str, ...] = ()
    kwargs: dict = field(default_factory=dict)
    description: str = ""
    stage: str = ""

    def resolve(self):
        mod, fn = self.target.split(":")
        return getattr(importlib.import_module(mod), fn)


TASKS: list[TaskSpec] = [
    TaskSpec("ingest", "src.ingestion.ingest:run", (), {}, "Load/generate raw CSVs, write manifest", "Data Ingestion"),
    TaskSpec("validate", "src.validation.validate:run", ("ingest",), {}, "Quality checks, quarantine bad rows, quality gate", "Data Validation"),
    TaskSpec("prepare", "src.preparation.prepare:run", ("validate",), {}, "Clean, impute, standardise, de-duplicate", "Data Preparation"),
    TaskSpec("transform", "src.transformation.transform:run", ("prepare",), {}, "Time-based split, activity/popularity tables, matrix", "Data Transformation"),
    TaskSpec("build_features", "src.features.build_features:run", ("transform",), {}, "Materialise feature store + feature catalog", "Feature Management"),
    TaskSpec("train_popularity", "src.training.train:train_model", ("build_features",), {"name": "popularity"}, "Baseline popularity recommender", "Model Training"),
    TaskSpec("train_item_knn", "src.training.train:train_model", ("build_features",), {"name": "item_knn"}, "Item-based collaborative filtering", "Model Training"),
    TaskSpec("train_hybrid_svd", "src.training.train:train_model", ("build_features",), {"name": "hybrid_svd"}, "Hybrid SVD (CF + content) recommender", "Model Training"),
    TaskSpec("train_surprise_svd", "src.training.train:train_model", ("build_features",), {"name": "surprise_svd"}, "scikit-surprise SVD rating model", "Model Training"),
    TaskSpec("evaluate", "src.evaluation.evaluate:run", ("train_popularity", "train_item_knn", "train_hybrid_svd", "train_surprise_svd"), {}, "Test-set metrics + comparison table", "Model Evaluation"),
]
BY_NAME = {t.name: t for t in TASKS}
