"""Evaluation metrics (binary relevance, top-K lists). Pure functions, unit-tested in tests/test_metrics.py."""
from __future__ import annotations

import numpy as np


def precision_at_k(recs: list, relevant: set, k: int) -> float:
    return len(set(recs[:k]) & relevant) / k


def recall_at_k(recs: list, relevant: set, k: int) -> float:
    return len(set(recs[:k]) & relevant) / len(relevant) if relevant else 0.0


def ndcg_at_k(recs: list, relevant: set, k: int) -> float:
    gains = np.array([1.0 if r in relevant else 0.0 for r in recs[:k]])
    disc = 1.0 / np.log2(np.arange(2, len(gains) + 2))
    ideal = disc[: min(len(relevant), k)].sum()
    return float((gains * disc).sum() / ideal) if ideal > 0 else 0.0


def catalog_coverage(all_recs: list[list], n_items: int) -> float:
    """Share of the catalogue that appears in at least one user's top-K list."""
    return len({i for r in all_recs for i in r}) / n_items


def intra_list_diversity(recs: list, genre_vectors: dict) -> float:
    """1 - mean pairwise cosine similarity of the items' genre vectors (higher = more diverse)."""
    vecs = np.array([genre_vectors[i] for i in recs], dtype=float)
    norm = np.linalg.norm(vecs, axis=1, keepdims=True)
    vecs = vecs / np.where(norm == 0, 1.0, norm)
    sim = vecs @ vecs.T
    n = len(recs)
    return float(1.0 - (sim.sum() - np.trace(sim)) / (n * (n - 1))) if n > 1 else 0.0


def rmse(y, yhat) -> float:
    return float(np.sqrt(np.mean((np.asarray(y) - np.asarray(yhat)) ** 2)))


def mae(y, yhat) -> float:
    return float(np.mean(np.abs(np.asarray(y) - np.asarray(yhat))))
