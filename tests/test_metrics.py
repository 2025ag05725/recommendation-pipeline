import math

import numpy as np
import pytest

from src.evaluation import metrics as M


def test_precision_recall():
    recs, rel = [1, 2, 3, 4, 5], {2, 5, 9}
    assert M.precision_at_k(recs, rel, 5) == pytest.approx(2 / 5)
    assert M.recall_at_k(recs, rel, 5) == pytest.approx(2 / 3)
    assert M.recall_at_k(recs, set(), 5) == 0.0


def test_ndcg_perfect_and_known_value():
    assert M.ndcg_at_k([1, 2, 3], {1, 2, 3}, 3) == pytest.approx(1.0)
    # one hit at rank 2 (0-based 1), one relevant item: DCG=1/log2(3), IDCG=1
    assert M.ndcg_at_k([7, 1, 8], {1}, 3) == pytest.approx(1 / math.log2(3))
    assert M.ndcg_at_k([7, 8, 9], {1}, 3) == 0.0


def test_coverage_and_diversity():
    assert M.catalog_coverage([[1, 2], [2, 3]], 10) == pytest.approx(0.3)
    same = {1: np.array([1, 0]), 2: np.array([1, 0])}
    diff = {1: np.array([1, 0]), 2: np.array([0, 1])}
    assert M.intra_list_diversity([1, 2], same) == pytest.approx(0.0)
    assert M.intra_list_diversity([1, 2], diff) == pytest.approx(1.0)


def test_rmse_mae():
    assert M.rmse([1, 3], [2, 5]) == pytest.approx(math.sqrt((1 + 4) / 2))
    assert M.mae([1, 3], [2, 5]) == pytest.approx(1.5)
