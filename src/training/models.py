"""Recommender models.

All models share one interface so training, tuning, evaluation and serving are model-agnostic:

    fit(train_df, store, users, items)            learn from TRAIN interactions (+ feature store)
    score(user_ids) -> (n_users, n_items) array   higher = better, columns follow `self.item_ids`
    recommend(user_ids, k)                        top-k item ids, excluding items seen in training
    predict_rating(user_ids, item_ids)            explicit rating (models without one return None)

Models
  PopularityRecommender  baseline: rank by item_popularity_count (feature) - non-personalised
  ItemKNNRecommender     personalised collaborative filtering: item-item cosine similarity on "liked" matrix
  HybridSVDRecommender   personalised hybrid: PureSVD latent CF score + content (genre-affinity feature) blend,
                         plus a biased-SVD explicit-rating head for RMSE / MAE
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse.linalg import svds

from src.features.feature_store import FeatureStore
from src.transformation.matrices import build_index, build_matrix


def _zscore_rows(x: np.ndarray) -> np.ndarray:
    return (x - x.mean(axis=1, keepdims=True)) / (x.std(axis=1, keepdims=True) + 1e-9)


class BaseRecommender:
    name = "base"

    def __init__(self, like_threshold: float = 4.0):
        self.like_threshold = like_threshold

    # -- subclass hooks
    def _fit(self, train: pd.DataFrame) -> None: ...
    def score(self, user_ids) -> np.ndarray: raise NotImplementedError
    def predict_rating(self, user_ids, item_ids):  # noqa: D401
        return None

    # -- shared
    def fit(self, train: pd.DataFrame, store: FeatureStore, users: pd.DataFrame, items: pd.DataFrame):
        self.store = store
        self.uidx, self.iidx = build_index(users, items)
        self.item_ids = np.array(sorted(self.iidx, key=self.iidx.get))
        self.global_mean = float(train["rating"].mean())
        self.R = build_matrix(train, self.uidx, self.iidx, "rating")               # users x items ratings
        self.seen = build_matrix(train, self.uidx, self.iidx, None).tocsr()        # users x items binary
        self.liked = build_matrix(train[train["rating"] >= self.like_threshold], self.uidx, self.iidx, None).tocsr()
        itf = store.get_item_features(self.item_ids)
        self.pop = itf["item_popularity_count"].to_numpy(dtype=float)
        self.bayes = itf["item_bayes_rating"].to_numpy(dtype=float)
        self._fit(train)
        return self

    def _rows(self, user_ids) -> np.ndarray:
        return np.array([self.uidx.get(int(u), -1) for u in user_ids])

    def _scores_with_fallback(self, user_ids) -> np.ndarray:
        """Personalised scores; users with no history (cold start) fall back to popularity."""
        s = self.score(user_ids)
        rows = self._rows(user_ids)
        cold = (rows < 0) | (np.asarray(self.seen[np.where(rows < 0, 0, rows)].sum(axis=1)).ravel() == 0)
        if cold.any():
            s[cold] = self.pop[None, :] + 1e-3 * self.bayes[None, :]
        return s

    def recommend(self, user_ids, k: int = 10) -> np.ndarray:
        user_ids = np.asarray(user_ids)
        s = self._scores_with_fallback(user_ids).astype(float)
        rows = self._rows(user_ids)
        known = rows >= 0
        if known.any():
            seen = self.seen[rows[known]].toarray().astype(bool)
            s[np.flatnonzero(known)[:, None].repeat(s.shape[1], 1)[seen], np.nonzero(seen)[1]] = -np.inf
        top = np.argpartition(-s, k, axis=1)[:, :k]
        order = np.argsort(-np.take_along_axis(s, top, 1), axis=1)
        return self.item_ids[np.take_along_axis(top, order, 1)]


class PopularityRecommender(BaseRecommender):
    name = "popularity"

    def _fit(self, train):
        self._score_vec = self.pop + 1e-3 * self.bayes   # count, ties broken by Bayesian rating

    def score(self, user_ids):
        return np.tile(self._score_vec, (len(user_ids), 1))

    def predict_rating(self, user_ids, item_ids):
        return self.store.get_item_features(item_ids)["item_bayes_rating"].to_numpy()

    def params(self):
        return {"model": self.name, "ranking": "item_popularity_count (+ bayes tie-break)", "rating_head": "item_bayes_rating"}


class ItemKNNRecommender(BaseRecommender):
    name = "item_knn"

    def __init__(self, n_neighbors: int = 50, shrink: float = 10.0, **kw):
        super().__init__(**kw)
        self.n_neighbors, self.shrink = n_neighbors, shrink

    def _fit(self, train):
        # Similarity is built in row-chunks so memory stays O(chunk x n_items) instead of O(n_items^2) (matters for MovieLens ~10k items)
        L = self.liked.astype(float).tocsr()
        n = L.shape[1]
        norm = np.sqrt(np.asarray(L.sum(axis=0)).ravel())            # sqrt(#likers) per item
        LT, blocks, step = L.T.tocsr(), [], 1000
        for start in range(0, n, step):
            end = min(start + step, n)
            sim = (LT[start:end] @ L).toarray() / (np.outer(norm[start:end], norm) + self.shrink)
            sim[np.arange(end - start), np.arange(start, end)] = 0.0   # no self-similarity
            if self.n_neighbors < n:                                   # keep only the strongest neighbours per item
                cut = -np.partition(-sim, self.n_neighbors - 1, axis=1)[:, self.n_neighbors - 1][:, None]
                sim = np.where(sim >= cut, sim, 0.0)
            blocks.append(sparse.csr_matrix(sim))
        self.sim = sparse.vstack(blocks).tocsr()

    def score(self, user_ids):
        rows = self._rows(user_ids)
        hist = self.liked[np.where(rows < 0, 0, rows)].astype(float)
        s = (hist @ self.sim).toarray()
        s[rows < 0] = 0.0
        return s + 1e-6 * self.pop[None, :] / max(self.pop.max(), 1)   # deterministic popularity tie-break

    def params(self):
        return {"model": self.name, "n_neighbors": self.n_neighbors, "shrink": self.shrink, "similarity": "cosine(liked)"}


class HybridSVDRecommender(BaseRecommender):
    name = "hybrid_svd"

    def __init__(self, k_implicit: int = 32, k_explicit: int = 16, alpha: float = 0.5, bias_reg: float = 10.0, **kw):
        super().__init__(**kw)
        self.k_implicit, self.k_explicit, self.alpha, self.bias_reg = k_implicit, k_explicit, alpha, bias_reg

    def _fit(self, train):
        # ---- explicit head: biased SVD on residual ratings (for RMSE / MAE)
        R, mu, reg = self.R.tocoo(), self.global_mean, self.bias_reg
        nu, ni = R.shape
        bi = np.bincount(R.col, weights=R.data - mu, minlength=ni) / (np.bincount(R.col, minlength=ni) + reg)
        bu = np.bincount(R.row, weights=R.data - mu - bi[R.col], minlength=nu) / (np.bincount(R.row, minlength=nu) + reg)
        resid = sparse.csr_matrix((R.data - mu - bu[R.row] - bi[R.col], (R.row, R.col)), shape=R.shape)
        k = min(self.k_explicit, min(resid.shape) - 1)
        U, S, Vt = svds(resid, k=k, random_state=0)
        self.bu, self.bi, self.P, self.Q = bu, bi, U * np.sqrt(S), (Vt.T * np.sqrt(S))
        # ---- implicit CF head: PureSVD on the binary "liked" matrix
        k2 = min(self.k_implicit, min(self.liked.shape) - 1)
        _, _, Vt2 = svds(self.liked.astype(float), k=k2, random_state=0)
        self.V = Vt2.T                                            # items x k2

    def score(self, user_ids):
        rows = self._rows(user_ids)
        hist = self.liked[np.where(rows < 0, 0, rows)].astype(float)
        cf = np.asarray((hist @ self.V) @ self.V.T)
        cf[rows < 0] = 0.0
        s = _zscore_rows(cf)
        if self.alpha > 0:                                       # content signal comes from the FEATURE STORE
            s = s + self.alpha * _zscore_rows(self.store.genre_affinity_matrix(user_ids, self.item_ids))
        return s

    def predict_rating(self, user_ids, item_ids):
        u = np.array([self.uidx.get(int(x), -1) for x in user_ids])
        i = np.array([self.iidx.get(int(x), -1) for x in item_ids])
        pred = np.full(len(u), self.global_mean)
        ku, ki = u >= 0, i >= 0
        pred[ku] += self.bu[u[ku]]
        pred[ki] += self.bi[i[ki]]
        both = ku & ki
        pred[both] += np.einsum("nk,nk->n", self.P[u[both]], self.Q[i[both]])
        return np.clip(pred, 0.5, 5.0)

    def params(self):
        return {"model": self.name, "k_implicit": self.k_implicit, "k_explicit": self.k_explicit, "alpha": self.alpha, "bias_reg": self.bias_reg}


class SurpriseSVDRecommender(BaseRecommender):
    """scikit-surprise `SVD` (biased matrix factorisation, SGD) trained on explicit ratings.

    Ranking score = predicted rating for every (user, item), computed in one vectorised step from the learned
    factors (`pu`, `qi`, `bu`, `bi`) instead of 1.8M `predict()` calls. Rating-prediction models are known to
    rank less well than ranking-optimised ones; it is included as the Surprise baseline for RMSE/MAE.
    """
    name = "surprise_svd"

    def __init__(self, n_factors: int = 50, reg_all: float = 0.02, n_epochs: int = 30, lr_all: float = 0.005, seed: int = 42, **kw):
        super().__init__(**kw)
        self.n_factors, self.reg_all, self.n_epochs, self.lr_all, self.seed = n_factors, reg_all, n_epochs, lr_all, seed

    def _fit(self, train):
        from surprise import SVD, Dataset, Reader
        data = Dataset.load_from_df(train[["user_id", "item_id", "rating"]], Reader(rating_scale=(0.5, 5.0)))
        ts = data.build_full_trainset()
        algo = SVD(n_factors=self.n_factors, n_epochs=self.n_epochs, lr_all=self.lr_all, reg_all=self.reg_all, random_state=self.seed)
        algo.fit(ts)
        nu, ni, k = len(self.uidx), len(self.iidx), self.n_factors
        self.mu = float(ts.global_mean)
        self.bu, self.bi, self.P, self.Q = np.zeros(nu), np.zeros(ni), np.zeros((nu, k)), np.zeros((ni, k))
        for inner in ts.all_users():            # copy factors into catalogue-wide arrays (users/items unseen in train stay 0)
            r = self.uidx[int(ts.to_raw_uid(inner))]
            self.bu[r], self.P[r] = algo.bu[inner], algo.pu[inner]
        for inner in ts.all_items():
            c = self.iidx[int(ts.to_raw_iid(inner))]
            self.bi[c], self.Q[c] = algo.bi[inner], algo.qi[inner]

    def score(self, user_ids):
        rows = self._rows(user_ids)
        r = np.where(rows < 0, 0, rows)
        s = self.mu + self.bu[r][:, None] + self.bi[None, :] + self.P[r] @ self.Q.T
        s[rows < 0] = self.mu
        return s

    def predict_rating(self, user_ids, item_ids):
        u = np.array([self.uidx.get(int(x), -1) for x in user_ids])
        i = np.array([self.iidx.get(int(x), -1) for x in item_ids])
        pred = np.full(len(u), self.mu)
        ku, ki = u >= 0, i >= 0
        pred[ku] += self.bu[u[ku]]
        pred[ki] += self.bi[i[ki]]
        both = ku & ki
        pred[both] += np.einsum("nk,nk->n", self.P[u[both]], self.Q[i[both]])
        return np.clip(pred, 0.5, 5.0)

    def params(self):
        return {"model": self.name, "library": "scikit-surprise SVD", "n_factors": self.n_factors, "reg_all": self.reg_all,
                "n_epochs": self.n_epochs, "lr_all": self.lr_all}


MODEL_CLASSES = {"popularity": PopularityRecommender, "item_knn": ItemKNNRecommender, "hybrid_svd": HybridSVDRecommender, "surprise_svd": SurpriseSVDRecommender}
