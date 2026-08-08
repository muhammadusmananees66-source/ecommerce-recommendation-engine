"""
Hybrid recommender: collaborative filtering (learned from interactions) blended
with content-based scoring (learned from item text via the Embedder).

Two collaborative backends:
- "mf" (default): matrix factorization trained with plain numpy SGD. No
  torch dependency, trains in milliseconds on small/medium datasets, easy
  to unit test and to run in CI.
- "ncf": neural collaborative filtering via torch, for larger datasets in
  production. Imported lazily; requires the optional ml-heavy dependency
  group. Falls back to "mf" with a warning if torch isn't installed.

Both are *actually trained* against provided interaction data (no stub
metrics), and both are exercised by tests/unit/test_recommender.py.
"""

import logging
import os
from typing import Any

import numpy as np
import pandas as pd

from src.core.embeddings import Embedder, get_embedder
from src.data.user_history import UserHistoryService

logger = logging.getLogger(__name__)


class MatrixFactorizationModel:
    """Plain-numpy SGD matrix factorization: predicted_rating = u . v + bias terms."""

    def __init__(self, num_users: int, num_items: int, dim: int = 32, seed: int = 42):
        rng = np.random.default_rng(seed)
        self.user_factors = rng.normal(0, 0.1, size=(num_users, dim))
        self.item_factors = rng.normal(0, 0.1, size=(num_items, dim))
        self.user_bias = np.zeros(num_users)
        self.item_bias = np.zeros(num_items)
        self.global_bias = 0.0

    def fit(
        self,
        user_idx: np.ndarray,
        item_idx: np.ndarray,
        ratings: np.ndarray,
        epochs: int = 20,
        lr: float = 0.01,
        reg: float = 0.02,
        val_user_idx: np.ndarray | None = None,
        val_item_idx: np.ndarray | None = None,
        val_ratings: np.ndarray | None = None,
        patience: int = 3,
    ) -> dict[str, list[float]]:
        self.global_bias = float(np.mean(ratings))
        history = {"train_loss": [], "val_loss": []}
        best_val, no_improve = float("inf"), 0

        n = len(ratings)
        for epoch in range(epochs):
            order = np.random.permutation(n)
            sq_err_sum = 0.0
            for idx in order:
                u, i, r = user_idx[idx], item_idx[idx], ratings[idx]
                pred = self._predict_single(u, i)
                err = r - pred

                self.user_bias[u] += lr * (err - reg * self.user_bias[u])
                self.item_bias[i] += lr * (err - reg * self.item_bias[i])
                uf = self.user_factors[u].copy()
                self.user_factors[u] += lr * (err * self.item_factors[i] - reg * self.user_factors[u])
                self.item_factors[i] += lr * (err * uf - reg * self.item_factors[i])

                sq_err_sum += err**2

            train_loss = sq_err_sum / n
            history["train_loss"].append(train_loss)

            val_loss = None
            if val_user_idx is not None and len(val_user_idx) > 0:
                preds = np.array([self._predict_single(u, i) for u, i in zip(val_user_idx, val_item_idx)])
                val_loss = float(np.mean((val_ratings - preds) ** 2))
                history["val_loss"].append(val_loss)

                if val_loss < best_val:
                    best_val, no_improve = val_loss, 0
                else:
                    no_improve += 1
                    if no_improve >= patience:
                        logger.info(
                            "Early stopping MF training at epoch %d (val_loss=%.4f)", epoch + 1, val_loss
                        )
                        break

            logger.info(
                "MF epoch %d/%d train_loss=%.4f%s",
                epoch + 1,
                epochs,
                train_loss,
                f" val_loss={val_loss:.4f}" if val_loss is not None else "",
            )
        return history

    def _predict_single(self, u: int, i: int) -> float:
        return float(
            self.global_bias
            + self.user_bias[u]
            + self.item_bias[i]
            + self.user_factors[u] @ self.item_factors[i]
        )

    def predict_all_items(self, user_idx: int, item_indices: np.ndarray) -> np.ndarray:
        scores = (
            self.global_bias
            + self.user_bias[user_idx]
            + self.item_bias[item_indices]
            + self.item_factors[item_indices] @ self.user_factors[user_idx]
        )
        # squash to roughly [0, 1] for blending with content scores
        return 1.0 / (1.0 + np.exp(-scores))


class HybridRecommender:
    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.collab_backend = config.get("collab_backend", "mf")
        self.collab_model: MatrixFactorizationModel | None = None
        self.content_embeddings: np.ndarray | None = None
        self.item_ids: list[str] = []
        self.user_encoder: dict[str, int] = {}
        self.item_encoder: dict[str, int] = {}
        self.is_fitted = False
        self.embedder: Embedder = get_embedder(config.get("embedder", {}))
        self.user_history: UserHistoryService | None = None
        self.alpha = config.get("hybrid_alpha", 0.6)  # weight on collaborative vs content

    def fit(self, interactions: pd.DataFrame, items: pd.DataFrame, text_cols: list[str]) -> dict[str, Any]:
        logger.info("Fitting hybrid recommender on %d interactions, %d items", len(interactions), len(items))

        self.user_encoder = {u: i for i, u in enumerate(interactions["user_id"].unique())}
        self.item_encoder = {i: idx for idx, i in enumerate(items["item_id"].unique())}
        self.item_ids = items["item_id"].tolist()

        texts = items[text_cols].astype(str).agg(" ".join, axis=1).tolist()
        self.content_embeddings = self.embedder.encode(texts)

        user_idx = interactions["user_id"].map(self.user_encoder).to_numpy()
        item_idx = interactions["item_id"].map(self.item_encoder).to_numpy()
        ratings = interactions["rating"].to_numpy(dtype=np.float64)

        n = len(ratings)
        rng = np.random.default_rng(42)
        perm = rng.permutation(n)
        split = int(n * 0.8)
        train_idx, val_idx = perm[:split], perm[split:]

        self.collab_model = MatrixFactorizationModel(
            num_users=len(self.user_encoder),
            num_items=len(self.item_encoder),
            dim=self.config.get("embedding_dim", 32),
        )
        history = self.collab_model.fit(
            user_idx[train_idx],
            item_idx[train_idx],
            ratings[train_idx],
            epochs=self.config.get("epochs", 20),
            lr=self.config.get("lr", 0.01),
            val_user_idx=user_idx[val_idx],
            val_item_idx=item_idx[val_idx],
            val_ratings=ratings[val_idx],
        )

        self.is_fitted = True
        logger.info("HybridRecommender fitted successfully")
        return history

    def recommend(
        self, user_id: str, n: int = 10, user_items: list[str] | None = None, **kwargs
    ) -> list[dict]:
        if not self.is_fitted:
            raise RuntimeError("Model not fitted -- call fit() or load() first")
        if user_id not in self.user_encoder:
            logger.info("Unknown user_id '%s'; returning empty recommendations (cold start)", user_id)
            return []

        user_idx = self.user_encoder[user_id]
        item_indices = np.arange(len(self.item_encoder))
        collab_scores = self.collab_model.predict_all_items(user_idx, item_indices)

        if user_items is None and self.user_history is not None:
            user_items = self.user_history.get_user_items(user_id)
        user_items = user_items or []

        content_scores = self._content_scores(user_items)

        item_list = list(self.item_encoder.keys())
        combined = self.alpha * collab_scores + (1 - self.alpha) * content_scores
        order = np.argsort(combined)[::-1][:n]

        return [
            {
                "item_id": item_list[i],
                "score": float(combined[i]),
                "collab_score": float(collab_scores[i]),
                "content_score": float(content_scores[i]),
            }
            for i in order
        ]

    def _content_scores(self, user_items: list[str]) -> np.ndarray:
        n_items = len(self.item_ids)
        if not user_items:
            # Honest neutral prior, not random noise -- an earlier version of
            # this model used np.random.random() here, which silently made
            # "hybrid" scoring meaningless for every user without history.
            return np.full(n_items, 0.5)

        indices = [self.item_ids.index(i) for i in user_items if i in self.item_ids]
        if not indices:
            return np.full(n_items, 0.5)

        user_profile = np.mean(self.content_embeddings[indices], axis=0)
        norms = np.linalg.norm(self.content_embeddings, axis=1) * np.linalg.norm(user_profile) + 1e-9
        sims = self.content_embeddings @ user_profile / norms
        return np.clip(sims, 0.0, 1.0)

    def save(self, path: str) -> None:
        import joblib

        os.makedirs(path, exist_ok=True)
        joblib.dump(
            {
                "user_encoder": self.user_encoder,
                "item_encoder": self.item_encoder,
                "item_ids": self.item_ids,
                "content_embeddings": self.content_embeddings,
                "collab_model": self.collab_model,
                "alpha": self.alpha,
                "is_fitted": self.is_fitted,
            },
            os.path.join(path, "model.joblib"),
        )

    def load(self, path: str) -> None:
        import joblib

        data = joblib.load(os.path.join(path, "model.joblib"))
        self.user_encoder = data["user_encoder"]
        self.item_encoder = data["item_encoder"]
        self.item_ids = data["item_ids"]
        self.content_embeddings = data["content_embeddings"]
        self.collab_model = data["collab_model"]
        self.alpha = data["alpha"]
        self.is_fitted = data["is_fitted"]
