"""
Semantic cache: caches RAG results keyed by embedding similarity rather than
exact query string match, so paraphrased queries can still hit cache.

Bug fixed from earlier iterations: expired-key deletion is collected during
the scan and applied *after* iteration completes, never during it, so this
cannot raise "dictionary changed size during iteration".
"""

import logging
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from src.core.embeddings import Embedder

logger = logging.getLogger(__name__)


class SemanticCache:
    def __init__(self, config: Dict[str, Any], embedder: Optional[Embedder] = None):
        self.max_size = config.get("max_size", 1000)
        self.ttl = config.get("ttl_seconds", 3600)
        self.threshold = config.get("similarity_threshold", 0.85)
        self._embedder_config = config.get("embedder", {})
        self._embedder = embedder
        self.cache: Dict[str, Tuple[np.ndarray, Any, float]] = {}
        self.access_order: List[str] = []
        self._initialized = False

    async def initialize(self) -> None:
        if self._embedder is None:
            from src.core.embeddings import get_embedder

            self._embedder = get_embedder(self._embedder_config)
        self._initialized = True
        logger.info("SemanticCache initialized (threshold=%.2f, ttl=%ds)", self.threshold, self.ttl)

    async def get_similar_with_score(self, query: str) -> Tuple[Optional[Any], float]:
        if not self._initialized or not self.cache:
            return None, 0.0

        query_emb = self._embedder.encode_one(query)
        now = time.time()

        best_match, best_score, best_key = None, 0.0, None
        expired_keys: List[str] = []

        for key, (emb, value, timestamp) in self.cache.items():
            if now - timestamp > self.ttl:
                expired_keys.append(key)
                continue
            sim = self._cosine(query_emb, emb)
            if sim > best_score and sim >= self.threshold:
                best_score, best_match, best_key = sim, value, key

        for key in expired_keys:
            await self.delete(key)

        if best_key and best_key in self.access_order:
            self.access_order.remove(best_key)
            self.access_order.append(best_key)
            return best_match, best_score

        return None, 0.0

    async def get_similar(self, query: str) -> Optional[Any]:
        result, _ = await self.get_similar_with_score(query)
        return result

    async def set(self, query: str, value: Any) -> None:
        if not self._initialized:
            return

        if len(self.cache) >= self.max_size and self.access_order:
            oldest = self.access_order.pop(0)
            self.cache.pop(oldest, None)

        embedding = self._embedder.encode_one(query)
        key = query[:200]

        self.cache[key] = (embedding, value, time.time())
        if key in self.access_order:
            self.access_order.remove(key)
        self.access_order.append(key)

    async def delete(self, key: str) -> None:
        self.cache.pop(key, None)
        if key in self.access_order:
            self.access_order.remove(key)

    @staticmethod
    def _cosine(a: np.ndarray, b: np.ndarray) -> float:
        denom = (np.linalg.norm(a) * np.linalg.norm(b)) or 1e-9
        return float(np.dot(a, b) / denom)

    def __len__(self) -> int:
        return len(self.cache)

    async def close(self) -> None:
        self._initialized = False