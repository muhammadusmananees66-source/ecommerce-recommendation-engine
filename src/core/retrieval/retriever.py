"""Multi-stage retrieval: vector search -> metadata filtering -> lexical re-boost."""

import logging
from typing import Any, Dict, List, Optional

import numpy as np

from src.core.retrieval.vector_store import VectorStore

logger = logging.getLogger(__name__)


class MultiStageRetriever:
    def __init__(self, config: Dict[str, Any], vector_store: Optional[VectorStore] = None):
        self.config = config
        self.vector_store = vector_store or VectorStore(config.get("vector_store", {}))
        self.stages = config.get("stages", ["vector", "filter", "rerank"])
        self._initialized = False

    async def initialize(self) -> None:
        await self.vector_store.initialize()
        self._initialized = True

    async def retrieve(
        self,
        query: str,
        query_embedding: np.ndarray,
        user_context: Optional[Dict] = None,
        max_docs: int = 20,
    ) -> List[Dict]:
        if not self._initialized:
            raise RuntimeError("MultiStageRetriever not initialized")

        docs: List[Dict] = []

        if "vector" in self.stages:
            docs = await self.vector_store.query(query_embedding, top_k=max_docs * 2)

        if "filter" in self.stages and docs:
            docs = self._apply_filters(docs, user_context)

        if "rerank" in self.stages and docs:
            docs = self._lexical_boost(query, docs)

        return docs[:max_docs]

    @staticmethod
    def _apply_filters(docs: List[Dict], user_context: Optional[Dict]) -> List[Dict]:
        if not user_context:
            return docs

        category = user_context.get("category")
        price_range = user_context.get("price_range")
        if not category and not price_range:
            return docs

        filtered = []
        for doc in docs:
            meta = doc.get("metadata", {})
            if category and meta.get("category") != category:
                continue
            if price_range:
                price = meta.get("price", 0)
                lo = price_range.get("min", 0)
                hi = price_range.get("max", float("inf"))
                if not (lo <= price <= hi):
                    continue
            filtered.append(doc)
        return filtered

    @staticmethod
    def _lexical_boost(query: str, docs: List[Dict]) -> List[Dict]:
        query_terms = set(query.lower().split())
        for doc in docs:
            text = doc.get("metadata", {}).get("text", "") or doc.get("text", "")
            matches = sum(1 for t in query_terms if t in text.lower())
            doc["score"] = doc.get("score", 0.0) * (1 + 0.1 * matches)
        return sorted(docs, key=lambda d: d.get("score", 0.0), reverse=True)

    async def close(self) -> None:
        await self.vector_store.close()
        self._initialized = False