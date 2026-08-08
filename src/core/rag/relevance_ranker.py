"""Ranks retrieved documents by relevance to the query before they reach the LLM."""

import logging
from typing import Any

import numpy as np

from src.core.embeddings import Embedder

logger = logging.getLogger(__name__)


class RelevanceRanker:
    def __init__(self, config: dict[str, Any], embedder: Embedder | None = None):
        self._embedder_config = config.get("embedder", {})
        self._embedder = embedder
        self.method = config.get("method", "hybrid")
        self.alpha = config.get("alpha", 0.6)  # weight on semantic vs lexical score

    def _get_embedder(self) -> Embedder:
        if self._embedder is None:
            from src.core.embeddings import get_embedder

            self._embedder = get_embedder(self._embedder_config)
        return self._embedder

    def rank(self, query: str, docs: list[dict], user_context: dict | None = None) -> list[dict]:
        if not docs:
            return []
        if self.method == "semantic":
            return self._rank_semantic(query, docs)
        if self.method == "lexical":
            return self._rank_lexical(query, docs)
        return self._rank_hybrid(query, docs)

    def _rank_semantic(self, query: str, docs: list[dict]) -> list[dict]:
        embedder = self._get_embedder()
        q = embedder.encode_one(query)
        texts = [d.get("text", "") for d in docs]
        doc_vecs = embedder.encode(texts)
        sims = doc_vecs @ q / (np.linalg.norm(doc_vecs, axis=1) * np.linalg.norm(q) + 1e-9)
        for doc, s in zip(docs, sims):
            doc["semantic_score"] = float(s)
            doc["score"] = float(s)
        return sorted(docs, key=lambda d: d["score"], reverse=True)

    def _rank_lexical(self, query: str, docs: list[dict]) -> list[dict]:
        from sklearn.feature_extraction.text import TfidfVectorizer

        texts = [d.get("text", "") for d in docs]
        vectorizer = TfidfVectorizer()
        try:
            matrix = vectorizer.fit_transform([query] + texts)
        except ValueError:
            # empty vocabulary (e.g. all-stopword or empty texts)
            for doc in docs:
                doc["lexical_score"] = 0.0
                doc["score"] = 0.0
            return docs
        sims = (matrix[0] @ matrix[1:].T).toarray().flatten()
        for doc, s in zip(docs, sims):
            doc["lexical_score"] = float(s)
            doc["score"] = float(s)
        return sorted(docs, key=lambda d: d["score"], reverse=True)

    def _rank_hybrid(self, query: str, docs: list[dict]) -> list[dict]:
        semantic = self._rank_semantic(query, [dict(d) for d in docs])
        lexical = self._rank_lexical(query, [dict(d) for d in docs])

        sem_by_id = {d.get("id", i): d["semantic_score"] for i, d in enumerate(semantic)}
        lex_by_id = {d.get("id", i): d["lexical_score"] for i, d in enumerate(lexical)}

        for i, doc in enumerate(docs):
            doc_id = doc.get("id", i)
            sem = sem_by_id.get(doc_id, 0.0)
            lex = lex_by_id.get(doc_id, 0.0)
            doc["semantic_score"] = sem
            doc["lexical_score"] = lex
            doc["score"] = self.alpha * sem + (1 - self.alpha) * lex

        return sorted(docs, key=lambda d: d["score"], reverse=True)
