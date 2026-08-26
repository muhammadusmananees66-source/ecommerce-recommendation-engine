"""
RAG evaluation metrics, computed from embeddings rather than requiring a
paid judge-LLM API call per evaluation (keeps CI evaluation runnable
offline/free; swapping in an LLM-judge is a valid upgrade path for higher
metric fidelity, not implemented here to keep this runnable without API keys).
"""

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np

from src.core.embeddings import Embedder, get_embedder

logger = logging.getLogger(__name__)


@dataclass
class RAGEvaluationResult:
    faithfulness: float
    answer_relevancy: float
    context_precision: float
    answer_similarity: float


class RAGEvaluator:
    def __init__(self, config: dict[str, Any], embedder: Embedder | None = None):
        self._embedder_config = config.get("embedder", {})
        self._embedder = embedder
        self.faithfulness_threshold = config.get("faithfulness_entailment_threshold", 0.6)

    async def initialize(self) -> None:
        if self._embedder is None:
            self._embedder = get_embedder(self._embedder_config)

    async def evaluate(
        self, query: str, answer: str, contexts: list[str], ground_truth: str | None = None
    ) -> RAGEvaluationResult:
        faithfulness = self._faithfulness(answer, contexts)
        relevancy = self._answer_relevancy(query, answer)
        precision = self._context_precision(query, contexts)
        similarity = self._answer_similarity(answer, ground_truth) if ground_truth else 0.0

        return RAGEvaluationResult(
            faithfulness=faithfulness,
            answer_relevancy=relevancy,
            context_precision=precision,
            answer_similarity=similarity,
        )

    def _faithfulness(self, answer: str, contexts: list[str]) -> float:
        """Fraction of answer sentences with high similarity to at least one context chunk."""
        if not contexts or not answer:
            return 0.0
        claims = [s.strip() for s in answer.split(".") if len(s.strip()) > 8]
        if not claims:
            return 0.0

        claim_vecs = self._embedder.encode(claims)
        context_vecs = self._embedder.encode(contexts)

        supported = 0
        for cv in claim_vecs:
            sims = context_vecs @ cv / (
                np.linalg.norm(context_vecs, axis=1) * np.linalg.norm(cv) + 1e-9
            )
            if np.max(sims) >= self.faithfulness_threshold:
                supported += 1
        return supported / len(claims)

    def _answer_relevancy(self, query: str, answer: str) -> float:
        if not answer:
            return 0.0
        q, a = self._embedder.encode_one(query), self._embedder.encode_one(answer)
        return float(np.clip(q @ a / (np.linalg.norm(q) * np.linalg.norm(a) + 1e-9), 0.0, 1.0))

    def _context_precision(self, query: str, contexts: list[str]) -> float:
        if not contexts:
            return 0.0
        q = self._embedder.encode_one(query)
        c = self._embedder.encode(contexts)
        sims = c @ q / (np.linalg.norm(c, axis=1) * np.linalg.norm(q) + 1e-9)
        return float(np.mean(sims >= self.faithfulness_threshold))

    def _answer_similarity(self, answer: str, ground_truth: str) -> float:
        a, g = self._embedder.encode_one(answer), self._embedder.encode_one(ground_truth)
        return float(np.clip(a @ g / (np.linalg.norm(a) * np.linalg.norm(g) + 1e-9), 0.0, 1.0))