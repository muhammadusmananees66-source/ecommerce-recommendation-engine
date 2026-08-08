"""
Groundedness checking: how well is the generated answer supported by the
retrieved documents. Two methods:
- "semantic" (default): max cosine similarity between answer and any doc.
  Cheap, no extra model download, works with the default tfidf embedder.
- "nli": sentence-level entailment via a cross-encoder NLI model. More
  accurate but requires the optional transformers+torch dependency group;
  imported lazily and falls back to "semantic" if unavailable.
"""

import logging
from typing import Any

import numpy as np

from src.core.embeddings import Embedder

logger = logging.getLogger(__name__)


class GroundednessChecker:
    def __init__(self, config: dict[str, Any], embedder: Embedder | None = None):
        self._embedder_config = config.get("embedder", {})
        self._embedder = embedder
        self.method = config.get("method", "semantic")
        self._nli_pipeline = None
        self._initialized = False

    async def initialize(self) -> None:
        if self._embedder is None:
            from src.core.embeddings import get_embedder

            self._embedder = get_embedder(self._embedder_config)

        if self.method == "nli":
            try:
                from transformers import pipeline

                self._nli_pipeline = pipeline("text-classification", model="roberta-large-mnli", device=-1)
            except ImportError as e:
                logger.warning(
                    "NLI groundedness requested but transformers not installed (%s); using semantic", e
                )
                self.method = "semantic"
        self._initialized = True

    async def check(self, query: str, response: str, docs: list[dict]) -> float:
        if not self._initialized or not docs or not response:
            return 0.0
        if self.method == "nli" and self._nli_pipeline:
            return self._check_nli(response, docs)
        return self._check_semantic(response, docs)

    def _check_semantic(self, response: str, docs: list[dict]) -> float:
        texts = [d.get("text", "") for d in docs if d.get("text")]
        if not texts:
            return 0.0
        response_vec = self._embedder.encode_one(response)
        doc_vecs = self._embedder.encode(texts)
        sims = (
            doc_vecs @ response_vec / (np.linalg.norm(doc_vecs, axis=1) * np.linalg.norm(response_vec) + 1e-9)
        )
        return float(np.clip(np.max(sims), 0.0, 1.0))

    def _check_nli(self, response: str, docs: list[dict]) -> float:
        claims = [s.strip() for s in response.split(".") if len(s.strip()) > 10]
        if not claims:
            return 0.0
        scores = []
        for claim in claims:
            best = 0.0
            for doc in docs:
                text = doc.get("text", "")
                if not text:
                    continue
                try:
                    result = self._nli_pipeline(f"{text} [SEP] {claim}")
                except Exception as e:
                    logger.warning("NLI inference failed (%s); skipping claim", e)
                    continue
                item = result[0] if isinstance(result, list) else result
                if item.get("label", "").upper() == "ENTAILMENT":
                    best = max(best, item.get("score", 0.0))
            scores.append(best)
        return float(np.mean(scores)) if scores else 0.0
