"""
Embedding abstraction.

Production RAG systems need a text -> vector function. This module provides
two real, working implementations instead of hard-coding sentence-transformers
everywhere (which pulls in torch and a multi-GB model download that many
CI/dev environments can't or don't want to pay for):

- TfidfEmbedder: pure scikit-learn, no network access, no GPU. Good enough
  for small/medium catalogs and for CI. This is the default.
- SentenceTransformerEmbedder: wraps sentence-transformers/all-MiniLM-L6-v2
  for real semantic embeddings in production. Imported lazily so the rest
  of the app works even if torch isn't installed.

Both implement the same interface: encode(texts) -> np.ndarray of shape
(n, dim), and encode_one(text) -> np.ndarray of shape (dim,).
"""

import logging
from typing import List, Protocol

import numpy as np

logger = logging.getLogger(__name__)


class Embedder(Protocol):
    dim: int

    def encode(self, texts: List[str]) -> np.ndarray: ...

    def encode_one(self, text: str) -> np.ndarray: ...


class TfidfEmbedder:
    """
    Deterministic, dependency-light embedder built on a hashing vectorizer
    so it needs no fit step and no vocabulary state to persist. Vectors are
    L2-normalized so cosine similarity behaves as expected.
    """

    def __init__(self, dim: int = 256):
        from sklearn.feature_extraction.text import HashingVectorizer

        self.dim = dim
        self._vectorizer = HashingVectorizer(
            n_features=dim, alternate_sign=False, norm="l2"
        )

    def encode(self, texts: List[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        matrix = self._vectorizer.transform(texts)
        return matrix.toarray().astype(np.float32)

    def encode_one(self, text: str) -> np.ndarray:
        return self.encode([text])[0]


class SentenceTransformerEmbedder:
    """
    Real semantic embeddings for production use. Requires the optional
    `sentence-transformers` + `torch` dependencies (see pyproject.toml
    [project.optional-dependencies] "ml-heavy" group). Import is lazy so
    importing this module doesn't require torch to be installed.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(model_name)
        self.dim = self._model.get_sentence_embedding_dimension()

    def encode(self, texts: List[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        return np.asarray(self._model.encode(texts), dtype=np.float32)

    def encode_one(self, text: str) -> np.ndarray:
        return self.encode([text])[0]


def get_embedder(config: dict) -> Embedder:
    """
    Factory. config["backend"] in {"tfidf", "sentence-transformers"}.
    Defaults to tfidf so the app can boot without heavy ML deps installed.
    Falls back to tfidf with a warning if sentence-transformers isn't
    installed, rather than crashing at startup.
    """
    backend = config.get("backend", "tfidf")
    if backend == "sentence-transformers":
        try:
            return SentenceTransformerEmbedder(config.get("model_name", "all-MiniLM-L6-v2"))
        except ImportError as e:
            logger.warning(
                "sentence-transformers backend requested but not installed (%s); "
                "falling back to tfidf embedder", e,
            )
    return TfidfEmbedder(dim=config.get("dim", 256))