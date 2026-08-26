"""Prometheus metrics for the RAG pipeline itself (distinct from HTTP-layer metrics)."""

import logging

from prometheus_client import Counter, Gauge, Histogram

logger = logging.getLogger(__name__)

RAG_REQUESTS = Counter("rag_requests_total", "Total RAG requests", ["model", "status"])
RAG_LATENCY = Histogram("rag_request_duration_seconds", "RAG pipeline latency", ["model"])
RAG_CONFIDENCE = Gauge("rag_confidence_score", "Most recent RAG confidence score")
RAG_GROUNDEDNESS = Gauge("rag_groundedness_score", "Most recent RAG groundedness score")
RAG_DOC_COUNT = Histogram("rag_retrieved_document_count", "Number of documents retrieved per query")
RAG_CACHE_HITS = Counter("rag_cache_hits_total", "Semantic cache hits")
RAG_CACHE_MISSES = Counter("rag_cache_misses_total", "Semantic cache misses")


class RAGMonitor:
    def record_request(self, model: str, status: str, latency_ms: float) -> None:
        RAG_REQUESTS.labels(model=model, status=status).inc()
        RAG_LATENCY.labels(model=model).observe(latency_ms / 1000.0)

    def record_confidence(self, value: float) -> None:
        RAG_CONFIDENCE.set(value)

    def record_groundedness(self, value: float) -> None:
        RAG_GROUNDEDNESS.set(value)

    def record_document_count(self, count: int) -> None:
        RAG_DOC_COUNT.observe(count)

    def record_cache_hit(self) -> None:
        RAG_CACHE_HITS.inc()

    def record_cache_miss(self) -> None:
        RAG_CACHE_MISSES.inc()