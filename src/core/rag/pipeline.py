"""
The RAG pipeline: cache check -> retrieve -> rank -> build context -> route LLM
-> generate -> check groundedness -> score confidence -> cache result.

Wrapped in an overall asyncio timeout so a hung dependency can't stall a
request indefinitely regardless of what any individual component does.
"""

import asyncio
import copy
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from src.core.embeddings import Embedder, get_embedder
from src.core.generation.llm_router import LLMRouter
from src.core.generation.response_generator import ResponseGenerator
from src.core.rag.context_builder import ContextBuilder
from src.core.rag.groundedness import GroundednessChecker
from src.core.rag.relevance_ranker import RelevanceRanker
from src.core.rag.semantic_cache import SemanticCache
from src.core.retrieval.retriever import MultiStageRetriever
from src.monitoring.rag_monitoring import RAGMonitor

logger = logging.getLogger(__name__)


@dataclass
class RAGResult:
    query: str
    retrieved_docs: List[Dict] = field(default_factory=list)
    generated_response: str = ""
    context: Dict = field(default_factory=dict)
    confidence_score: float = 0.0
    latency_ms: float = 0.0
    sources: List[Dict] = field(default_factory=list)
    groundedness_score: float = 0.0
    model_used: str = "unknown"
    cached: bool = False
    cache_similarity: float = 0.0


class RAGPipeline:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.timeout = config.get("timeout", 30)

        self.embedder: Embedder = get_embedder(config.get("embedder", {}))
        self.retriever = MultiStageRetriever(config.get("retriever", {}))
        self.llm_router = LLMRouter(config.get("llm", {}))
        self.response_generator = ResponseGenerator(config.get("generator", {}))
        self.groundedness_checker = GroundednessChecker(config.get("groundedness", {}), embedder=self.embedder)
        self.semantic_cache = SemanticCache(config.get("cache", {}), embedder=self.embedder)
        self.context_builder = ContextBuilder(config.get("context", {}))
        self.relevance_ranker = RelevanceRanker(config.get("ranking", {}), embedder=self.embedder)
        self.monitor = RAGMonitor()

        self.response_generator.llm_router = self.llm_router
        self._initialized = False

    async def initialize(self) -> None:
        await self.retriever.initialize()
        await self.llm_router.initialize()
        await self.response_generator.initialize()
        await self.semantic_cache.initialize()
        await self.groundedness_checker.initialize()
        self._initialized = True
        logger.info("RAGPipeline initialized")

    async def query(
        self,
        query: str,
        user_context: Optional[Dict] = None,
        max_docs: int = 10,
        temperature: float = 0.7,
    ) -> RAGResult:
        if not self._initialized:
            raise RuntimeError("RAGPipeline not initialized")
        try:
            return await asyncio.wait_for(
                self._query_internal(query, user_context, max_docs, temperature),
                timeout=self.timeout,
            )
        except asyncio.TimeoutError:
            logger.error("RAG query timed out after %ds: %r", self.timeout, query)
            raise TimeoutError(f"RAG query timed out after {self.timeout}s")

    async def _query_internal(
        self, query: str, user_context: Optional[Dict], max_docs: int, temperature: float
    ) -> RAGResult:
        start = time.time()

        cached, similarity = await self.semantic_cache.get_similar_with_score(query)
        if cached:
            result = copy.deepcopy(cached)
            result.cached = True
            result.cache_similarity = similarity
            self.monitor.record_cache_hit()
            return result
        self.monitor.record_cache_miss()

        embedding = self.embedder.encode_one(query)
        docs = await self.retriever.retrieve(query, embedding, user_context, max_docs * 2)
        docs = self.relevance_ranker.rank(query, docs, user_context)[:max_docs]

        context = self.context_builder.build(query, docs, user_context)
        llm_config = self.llm_router.route(query, context)

        prompt = f"Query: {context['query']}\n\n{context['text']}"
        response, model_used = await self.response_generator.generate_with_fallback(
            prompt=prompt, model=llm_config["model"], temperature=temperature,
        )

        groundedness = await self.groundedness_checker.check(query, response, docs)
        confidence = self._calculate_confidence(docs, response, groundedness)

        result = RAGResult(
            query=query,
            retrieved_docs=docs,
            generated_response=response,
            context=context,
            confidence_score=confidence,
            latency_ms=(time.time() - start) * 1000,
            sources=docs[:3],
            groundedness_score=groundedness,
            model_used=model_used,
            cached=False,
        )

        await self.semantic_cache.set(query, result)

        self.monitor.record_request(model_used, "success", result.latency_ms)
        self.monitor.record_confidence(confidence)
        self.monitor.record_groundedness(groundedness)
        self.monitor.record_document_count(len(docs))

        return result

    @staticmethod
    def _calculate_confidence(docs: List[Dict], response: str, groundedness: float) -> float:
        if not docs:
            return 0.0
        import numpy as np

        avg_retrieval_score = float(np.mean([d.get("score", 0.0) for d in docs]))
        length_factor = min(1.0, len(response) / 200.0)
        return round(0.4 * avg_retrieval_score + 0.2 * length_factor + 0.4 * groundedness, 4)

    async def close(self) -> None:
        await self.retriever.close()
        await self.semantic_cache.close()
        self._initialized = False