import time

import numpy as np
import pytest

from src.api.middleware.rate_limit import RateLimitMiddleware
from src.core.embeddings import get_embedder
from src.core.rag.semantic_cache import SemanticCache
from src.core.retrieval.vector_store import VectorStore


@pytest.mark.asyncio
async def test_vector_store_empty_returns_empty_no_fabrication():
    vs = VectorStore({})
    await vs.initialize()
    result = await vs.query(np.random.rand(256), top_k=5)
    assert result == []


@pytest.mark.asyncio
async def test_vector_store_bounded_fifo_eviction():
    vs = VectorStore({"max_size": 3})
    await vs.initialize()
    embedder = get_embedder({})
    texts = [f"document number {i}" for i in range(5)]
    vecs = embedder.encode(texts)
    metas = [{"id": f"d{i}", "text": t} for i, t in enumerate(texts)]
    vs.add_documents(vecs, metas)
    assert len(vs) == 3  # capped, not unbounded growth


@pytest.mark.asyncio
async def test_vector_store_surfaces_text_at_top_level():
    """Regression test: retrieved docs must expose 'text' at top level,
    not only nested under metadata, or downstream context building fails silently."""
    vs = VectorStore({})
    await vs.initialize()
    embedder = get_embedder({})
    vecs = embedder.encode(["hello world"])
    vs.add_documents(vecs, [{"id": "d1", "text": "hello world"}])
    results = await vs.query(embedder.encode_one("hello world"), top_k=1)
    assert results[0]["text"] == "hello world"


@pytest.mark.asyncio
async def test_semantic_cache_survives_mass_expiry_without_crashing():
    """Regression test: deleting expired entries during iteration used to
    raise RuntimeError: dictionary changed size during iteration."""
    cache = SemanticCache({"ttl_seconds": 0, "similarity_threshold": 0.99})
    await cache.initialize()

    class FakeResult:
        def __init__(self, q):
            self.query = q

    for i in range(10):
        await cache.set(f"query {i}", FakeResult(f"answer {i}"))

    result, score = await cache.get_similar_with_score("query 0")
    assert result is None
    assert len(cache) == 0


@pytest.mark.asyncio
async def test_semantic_cache_hits_on_repeated_query():
    cache = SemanticCache({"ttl_seconds": 3600, "similarity_threshold": 0.5})
    await cache.initialize()

    class FakeResult:
        def __init__(self, q):
            self.query = q

    await cache.set("what is the refund policy", FakeResult("30 days"))
    result, score = await cache.get_similar_with_score("what is the refund policy")
    assert result is not None
    assert score > 0.5


def test_rate_limiter_fallback_actually_disables_redis_client_on_connect_failure():
    """Regression test: an earlier version left self.redis_client set to a
    broken (but truthy) client object when the connection failed, so the
    memory fallback path was silently unreachable."""

    class DummyApp:
        pass

    mw = RateLimitMiddleware(
        DummyApp(),
        config={"redis": {"host": "localhost", "port": 1, "socket_timeout": 1}},
    )
    assert mw.redis_client is None


def test_rate_limiter_memory_fallback_enforces_limit():
    class DummyApp:
        pass

    mw = RateLimitMiddleware(
        DummyApp(),
        config={
            "redis": {"host": "localhost", "port": 1, "socket_timeout": 1},
            "rate_limiting": {"requests_per_minute": 2, "window": 60},
        },
    )
    now = int(time.time())
    assert mw._check_memory("1.2.3.4", now)[0] is True
    assert mw._check_memory("1.2.3.4", now)[0] is True
    assert mw._check_memory("1.2.3.4", now)[0] is False