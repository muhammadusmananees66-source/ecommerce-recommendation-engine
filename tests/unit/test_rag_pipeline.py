import pytest

from src.core.rag.pipeline import RAGPipeline


def _pipeline_config():
    return {
        "timeout": 10,
        "llm": {"available_models": ["local-echo"], "fallback_chain": ["local-echo"]},
        "generator": {"api_keys": {}, "timeout": 5},
        "cache": {"ttl_seconds": 3600, "similarity_threshold": 0.9},
    }


@pytest.fixture
async def seeded_pipeline():
    pipeline = RAGPipeline(_pipeline_config())
    await pipeline.initialize()
    docs = [
        {"id": "d1", "text": "Our refund policy allows returns within 30 days."},
        {"id": "d2", "text": "Shipping takes 3-5 business days."},
        {"id": "d3", "text": "Refunds go back to the original payment method."},
    ]
    vecs = pipeline.embedder.encode([d["text"] for d in docs])
    pipeline.retriever.vector_store.add_documents(vecs, [{"id": d["id"], "text": d["text"]} for d in docs])
    yield pipeline
    await pipeline.close()


@pytest.mark.asyncio
async def test_query_retrieves_relevant_documents(seeded_pipeline):
    pipeline = seeded_pipeline
    result = await pipeline.query("What is your refund policy?")
    assert result.retrieved_docs
    assert any("refund" in d["text"].lower() for d in result.retrieved_docs)


@pytest.mark.asyncio
async def test_groundedness_reflects_actual_retrieved_text(seeded_pipeline):
    pipeline = seeded_pipeline
    result = await pipeline.query("What is your refund policy?")
    assert result.groundedness_score > 0.0


@pytest.mark.asyncio
async def test_empty_store_returns_zero_confidence_not_fabricated_docs(seeded_pipeline):
    pipeline = RAGPipeline(_pipeline_config())
    await pipeline.initialize()
    result = await pipeline.query("anything at all")
    assert result.retrieved_docs == []
    assert result.confidence_score == 0.0
    await pipeline.close()


@pytest.mark.asyncio
async def test_repeated_query_hits_semantic_cache(seeded_pipeline):
    pipeline = seeded_pipeline
    first = await pipeline.query("What is your refund policy?")
    second = await pipeline.query("What is your refund policy?")
    assert first.cached is False
    assert second.cached is True


@pytest.mark.asyncio
async def test_query_raises_before_initialize():
    pipeline = RAGPipeline(_pipeline_config())
    with pytest.raises(RuntimeError):
        await pipeline.query("test")