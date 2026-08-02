import logging
import uuid

from fastapi import APIRouter, HTTPException, Request

from src.api.schemas.request import RAGRequest
from src.api.schemas.response import RAGResponse

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/rag", tags=["rag"])


@router.post("/query", response_model=RAGResponse)
async def rag_query(request: Request, body: RAGRequest) -> RAGResponse:
    container = request.app.state.container
    pipeline = container.rag_pipeline

    user_context = dict(body.context or {})
    if body.user_id:
        user_context["user_id"] = body.user_id

    try:
        result = await pipeline.query(
            query=body.query,
            user_context=user_context,
            max_docs=body.max_docs,
            temperature=body.temperature,
        )
    except TimeoutError as e:
        raise HTTPException(status_code=504, detail=str(e))
    except Exception as e:
        logger.error("RAG query failed: %s", e, exc_info=True)
        raise HTTPException(status_code=502, detail="Upstream RAG pipeline failed")

    return RAGResponse(
        request_id=getattr(request.state, "request_id", str(uuid.uuid4())),
        query=result.query,
        response=result.generated_response,
        confidence_score=result.confidence_score,
        groundedness_score=result.groundedness_score,
        sources=result.sources,
        latency_ms=result.latency_ms,
        model_used=result.model_used,
        cached=result.cached,
    )
