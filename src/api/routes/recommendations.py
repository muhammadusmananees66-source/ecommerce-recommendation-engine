import logging
import uuid

from fastapi import APIRouter, HTTPException, Request

from src.api.schemas.request import RecommendationRequest
from src.api.schemas.response import RecommendationResponse

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/recommendations", tags=["recommendations"])


@router.post("", response_model=RecommendationResponse)
async def get_recommendations(request: Request, body: RecommendationRequest) -> RecommendationResponse:
    container = request.app.state.container
    predictor = container.predictor

    try:
        recommendations = await predictor.get_recommendations(user_id=body.user_id, n=body.n)
    except Exception as e:
        logger.error("Recommendation request failed: %s", e, exc_info=True)
        raise HTTPException(status_code=502, detail="Recommendation service failed")

    return RecommendationResponse(
        request_id=getattr(request.state, "request_id", str(uuid.uuid4())),
        user_id=body.user_id,
        recommendations=recommendations,
        total_count=len(recommendations),
        model_version=predictor.model_version,
    )