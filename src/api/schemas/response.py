from typing import Any

from pydantic import BaseModel


class RAGResponse(BaseModel):
    request_id: str
    query: str
    response: str
    confidence_score: float
    groundedness_score: float
    sources: list[dict[str, Any]]
    latency_ms: float
    model_used: str
    cached: bool


class RecommendationResponse(BaseModel):
    request_id: str
    user_id: str
    recommendations: list[dict[str, Any]]
    total_count: int
    model_version: str