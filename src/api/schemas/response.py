from typing import Any, Dict, List

from pydantic import BaseModel


class RAGResponse(BaseModel):
    request_id: str
    query: str
    response: str
    confidence_score: float
    groundedness_score: float
    sources: List[Dict[str, Any]]
    latency_ms: float
    model_used: str
    cached: bool


class RecommendationResponse(BaseModel):
    request_id: str
    user_id: str
    recommendations: List[Dict[str, Any]]
    total_count: int
    model_version: str