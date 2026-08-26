from typing import Any

from pydantic import BaseModel, Field


class RAGRequest(BaseModel):
    query: str = Field(..., min_length=1, description="User query")
    user_id: str | None = None
    max_docs: int = Field(10, ge=1, le=50)
    temperature: float = Field(0.7, ge=0.0, le=2.0)
    context: dict[str, Any] | None = None


class RecommendationRequest(BaseModel):
    user_id: str = Field(..., min_length=1)
    n: int = Field(10, ge=1, le=100)