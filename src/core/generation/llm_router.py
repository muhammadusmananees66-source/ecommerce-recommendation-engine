"""Routes a query to the best available LLM given cost/latency/quality tradeoffs."""

import logging
from typing import Any

logger = logging.getLogger(__name__)

# Illustrative capability table. In a real deployment these numbers should
# come from your own measured latency/cost, not vendor marketing figures,
# and should be refreshed periodically -- treat this as a config default,
# not a hardcoded truth.
DEFAULT_MODEL_CAPABILITIES = {
    "local-echo": {"cost": 0.0, "latency": 1, "quality": 0.10, "max_tokens": 4096},
    "gpt-4o": {"cost": 0.005, "latency": 600, "quality": 0.94, "max_tokens": 128_000},
    "gpt-4o-mini": {"cost": 0.0006, "latency": 300, "quality": 0.85, "max_tokens": 128_000},
    "claude-sonnet-5": {"cost": 0.003, "latency": 500, "quality": 0.95, "max_tokens": 200_000},
    "gemini-2.0-flash": {"cost": 0.0004, "latency": 250, "quality": 0.87, "max_tokens": 1_000_000},
}


class LLMRouter:
    def __init__(self, config: dict[str, Any]):
        self.capabilities = {**DEFAULT_MODEL_CAPABILITIES, **config.get("model_capabilities", {})}
        self.available = config.get("available_models", list(self.capabilities.keys()))
        self.fallback_chain = config.get("fallback_chain", self.available)
        self._initialized = False

    async def initialize(self) -> None:
        self._initialized = True

    def route(self, query: str, context: dict[str, Any]) -> dict[str, Any]:
        if not self._initialized:
            raise RuntimeError("LLMRouter not initialized")

        complexity = self._estimate_complexity(query)
        scored = []
        for model in self.available:
            caps = self.capabilities.get(model)
            if not caps:
                continue
            quality_term = caps["quality"] ** (1 + complexity)
            latency_term = 1.0 / (caps["latency"] + 1)
            cost_term = 1.0 / (caps["cost"] + 1e-4)
            score = 0.5 * quality_term + 0.3 * latency_term + 0.2 * min(cost_term, 1.0)
            scored.append((model, score, caps))

        if not scored:
            raise RuntimeError("No available LLM models configured")

        scored.sort(key=lambda x: x[1], reverse=True)
        model, score, caps = scored[0]
        return {"model": model, "score": score, "max_tokens": caps["max_tokens"]}

    def get_fallback_chain(self, primary: str) -> list[str]:
        chain = [m for m in self.fallback_chain if m != primary]
        return chain

    @staticmethod
    def _estimate_complexity(query: str) -> float:
        words = len(query.split())
        has_question = "?" in query
        return min(1.0, 0.2 + 0.3 * (words / 20) + (0.2 if has_question else 0.0))