"""Generic Redis-backed cache manager used by the API layer (distinct from the
RAG semantic cache, which matches by embedding similarity rather than exact key)."""

import json
import logging
from typing import Any

logger = logging.getLogger(__name__)


class CacheManager:
    def __init__(self, redis_client, config: dict[str, Any]):
        self.redis = redis_client
        self.ttl = config.get("ttl", 300)
        self.prefix = config.get("prefix", "cache:")

    async def get(self, key: str) -> Any | None:
        if self.redis is None:
            return None
        try:
            value = self.redis.get(f"{self.prefix}{key}")
            return json.loads(value) if value else None
        except Exception as e:
            logger.warning("Cache get failed for key '%s': %s", key, e)
            return None

    async def set(self, key: str, value: Any, ttl: int | None = None) -> None:
        if self.redis is None:
            return
        try:
            self.redis.setex(f"{self.prefix}{key}", ttl or self.ttl, json.dumps(value, default=str))
        except Exception as e:
            logger.warning("Cache set failed for key '%s': %s", key, e)

    async def delete(self, key: str) -> None:
        if self.redis is None:
            return
        try:
            self.redis.delete(f"{self.prefix}{key}")
        except Exception as e:
            logger.warning("Cache delete failed for key '%s': %s", key, e)