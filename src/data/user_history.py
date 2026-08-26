"""
Tracks which items a user has interacted with, used to build the content-based
half of the hybrid recommender's score at serving time.

Redis is the primary store (durable, shared across replicas); falls back to
an in-process dict if Redis is unreachable, which is honestly a single-pod,
non-durable fallback -- acceptable for degraded-mode serving, not a
substitute for Redis actually being up in production.
"""

import logging
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


class UserHistoryService:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self._memory: Dict[str, List[Dict]] = {}
        self.redis_client = self._try_connect(config)

    def _try_connect(self, config: Dict[str, Any]):
        try:
            import redis

            client = redis.Redis(
                host=config.get("redis_host", "localhost"),
                port=config.get("redis_port", 6379),
                db=config.get("redis_db", 0),
                decode_responses=True,
                socket_timeout=config.get("socket_timeout", 2),
                socket_connect_timeout=config.get("socket_timeout", 2),
            )
            client.ping()
            return client
        except Exception as e:
            logger.warning("Redis unavailable for user history (%s); using in-memory fallback", e)
            return None

    def get_user_items(self, user_id: str) -> List[str]:
        if self.redis_client is not None:
            try:
                return list(self.redis_client.smembers(f"user_items:{user_id}"))
            except Exception as e:
                logger.error("Redis read failed (%s); falling back to memory for this call", e)

        return [entry["item_id"] for entry in self._memory.get(user_id, [])]

    def add_user_item(self, user_id: str, item_id: str, rating: float = 1.0) -> None:
        if self.redis_client is not None:
            try:
                key = f"user_items:{user_id}"
                self.redis_client.sadd(key, item_id)
                self.redis_client.expire(key, 60 * 60 * 24 * 30)
            except Exception as e:
                logger.error("Redis write failed (%s); recording in memory only", e)

        self._memory.setdefault(user_id, []).append({"item_id": item_id, "rating": rating})