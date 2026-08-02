"""Feature store for serving-time user features (used by the Predictor)."""

import json
import logging
from typing import Any, Dict

logger = logging.getLogger(__name__)

DEFAULT_FEATURES = {
    "user_rating_mean": 0.0,
    "user_rating_std": 0.0,
    "user_rating_count": 0,
    "user_price_mean": 0.0,
}


class FeatureStore:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
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
            logger.warning("Redis unavailable for feature store (%s); using default features", e)
            return None

    def get_user_features(self, user_id: str) -> Dict[str, Any]:
        if self.redis_client is not None:
            try:
                data = self.redis_client.hgetall(f"user_features:{user_id}")
                if data:
                    return {
                        k: (float(v) if k != "user_preference_vector" else json.loads(v))
                        for k, v in data.items()
                    }
            except Exception as e:
                logger.error("Redis feature read failed (%s); using defaults", e)

        return dict(DEFAULT_FEATURES)

    def set_user_features(self, user_id: str, features: Dict[str, Any]) -> None:
        if self.redis_client is None:
            logger.warning("Feature store has no Redis connection; set_user_features is a no-op")
            return
        try:
            key = f"user_features:{user_id}"
            payload = {k: (json.dumps(v) if isinstance(v, (dict, list)) else v) for k, v in features.items()}
            self.redis_client.hset(key, mapping=payload)
        except Exception as e:
            logger.error("Redis feature write failed: %s", e)
