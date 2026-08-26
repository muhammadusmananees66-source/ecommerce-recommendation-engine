"""
Rate limiting middleware, Redis-backed with an in-memory fallback.

Bug fixed from earlier iterations: the fallback path is now verified to
actually engage when Redis is unreachable (see tests/unit/test_rate_limit.py,
which points this middleware at a closed port and confirms the memory path
is used, not a silently-broken Redis client kept truthy).
"""

import logging
import time
from typing import Any

from fastapi import HTTPException, Request, status
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger(__name__)

EXEMPT_PATHS = ("/api/v1/health", "/api/v1/ready", "/metrics")


class RateLimitMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, config: dict[str, Any] | None = None):
        super().__init__(app)
        config = config or {}
        self.rate_limit = config.get("rate_limiting", {}).get("requests_per_minute", 100)
        self.window = config.get("rate_limiting", {}).get("window", 60)
        self.max_tracked_clients = config.get("rate_limiting", {}).get("max_tracked_clients", 20_000)

        self.redis_client = None
        self._memory_store: dict[str, list] = {}

        redis_config = config.get("redis", {})
        if redis_config.get("enabled", True):
            self.redis_client = self._try_connect_redis(redis_config)

    def _try_connect_redis(self, redis_config: dict[str, Any]):
        try:
            import redis

            client = redis.Redis(
                host=redis_config.get("host", "localhost"),
                port=redis_config.get("port", 6379),
                db=redis_config.get("db", 0),
                decode_responses=True,
                socket_timeout=redis_config.get("socket_timeout", 2),
                socket_connect_timeout=redis_config.get("socket_timeout", 2),
            )
            client.ping()
            logger.info("Rate limiter connected to Redis")
            return client
        except Exception as e:
            logger.warning("Redis unavailable for rate limiting (%s); using in-memory fallback", e)
            return None  # explicit: caller must not hold a half-broken client

    async def dispatch(self, request: Request, call_next):
        if any(request.url.path.startswith(p) for p in EXEMPT_PATHS):
            return await call_next(request)

        client_ip = request.client.host if request.client else "unknown"
        now = int(time.time())

        if self.redis_client is not None:
            allowed, remaining = self._check_redis(client_ip)
        else:
            allowed, remaining = self._check_memory(client_ip, now)

        if not allowed:
            logger.warning("Rate limit exceeded for %s", client_ip)
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail=f"Rate limit exceeded: {self.rate_limit} requests per {self.window}s",
                headers={
                    "X-RateLimit-Limit": str(self.rate_limit),
                    "X-RateLimit-Remaining": "0",
                    "X-RateLimit-Reset": str(now + self.window),
                },
            )

        response = await call_next(request)
        response.headers["X-RateLimit-Limit"] = str(self.rate_limit)
        response.headers["X-RateLimit-Remaining"] = str(remaining)
        return response

    def _check_redis(self, client_ip: str) -> tuple[bool, int]:
        try:
            key = f"rate_limit:{client_ip}"
            pipe = self.redis_client.pipeline()
            pipe.incr(key)
            pipe.expire(key, self.window)
            count, _ = pipe.execute()
            remaining = max(0, self.rate_limit - count)
            return count <= self.rate_limit, remaining
        except Exception as e:
            logger.error("Redis rate check failed mid-request (%s); using memory for this request", e)
            return self._check_memory(client_ip, int(time.time()))

    def _check_memory(self, client_ip: str, now: int) -> tuple[bool, int]:
        bucket = self._memory_store.setdefault(client_ip, [])
        bucket[:] = [t for t in bucket if now - t < self.window]

        if len(bucket) >= self.rate_limit:
            return False, 0

        bucket.append(now)

        if len(self._memory_store) > self.max_tracked_clients:
            self._memory_store = {k: v for k, v in self._memory_store.items() if v}

        return True, max(0, self.rate_limit - len(bucket))