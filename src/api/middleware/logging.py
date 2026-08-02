"""Structured logging middleware with a correlation ID threaded through each request."""

import logging
import time
import uuid

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger(__name__)


class LoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        request_id = str(uuid.uuid4())
        request.state.request_id = request_id
        start = time.time()

        logger.info(
            "request.start",
            extra={"request_id": request_id, "method": request.method, "path": request.url.path},
        )

        response = await call_next(request)

        duration_ms = (time.time() - start) * 1000
        response.headers["X-Request-ID"] = request_id
        logger.info(
            "request.end",
            extra={
                "request_id": request_id,
                "method": request.method,
                "path": request.url.path,
                "status_code": response.status_code,
                "duration_ms": round(duration_ms, 2),
            },
        )
        return response
