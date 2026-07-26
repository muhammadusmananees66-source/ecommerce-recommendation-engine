"""Prometheus HTTP metrics middleware."""

import time

from fastapi import Request
from prometheus_client import Counter, Histogram
from starlette.middleware.base import BaseHTTPMiddleware

HTTP_REQUESTS = Counter(
    "http_requests_total", "Total HTTP requests", ["method", "endpoint", "status"]
)
HTTP_LATENCY = Histogram(
    "http_request_duration_seconds", "HTTP request latency", ["method", "endpoint"]
)


class MetricsMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        start = time.time()
        response = await call_next(request)
        duration = time.time() - start

        endpoint = request.url.path
        HTTP_REQUESTS.labels(method=request.method, endpoint=endpoint, status=str(response.status_code)).inc()
        HTTP_LATENCY.labels(method=request.method, endpoint=endpoint).observe(duration)
        return response