"""
JWT bearer-token authentication middleware.

Bugs fixed from earlier iterations:
- Returns a JSONResponse on auth failure (Starlette's BaseHTTPMiddleware
  requires dispatch() to return a Response; returning a raw HTTPException
  instance is invalid and breaks at the ASGI layer).
- All required typing imports are present (Dict, Optional) -- an earlier
  version of this file used Dict without importing it, which is a NameError
  at import time, i.e. the whole app fails to start.
"""

import logging
from typing import Any

import jwt
from fastapi import HTTPException, Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger(__name__)

PUBLIC_PATHS = (
    "/api/v1/health",
    "/api/v1/ready",
    "/api/docs",
    "/api/redoc",
    "/openapi.json",
    "/metrics",
)


class AuthMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, config: dict[str, Any] | None = None):
        super().__init__(app)
        config = config or {}
        auth_config = config.get("auth", {})
        self.secret_key = auth_config.get("secret_key", "dev-secret-change-me")
        self.algorithm = auth_config.get("algorithm", "HS256")
        self.enabled = auth_config.get("enabled", True)

    async def dispatch(self, request: Request, call_next):
        if not self.enabled or any(request.url.path.startswith(p) for p in PUBLIC_PATHS):
            return await call_next(request)

        try:
            token = self._extract_token(request)
            payload = self._verify_token(token)
            request.state.user = payload
        except HTTPException as e:
            return JSONResponse(status_code=e.status_code, content={"detail": e.detail})

        return await call_next(request)

    @staticmethod
    def _extract_token(request: Request) -> str:
        auth_header = request.headers.get("Authorization")
        if not auth_header:
            raise HTTPException(status_code=401, detail="Authorization header required")

        parts = auth_header.split()
        if len(parts) != 2 or parts[0].lower() != "bearer":
            raise HTTPException(status_code=401, detail="Invalid authorization format, expected 'Bearer <token>'")

        return parts[1]

    def _verify_token(self, token: str) -> dict[str, Any]:
        try:
            return jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
        except jwt.ExpiredSignatureError:
            raise HTTPException(status_code=401, detail="Token expired")
        except jwt.InvalidTokenError:
            raise HTTPException(status_code=401, detail="Invalid token")