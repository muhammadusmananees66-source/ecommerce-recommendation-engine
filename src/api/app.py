"""
FastAPI application factory.

Bugs fixed from earlier iterations, both verified with tests
(tests/integration/test_api.py):
- CORS: allow_origin_regex now actually matches http://localhost:<any port>
  (previous versions used `:*` which is not valid "any digits" regex syntax
  and silently rejected every localhost origin).
- The global exception handler logs full details server-side but returns
  only a generic message + request_id to the client -- no raw exception
  text, stack trace, or internal file paths leak to callers.
"""

import logging
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from starlette.responses import Response

from src.api.dependencies.container import Container
from src.api.middleware.auth import AuthMiddleware
from src.api.middleware.logging import LoggingMiddleware
from src.api.middleware.metrics import MetricsMiddleware
from src.api.middleware.rate_limit import RateLimitMiddleware
from src.api.routes import rag, recommendations
from src.utils.config import load_config

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting application...")
    config = load_config(app.state.config_path if hasattr(app.state, "config_path") else None)
    container = Container(config)
    await container.init()
    app.state.container = container
    app.state.config = config
    app.state.start_time = time.time()
    yield
    logger.info("Shutting down application...")
    await container.shutdown()


# def create_app(config_path: str = None) -> FastAPI:
def create_app(config_path: str | None = None) -> FastAPI:
    app = FastAPI(
        title="RAG + Recommendation Engine",
        description="RAG-based question answering combined with hybrid recommendations",
        version="1.0.0",
        lifespan=lifespan,
        docs_url="/api/docs",
        redoc_url="/api/redoc",
    )
    app.state.config_path = config_path

    config = load_config(config_path)
    cors_config = config.get("cors", {})

    app.add_middleware(
        CORSMiddleware,
        # Fixed: `:*` is not valid "zero-or-more-digits" regex; `(:\d+)?`
        # correctly matches an optional port, so localhost with any port
        # (or no port) actually matches, verified in test_api.py.
        allow_origin_regex=cors_config.get(
            "allow_origin_regex", r"https://.*\.example\.com|http://localhost(:\d+)?"
        ),
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.add_middleware(MetricsMiddleware)
    app.add_middleware(LoggingMiddleware)
    app.add_middleware(RateLimitMiddleware, config=config)
    app.add_middleware(AuthMiddleware, config=config)

    @app.get("/api/v1/health")
    async def health(request: Request):
        return {"status": "healthy", "uptime_s": time.time() - request.app.state.start_time, "version": "1.0.0"}

    @app.get("/api/v1/ready")
    async def ready(request: Request):
        container = getattr(request.app.state, "container", None)
        if container is None or container.rag_pipeline is None:
            return JSONResponse(status_code=503, content={"status": "not ready"})
        return {"status": "ready"}

    @app.get("/metrics")
    async def metrics():
        return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)

    @app.exception_handler(Exception)
    # async def global_exception_handler(request: Request, exc: Exception):
    #     request_id = getattr(request.state, "request_id", str(uuid.uuid4()))
    #     logger.error("Unhandled exception [%s]: %s", request_id, exc, exc_info=True)
    #     return JSONResponse(
    async def global_exception_handler(request: Request, exc: Exception):
        request_id = getattr(request.state, "request_id", str(uuid.uuid4()))
        logger.exception("Unhandled exception [%s]: %s", request_id, exc)
        return JSONResponse(
            status_code=500,
            content={
                "error": "internal_server_error",
                "request_id": request_id,
                "message": "An unexpected error occurred. Please try again or contact support with this request ID.",
            },
        )

    app.include_router(rag.router)
    app.include_router(recommendations.router)

    return app