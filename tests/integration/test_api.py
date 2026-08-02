import jwt
import pytest
from fastapi.testclient import TestClient

from src.api.app import create_app


@pytest.fixture
def client():
    app = create_app()
    with TestClient(app) as c:
        yield c


@pytest.fixture
def auth_headers():
    token = jwt.encode({"sub": "test-user"}, "dev-secret-change-me", algorithm="HS256")
    return {"Authorization": f"Bearer {token}"}


def test_health_endpoint_is_public(client):
    r = client.get("/api/v1/health")
    assert r.status_code == 200
    assert r.json()["status"] == "healthy"


def test_ready_endpoint_is_public(client):
    r = client.get("/api/v1/ready")
    assert r.status_code == 200


def test_metrics_endpoint_exposes_prometheus_format(client):
    r = client.get("/metrics")
    assert r.status_code == 200
    assert "http_requests_total" in r.text


def test_rag_query_requires_auth(client):
    r = client.post("/api/v1/rag/query", json={"query": "hello"})
    assert r.status_code == 401


def test_rag_query_rejects_malformed_token(client):
    r = client.post(
        "/api/v1/rag/query",
        json={"query": "hello"},
        headers={"Authorization": "Bearer not-a-real-token"},
    )
    assert r.status_code == 401


def test_rag_query_succeeds_with_valid_token(client, auth_headers):
    r = client.post("/api/v1/rag/query", json={"query": "What is the refund policy?"}, headers=auth_headers)
    assert r.status_code == 200
    body = r.json()
    assert "response" in body
    assert body["model_used"] == "local-echo"


def test_rag_query_validates_input(client, auth_headers):
    r = client.post("/api/v1/rag/query", json={"query": ""}, headers=auth_headers)
    assert r.status_code == 422  # pydantic min_length validation


def test_recommendations_endpoint_with_no_trained_model_returns_empty(client, auth_headers):
    r = client.post("/api/v1/recommendations", json={"user_id": "u1", "n": 5}, headers=auth_headers)
    assert r.status_code == 200
    assert r.json()["recommendations"] == []


def test_cors_allows_localhost_with_port(client):
    r = client.options(
        "/api/v1/health",
        headers={"Origin": "http://localhost:3000", "Access-Control-Request-Method": "GET"},
    )
    assert r.headers.get("access-control-allow-origin") == "http://localhost:3000"


def test_unhandled_exception_does_not_leak_internals(client, monkeypatch, auth_headers):
    """The global exception handler must return a generic message, not str(exc)."""

    async def boom(*args, **kwargs):
        raise ValueError("some internal secret detail: db_password=hunter2")

    monkeypatch.setattr(
        "src.api.dependencies.container.Container.rag_pipeline",
        property(lambda self: type("P", (), {"query": boom})()),
        raising=False,
    )
    r = client.post("/api/v1/rag/query", json={"query": "trigger"}, headers=auth_headers)
    assert "hunter2" not in r.text
