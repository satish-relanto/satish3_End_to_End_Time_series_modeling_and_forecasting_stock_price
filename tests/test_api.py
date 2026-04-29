"""API smoke tests for the FastAPI application."""

from fastapi.testclient import TestClient

from api.main import app


def test_health_endpoint():
    # TestClient exercises the ASGI app in-process without starting uvicorn.
    client = TestClient(app)
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_root_redirects_to_docs():
    client = TestClient(app)
    response = client.get("/", follow_redirects=False)

    assert response.status_code == 307
    assert response.headers["location"] == "/docs"
