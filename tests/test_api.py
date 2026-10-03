"""Unit tests for FastAPI endpoints."""

from fastapi.testclient import TestClient
from app.api import app

client = TestClient(app)


def test_health_endpoint():
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "models_dir" in data


def test_models_endpoint():
    response = client.get("/api/models")
    assert response.status_code == 200
    data = response.json()
    assert "models" in data
    assert "default_tau" in data
    assert isinstance(data["models"], list)
    assert 0.0 <= data["default_tau"] <= 1.0


def test_samples_endpoint():
    response = client.get("/api/samples")
    assert response.status_code == 200
    data = response.json()
    assert "samples" in data
    assert "count" in data
    assert isinstance(data["samples"], list)
