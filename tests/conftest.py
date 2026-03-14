"""
tests/conftest.py
=================
Pytest fixtures that mock all GCP services so tests never make real API calls.
"""

import os
from typing import AsyncGenerator, Generator
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient


# ── Ensure test environment variables are set ─────────────────────────────────

@pytest.fixture(autouse=True)
def mock_settings(monkeypatch):
    """
    Override all GCP-related settings with test values so no real
    credentials or project IDs are needed.
    """
    monkeypatch.setenv("GCP_PROJECT", "test-project")
    monkeypatch.setenv("GCP_LOCATION", "us-central1")
    monkeypatch.setenv("VERTEX_INDEX_ID", "projects/test/locations/us-central1/indexes/test-idx")
    monkeypatch.setenv("VERTEX_ENDPOINT_ID", "projects/test/locations/us-central1/indexEndpoints/test-ep")
    monkeypatch.setenv("VERTEX_INDEX_DEPLOYED_ID", "test-deployed-idx")
    monkeypatch.setenv("GCS_BUCKET", "test-bucket")
    monkeypatch.setenv("GEMINI_MODEL", "gemini-1.5-pro")
    monkeypatch.setenv("FIREBASE_PROJECT_ID", "test-firebase-project")
    monkeypatch.setenv("APP_ENV", "testing")
    monkeypatch.setenv("LOG_LEVEL", "DEBUG")

    # Clear the lru_cache so new env values are picked up
    from config.settings import get_settings
    get_settings.cache_clear()

    yield

    get_settings.cache_clear()


# ── Mock Vertex AI MatchingEngineIndexEndpoint ────────────────────────────────

@pytest.fixture
def mock_vertex():
    """
    Mock Vertex AI Vector Search find_neighbors to return fake results
    without making any real API calls.
    """
    mock_neighbor_1 = MagicMock()
    mock_neighbor_1.id = "doc1__chunk_0001"
    mock_neighbor_1.distance = 0.1

    mock_neighbor_2 = MagicMock()
    mock_neighbor_2.id = "doc2__chunk_0002"
    mock_neighbor_2.distance = 0.3

    mock_endpoint = MagicMock()
    mock_endpoint.find_neighbors.return_value = [
        [mock_neighbor_1, mock_neighbor_2],
    ]

    with patch(
        "google.cloud.aiplatform.MatchingEngineIndexEndpoint",
        return_value=mock_endpoint,
    ) as mock_cls:
        yield mock_endpoint


# ── Mock Gemini GenerativeModel ───────────────────────────────────────────────

@pytest.fixture
def mock_gemini():
    """
    Mock Vertex AI Gemini model for both streaming and non-streaming calls.
    """
    # Mock for single (non-streaming) response
    mock_response = MagicMock()
    mock_response.text = '{"score": 0.85}'

    # Mock for streaming response
    mock_stream_chunk = MagicMock()
    mock_stream_chunk.text = "This is a test response token."

    # Async iterator for streaming
    async def mock_stream():
        yield mock_stream_chunk

    mock_model = MagicMock()
    mock_model.generate_content_async = AsyncMock(return_value=mock_response)

    with patch(
        "vertexai.generative_models.GenerativeModel",
        return_value=mock_model,
    ) as mock_cls:
        yield mock_model


# ── Mock Firebase ─────────────────────────────────────────────────────────────

@pytest.fixture
def mock_firebase():
    """Mock Firebase Admin SDK token verification."""
    with patch("firebase_admin.auth.verify_id_token") as mock_verify:
        mock_verify.return_value = {
            "uid": "test-uid-123",
            "email": "test@example.com",
            "email_verified": True,
        }
        yield mock_verify


# ── Mock GCS ─────────────────────────────────────────────────────────────────

@pytest.fixture
def mock_gcs():
    """Mock Google Cloud Storage client."""
    mock_blob = MagicMock()
    mock_blob.download_as_text.return_value = '[{"chunk_id": "doc1__chunk_0001", "text": "Test chunk text", "metadata": {"file": "test.pdf", "heading": "Section 1"}}]'
    mock_blob.upload_from_filename.return_value = None
    mock_blob.upload_from_string.return_value = None
    mock_blob.name = "raw/test.pdf"
    mock_blob.updated = None
    mock_blob.size = 1024

    mock_bucket = MagicMock()
    mock_bucket.blob.return_value = mock_blob

    mock_client = MagicMock()
    mock_client.bucket.return_value = mock_bucket
    mock_client.list_blobs.return_value = [mock_blob]

    with patch("google.cloud.storage.Client", return_value=mock_client) as mock_cls:
        yield mock_client


# ── Mock Embedding Model ─────────────────────────────────────────────────────

@pytest.fixture
def mock_embedding_model():
    """Mock Vertex AI TextEmbeddingModel."""
    mock_embedding = MagicMock()
    mock_embedding.values = [0.1] * 768

    mock_model = MagicMock()
    mock_model.get_embeddings.return_value = [mock_embedding]

    with patch(
        "vertexai.language_models.TextEmbeddingModel.from_pretrained",
        return_value=mock_model,
    ) as mock_cls:
        yield mock_model


# ── Test client ───────────────────────────────────────────────────────────────

@pytest.fixture
def test_client(mock_vertex, mock_gemini, mock_gcs, mock_firebase) -> TestClient:
    """
    FastAPI TestClient with all GCP services mocked.

    Patches are applied before importing the app to ensure mocks
    are in place during module initialisation.
    """
    with patch("vertexai.init"):
        with patch("services.auth_service.init_firebase"):
            from backend.main import app
            client = TestClient(app)
            yield client
