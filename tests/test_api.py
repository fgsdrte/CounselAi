"""
tests/test_api.py
=================
Integration tests for CounselAI API endpoints:
  - Health check
  - Chat streaming
  - Document upload validation
  - Auth token verification
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ── Test: GET /api/health ─────────────────────────────────────────────────────

def test_health_endpoint(test_client):
    """Health endpoint should return status ok with version and timestamp."""
    response = test_client.get("/api/health")

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["version"] == "1.0.0"
    assert "timestamp" in data


# ── Test: POST /api/chat streams response ─────────────────────────────────────

def test_chat_endpoint_streams(test_client):
    """Chat endpoint should return a streaming response with SSE events."""
    # Mock the RAG pipeline to yield test tokens
    async def mock_pipeline(query):
        yield "Hello "
        yield "world!"

    with patch("backend.api.routes.chat.run_rag_pipeline", side_effect=mock_pipeline):
        response = test_client.post(
            "/api/chat",
            json={"message": "What is Section 302 of IPC?"},
        )

        assert response.status_code == 200
        assert "text/event-stream" in response.headers.get("content-type", "")

        # Check that SSE events are present in the response body
        body = response.text
        assert "data:" in body
        assert "data: [DONE]" in body


# ── Test: POST /api/documents/upload rejects non-PDF ─────────────────────────

def test_upload_rejects_non_pdf(test_client):
    """Upload endpoint should reject files that are not PDF/DOCX/DOC."""
    # Create a fake .txt file
    response = test_client.post(
        "/api/documents/upload",
        files={"file": ("test.txt", b"Hello world", "text/plain")},
    )

    assert response.status_code == 422
    data = response.json()
    assert "not allowed" in data["detail"].lower() or ".txt" in data["detail"]


# ── Test: POST /api/auth/verify with invalid token ───────────────────────────

def test_auth_verify_invalid_token(test_client):
    """Auth verify should return valid=false for an invalid token."""
    with patch(
        "services.auth_service.verify_token",
        side_effect=Exception("Invalid token"),
    ):
        response = test_client.post(
            "/api/auth/verify",
            json={"id_token": "invalid-token-xyz"},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["valid"] is False


# ── Test: POST /api/chat rejects empty message ───────────────────────────────

def test_chat_rejects_empty_message(test_client):
    """Chat endpoint should reject an empty message."""
    response = test_client.post(
        "/api/chat",
        json={"message": ""},
    )

    assert response.status_code == 422


# ── Test: POST /api/chat rejects oversized message ───────────────────────────

def test_chat_rejects_oversized_message(test_client):
    """Chat endpoint should reject messages over 2000 characters."""
    long_message = "x" * 2001
    response = test_client.post(
        "/api/chat",
        json={"message": long_message},
    )

    assert response.status_code == 422
