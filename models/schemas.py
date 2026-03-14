"""
models/schemas.py
=================
Pydantic models for request/response validation across CounselAI API.
All models use str_strip_whitespace for clean input handling.
"""

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


# ── Chat ──────────────────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    """Incoming chat message from the user."""

    model_config = ConfigDict(str_strip_whitespace=True)

    message: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="User's legal question or query",
    )
    session_id: Optional[str] = Field(
        default=None,
        description="Optional session ID for conversation continuity",
    )


class SourceChunk(BaseModel):
    """A single source chunk returned alongside a chat response."""

    model_config = ConfigDict(str_strip_whitespace=True)

    chunk_id: str = Field(..., description="Unique identifier of the chunk")
    file: str = Field(default="", description="Source document filename")
    heading: str = Field(default="", description="Section heading in the source")
    score: float = Field(default=0.0, description="Relevance score (0.0–1.0)")


class ChatResponse(BaseModel):
    """Full (non-streaming) chat response."""

    model_config = ConfigDict(str_strip_whitespace=True)

    response: str = Field(..., description="Generated answer text")
    sources: list[SourceChunk] = Field(
        default_factory=list,
        description="Source chunks used to generate the answer",
    )


# ── Document upload ───────────────────────────────────────────────────────────

class UploadResponse(BaseModel):
    """Response after a document upload request."""

    model_config = ConfigDict(str_strip_whitespace=True)

    filename: str = Field(..., description="Name of the uploaded file")
    status: str = Field(default="processing", description="Processing status")
    message: str = Field(default="", description="Human-readable status message")


class DocumentInfo(BaseModel):
    """Metadata about a stored document."""

    model_config = ConfigDict(str_strip_whitespace=True)

    filename: str = Field(..., description="Document filename")
    uploaded_at: str = Field(
        default="",
        description="ISO-8601 upload timestamp",
    )
    size_bytes: int = Field(default=0, description="File size in bytes")


# ── Auth ──────────────────────────────────────────────────────────────────────

class AuthRequest(BaseModel):
    """Firebase ID token verification request."""

    model_config = ConfigDict(str_strip_whitespace=True)

    id_token: str = Field(..., description="Firebase ID token (JWT)")


class AuthResponse(BaseModel):
    """Result of Firebase token verification."""

    model_config = ConfigDict(str_strip_whitespace=True)

    uid: str = Field(default="", description="Firebase user UID")
    email: str = Field(default="", description="User's email address")
    valid: bool = Field(default=False, description="Whether the token is valid")
