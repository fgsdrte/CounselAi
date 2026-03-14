"""
config/settings.py
==================
Application settings powered by pydantic-settings.
All values are read from environment variables / .env file.
Never hardcode GCP project IDs, credentials, or bucket names.

Usage:
    from config.settings import get_settings
    settings = get_settings()
"""

from functools import lru_cache
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """CounselAI application settings — all sourced from env vars."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
    )

    # ── Google Cloud ──────────────────────────────────────────
    GCP_PROJECT: str = Field(..., description="GCP project ID")
    GCP_LOCATION: str = Field(default="us-central1", description="GCP region")

    # ── Vertex AI Vector Search ───────────────────────────────
    VERTEX_INDEX_ID: str = Field(..., description="Vertex AI Vector Search index resource ID")
    VERTEX_ENDPOINT_ID: str = Field(..., description="Vertex AI index endpoint resource ID")
    VERTEX_INDEX_DEPLOYED_ID: str = Field(
        default="",
        description="Deployed index ID on the endpoint",
    )

    # ── Cloud Storage ─────────────────────────────────────────
    GCS_BUCKET: str = Field(..., description="GCS bucket for document storage")

    # ── Gemini ────────────────────────────────────────────────
    GEMINI_MODEL: str = Field(default="gemini-1.5-pro", description="Gemini model name")

    # ── Firebase Auth ─────────────────────────────────────────
    FIREBASE_PROJECT_ID: str = Field(..., description="Firebase project ID")

    # ── Application ───────────────────────────────────────────
    APP_ENV: str = Field(default="development", description="Application environment")
    LOG_LEVEL: str = Field(default="INFO", description="Logging level")
    MAX_UPLOAD_SIZE_MB: int = Field(default=20, description="Max upload size in MB")
    ALLOWED_ORIGINS: list[str] = Field(
        default=["http://localhost:3000", "http://localhost:8000"],
        description="CORS allowed origins",
    )


@lru_cache()
def get_settings() -> Settings:
    """Return cached singleton Settings instance."""
    return Settings()  # type: ignore[call-arg]
