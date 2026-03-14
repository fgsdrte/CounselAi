"""
backend/main.py
===============
FastAPI application entry point for CounselAI API.

Configures CORS, includes all route modules, initialises Vertex AI
and Firebase on startup, and provides a global exception handler.

Run locally:
    uvicorn backend.main:app --reload --port 8000
"""

import logging
import traceback

import vertexai
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from config.logging import setup_logging
from config.settings import get_settings
from services.auth_service import init_firebase

# Import route modules
from backend.api.routes import auth, chat, documents, health

logger = logging.getLogger(__name__)

# ── Create FastAPI app ────────────────────────────────────────────────────────

app = FastAPI(
    title="CounselAI API",
    version="1.0.0",
    description="Indian Legal RAG Chatbot — powered by Vertex AI & Gemini",
)


# ── CORS middleware ───────────────────────────────────────────────────────────

@app.on_event("startup")
async def _configure_cors() -> None:
    """Configure CORS with allowed origins from settings (runs at startup)."""
    pass  # CORS is added below synchronously so it's available immediately


settings = get_settings()
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Include routers ──────────────────────────────────────────────────────────

app.include_router(health.router)
app.include_router(chat.router)
app.include_router(documents.router)
app.include_router(auth.router)


# ── Startup event ─────────────────────────────────────────────────────────────

@app.on_event("startup")
async def startup_event() -> None:
    """
    Initialise external services on application startup:
      1. Configure structured logging
      2. Initialise Vertex AI SDK
      3. Initialise Firebase Admin SDK
    """
    # 1. Logging
    setup_logging()
    logger.info("CounselAI API starting up...")

    # 2. Vertex AI
    try:
        vertexai.init(
            project=settings.GCP_PROJECT,
            location=settings.GCP_LOCATION,
        )
        logger.info(
            f"Vertex AI initialised — project={settings.GCP_PROJECT}, "
            f"location={settings.GCP_LOCATION}"
        )
    except Exception as exc:
        logger.error(f"Failed to initialise Vertex AI: {exc}", exc_info=True)

    # 3. Firebase
    try:
        init_firebase()
        logger.info("Firebase Admin SDK initialised")
    except Exception as exc:
        logger.error(f"Failed to initialise Firebase: {exc}", exc_info=True)

    logger.info("CounselAI API startup complete ✓")


# ── Global exception handler ─────────────────────────────────────────────────

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """
    Catch-all exception handler that returns a structured JSON error
    instead of an HTML 500 page.
    """
    logger.error(
        f"Unhandled exception on {request.method} {request.url}: {exc}",
        exc_info=True,
    )
    return JSONResponse(
        status_code=500,
        content={
            "error": str(exc),
            "status": 500,
        },
    )
