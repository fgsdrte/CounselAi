"""
backend/api/routes/health.py
============================
Health check endpoint for CounselAI API.
Used by Cloud Run, load balancers, and monitoring.

GET /api/health → { status, version, timestamp }
"""

from datetime import datetime, timezone

from fastapi import APIRouter

router = APIRouter(prefix="/api", tags=["health"])


@router.get("/health")
async def health_check() -> dict:
    """
    Health check endpoint.

    Returns:
        JSON with service status, version, and current timestamp.
    """
    return {
        "status": "ok",
        "version": "1.0.0",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
