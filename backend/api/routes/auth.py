"""
backend/api/routes/auth.py
==========================
Firebase authentication endpoints and dependency for CounselAI API.

POST /api/auth/verify  — verify a Firebase ID token
get_current_user       — FastAPI dependency for protected routes
"""

import logging

from fastapi import APIRouter, Header, HTTPException, status

from models.schemas import AuthRequest, AuthResponse
from services.auth_service import verify_token

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/auth", tags=["auth"])


@router.post("/verify", response_model=AuthResponse)
async def verify_firebase_token(request: AuthRequest) -> AuthResponse:
    """
    Verify a Firebase ID token and return the decoded user info.

    Args:
        request: AuthRequest containing the Firebase JWT id_token.

    Returns:
        AuthResponse with uid, email, and valid flag.
    """
    try:
        decoded = verify_token(request.id_token)
        return AuthResponse(
            uid=decoded.get("uid", ""),
            email=decoded.get("email", ""),
            valid=True,
        )
    except Exception as exc:
        logger.warning(f"Token verification failed: {exc}")
        return AuthResponse(
            uid="",
            email="",
            valid=False,
        )


async def get_current_user(
    authorization: str = Header(..., alias="Authorization"),
) -> dict:
    """
    FastAPI dependency that extracts and verifies a Firebase token
    from the Authorization header.

    Expected header format: "Bearer <id_token>"

    Args:
        authorization: The Authorization header value.

    Returns:
        Decoded token claims dict (uid, email, etc.).

    Raises:
        HTTPException 401: If the token is missing, malformed, or invalid.
    """
    if not authorization.startswith("Bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authorization header must start with 'Bearer '",
        )

    token = authorization[len("Bearer "):]

    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token is missing from Authorization header",
        )

    try:
        decoded = verify_token(token)
        return decoded
    except Exception as exc:
        logger.warning(f"Auth dependency — token invalid: {exc}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Invalid or expired token: {exc}",
        )
