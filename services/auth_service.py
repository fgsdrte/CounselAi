"""
services/auth_service.py
========================
Firebase Authentication helpers for CounselAI.
- init_firebase()  — one-time initialisation of the Firebase Admin SDK
- verify_token()   — verify a Firebase ID token (JWT) and return decoded claims

Usage:
    from services.auth_service import init_firebase, verify_token
    init_firebase()
    claims = verify_token(id_token_string)
"""

import logging
from typing import Optional

import firebase_admin
from firebase_admin import auth, credentials

from config.settings import get_settings

logger = logging.getLogger(__name__)

_firebase_app: Optional[firebase_admin.App] = None


def init_firebase() -> None:
    """
    Initialise the Firebase Admin SDK app (idempotent — safe to call multiple times).
    Uses Application Default Credentials (ADC) on GCP, or
    GOOGLE_APPLICATION_CREDENTIALS env var locally.
    """
    global _firebase_app

    if _firebase_app is not None:
        logger.debug("Firebase app already initialised — skipping")
        return

    settings = get_settings()

    try:
        # Use default credentials (ADC); project is set explicitly
        cred = credentials.ApplicationDefault()
        _firebase_app = firebase_admin.initialize_app(
            cred,
            options={"projectId": settings.FIREBASE_PROJECT_ID},
        )
        logger.info(
            f"Firebase Admin SDK initialised — project={settings.FIREBASE_PROJECT_ID}"
        )
    except ValueError:
        # Already initialised (e.g. in tests)
        _firebase_app = firebase_admin.get_app()
        logger.debug("Firebase app was already initialised externally")


def verify_token(id_token: str) -> dict:
    """
    Verify a Firebase ID token and return the decoded claims.

    Args:
        id_token: Firebase JWT ID token string from the client.

    Returns:
        Dict of decoded token claims (uid, email, email_verified, etc.).

    Raises:
        firebase_admin.auth.InvalidIdTokenError: If the token is invalid.
        firebase_admin.auth.ExpiredIdTokenError: If the token has expired.
        firebase_admin.auth.RevokedIdTokenError: If the token has been revoked.
    """
    logger.info("Verifying Firebase ID token")

    try:
        decoded = auth.verify_id_token(id_token)
        logger.info(f"Token verified — uid={decoded.get('uid')}")
        return decoded
    except Exception as exc:
        logger.warning(f"Token verification failed: {exc}")
        raise
