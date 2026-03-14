"""
config/logging.py
=================
Configures structured logging for CounselAI.
- JSON format in production  (APP_ENV=production)
- Pretty human-readable format in development

Usage:
    from config.logging import setup_logging
    setup_logging()
"""

import logging
import logging.config
import sys
from typing import Optional


def setup_logging(log_level: Optional[str] = None, app_env: Optional[str] = None) -> None:
    """
    Configure application-wide logging.

    Args:
        log_level: Override log level (e.g. 'DEBUG', 'INFO'). Defaults to settings.LOG_LEVEL.
        app_env:   Override environment. Defaults to settings.APP_ENV.
    """
    # Late import to avoid circular dependency during startup
    from config.settings import get_settings

    settings = get_settings()
    level = (log_level or settings.LOG_LEVEL).upper()
    env = app_env or settings.APP_ENV

    if env == "production":
        # JSON structured logging for Cloud Logging / log aggregation
        formatter_class = "logging.Formatter"
        fmt = '{"time": "%(asctime)s", "level": "%(levelname)s", "logger": "%(name)s", "message": "%(message)s"}'
        datefmt = "%Y-%m-%dT%H:%M:%S%z"
    else:
        # Pretty coloured output for local development
        formatter_class = "logging.Formatter"
        fmt = "%(asctime)s  %(levelname)-8s  [%(name)s]  %(message)s"
        datefmt = "%H:%M:%S"

    config = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "default": {
                "class": formatter_class,
                "format": fmt,
                "datefmt": datefmt,
            },
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "formatter": "default",
                "stream": "ext://sys.stdout",
            },
        },
        "root": {
            "level": level,
            "handlers": ["console"],
        },
        # Silence noisy third-party loggers
        "loggers": {
            "google": {"level": "WARNING"},
            "urllib3": {"level": "WARNING"},
            "httpx": {"level": "WARNING"},
            "grpc": {"level": "WARNING"},
        },
    }

    logging.config.dictConfig(config)
    logging.getLogger(__name__).info(
        f"Logging configured — level={level}, env={env}"
    )
