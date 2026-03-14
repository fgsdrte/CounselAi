"""
services/gcs_storage.py
=======================
Google Cloud Storage helper for uploading, downloading, and listing files
in the CounselAI document bucket.

Usage:
    from services.gcs_storage import get_gcs_service
    gcs = get_gcs_service()
    uri = gcs.upload_file("/tmp/doc.pdf", "raw/doc.pdf")
"""

import json
import logging
import time
from typing import Optional

from google.cloud import storage
from tenacity import retry, stop_after_attempt, wait_exponential

from config.settings import get_settings

logger = logging.getLogger(__name__)


class GCSStorageService:
    """Google Cloud Storage operations for CounselAI."""

    def __init__(self) -> None:
        """Initialise the GCS client and bucket reference."""
        settings = get_settings()
        self._bucket_name = settings.GCS_BUCKET
        self._client = storage.Client(project=settings.GCP_PROJECT)
        self._bucket = self._client.bucket(self._bucket_name)
        logger.info(f"GCSStorageService initialised — bucket={self._bucket_name}")

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        reraise=True,
    )
    def upload_file(self, local_path: str, destination_blob: str) -> str:
        """
        Upload a local file to GCS.

        Args:
            local_path:        Absolute path to the local file.
            destination_blob:  Destination blob name (e.g. "raw/document.pdf").

        Returns:
            GCS URI string (gs://bucket/blob).
        """
        start = time.perf_counter()
        logger.info(f"GCS upload: {local_path} → gs://{self._bucket_name}/{destination_blob}")

        blob = self._bucket.blob(destination_blob)
        blob.upload_from_filename(local_path)

        uri = f"gs://{self._bucket_name}/{destination_blob}"
        elapsed = time.perf_counter() - start
        logger.info(f"GCS upload complete in {elapsed:.2f}s → {uri}")
        return uri

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        reraise=True,
    )
    def download_json(self, blob_name: str) -> dict:
        """
        Download and parse a JSON blob from GCS.

        Args:
            blob_name: Full blob path inside the bucket.

        Returns:
            Parsed JSON as a Python dict.
        """
        start = time.perf_counter()
        logger.info(f"GCS download JSON: gs://{self._bucket_name}/{blob_name}")

        blob = self._bucket.blob(blob_name)
        content = blob.download_as_text(encoding="utf-8")
        data = json.loads(content)

        elapsed = time.perf_counter() - start
        logger.info(f"GCS download JSON complete in {elapsed:.2f}s")
        return data

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        reraise=True,
    )
    def list_files(self, prefix: str) -> list[str]:
        """
        List all blob names under the given prefix.

        Args:
            prefix: GCS prefix to list (e.g. "raw/").

        Returns:
            List of blob name strings.
        """
        start = time.perf_counter()
        logger.info(f"GCS list files: gs://{self._bucket_name}/{prefix}")

        blobs = self._client.list_blobs(self._bucket_name, prefix=prefix)
        file_names = [blob.name for blob in blobs]

        elapsed = time.perf_counter() - start
        logger.info(
            f"GCS list files complete in {elapsed:.2f}s — {len(file_names)} files found"
        )
        return file_names


# ── Singleton ─────────────────────────────────────────────────────────────────

_instance: Optional[GCSStorageService] = None


def get_gcs_service() -> GCSStorageService:
    """Return a singleton GCSStorageService instance."""
    global _instance
    if _instance is None:
        _instance = GCSStorageService()
    return _instance
