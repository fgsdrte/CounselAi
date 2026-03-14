"""
backend/api/routes/documents.py
===============================
Document upload and listing endpoints.

POST /api/documents/upload  — upload a legal document (PDF/DOCX)
GET  /api/documents         — list all uploaded documents
"""

import logging
import os
import tempfile
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, HTTPException, UploadFile, File, status

from config.settings import get_settings
from models.schemas import DocumentInfo, UploadResponse
from services.gcs_storage import get_gcs_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/documents", tags=["documents"])

# Allowed file extensions
_ALLOWED_EXTENSIONS = {".pdf", ".docx", ".doc"}


def _run_ingestion_pipeline() -> None:
    """
    Trigger the data ingestion pipeline as a background task.

    Imports are done inside the function to avoid circular imports
    and heavy module loading at import time.
    """
    try:
        from data_pipeline.index_builder import run_pipeline
        settings = get_settings()

        run_pipeline(
            input_dir="",  # Will scan GCS in production
            project=settings.GCP_PROJECT,
            location=settings.GCP_LOCATION,
            index_id=settings.VERTEX_INDEX_ID,
            endpoint_id=settings.VERTEX_ENDPOINT_ID,
            gcs_bucket=settings.GCS_BUCKET,
        )
        logger.info("Background ingestion pipeline completed successfully")
    except Exception as exc:
        logger.error(f"Background ingestion pipeline failed: {exc}", exc_info=True)


@router.post("/upload", response_model=UploadResponse)
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
) -> UploadResponse:
    """
    Upload a legal document for indexing.

    Accepts PDF, DOCX, and DOC files up to MAX_UPLOAD_SIZE_MB.
    The file is saved to GCS and the ingestion pipeline is triggered
    as a background task.

    Args:
        background_tasks: FastAPI background task manager.
        file:             Uploaded file (multipart/form-data field "file").

    Returns:
        UploadResponse with filename, status, and a human-readable message.
    """
    settings = get_settings()
    max_size_bytes = settings.MAX_UPLOAD_SIZE_MB * 1024 * 1024

    # Validate filename and extension
    if not file.filename:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Filename is required",
        )

    extension = Path(file.filename).suffix.lower()
    if extension not in _ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=(
                f"File type '{extension}' is not allowed. "
                f"Accepted types: {', '.join(_ALLOWED_EXTENSIONS)}"
            ),
        )

    # Read file content and check size
    content = await file.read()
    if len(content) > max_size_bytes:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=(
                f"File size ({len(content) / 1024 / 1024:.1f} MB) exceeds "
                f"maximum allowed size ({settings.MAX_UPLOAD_SIZE_MB} MB)"
            ),
        )

    logger.info(
        f"Document upload — filename={file.filename}, "
        f"size={len(content)} bytes, type={extension}"
    )

    # Save to a temporary file, then upload to GCS
    try:
        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=extension,
        ) as tmp:
            tmp.write(content)
            tmp_path = tmp.name

        gcs = get_gcs_service()
        destination_blob = f"raw/{file.filename}"
        gcs.upload_file(local_path=tmp_path, destination_blob=destination_blob)

    except Exception as exc:
        logger.error(f"Failed to upload to GCS: {exc}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to store document: {exc}",
        )
    finally:
        # Clean up temp file
        if "tmp_path" in locals():
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

    # Trigger ingestion pipeline in the background
    background_tasks.add_task(_run_ingestion_pipeline)

    return UploadResponse(
        filename=file.filename,
        status="processing",
        message=(
            f"Document '{file.filename}' uploaded successfully. "
            f"Indexing has started and will complete shortly."
        ),
    )


@router.get("")
async def list_documents() -> list[DocumentInfo]:
    """
    List all uploaded documents from the GCS 'raw/' prefix.

    Returns:
        List of DocumentInfo objects with filename, upload time, and size.
    """
    try:
        gcs = get_gcs_service()
        settings = get_settings()

        from google.cloud import storage

        client = storage.Client(project=settings.GCP_PROJECT)
        bucket = client.bucket(settings.GCS_BUCKET)

        blobs = client.list_blobs(settings.GCS_BUCKET, prefix="raw/")
        documents: list[DocumentInfo] = []

        for blob in blobs:
            # Skip the prefix-only blob (if it exists)
            filename = blob.name.replace("raw/", "", 1)
            if not filename:
                continue

            documents.append(
                DocumentInfo(
                    filename=filename,
                    uploaded_at=blob.updated.isoformat() if blob.updated else "",
                    size_bytes=blob.size or 0,
                )
            )

        logger.info(f"Listed {len(documents)} documents from GCS")
        return documents

    except Exception as exc:
        logger.error(f"Failed to list documents: {exc}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Unable to list documents: {exc}",
        )
