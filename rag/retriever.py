"""
rag/retriever.py
================
Retrieves the most similar document chunks for a given query embedding
using Vertex AI Vector Search, then enriches results with full chunk text
and metadata from GCS.

Usage:
    from rag.retriever import retrieve
    chunks = retrieve(query_embedding, top_k=5)
"""

import json
import logging
import time
from typing import Optional

from config.settings import get_settings
from services.gcs_storage import get_gcs_service
from services.vertex_search import get_vertex_search_service

logger = logging.getLogger(__name__)


def _fetch_chunk_from_gcs(chunk_id: str) -> Optional[dict]:
    """
    Attempt to find a chunk's text and metadata from GCS chunk_metadata/ blobs.

    The index_builder stores chunks as JSON arrays in blobs under the
    'chunk_metadata/' prefix. We scan available blobs to find the matching
    chunk_id.

    Args:
        chunk_id: The unique identifier of the chunk.

    Returns:
        Dict with 'text' and 'metadata' keys, or None if not found.
    """
    try:
        gcs = get_gcs_service()
        settings = get_settings()
        prefix = "chunk_metadata/"

        blob_names = gcs.list_files(prefix)

        for blob_name in blob_names:
            if not blob_name.endswith(".json"):
                continue
            try:
                data = gcs.download_json(blob_name)
                # data is a list of chunk records
                if isinstance(data, list):
                    for record in data:
                        if record.get("chunk_id") == chunk_id:
                            return {
                                "text": record.get("text", ""),
                                "metadata": record.get("metadata", {}),
                            }
            except Exception:
                # Skip individual blob failures
                continue

    except Exception as exc:
        logger.warning(f"GCS fetch failed for chunk {chunk_id}: {exc}")

    return None


def retrieve(
    query_embedding: list[float],
    top_k: int = 5,
) -> list[dict]:
    """
    Retrieve the top-k most relevant document chunks for a query embedding.

    Uses Vertex AI Vector Search for similarity search, then fetches full
    chunk text from GCS. Falls back gracefully if GCS fetch fails.

    Args:
        query_embedding: 768-dim float vector from embed_query().
        top_k:           Number of chunks to retrieve.

    Returns:
        List of dicts, each containing:
            - chunk_id:  str
            - score:     float (similarity score)
            - text:      str   (full chunk text, empty if GCS fetch failed)
            - metadata:  dict  (source file, heading, etc.)
    """
    start = time.perf_counter()
    logger.info(f"retrieve — searching for top {top_k} chunks")

    # Step 1: Similarity search via Vertex AI
    search_service = get_vertex_search_service()
    raw_results = search_service.query(embedding=query_embedding, top_k=top_k)

    elapsed_search = time.perf_counter() - start
    logger.info(
        f"retrieve — vector search returned {len(raw_results)} results "
        f"in {elapsed_search:.2f}s"
    )

    # Step 2: Enrich with text + metadata from GCS
    enriched: list[dict] = []

    for result in raw_results:
        chunk_id = result["id"]
        score = result.get("score", 0.0)

        chunk_data = _fetch_chunk_from_gcs(chunk_id)

        if chunk_data:
            enriched.append({
                "chunk_id": chunk_id,
                "score": score,
                "text": chunk_data["text"],
                "metadata": chunk_data["metadata"],
            })
        else:
            # Graceful fallback: return chunk_id and score without text
            logger.warning(
                f"retrieve — could not fetch text for chunk {chunk_id}, "
                f"returning id and score only"
            )
            enriched.append({
                "chunk_id": chunk_id,
                "score": score,
                "text": "",
                "metadata": {},
            })

    elapsed_total = time.perf_counter() - start
    logger.info(f"retrieve completed in {elapsed_total:.2f}s — {len(enriched)} chunks")

    return enriched
