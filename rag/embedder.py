"""
rag/embedder.py
===============
Embeds a text query into a 768-dimensional float vector using
Vertex AI's text-embedding-004 model.

Usage:
    from rag.embedder import embed_query
    vector = embed_query("What is Section 302 of IPC?")
"""

import logging
import time
from typing import Optional

import vertexai
from tenacity import retry, stop_after_attempt, wait_exponential
from vertexai.language_models import TextEmbeddingModel

from config.settings import get_settings

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

EMBEDDING_MODEL: str = "text-embedding-004"
EMBEDDING_DIM: int = 768

# ── Module-level model cache ─────────────────────────────────────────────────

_model: Optional[TextEmbeddingModel] = None


def _get_model() -> TextEmbeddingModel:
    """Lazily initialise and cache the embedding model."""
    global _model
    if _model is None:
        settings = get_settings()
        vertexai.init(project=settings.GCP_PROJECT, location=settings.GCP_LOCATION)
        _model = TextEmbeddingModel.from_pretrained(EMBEDDING_MODEL)
        logger.info(f"Embedding model loaded: {EMBEDDING_MODEL}")
    return _model


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    reraise=True,
)
def embed_query(text: str) -> list[float]:
    """
    Embed a single text string into a 768-dimensional float vector.

    Args:
        text: The query text to embed (e.g. a user's legal question).

    Returns:
        A list of 768 floats representing the text embedding.

    Raises:
        Exception: Re-raised after 3 retry attempts if the API call fails.
    """
    start = time.perf_counter()
    logger.info(f"embed_query — embedding text ({len(text)} chars)")

    model = _get_model()
    embeddings = model.get_embeddings([text])

    if not embeddings or not embeddings[0].values:
        raise ValueError("Embedding API returned empty result")

    vector = embeddings[0].values
    elapsed = time.perf_counter() - start
    logger.info(
        f"embed_query completed in {elapsed:.2f}s — "
        f"vector dim={len(vector)}"
    )

    return vector
