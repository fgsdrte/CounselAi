"""
services/vertex_search.py
=========================
Client for Vertex AI Vector Search (Matching Engine).
Wraps MatchingEngineIndexEndpoint.find_neighbors for similarity search.

Usage:
    from services.vertex_search import get_vertex_search_service
    service = get_vertex_search_service()
    results = service.query(embedding_vector, top_k=5)
"""

import logging
import time
from typing import Optional

from google.cloud import aiplatform
from tenacity import retry, stop_after_attempt, wait_exponential

from config.settings import get_settings

logger = logging.getLogger(__name__)


class VertexSearchService:
    """Vertex AI Vector Search (Matching Engine) query client."""

    def __init__(self) -> None:
        """Initialise the Matching Engine index endpoint client."""
        settings = get_settings()
        self._endpoint_id = settings.VERTEX_ENDPOINT_ID
        self._deployed_index_id = settings.VERTEX_INDEX_DEPLOYED_ID
        self._endpoint = aiplatform.MatchingEngineIndexEndpoint(
            index_endpoint_name=self._endpoint_id,
        )
        logger.info(
            f"VertexSearchService initialised — endpoint={self._endpoint_id}"
        )

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        reraise=True,
    )
    def query(
        self,
        embedding: list[float],
        top_k: int = 5,
    ) -> list[dict]:
        """
        Find nearest neighbours for the given embedding vector.

        Args:
            embedding: 768-dim float vector from text-embedding-004.
            top_k:     Number of nearest neighbours to return.

        Returns:
            List of dicts with keys: id, distance, score.
            Score is computed as 1 / (1 + distance) for easier interpretation.
        """
        start = time.perf_counter()
        logger.info(f"VertexSearchService.query — top_k={top_k}")

        try:
            response = self._endpoint.find_neighbors(
                deployed_index_id=self._deployed_index_id,
                queries=[embedding],
                num_neighbors=top_k,
            )

            results: list[dict] = []
            if response and len(response) > 0:
                for neighbor in response[0]:
                    distance = neighbor.distance
                    results.append({
                        "id": neighbor.id,
                        "distance": distance,
                        "score": 1.0 / (1.0 + distance) if distance >= 0 else 0.0,
                    })

            elapsed = time.perf_counter() - start
            logger.info(
                f"VertexSearchService.query returned {len(results)} results "
                f"in {elapsed:.2f}s"
            )
            return results

        except Exception as exc:
            elapsed = time.perf_counter() - start
            logger.error(
                f"VertexSearchService.query failed after {elapsed:.2f}s: {exc}",
                exc_info=True,
            )
            raise


# ── Singleton ─────────────────────────────────────────────────────────────────

_instance: Optional[VertexSearchService] = None


def get_vertex_search_service() -> VertexSearchService:
    """Return a singleton VertexSearchService instance."""
    global _instance
    if _instance is None:
        _instance = VertexSearchService()
    return _instance
