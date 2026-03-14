"""
rag/reranker.py
===============
Re-ranks retrieved chunks using Gemini 1.5 Pro to score relevance
against the original user query. Falls back gracefully if the LLM
call fails (returns original order truncated to top_n).

Usage:
    from rag.reranker import rerank
    top_chunks = rerank(query, retrieved_chunks, top_n=3)
"""

import json
import logging
import time
from typing import Optional

from config.prompts import RERANK_PROMPT_TEMPLATE
from services.gemini_service import get_gemini_service

logger = logging.getLogger(__name__)


async def rerank(
    query: str,
    chunks: list[dict],
    top_n: int = 3,
) -> list[dict]:
    """
    Re-rank retrieved chunks by relevance using Gemini 1.5 Pro.

    For each chunk, asks Gemini to score its relevance (0.0–1.0) to the query.
    Sorts by score descending and returns the top_n results.

    If the Gemini call fails for any chunk, that chunk keeps its original
    retrieval score. If ALL calls fail, returns the first top_n chunks unchanged.

    Args:
        query:  The user's original legal question.
        chunks: List of chunk dicts from the retriever (must have 'text' key).
        top_n:  Number of top-scoring chunks to return.

    Returns:
        List of top_n chunk dicts, sorted by reranker score (highest first).
        Each dict gets an added 'rerank_score' key.
    """
    start = time.perf_counter()
    logger.info(f"rerank — scoring {len(chunks)} chunks, will return top {top_n}")

    if not chunks:
        return []

    gemini = get_gemini_service()
    scored_chunks: list[dict] = []

    for i, chunk in enumerate(chunks):
        chunk_text = chunk.get("text", "")
        if not chunk_text:
            # No text → keep original score
            chunk_copy = {**chunk, "rerank_score": chunk.get("score", 0.0)}
            scored_chunks.append(chunk_copy)
            continue

        prompt = RERANK_PROMPT_TEMPLATE.format(
            query=query,
            chunk_text=chunk_text[:2000],  # Truncate to avoid token limits
        )

        try:
            response_text = await gemini.single_response(
                prompt=prompt,
                system="You are a relevance scorer. Return only valid JSON.",
            )

            # Parse the score from the JSON response
            # Handle cases where Gemini wraps JSON in markdown code blocks
            cleaned = response_text.strip()
            if cleaned.startswith("```"):
                # Remove markdown code fence
                lines = cleaned.split("\n")
                cleaned = "\n".join(
                    line for line in lines
                    if not line.strip().startswith("```")
                )

            parsed = json.loads(cleaned)
            score = float(parsed.get("score", 0.0))
            score = max(0.0, min(1.0, score))  # Clamp to [0, 1]

            chunk_copy = {**chunk, "rerank_score": score}
            scored_chunks.append(chunk_copy)
            logger.debug(f"rerank — chunk {i}: score={score:.3f}")

        except Exception as exc:
            logger.warning(
                f"rerank — Gemini scoring failed for chunk {i}: {exc}. "
                f"Falling back to retrieval score."
            )
            chunk_copy = {**chunk, "rerank_score": chunk.get("score", 0.0)}
            scored_chunks.append(chunk_copy)

    # Sort by rerank_score descending
    scored_chunks.sort(key=lambda c: c["rerank_score"], reverse=True)

    top_results = scored_chunks[:top_n]

    elapsed = time.perf_counter() - start
    logger.info(
        f"rerank completed in {elapsed:.2f}s — "
        f"top scores: {[c['rerank_score'] for c in top_results]}"
    )

    return top_results
