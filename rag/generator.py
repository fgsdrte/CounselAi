"""
rag/generator.py
================
Generates a streaming response from Gemini 1.5 Pro, grounded in the
retrieved and reranked context chunks.

Usage:
    from rag.generator import generate
    async for token in generate(query, context_chunks):
        print(token, end="")
"""

import logging
import time
from typing import AsyncGenerator

from config.prompts import SYSTEM_PROMPT
from services.gemini_service import get_gemini_service

logger = logging.getLogger(__name__)


def _build_context(chunks: list[dict]) -> str:
    """
    Build a formatted context string from reranked chunks.

    Each chunk is labelled with its source file and heading for
    provenance tracking.

    Args:
        chunks: List of chunk dicts with 'text' and 'metadata' keys.

    Returns:
        Formatted context string ready for inclusion in the prompt.
    """
    parts: list[str] = []

    for i, chunk in enumerate(chunks, 1):
        metadata = chunk.get("metadata", {})
        source_file = metadata.get("file", "Unknown source")
        heading = metadata.get("heading", "No heading")
        text = chunk.get("text", "")

        parts.append(
            f"--- Context Chunk {i} ---\n"
            f"[Source: {source_file} | {heading}]\n"
            f"{text}\n"
        )

    return "\n".join(parts)


async def generate(
    query: str,
    context_chunks: list[dict],
) -> AsyncGenerator[str, None]:
    """
    Generate a streaming response to the user's legal query using Gemini 1.5 Pro.

    Builds a grounded prompt from the context chunks and the system prompt,
    then streams tokens as they arrive from the model.

    Args:
        query:           The user's original legal question.
        context_chunks:  List of reranked chunk dicts with text and metadata.

    Yields:
        Individual text tokens from the Gemini response stream.
    """
    start = time.perf_counter()
    logger.info(
        f"generate — streaming response for query ({len(query)} chars) "
        f"with {len(context_chunks)} context chunks"
    )

    context_str = _build_context(context_chunks)

    user_prompt = (
        f"### Provided Legal Context\n\n"
        f"{context_str}\n\n"
        f"### User Question\n\n"
        f"{query}\n\n"
        f"### Instructions\n\n"
        f"Answer the user's question based ONLY on the provided context above. "
        f"Cite specific sources. If the context does not contain the answer, "
        f"clearly state that you could not find the information.\n"
    )

    gemini = get_gemini_service()

    try:
        async for token in gemini.stream_response(
            prompt=user_prompt,
            system=SYSTEM_PROMPT,
        ):
            yield token

    except Exception as exc:
        logger.error(f"generate — streaming failed: {exc}", exc_info=True)
        yield (
            "\n\n**Error:** I encountered an issue generating a response. "
            "Please try again or rephrase your question."
        )

    finally:
        elapsed = time.perf_counter() - start
        logger.info(f"generate completed in {elapsed:.2f}s")
