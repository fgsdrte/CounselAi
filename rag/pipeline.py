"""
rag/pipeline.py
===============
Orchestrates the full RAG pipeline:
    embed_query → retrieve → rerank → generate

Accepts a user query string and returns a streaming async generator
that yields response tokens.

Usage:
    from rag.pipeline import run_rag_pipeline
    async for token in run_rag_pipeline("What is Section 302 IPC?"):
        print(token, end="")
"""

import logging
import time
from typing import AsyncGenerator

from rag.embedder import embed_query
from rag.generator import generate
from rag.reranker import rerank
from rag.retriever import retrieve

logger = logging.getLogger(__name__)


async def run_rag_pipeline(
    query: str,
) -> AsyncGenerator[str, None]:
    """
    Full RAG pipeline: embed → retrieve → rerank → generate (streaming).

    Each step is logged with precise timing. On failure at any step,
    yields a user-friendly error message instead of crashing.

    Args:
        query: The user's legal question.

    Yields:
        Text tokens from the generated response.
    """
    pipeline_start = time.perf_counter()
    logger.info(f"RAG pipeline started — query: {query[:100]}...")

    # ── Step 1: Embed the query ───────────────────────────────────────────
    try:
        step_start = time.perf_counter()
        query_embedding = embed_query(query)
        elapsed = time.perf_counter() - step_start
        logger.info(f"[RAG] Step 1/4 — embed_query completed in {elapsed:.2f}s")
    except Exception as exc:
        logger.error(f"[RAG] Step 1/4 — embed_query FAILED: {exc}", exc_info=True)
        yield (
            "I'm sorry, I encountered an error while processing your query. "
            "Please try again in a moment."
        )
        return

    # ── Step 2: Retrieve similar chunks ───────────────────────────────────
    try:
        step_start = time.perf_counter()
        chunks = retrieve(query_embedding=query_embedding, top_k=5)
        elapsed = time.perf_counter() - step_start
        logger.info(
            f"[RAG] Step 2/4 — retrieve completed in {elapsed:.2f}s "
            f"({len(chunks)} chunks)"
        )
    except Exception as exc:
        logger.error(f"[RAG] Step 2/4 — retrieve FAILED: {exc}", exc_info=True)
        yield (
            "I'm sorry, I was unable to search the legal document database. "
            "Please try again shortly."
        )
        return

    if not chunks:
        logger.warning("[RAG] Step 2/4 — no chunks retrieved")
        yield (
            "I could not find any relevant legal documents for your query. "
            "Please try rephrasing your question or uploading relevant documents."
        )
        return

    # ── Step 3: Rerank chunks ─────────────────────────────────────────────
    try:
        step_start = time.perf_counter()
        top_chunks = await rerank(query=query, chunks=chunks, top_n=3)
        elapsed = time.perf_counter() - step_start
        logger.info(
            f"[RAG] Step 3/4 — rerank completed in {elapsed:.2f}s "
            f"({len(top_chunks)} top chunks)"
        )
    except Exception as exc:
        logger.error(f"[RAG] Step 3/4 — rerank FAILED: {exc}", exc_info=True)
        # Fallback: use original chunks without reranking
        top_chunks = chunks[:3]
        logger.info("[RAG] Step 3/4 — falling back to top-3 unreranked chunks")

    # ── Step 4: Generate streaming response ───────────────────────────────
    try:
        step_start = time.perf_counter()
        token_count = 0
        async for token in generate(query=query, context_chunks=top_chunks):
            token_count += 1
            yield token
        elapsed = time.perf_counter() - step_start
        logger.info(
            f"[RAG] Step 4/4 — generate completed in {elapsed:.2f}s "
            f"({token_count} tokens streamed)"
        )
    except Exception as exc:
        logger.error(f"[RAG] Step 4/4 — generate FAILED: {exc}", exc_info=True)
        yield (
            "I'm sorry, I encountered an error generating the response. "
            "Please try again."
        )

    pipeline_elapsed = time.perf_counter() - pipeline_start
    logger.info(f"RAG pipeline completed in {pipeline_elapsed:.2f}s total")
