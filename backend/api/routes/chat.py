"""
backend/api/routes/chat.py
==========================
Chat endpoint that streams RAG-powered responses as Server-Sent Events.

POST /api/chat  →  SSE stream of generated tokens
"""

import logging

from fastapi import APIRouter, HTTPException, status
from fastapi.responses import StreamingResponse

from models.schemas import ChatRequest
from rag.pipeline import run_rag_pipeline

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["chat"])


async def _sse_generator(query: str):
    """
    Wrap the RAG pipeline generator into SSE-formatted events.

    Each token is sent as: data: {token}\\n\\n
    A final event signals completion: data: [DONE]\\n\\n
    """
    try:
        async for token in run_rag_pipeline(query):
            yield f"data: {token}\n\n"
        yield "data: [DONE]\n\n"
    except Exception as exc:
        logger.error(f"SSE stream error: {exc}", exc_info=True)
        yield f"data: [ERROR] {exc}\n\n"
        yield "data: [DONE]\n\n"


@router.post("/chat")
async def chat(request: ChatRequest) -> StreamingResponse:
    """
    Process a user's legal query through the RAG pipeline and
    stream the response as Server-Sent Events.

    Request body:
        - message: str (1–2000 chars, required)
        - session_id: str (optional)

    Returns:
        StreamingResponse with media_type text/event-stream.
    """
    if not request.message or not request.message.strip():
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Message must not be empty",
        )

    if len(request.message) > 2000:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Message must not exceed 2000 characters",
        )

    logger.info(
        f"Chat request — message_len={len(request.message)}, "
        f"session_id={request.session_id}"
    )

    return StreamingResponse(
        content=_sse_generator(request.message),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
