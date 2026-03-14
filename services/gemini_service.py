"""
services/gemini_service.py
==========================
Wrapper around Vertex AI Gemini 1.5 Pro for both streaming and single-shot
text generation. Safety settings are set to BLOCK_NONE for legal content.

Usage:
    from services.gemini_service import get_gemini_service
    service = get_gemini_service()
    async for token in service.stream_response(prompt, system):
        print(token, end="")
"""

import logging
import time
from typing import AsyncGenerator, Optional

import vertexai
from vertexai.generative_models import (
    GenerationConfig,
    GenerativeModel,
    HarmBlockThreshold,
    HarmCategory,
    Part,
)

from config.settings import get_settings

logger = logging.getLogger(__name__)

# ── Safety settings: BLOCK_NONE for all categories (legal content needs this)
_SAFETY_SETTINGS = {
    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
}

# ── Default generation config
_GENERATION_CONFIG = GenerationConfig(
    temperature=0.2,
    top_p=0.8,
    max_output_tokens=1024,
)


class GeminiService:
    """Vertex AI Gemini 1.5 Pro wrapper for CounselAI."""

    def __init__(self, model_name: Optional[str] = None) -> None:
        """
        Initialise the Gemini generative model.

        Args:
            model_name: Override model name. Defaults to settings.GEMINI_MODEL.
        """
        settings = get_settings()
        self._model_name = model_name or settings.GEMINI_MODEL
        self._model = GenerativeModel(
            self._model_name,
            safety_settings=_SAFETY_SETTINGS,
            generation_config=_GENERATION_CONFIG,
        )
        logger.info(f"GeminiService initialised with model={self._model_name}")

    async def stream_response(
        self,
        prompt: str,
        system: str = "",
    ) -> AsyncGenerator[str, None]:
        """
        Stream tokens from Gemini for the given prompt.

        Args:
            prompt:  User prompt / conversation content.
            system:  System instruction string.

        Yields:
            Individual text tokens as they arrive.
        """
        start = time.perf_counter()
        logger.info("GeminiService.stream_response — starting stream")

        try:
            # Build a model instance with the system instruction for this call
            model = GenerativeModel(
                self._model_name,
                system_instruction=system if system else None,
                safety_settings=_SAFETY_SETTINGS,
                generation_config=_GENERATION_CONFIG,
            )

            response = await model.generate_content_async(
                [Part.from_text(prompt)],
                stream=True,
            )

            async for chunk in response:
                if chunk.text:
                    yield chunk.text

        except Exception as exc:
            logger.error(f"GeminiService.stream_response failed: {exc}", exc_info=True)
            yield f"\n\n[Error generating response: {exc}]"
        finally:
            elapsed = time.perf_counter() - start
            logger.info(f"GeminiService.stream_response completed in {elapsed:.2f}s")

    async def single_response(
        self,
        prompt: str,
        system: str = "",
    ) -> str:
        """
        Get a single (non-streaming) response from Gemini.

        Args:
            prompt:  User prompt.
            system:  System instruction string.

        Returns:
            Full response text.
        """
        start = time.perf_counter()
        logger.info("GeminiService.single_response — calling Gemini")

        try:
            model = GenerativeModel(
                self._model_name,
                system_instruction=system if system else None,
                safety_settings=_SAFETY_SETTINGS,
                generation_config=_GENERATION_CONFIG,
            )

            response = await model.generate_content_async(
                [Part.from_text(prompt)],
                stream=False,
            )
            result = response.text
            logger.info(
                f"GeminiService.single_response completed in "
                f"{time.perf_counter() - start:.2f}s  "
                f"({len(result)} chars)"
            )
            return result

        except Exception as exc:
            logger.error(f"GeminiService.single_response failed: {exc}", exc_info=True)
            raise


# ── Singleton ─────────────────────────────────────────────────────────────────

_instance: Optional[GeminiService] = None


def get_gemini_service() -> GeminiService:
    """Return a singleton GeminiService instance."""
    global _instance
    if _instance is None:
        _instance = GeminiService()
    return _instance
