"""
tests/test_rag.py
=================
Unit tests for the RAG pipeline components:
  - embed_query
  - retrieve
  - reranker
  - pipeline end-to-end (mocked)
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio


# ── Test: embed_query returns a 768-dim vector ───────────────────────────────

def test_embed_query_returns_vector(mock_embedding_model):
    """embed_query should return a list of 768 floats."""
    with patch("vertexai.init"):
        # Clear module-level cache
        import rag.embedder
        rag.embedder._model = None

        with patch(
            "vertexai.language_models.TextEmbeddingModel.from_pretrained",
            return_value=mock_embedding_model,
        ):
            from rag.embedder import embed_query
            rag.embedder._model = mock_embedding_model

            vector = embed_query("What is Section 302 of IPC?")

            assert isinstance(vector, list)
            assert len(vector) == 768
            assert all(isinstance(v, float) for v in vector)


# ── Test: retrieve returns chunks ─────────────────────────────────────────────

def test_retrieve_returns_chunks(mock_vertex, mock_gcs):
    """retrieve should return a list of dicts with expected keys."""
    with patch(
        "services.vertex_search.get_vertex_search_service",
    ) as mock_get_svc:
        mock_svc = MagicMock()
        mock_svc.query.return_value = [
            {"id": "doc1__chunk_0001", "distance": 0.1, "score": 0.91},
            {"id": "doc2__chunk_0002", "distance": 0.3, "score": 0.77},
        ]
        mock_get_svc.return_value = mock_svc

        with patch("services.gcs_storage.get_gcs_service") as mock_get_gcs:
            mock_gcs_svc = MagicMock()
            mock_gcs_svc.list_files.return_value = ["chunk_metadata/chunks_001.json"]
            mock_gcs_svc.download_json.return_value = [
                {
                    "chunk_id": "doc1__chunk_0001",
                    "text": "Section 302 of IPC deals with punishment for murder.",
                    "metadata": {"file": "ipc.pdf", "heading": "Section 302"},
                },
            ]
            mock_get_gcs.return_value = mock_gcs_svc

            from rag.retriever import retrieve

            query_embedding = [0.1] * 768
            results = retrieve(query_embedding, top_k=2)

            assert isinstance(results, list)
            assert len(results) == 2

            # First chunk should have text (found in GCS)
            assert results[0]["chunk_id"] == "doc1__chunk_0001"
            assert results[0]["score"] == 0.91
            assert "Section 302" in results[0]["text"]

            # Second chunk may not have text (not in GCS mock data)
            assert results[1]["chunk_id"] == "doc2__chunk_0002"


# ── Test: reranker sorts by score ─────────────────────────────────────────────

@pytest.mark.asyncio
async def test_reranker_sorts_by_score():
    """rerank should sort chunks by Gemini-assigned score descending."""
    mock_service = MagicMock()
    mock_service.single_response = AsyncMock(
        side_effect=[
            '{"score": 0.3}',   # Low score for chunk 1
            '{"score": 0.9}',   # High score for chunk 2
            '{"score": 0.6}',   # Medium score for chunk 3
        ]
    )

    chunks = [
        {"chunk_id": "c1", "text": "Chunk one text", "score": 0.5, "metadata": {}},
        {"chunk_id": "c2", "text": "Chunk two text", "score": 0.5, "metadata": {}},
        {"chunk_id": "c3", "text": "Chunk three text", "score": 0.5, "metadata": {}},
    ]

    with patch(
        "rag.reranker.get_gemini_service",
        return_value=mock_service,
    ):
        from rag.reranker import rerank
        result = await rerank("test query", chunks, top_n=3)

    assert len(result) == 3
    # Should be sorted: c2 (0.9) > c3 (0.6) > c1 (0.3)
    assert result[0]["chunk_id"] == "c2"
    assert result[0]["rerank_score"] == 0.9
    assert result[1]["chunk_id"] == "c3"
    assert result[2]["chunk_id"] == "c1"


# ── Test: pipeline end-to-end (fully mocked) ─────────────────────────────────

@pytest.mark.asyncio
async def test_pipeline_end_to_end():
    """run_rag_pipeline should stream tokens from embed → retrieve → rerank → generate."""
    # Mock embed_query
    with patch("rag.pipeline.embed_query", return_value=[0.1] * 768):
        # Mock retrieve
        mock_chunks = [
            {
                "chunk_id": "test__chunk_0001",
                "score": 0.9,
                "text": "Test legal text about Section 302.",
                "metadata": {"file": "test.pdf", "heading": "Section 302"},
            },
        ]
        with patch("rag.pipeline.retrieve", return_value=mock_chunks):
            # Mock rerank
            reranked = [{**mock_chunks[0], "rerank_score": 0.95}]

            async def mock_rerank(query, chunks, top_n=3):
                return reranked

            with patch("rag.pipeline.rerank", side_effect=mock_rerank):
                # Mock generate

                async def mock_generate(query, context_chunks):
                    yield "This "
                    yield "is "
                    yield "a test."

                with patch("rag.pipeline.generate", side_effect=mock_generate):
                    from rag.pipeline import run_rag_pipeline

                    tokens = []
                    async for token in run_rag_pipeline("What is Section 302?"):
                        tokens.append(token)

                    combined = "".join(tokens)
                    assert "This is a test." == combined
