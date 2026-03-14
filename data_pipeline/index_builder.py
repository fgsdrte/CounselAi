"""
data_pipeline/index_builder.py
===============================
Orchestrates the full ingestion pipeline:
  1. Parse PDF / DOCX  →  raw pages / sections
  2. Clean text        →  normalised text
  3. Chunk             →  overlapping chunks
  4. Embed             →  Vertex AI text-embedding-004 vectors
  5. Upsert            →  Vertex AI Vector Search index

Run this script once per batch of new legal documents.
Re-running is safe — Vertex AI upserts (update-or-insert) by chunk_id.

Usage:
    python -m data_pipeline.index_builder \
        --input-dir ./docs/raw \
        --project   your-gcp-project-id \
        --location  us-central1 \
        --index-id  your-vertex-index-id \
        --endpoint-id your-vertex-endpoint-id

Environment variables (alternative to flags):
    GCP_PROJECT, GCP_LOCATION, VERTEX_INDEX_ID, VERTEX_ENDPOINT_ID
"""

import argparse
import json
import logging
import os
import time
from pathlib import Path
from typing import Optional

# Google Cloud clients
from google.cloud import aiplatform
from google.cloud.aiplatform.matching_engine import MatchingEngineIndex
from vertexai.language_models import TextEmbeddingModel
import vertexai

# Local pipeline modules
from data_pipeline.parse_pdf  import parse_pdf
from data_pipeline.parse_docx import parse_docx
from data_pipeline.clean      import clean_sections
from rag.chunker               import chunk_sections

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ── Configuration ─────────────────────────────────────────────────────────────

EMBEDDING_MODEL   = "text-embedding-004"   # Vertex AI embedding model
EMBEDDING_DIM     = 768                    # Dimensionality of text-embedding-004
BATCH_SIZE_EMBED  = 5                      # Chunks per embedding API call (stay under quota)
BATCH_SIZE_UPSERT = 100                    # Chunks per Vector Search upsert call
SUPPORTED_EXTS    = {".pdf", ".docx", ".doc"}


# ── Step 1: Parse ──────────────────────────────────────────────────────────────

def parse_file(file_path: Path) -> list[dict]:
    """Auto-detect file type and parse into raw sections/pages."""
    ext = file_path.suffix.lower()
    if ext == ".pdf":
        return parse_pdf(str(file_path))
    elif ext in {".docx", ".doc"}:
        return parse_docx(str(file_path))
    else:
        raise ValueError(f"Unsupported file type: {ext}")


# ── Step 2+3: Clean + Chunk ───────────────────────────────────────────────────

def process_file(file_path: Path) -> list[dict]:
    """Parse → Clean → Chunk a single file. Returns list of chunk dicts."""
    logger.info(f"Processing: {file_path.name}")
    raw      = parse_file(file_path)
    cleaned  = clean_sections(raw)
    chunks   = chunk_sections(cleaned, source_doc_id=file_path.name)
    logger.info(f"  → {len(chunks)} chunks")
    return chunks


# ── Step 4: Embed ─────────────────────────────────────────────────────────────

def embed_chunks(chunks: list[dict], model: TextEmbeddingModel) -> list[dict]:
    """
    Add an 'embedding' key to each chunk dict.
    Calls Vertex AI in batches to respect rate limits.
    """
    texts = [c["text"] for c in chunks]
    embeddings = []

    logger.info(f"  Embedding {len(texts)} chunks in batches of {BATCH_SIZE_EMBED}...")

    for i in range(0, len(texts), BATCH_SIZE_EMBED):
        batch = texts[i : i + BATCH_SIZE_EMBED]
        try:
            results = model.get_embeddings(batch)
            embeddings.extend([r.values for r in results])
        except Exception as e:
            logger.error(f"  Embedding batch {i//BATCH_SIZE_EMBED} failed: {e}")
            # Fill with zero vectors so the pipeline doesn't crash; flag for retry
            embeddings.extend([[0.0] * EMBEDDING_DIM for _ in batch])
        time.sleep(0.3)  # gentle rate limiting

    for chunk, emb in zip(chunks, embeddings):
        chunk["embedding"] = emb

    return chunks


# ── Step 5: Upsert to Vertex AI Vector Search ─────────────────────────────────

def upsert_to_vertex(
    chunks: list[dict],
    index: MatchingEngineIndex,
) -> None:
    """
    Push embedded chunks to Vertex AI Vector Search (batch upsert).
    Each chunk is stored as:
        id          = chunk_id  (e.g. "ipc_2023.pdf__chunk_0001")
        embedding   = float[]
        restricts   = [ { namespace: "file", allow: [filename] }, ... ]
    """
    datapoints = []

    for chunk in chunks:
        if not chunk.get("embedding"):
            logger.warning(f"  Skipping chunk without embedding: {chunk['chunk_id']}")
            continue

        # Vertex AI allows filtering via 'restricts' (categorical metadata)
        restricts = []
        if chunk["metadata"].get("file"):
            restricts.append({
                "namespace": "file",
                "allow_list": [chunk["metadata"]["file"]],
            })
        if chunk["metadata"].get("heading"):
            restricts.append({
                "namespace": "heading",
                "allow_list": [chunk["metadata"]["heading"][:100]],
            })

        datapoints.append({
            "datapoint_id": chunk["chunk_id"],
            "feature_vector": chunk["embedding"],
            "restricts": restricts,
        })

    logger.info(f"  Upserting {len(datapoints)} datapoints to Vertex AI...")

    for i in range(0, len(datapoints), BATCH_SIZE_UPSERT):
        batch = datapoints[i : i + BATCH_SIZE_UPSERT]
        index.upsert_datapoints(datapoints=batch)
        logger.info(f"    Upserted batch {i//BATCH_SIZE_UPSERT + 1} "
                    f"({len(batch)} points)")
        time.sleep(0.5)

    logger.info("  Upsert complete.")


# ── Metadata store (GCS JSON) ─────────────────────────────────────────────────

def save_chunk_metadata_to_gcs(
    chunks: list[dict],
    project: str,
    bucket_name: str,
    blob_prefix: str = "chunk_metadata/",
) -> None:
    """
    Save chunk text + metadata as JSON blobs in GCS.
    This allows the retriever to fetch full chunk text by chunk_id
    without re-querying the source documents.
    """
    from google.cloud import storage

    client = storage.Client(project=project)
    bucket = client.bucket(bucket_name)

    records = []
    for chunk in chunks:
        record = {
            "chunk_id":    chunk["chunk_id"],
            "doc_id":      chunk["doc_id"],
            "chunk_index": chunk["chunk_index"],
            "text":        chunk["text"],
            "char_count":  chunk["char_count"],
            "metadata":    chunk["metadata"],
        }
        records.append(record)

    blob_name = f"{blob_prefix}chunks_{int(time.time())}.json"
    blob = bucket.blob(blob_name)
    blob.upload_from_string(
        json.dumps(records, ensure_ascii=False, indent=2),
        content_type="application/json",
    )
    logger.info(f"  Saved {len(records)} chunk metadata records → gs://{bucket_name}/{blob_name}")


# ── Main pipeline ─────────────────────────────────────────────────────────────

def run_pipeline(
    input_dir: str,
    project: str,
    location: str,
    index_id: str,
    endpoint_id: str,
    gcs_bucket: Optional[str] = None,
) -> None:
    """
    Full pipeline: scan input_dir → parse → clean → chunk → embed → upsert.

    Args:
        input_dir:   Local directory containing PDF/DOCX files.
        project:     GCP project ID.
        location:    GCP region (e.g. "us-central1").
        index_id:    Vertex AI Vector Search index resource ID.
        endpoint_id: Vertex AI Vector Search index endpoint resource ID.
        gcs_bucket:  Optional GCS bucket name for metadata storage.
    """
    # Initialise Vertex AI SDK
    vertexai.init(project=project, location=location)
    aiplatform.init(project=project, location=location)

    # Load embedding model
    logger.info(f"Loading embedding model: {EMBEDDING_MODEL}")
    embed_model = TextEmbeddingModel.from_pretrained(EMBEDDING_MODEL)

    # Load Vector Search index
    logger.info(f"Connecting to Vertex AI Vector Search index: {index_id}")
    index = aiplatform.MatchingEngineIndex(index_name=index_id)

    # Discover input files
    input_path = Path(input_dir)
    files = [f for f in input_path.rglob("*") if f.suffix.lower() in SUPPORTED_EXTS]

    if not files:
        logger.warning(f"No supported files found in {input_dir}")
        return

    logger.info(f"Found {len(files)} file(s) to ingest:")
    for f in files:
        logger.info(f"  {f.name}")

    all_chunks = []

    for file_path in files:
        try:
            chunks = process_file(file_path)
            chunks = embed_chunks(chunks, embed_model)
            upsert_to_vertex(chunks, index)
            all_chunks.extend(chunks)
        except Exception as e:
            logger.error(f"Failed to process {file_path.name}: {e}", exc_info=True)
            continue

    # Optional: save metadata to GCS for the retriever
    if gcs_bucket and all_chunks:
        save_chunk_metadata_to_gcs(all_chunks, project, gcs_bucket)

    logger.info(
        f"\n{'='*50}\n"
        f"Pipeline complete.\n"
        f"  Files processed : {len(files)}\n"
        f"  Total chunks    : {len(all_chunks)}\n"
        f"{'='*50}"
    )


# ── CLI entry point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CounselAI — data ingestion pipeline")
    parser.add_argument("--input-dir",    required=True,  help="Directory with PDF/DOCX files")
    parser.add_argument("--project",      default=os.getenv("GCP_PROJECT"),      help="GCP project ID")
    parser.add_argument("--location",     default=os.getenv("GCP_LOCATION", "us-central1"))
    parser.add_argument("--index-id",     default=os.getenv("VERTEX_INDEX_ID"),  help="Vertex AI index resource ID")
    parser.add_argument("--endpoint-id",  default=os.getenv("VERTEX_ENDPOINT_ID"))
    parser.add_argument("--gcs-bucket",   default=os.getenv("GCS_BUCKET"),       help="GCS bucket for metadata")
    args = parser.parse_args()

    if not args.project:
        parser.error("--project or GCP_PROJECT env var required")
    if not args.index_id:
        parser.error("--index-id or VERTEX_INDEX_ID env var required")

    run_pipeline(
        input_dir   = args.input_dir,
        project     = args.project,
        location    = args.location,
        index_id    = args.index_id,
        endpoint_id = args.endpoint_id,
        gcs_bucket  = args.gcs_bucket,
    )