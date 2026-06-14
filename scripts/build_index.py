"""
build_index.py

Rebuilds the ChromaDB vector index from scratch.
Run this whenever you:
  - Add new PDF lecture notes to data/pdfs/
  - Suspect the index is corrupted
  - Want a clean rebuild

Usage:
    python -m scripts.build_index
"""

import logging
import shutil
from pathlib import Path

from src.pipeline.pdf_extractor import extract_all_pdfs
from src.pipeline.chunker import chunk_pages
from src.pipeline.indexer import get_collection, index_chunks

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# ── Paths ──────────────────────────────────────────────────────────────────────
PDF_DIR    = Path("data/pdfs")
CHROMA_DIR = Path("data/chroma_db")


def wipe_index() -> None:
    """Deletes the existing ChromaDB directory so we start clean."""
    if CHROMA_DIR.exists():
        shutil.rmtree(CHROMA_DIR)
        logger.info(f"Wiped existing index at {CHROMA_DIR}")
    else:
        logger.info("No existing index found — starting fresh.")


def build_index() -> None:
    """
    Full pipeline: wipe → extract → chunk → embed → index.
    Logs a summary at the end.
    """
    logger.info("=" * 50)
    logger.info("Starting full index rebuild")
    logger.info("=" * 50)

    # Step 1 — wipe
    wipe_index()

    # Step 2 — extract
    logger.info("Step 1/3 — Extracting text from PDFs...")
    pages = extract_all_pdfs(str(PDF_DIR))
    logger.info(f"Extracted {len(pages)} pages from {PDF_DIR}")

    if not pages:
        logger.error("No pages extracted — check that PDFs exist in data/pdfs/")
        return

    # Step 3 — chunk
    logger.info("Step 2/3 — Chunking pages...")
    chunks = chunk_pages(pages)
    logger.info(f"Created {len(chunks)} chunks")

    # Step 4 — index
    logger.info("Step 3/3 — Embedding and indexing chunks...")
    collection = get_collection()
    index_chunks(chunks, collection)

    logger.info("=" * 50)
    logger.info(f"Index rebuild complete")
    logger.info(f"  PDFs processed : {len(set(p['source'] for p in pages))}")
    logger.info(f"  Pages extracted: {len(pages)}")
    logger.info(f"  Chunks indexed : {len(chunks)}")
    logger.info("=" * 50)


if __name__ == "__main__":
    build_index()