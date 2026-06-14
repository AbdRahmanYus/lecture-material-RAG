import sys
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import chromadb
from chromadb.utils import embedding_functions

logger = logging.getLogger(__name__)

# Configuration
CHROMA_DB_DIR   = "data/chroma_db"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
COLLECTION_NAME = "control_engineering_notes"
BATCH_SIZE      = 50  # Safe for 8GB RAM


def get_collection(reset: bool = False):
    """
    Connect to ChromaDB and return the collection.
    Creates the collection if it doesn't exist.
    If reset=True, wipes and rebuilds from scratch.
    """
    Path(CHROMA_DB_DIR).mkdir(parents=True, exist_ok=True)

    client = chromadb.PersistentClient(path=CHROMA_DB_DIR)

    emb_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL
    )

    if reset:
        try:
            client.delete_collection(COLLECTION_NAME)
            logger.info(f"Deleted existing collection: {COLLECTION_NAME}")
        except Exception:
            pass

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=emb_fn,
        metadata={"hnsw:space": "cosine"}
    )

    return collection


def index_chunks(chunks: list, reset: bool = False):
    """
    Embed all chunks and store them in ChromaDB.
    Skips indexing if data already exists (unless reset=True).
    """
    collection = get_collection(reset=reset)

    existing = collection.count()
    if existing > 0 and not reset:
        logger.info(f"Collection already has {existing} chunks — skipping indexing.")
        logger.info("Pass reset=True to rebuild from scratch.")
        return collection

    total = len(chunks)
    logger.info(f"Indexing {total} chunks — this may take a few minutes...")

    for i in range(0, total, BATCH_SIZE):
        batch = chunks[i: i + BATCH_SIZE]
        batch_num = (i // BATCH_SIZE) + 1
        total_batches = (total + BATCH_SIZE - 1) // BATCH_SIZE

        print(f"  Batch {batch_num}/{total_batches} — chunks {i} to {i + len(batch)}")

        collection.add(
            ids=[c["chunk_id"] for c in batch],
            documents=[c["text"] for c in batch],
            metadatas=[{
                "source": c["source"],
                "page": c["page"],
                "chunk_index": c["chunk_index"]
            } for c in batch]
        )

    logger.info(f"Indexing complete — {collection.count()} chunks stored")
    return collection


def query_collection(query: str, n_results: int = 5) -> list:
    """
    Find the most relevant chunks for a given question.
    Returns a list of dicts with text, source, page, and distance.
    """
    collection = get_collection()

    if collection.count() == 0:
        raise RuntimeError("Collection is empty. Run index_chunks() first.")

    results = collection.query(
        query_texts=[query],
        n_results=n_results
    )

    hits = []
    for i in range(len(results["documents"][0])):
        hits.append({
            "text":     results["documents"][0][i],
            "source":   results["metadatas"][0][i]["source"],
            "page":     results["metadatas"][0][i]["page"],
            "distance": results["distances"][0][i]
        })

    return hits


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    from src.pipeline.pdf_extractor import extract_all_pdfs
    from src.pipeline.chunker import chunk_pages

    # Run the full pipeline
    pages = extract_all_pdfs("data/pdfs")
    chunks = chunk_pages(pages)
    collection = index_chunks(chunks)

    # Test a real question
    print("\n--- Testing retrieval ---")
    question = "What is state space representation?"
    hits = query_collection(question, n_results=3)

    for i, hit in enumerate(hits, 1):
        print(f"\nResult {i} | {hit['source']} p.{hit['page']} | distance: {hit['distance']:.4f}")
        print(hit["text"][:300])