from pathlib import Path
import sys
import logging
from typing import List, Dict, Any

# Make sure Python can find modules in the project root
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Configure logging
logger = logging.getLogger(__name__)

# These are our tuning parameters
CHUNK_SIZE = 512      # Max characters per chunk
CHUNK_OVERLAP = 64    # How many characters to repeat between chunks


def _find_natural_break_point(text: str, start: int, end: int) -> int:
    """
    Find a natural break point (newline or period) within the specified range.
    Falls back to hard cut at end if no natural break is found.
    
    Args:
        text: The text to search
        start: Start position for search
        end: End position for search
    
    Returns:
        The position of the break point
    """
    # Try newline first
    break_point = text.rfind("\n", start, end)
    if break_point > start:
        return break_point
    
    # Try period next
    break_point = text.rfind(". ", start, end)
    if break_point > start:
        return break_point + 1  # Include the period
    
    # No natural break found, hard cut
    return end


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    """
    Split a string into overlapping chunks at natural break points.
    
    Args:
        text: The text to chunk
        chunk_size: Maximum characters per chunk
        overlap: Number of characters to repeat between chunks
    
    Returns:
        A list of text chunks
    """
    if not isinstance(text, str):
        raise TypeError(f"Expected str, got {type(text).__name__}")
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}")
    if overlap < 0:
        raise ValueError(f"overlap cannot be negative, got {overlap}")
    
    # If the whole text fits in one chunk, just return it
    stripped_text = text.strip()
    if not stripped_text or len(stripped_text) <= chunk_size:
        return [stripped_text] if stripped_text else []

    chunks = []
    start = 0

    while start < len(text):
        end = min(start + chunk_size, len(text))
        
        if end < len(text):
            break_point = _find_natural_break_point(text, start, end)
        else:
            break_point = len(text)

        chunk = text[start:break_point].strip()
        if chunk:
            chunks.append(chunk)

        # Move forward but step back by overlap amount
        start = max(0, break_point - overlap)
        
        # Prevent infinite loops with very small overlap
        if start == break_point - overlap and start > 0:
            start = break_point

    return chunks


def chunk_pages(pages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Takes the list of page dicts from pdf_extractor
    and returns a flat list of chunk dicts.
    
    Args:
        pages: List of page dictionaries with 'text', 'source', and 'page' keys
    
    Returns:
        List of chunk dictionaries with metadata
    """
    if not isinstance(pages, list):
        raise TypeError(f"Expected list, got {type(pages).__name__}")
    
    all_chunks = []

    for page in pages:
        text = page.get("text", "")
        if not text.strip():
            continue  # Skip empty pages

        chunks = chunk_text(text)

        for i, chunk in enumerate(chunks):
            chunk_data = {
                "chunk_id": f"{page['source']}__p{page['page']}__c{i}",
                "source": page["source"],
                "page": page["page"],
                "text": chunk,
                "chunk_index": i
            }
            all_chunks.append(chunk_data)

    logger.info(f"Created {len(all_chunks)} chunks from {len(pages)} pages")
    return all_chunks


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    from src.pipeline.pdf_extractor import extract_all_pdfs

    try:
        # Step 1: Extract
        pages = extract_all_pdfs("data/pdfs")
        logger.info(f"Extracted {len(pages)} pages")

        # Step 2: Chunk
        chunks = chunk_pages(pages)

        # Inspect a few chunks
        if chunks:
            print("\n--- Sample chunk ---")
            sample_index = min(10, len(chunks) - 1)
            sample_chunk = chunks[sample_index]
            print(f"ID: {sample_chunk['chunk_id']}")
            print(f"Source: {sample_chunk['source']}")
            print(f"Page: {sample_chunk['page']}")
            print(f"Text: {sample_chunk['text'][:200]}")
        else:
            logger.warning("No chunks were created from the PDFs")
    
    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
    except Exception as e:
        logger.error(f"Error during processing: {e}", exc_info=True)