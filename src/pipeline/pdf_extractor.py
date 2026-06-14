import os
import fitz  # PyMuPDF
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

def extract_pdf(pdf_path):
    """
    Extract text from every page of a PDF.
    Returns a list of dictionaries — one per page.
    """
    path = Path(pdf_path)  # Path() handles Windows/Linux differences cleanly
    doc = fitz.open(str(path))

    pages = []  # We'll collect results here

    for i, page in enumerate(doc):
        text = page.get_text("text").strip()

        if text:  # Only keep pages with actual content
            page_data = {
                "source": path.name,        # Just the filename e.g. "lec1.pdf"
                "page": i + 1,              # Human-readable page number
                "text": text,
                "total_pages": len(doc)
            }
            pages.append(page_data)

    doc.close()

    logger.info(f"Extracted {len(pages)} pages from {path.name}")
    return pages


def extract_all_pdfs(pdf_dir):
    """
    Extract text from ALL PDFs in a folder.
    Returns one flat list of page dicts from every PDF combined.
    """
    pdf_dir = Path(pdf_dir)

    # Find all .pdf files in the folder
    pdf_files = list(pdf_dir.glob("*.pdf"))
    logger.info(f"Found {len(pdf_files)} PDFs in {pdf_dir}")

    all_pages = []

    for pdf_path in pdf_files:
        pages = extract_pdf(str(pdf_path))
        all_pages.extend(pages)  # extend adds all items, append would add a list inside a list

    logger.info(f"Total pages extracted across all PDFs: {len(all_pages)}")
    return all_pages



# ── Test it ───────────────────────────────────────────
if __name__ == "__main__":
    all_pages = extract_all_pdfs("data/pdfs")

    print("\nLast page extracted:")
    print(all_pages[-1]["source"], "| page", all_pages[-1]["page"])
    print(all_pages[-1]["text"][:200])