# Control Engineering RAG Study Assistant

A Retrieval-Augmented Generation study assistant built from scratch
using 20 Control Engineering lecture notes.

## What it does
- Answers Control Engineering questions grounded in your lecture notes
- Analyses transfer functions: poles, stability, Routh-Hurwitz, state space
- Cites the source PDF and page number for every answer

## Stack
| Component | Technology |
|---|---|
| PDF extraction | PyMuPDF |
| Embeddings | sentence-transformers (all-MiniLM-L6-v2) |
| Vector store | ChromaDB |
| LLM | OpenRouter free tier |
| Formula solver | SymPy |
| UI | Streamlit |

## Setup
1. Add PDF lecture notes to `data/pdfs/`
2. Create `.env` with `OPENROUTER_API_KEY=your_key_here`
3. Install dependencies: `pip install -r requirements.txt`
4. Build the index: `python -m scripts.build_index`
5. Run the app: `streamlit run src/ui/app.py`