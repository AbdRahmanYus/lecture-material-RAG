"""
query_engine.py

Orchestrates the full RAG pipeline:
  1. Retrieve relevant chunks from ChromaDB
  2. Build a prompt combining the question and retrieved context
  3. Send the prompt to an LLM via OpenRouter
  4. Return the answer
"""

import os
import logging
import requests
from dotenv import load_dotenv

from src.pipeline.indexer import query_collection

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# ── Configuration ──────────────────────────────────────────────────────────────
load_dotenv()  # reads your .env file and loads variables into the environment


OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_URL     = "https://openrouter.ai/api/v1/chat/completions"
PRIMARY_MODEL = "minimax/minimax-m2.5:free"

FALLBACK_MODELS = [
    "nvidia/nemotron-3-super-120b-a12b:free",
    "arcee-ai/trinity-large-thinking:free",
    "google/gemma-4-31b-it:free",
]



# MODEL              = "minimax/minimax-m2.5:free" #"deepseek/deepseek-v4-flash:free" 
#"meta-llama/llama-3-8b-instruct:free"
MAX_TOKENS         = 512


# ── Function 1: Build the prompt ───────────────────────────────────────────────
def build_prompt(question: str, chunks: list[dict]) -> str:
    """
    Combines the retrieved chunks and the user's question into a single
    prompt string that instructs the LLM how to behave.

    Args:
        question: The user's original question.
        chunks:   List of chunk dicts returned by query_collection().

    Returns:
        A formatted prompt string.
    """
    if not question or not question.strip():
        raise ValueError("Question cannot be empty.")
    if not chunks:
        raise ValueError("No chunks provided — cannot build prompt.")

    # Join the text of each chunk, labelled by source and page
    context_blocks = []
    for i, chunk in enumerate(chunks, start=1):
        label = f"[{i}] {chunk['source']} — page {chunk['page']}"
        context_blocks.append(f"{label}\n{chunk['text']}")

    context = "\n\n".join(context_blocks)

    prompt = f"""You are a helpful Control Engineering teaching assistant.
Use ONLY the lecture note excerpts below to answer the question.
If the answer is not in the excerpts, say "I could not find that in the lecture notes."

--- LECTURE NOTE EXCERPTS ---
{context}
--- END OF EXCERPTS ---

Question: {question}

Answer:"""

    return prompt





# ── Function 2: Call the LLM ───────────────────────────────────────────────────
def ask_llm(prompt: str) -> str:
    """
    Sends a prompt to OpenRouter. Tries the primary model first,
    then falls back through FALLBACK_MODELS on 429 rate-limit errors.

    Args:
        prompt: The fully formatted prompt string.

    Returns:
        The model's answer as a plain string.

    Raises:
        ValueError:   If the API key is missing.
        RuntimeError: If all models fail.
    """
    if not OPENROUTER_API_KEY:
        raise ValueError(
            "OPENROUTER_API_KEY not found. "
            "Check your .env file has: OPENROUTER_API_KEY=sk-or-v1-..."
        )

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type":  "application/json",
    }

    models_to_try = [PRIMARY_MODEL] + FALLBACK_MODELS

    for model in models_to_try:
        logger.info(f"Trying model: {model}")

        payload = {
            "model": model,
            "max_tokens": MAX_TOKENS,
            "messages": [
                {"role": "user", "content": prompt}
            ],
        }

        try:
            response = requests.post(
                OPENROUTER_URL, headers=headers, json=payload, timeout=30
            )

            if response.status_code == 429:
                logger.warning(f"Model {model} is rate-limited — trying next fallback")
                continue  # skip to the next model in the list

            response.raise_for_status()

            data = response.json()
            answer = data["choices"][0]["message"]["content"].strip()
            logger.info(f"Answer received from model: {model}")
            return answer

        except requests.exceptions.Timeout:
            logger.warning(f"Model {model} timed out — trying next fallback")
            continue
        except requests.exceptions.HTTPError as e:
            raise RuntimeError(
                f"OpenRouter API error: {e.response.status_code} — {e.response.text}"
            )
        except (KeyError, IndexError):
            raise RuntimeError(f"Unexpected response format from OpenRouter: {response.json()}")

    raise RuntimeError(
        "All models failed or were rate-limited. Try again in a few minutes."
    )

# ── Function 2: Call the LLM ───────────────────────────────────────────────────
# def ask_llm(prompt: str) -> str:
#     """
#     Sends a prompt to the OpenRouter API and returns the model's reply.

#     Args:
#         prompt: The fully formatted prompt string.

#     Returns:
#         The model's answer as a plain string.

#     Raises:
#         ValueError:   If the API key is missing.
#         RuntimeError: If the API call fails.
#     """
#     if not OPENROUTER_API_KEY:
#         raise ValueError(
#             "OPENROUTER_API_KEY not found. "
#             "Check your .env file has: OPENROUTER_API_KEY=sk-or-v1-..."
#         )

#     headers = {
#         "Authorization": f"Bearer {OPENROUTER_API_KEY}",
#         "Content-Type":  "application/json",
#     }

#     payload = {
#         "model": MODEL,
#         "max_tokens": MAX_TOKENS,
#         "messages": [
#             {"role": "user", "content": prompt}
#         ],
#     }

#     logger.info(f"Sending request to OpenRouter — model: {MODEL}")

#     try:
#         response = requests.post(OPENROUTER_URL, headers=headers, json=payload, timeout=30)
#         response.raise_for_status()  # raises an exception for 4xx/5xx status codes
#     except requests.exceptions.Timeout:
#         raise RuntimeError("Request timed out. Check your internet connection.")
#     except requests.exceptions.HTTPError as e:
#         raise RuntimeError(f"OpenRouter API error: {e.response.status_code} — {e.response.text}")
#     except requests.exceptions.RequestException as e:
#         raise RuntimeError(f"Network error: {e}")

#     data = response.json()

#     try:
#         answer = data["choices"][0]["message"]["content"].strip()
#     except (KeyError, IndexError) as e:
#         raise RuntimeError(f"Unexpected response format from OpenRouter: {data}")

#     return answer


# ── Function 3: Full pipeline ──────────────────────────────────────────────────
def answer_question(question: str, n_results: int = 3) -> dict:
    """
    Full RAG pipeline: retrieve → build prompt → call LLM → return answer.

    Args:
        question:  The user's question.
        n_results: How many chunks to retrieve from ChromaDB.

    Returns:
        A dict with keys:
            'question' : the original question
            'answer'   : the LLM's answer
            'sources'  : list of (source, page) tuples used as context
    """
    if not question or not question.strip():
        raise ValueError("Question cannot be empty.")

    logger.info(f"Question received: {question}")

    # Step 1 — retrieve relevant chunks
    chunks = query_collection(question, n_results=n_results)
    logger.info(f"Retrieved {len(chunks)} chunks from ChromaDB")

    # Step 2 — build the prompt
    prompt = build_prompt(question, chunks)

    # Step 3 — call the LLM
    answer = ask_llm(prompt)
    logger.info("Answer received from LLM")

    # Step 4 — collect source references
    sources = [(c["source"], c["page"]) for c in chunks]

    return {
        "question": question,
        "answer":   answer,
        "sources":  sources,
    }


# ── Quick test ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    test_question = input("Enter your question: ")
    result = answer_question(test_question)

    print("\n" + "="*60)
    print(f"QUESTION: {result['question']}")
    print("="*60)
    print(f"ANSWER:\n{result['answer']}")
    print("="*60)
    print("SOURCES:")
    for source, page in result['sources']:
        print(f"  • {source} — page {page}")