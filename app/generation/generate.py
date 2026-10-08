"""
Answer generation via the Gemini API (google-genai SDK).

Uses models.generate_content (fully supported per the current docs, with a
stable config surface) rather than the newer Interactions API, because
self-consistency sampling depends on per-call temperature control. Swapping
to client.interactions.create later is a contained change if desired.

Env vars used:
  GEMINI_MODEL     - model id (default: gemini-3.8-flash)
  GEMINI_API_KEY   - read from the environment by genai.Client()

The client is constructed lazily so importing this module never requires an
API key (unit tests / CI import this chain without one).
"""
import os
from functools import lru_cache

from google import genai
from google.genai import types

from app.retrieval.retriever import RetrievedChunk

MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.8-flash")


@lru_cache(maxsize=1)
def get_client() -> genai.Client:
    return genai.Client()


PROMPT_TEMPLATE = """You are answering questions about a company's SEC filings \
using only the context below. If the context does not support the question, \
say so plainly rather than guessing.

Context:
{context}

Question: {question}

Answer concisely in plain text, no markdown, citing the figure or statement you relied on."""


def build_context(chunks: list[RetrievedChunk]) -> str:
    return "\n\n".join(
        f"[{c.section} | {c.page_ref or 'n/a'}] {c.text}" for c in chunks
    )


def generate_answer(question: str, chunks: list[RetrievedChunk], temperature: float = 0.3) -> str:
    context = build_context(chunks)
    prompt = PROMPT_TEMPLATE.format(context=context, question=question)
    resp = get_client().models.generate_content(
        model=MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(temperature=temperature, max_output_tokens=300),
    )
    return (resp.text or "").strip()
