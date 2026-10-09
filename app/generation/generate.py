"""
Answer generation via the Gemini API (google-genai SDK).

Uses models.generate_content (fully supported per the current docs, with a
stable config surface) rather than the newer Interactions API, because
self-consistency sampling depends on per-call temperature control. Swapping
to client.interactions.create later is a contained change if desired.

Env vars used:
  GEMINI_MODEL     - model id (default: gemini-3.8-flash)
  GEMINI_API_KEY   - read from the environment by genai.Client()
  GEMINI_THINKING_LEVEL - reasoning level (default: low for concise QA)
  GENERATION_MAX_OUTPUT_TOKENS - answer + thinking ceiling (default: 1000)

The client is constructed lazily so importing this module never requires an
API key (unit tests / CI import this chain without one).
"""
import os
import time
from functools import lru_cache

from google import genai
from google.genai import types
from google.genai.errors import ClientError, ServerError

from app.retrieval.retriever import RetrievedChunk

MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.8-flash")
THINKING_LEVEL = os.environ.get("GEMINI_THINKING_LEVEL", "low")
MAX_OUTPUT_TOKENS = int(os.environ.get("GENERATION_MAX_OUTPUT_TOKENS", "1000"))
MAX_RETRY_DELAY_SECONDS = 60.0
INSUFFICIENT_CONTEXT_RESPONSE = (
    "The provided context is insufficient to answer this question."
)


@lru_cache(maxsize=1)
def get_client() -> genai.Client:
    return genai.Client()


PROMPT_TEMPLATE = """You are answering questions about a company's SEC filings \
using only the context below. If the context does not contain enough evidence \
to answer the question, reply exactly: "{insufficient_response}" Do not infer \
missing facts or use outside knowledge.

Context:
{context}

Question: {question}

Answer concisely in plain text, no markdown, citing the figure or statement you relied on."""


def build_context(chunks: list[RetrievedChunk]) -> str:
    return "\n\n".join(
        f"[{c.section} | {c.page_ref or 'n/a'}] {c.text}" for c in chunks
    )


def build_prompt(question: str, chunks: list[RetrievedChunk]) -> str:
    return PROMPT_TEMPLATE.format(
        context=build_context(chunks),
        question=question,
        insufficient_response=INSUFFICIENT_CONTEXT_RESPONSE,
    )


def _retry_delay(error: ClientError | ServerError, attempt: int) -> float:
    details = error.details.get("error", {}).get("details", [])
    for detail in details:
        retry_delay = detail.get("retryDelay", "")
        if retry_delay.endswith("s"):
            try:
                return float(retry_delay[:-1]) + 1.0
            except ValueError:
                pass
    return float(2 ** (attempt + 1))


def _generate_content(prompt: str, temperature: float):
    for attempt in range(5):
        try:
            return get_client().models.generate_content(
                model=MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=temperature,
                    max_output_tokens=MAX_OUTPUT_TOKENS,
                    thinking_config=types.ThinkingConfig(
                        thinking_level=THINKING_LEVEL,
                    ),
                ),
            )
        except (ClientError, ServerError) as error:
            if error.code not in (429, 503) or attempt == 4:
                raise
            delay = _retry_delay(error, attempt)
            if delay > MAX_RETRY_DELAY_SECONDS:
                raise
            print(f"Gemini generation unavailable; retrying in {delay:.0f}s")
            time.sleep(delay)
    raise RuntimeError("unreachable")


def generate_answer(question: str, chunks: list[RetrievedChunk], temperature: float = 0.3) -> str:
    prompt = build_prompt(question, chunks)
    resp = _generate_content(prompt, temperature)
    return (resp.text or "").strip()
