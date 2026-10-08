"""
Embeddings via the Gemini API (google-genai SDK).

Replaces the previous local sentence-transformers model. Env vars used:
  EMBEDDING_MODEL  - model id (default: gemini-embedding-2)
  EMBEDDING_DIM    - output dimensionality; MUST match VECTOR(n) in
                     db/schema.sql (requesting < 3072 dims also makes the
                     API return normalized vectors)

gemini-embedding-2 specifics that differ from the old local model:
  - Multiple raw string inputs in ONE embed_content call are aggregated into
    a single embedding; each text is therefore wrapped in its own Content
    object so every chunk gets its own vector.
  - Embedding requests are capped at 100 inputs, so embed_batch slices
    accordingly.
  - Vectors are defensively re-normalized here so the "dot product == cosine
    similarity" assumption in self_consistency._text_agreement always holds.

The client is constructed lazily so importing this module never requires an
API key (unit tests / CI import this chain without one).
"""
import os
from functools import lru_cache
from math import sqrt

from google import genai
from google.genai import types

EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "gemini-embedding-2")
EMBEDDING_DIM = int(os.environ.get("EMBEDDING_DIM", "768"))
MAX_INPUTS_PER_REQUEST = 100  # per-request cap for Gemini embedding models


@lru_cache(maxsize=1)
def get_client() -> genai.Client:
    # genai.Client() reads GEMINI_API_KEY (or GOOGLE_API_KEY) from the env.
    return genai.Client()


def _normalize(vec: list[float]) -> list[float]:
    norm = sqrt(sum(x * x for x in vec)) or 1.0
    return [x / norm for x in vec]


def _request_embeddings(texts: list[str]) -> list[list[float]]:
    result = get_client().models.embed_content(
        model=EMBEDDING_MODEL,
        contents=[
            types.Content(role="user", parts=[types.Part(text=t)]) for t in texts
        ],
        config=types.EmbedContentConfig(output_dimensionality=EMBEDDING_DIM),
    )
    return [_normalize(list(e.values)) for e in result.embeddings]


def embed_text(text: str) -> list[float]:
    return embed_batch([text])[0]


def embed_batch(texts: list[str]) -> list[list[float]]:
    vectors: list[list[float]] = []
    for start in range(0, len(texts), MAX_INPUTS_PER_REQUEST):
        vectors.extend(_request_embeddings(texts[start:start + MAX_INPUTS_PER_REQUEST]))
    return vectors
