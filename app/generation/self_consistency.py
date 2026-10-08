"""
Generates N independent samples for the same question/context and computes
an agreement score across them (Signal 2, Section 8 of the design doc).

Two agreement modes:
  - numeric: extracts the first numeric figure from each sample and checks
    whether samples agree within a tolerance. Used for numeric-fact questions
    (the dominant case for financial QA: revenue, margin, growth %, etc).
  - text: falls back to pairwise text-embedding similarity when no numbers
    are found (descriptive questions).
"""
import os
import re
from dataclasses import dataclass
from statistics import mean
from itertools import combinations

from app.generation.generate import generate_answer
from app.retrieval.embed import embed_batch
from app.retrieval.retriever import RetrievedChunk

N_SAMPLES = int(os.environ.get("SELF_CONSISTENCY_N", 5))
NUMERIC_TOLERANCE_PCT = 0.02  # 2% relative tolerance for "same" figure

NUMBER_RE = re.compile(r"[-+]?\$?\d[\d,]*\.?\d*\s?(?:%|billion|million|B|M)?", re.I)


@dataclass
class SelfConsistencyResult:
    samples: list[str]
    self_consistency_score: float   # 0-1, higher = more agreement
    extracted_numbers: list[float | None]


def _extract_number(text: str) -> float | None:
    match = NUMBER_RE.search(text)
    if not match:
        return None
    raw = match.group(0).lower()
    raw = raw.replace(",", "").replace("$", "").replace("%", "").strip()
    multiplier = 1.0
    if "billion" in raw or raw.endswith("b"):
        multiplier = 1e9
        raw = raw.replace("billion", "").replace("b", "")
    elif "million" in raw or raw.endswith("m"):
        multiplier = 1e6
        raw = raw.replace("million", "").replace("m", "")
    try:
        return float(raw.strip()) * multiplier
    except ValueError:
        return None


def _numeric_agreement(numbers: list[float | None]) -> float | None:
    valid = [n for n in numbers if n is not None]
    if len(valid) < 2:
        return None
    pairs_agree = []
    for a, b in combinations(valid, 2):
        if a == 0 and b == 0:
            pairs_agree.append(1.0)
            continue
        rel_diff = abs(a - b) / max(abs(a), abs(b), 1e-9)
        pairs_agree.append(1.0 if rel_diff <= NUMERIC_TOLERANCE_PCT else 0.0)
    return mean(pairs_agree)


def _text_agreement(samples: list[str]) -> float:
    vecs = embed_batch(samples)
    sims = []
    for a, b in combinations(vecs, 2):
        num = sum(x * y for x, y in zip(a, b))
        sims.append(num)  # embeddings are normalized -> dot product == cosine sim
    return mean(sims) if sims else 0.0


def run_self_consistency(question: str, chunks: list[RetrievedChunk],
                         n: int = N_SAMPLES) -> SelfConsistencyResult:
    samples = [generate_answer(question, chunks, temperature=0.7) for _ in range(n)]
    numbers = [_extract_number(s) for s in samples]

    numeric_score = _numeric_agreement(numbers)
    score = numeric_score if numeric_score is not None else _text_agreement(samples)

    return SelfConsistencyResult(
        samples=samples, self_consistency_score=score, extracted_numbers=numbers
    )
