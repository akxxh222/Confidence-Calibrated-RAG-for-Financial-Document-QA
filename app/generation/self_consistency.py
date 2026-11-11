"""
Generates N independent samples for the same question/context and computes
an agreement score across them. See docs/architecture.md for signal definitions.

Two agreement modes:
  - numeric: extracts the first numeric figure from each sample and checks
    whether samples agree within a tolerance. Used for numeric-fact questions
    (the dominant case for financial QA: revenue, margin, growth %, etc).
  - text: falls back to pairwise text-embedding similarity when no numbers
    are found (descriptive questions).
"""
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from statistics import mean
from itertools import combinations

from google.genai.errors import APIError

from app.generation.generate import generate_answer
from app.retrieval.embed import embed_batch
from app.retrieval.retriever import RetrievedChunk

N_SAMPLES = int(os.environ.get("SELF_CONSISTENCY_N", 5))
MIN_SAMPLES = int(os.environ.get("SELF_CONSISTENCY_MIN_SAMPLES", 3))
MAX_WORKERS = int(os.environ.get("SELF_CONSISTENCY_WORKERS", 3))
NUMERIC_TOLERANCE_PCT = 0.02  # 2% relative tolerance for "same" figure

NUMBER_RE = re.compile(
    r"(?P<open>\()?\s*(?P<sign>[-+]?)\s*(?P<currency>\$?)\s*"
    r"(?P<number>\d[\d,]*(?:\.\d+)?)\s*"
    r"(?P<unit>billion\b|million\b|%|B\b|M\b)?\s*(?P<close>\))?",
    re.I,
)


@dataclass
class SelfConsistencyResult:
    samples: list[str]
    self_consistency_score: float   # 0-1, higher = more agreement
    extracted_numbers: list[float | None]


def _extract_number(text: str) -> float | None:
    parsed: list[tuple[float, bool, bool]] = []
    for match in NUMBER_RE.finditer(text):
        try:
            value = float(match.group("number").replace(",", ""))
        except ValueError:
            continue

        unit = (match.group("unit") or "").lower()
        if unit in ("billion", "b"):
            value *= 1e9
        elif unit in ("million", "m"):
            value *= 1e6

        if match.group("sign") == "-" or (
            match.group("open") and match.group("close")
        ):
            value = -value

        is_plain_year = (
            1900 <= value <= 2100
            and not match.group("currency")
            and not unit
            and not match.group("sign")
            and not match.group("open")
        )
        is_explicit_figure = bool(
            match.group("currency")
            or unit
            or match.group("sign")
            or match.group("open")
        )
        parsed.append((value, is_plain_year, is_explicit_figure))

    for value, _is_plain_year, is_explicit_figure in parsed:
        if is_explicit_figure:
            return value
    for value, is_plain_year, _is_explicit_figure in parsed:
        if not is_plain_year:
            return value
    return parsed[0][0] if parsed else None


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


def score_samples(samples: list[str]) -> SelfConsistencyResult:
    """Score already-collected samples, including resumable experiment runs."""
    numbers = [_extract_number(sample) for sample in samples]
    numeric_score = _numeric_agreement(numbers)
    score = numeric_score if numeric_score is not None else _text_agreement(samples)
    return SelfConsistencyResult(
        samples=samples,
        self_consistency_score=score,
        extracted_numbers=numbers,
    )


def run_self_consistency(question: str, chunks: list[RetrievedChunk],
                         n: int = N_SAMPLES,
                         workers: int = MAX_WORKERS) -> SelfConsistencyResult:
    """Generate independent answers in bounded parallel batches.

    Gemini occasionally returns a transient error for one sample even when the
    surrounding calls succeed. A single failed sample should not discard the
    successful LLM answers. We still require multiple samples so calibration
    never treats one generation as a consistency measurement. Preserve sample
    submission order so network completion order cannot choose the answer.
    """
    if n < 2 or workers < 1 or MIN_SAMPLES < 2:
        raise ValueError("self-consistency requires >=2 samples and >=1 worker")
    required = min(MIN_SAMPLES, n)
    samples_by_index = {}
    last_error = None
    completed = 0
    batch_size = min(workers, n)
    with ThreadPoolExecutor(max_workers=batch_size) as executor:
        for batch_start in range(0, n, batch_size):
            futures = {
                executor.submit(generate_answer, question, chunks, temperature=0.7): index
                for index in range(batch_start, min(batch_start + batch_size, n))
            }
            try:
                for future in as_completed(futures):
                    completed += 1
                    try:
                        sample = future.result()
                    except APIError as error:
                        # Each generation already used its retry policy. Stop
                        # scheduling batches on quota or configuration errors.
                        if error.code not in (500, 502, 503, 504):
                            raise
                        last_error = error
                        if len(samples_by_index) + n - completed < required:
                            raise
                    else:
                        if sample:
                            samples_by_index[futures[future]] = sample
            except Exception:
                for future in futures:
                    future.cancel()
                raise

    samples = [samples_by_index[index] for index in sorted(samples_by_index)]
    if len(samples) < required:
        if last_error is not None:
            raise last_error
        raise RuntimeError(
            f"Gemini returned only {len(samples)} usable samples; "
            f"at least {required} are required"
        )
    return score_samples(samples)
