import pytest
from concurrent.futures import Future
from threading import Barrier, Lock
from google.genai.errors import ClientError, ServerError

import app.generation.self_consistency as consistency_module
from app.generation.self_consistency import (
    _extract_number,
    _numeric_agreement,
    score_samples,
)


def test_extract_plain_number():
    assert _extract_number("Revenue was 383.285 billion.") == 383.285e9


def test_extract_percentage():
    assert _extract_number("Growth was 4%") == 4.0


def test_extract_skips_year_before_financial_answer():
    assert _extract_number("In 2025, revenue was $1.2 billion.") == 1.2e9


def test_extract_normalizes_equivalent_million_and_billion_scales():
    billion = _extract_number("Revenue was $1.2 billion.")
    million = _extract_number("Revenue was $1,200 million.")
    assert billion == million == 1.2e9


def test_extract_parenthesized_financial_value_as_negative():
    assert _extract_number("Operating loss was ($5.2 million).") == -5.2e6


def test_extract_range_uses_first_bound_consistently():
    assert _extract_number("Gross margin was between 10% and 12%.") == 10.0


def test_extract_skips_multiple_years_before_first_financial_figure():
    text = "In 2025 and 2024, revenue was $1.2 billion versus $1.1 billion."
    assert _extract_number(text) == 1.2e9


def test_extract_prioritizes_scaled_figure_after_full_date():
    text = (
        "Microsoft's total revenue for the year ended June 30, 2026 "
        "was $331,839 million."
    )
    assert _extract_number(text) == 331839e6


def test_extract_does_not_treat_company_initial_as_million_unit():
    text = (
        "In fiscal year 2025, Microsoft's gross margin was "
        "$193,893 million."
    )
    assert _extract_number(text) == 193893e6


def test_extract_none_when_no_number():
    assert _extract_number("Revenue grew significantly year over year.") is None


def test_numeric_agreement_all_same():
    assert _numeric_agreement([100.0, 100.0, 100.5]) == 1.0


def test_numeric_agreement_disagreement():
    score = _numeric_agreement([100.0, 50.0])
    assert score == 0.0


def test_numeric_agreement_insufficient_data_returns_none():
    assert _numeric_agreement([None, None]) is None
    assert _numeric_agreement([100.0]) is None


def test_score_samples_supports_resumable_numeric_runs():
    result = score_samples([
        "Revenue was $1.2 billion.",
        "Revenue was $1,200 million.",
    ])

    assert result.extracted_numbers == [1.2e9, 1.2e9]
    assert result.self_consistency_score == 1.0


def test_run_self_consistency_stops_after_quota_failure(monkeypatch):
    calls = []
    quota_error = ClientError(429, {"error": {"message": "quota exhausted"}})

    def generate(*_args, **_kwargs):
        calls.append(1)
        raise quota_error

    monkeypatch.setattr(consistency_module, "generate_answer", generate)
    with pytest.raises(ClientError) as caught:
        consistency_module.run_self_consistency("question", [], n=5, workers=1)

    assert caught.value is quota_error
    assert len(calls) == 1


def test_run_self_consistency_tolerates_one_provider_failure(monkeypatch):
    outcomes = iter([
        "Revenue was $100 million.",
        ServerError(504, {"error": {"message": "timeout"}}),
        "Revenue was $100 million.",
        "Revenue was $100 million.",
        "Revenue was $100 million.",
    ])

    def generate(*_args, **_kwargs):
        outcome = next(outcomes)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    monkeypatch.setattr(consistency_module, "generate_answer", generate)
    result = consistency_module.run_self_consistency("question", [], n=5)

    assert len(result.samples) == 4
    assert result.self_consistency_score == 1.0


def test_generations_overlap_without_exceeding_worker_limit(monkeypatch):
    barrier = Barrier(3)
    lock = Lock()
    active = 0
    peak_active = 0
    calls = 0

    def generate(*_args, **_kwargs):
        nonlocal active, peak_active, calls
        with lock:
            calls += 1
            call_number = calls
            active += 1
            peak_active = max(peak_active, active)
        try:
            if call_number <= 3:
                barrier.wait(timeout=5)
            return "Revenue was $100 million."
        finally:
            with lock:
                active -= 1

    monkeypatch.setattr(consistency_module, "generate_answer", generate)
    result = consistency_module.run_self_consistency("question", [], n=5, workers=3)

    assert calls == 5
    assert peak_active == 3
    assert len(result.samples) == 5
    assert result.self_consistency_score == 1.0


def test_primary_answer_keeps_submission_order(monkeypatch):
    class InlineExecutor:
        def __init__(self, **_kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            pass

        def submit(self, fn, *args, **kwargs):
            future = Future()
            future.set_result(fn(*args, **kwargs))
            return future

    answers = iter(f"Revenue was ${value} million." for value in range(100, 105))
    monkeypatch.setattr(consistency_module, "generate_answer", lambda *a, **kw: next(answers))
    monkeypatch.setattr(consistency_module, "ThreadPoolExecutor", InlineExecutor)
    # Visit completed futures in reverse order to simulate out-of-order replies.
    monkeypatch.setattr(consistency_module, "as_completed", lambda futures: reversed(futures))

    result = consistency_module.run_self_consistency("question", [], n=5, workers=3)

    assert result.samples[0] == "Revenue was $100 million."
    assert result.samples == [f"Revenue was ${value} million." for value in range(100, 105)]
    assert len(result.samples) == 5


def test_parallel_quota_failure_does_not_start_another_batch(monkeypatch):
    calls = []

    def generate(*_args, **_kwargs):
        calls.append(1)
        raise ClientError(429, {"error": {"message": "quota exhausted"}})

    monkeypatch.setattr(consistency_module, "generate_answer", generate)
    with pytest.raises(ClientError):
        consistency_module.run_self_consistency("question", [], n=5, workers=3)

    assert 1 <= len(calls) <= 3
