from types import SimpleNamespace

from google.genai.errors import ClientError, ServerError

from app.generation import generate
from app.generation.generate import (
    INSUFFICIENT_CONTEXT_RESPONSE,
    build_prompt,
)
from app.retrieval.retriever import RetrievedChunk


def test_prompt_requires_canonical_insufficient_context_response():
    chunks = [
        RetrievedChunk(
            chunk_id=1,
            doc_id=1,
            section="MD&A",
            text="Revenue increased year over year.",
            page_ref=None,
            similarity=0.8,
        )
    ]

    prompt = build_prompt("What was the CEO's home address?", chunks)

    assert INSUFFICIENT_CONTEXT_RESPONSE in prompt
    assert "using only the context" in prompt


def test_generate_answer_retries_transient_model_unavailable(monkeypatch):
    class Models:
        def __init__(self):
            self.calls = 0
            self.last_config = None

        def generate_content(self, **kwargs):
            self.calls += 1
            self.last_config = kwargs["config"]
            if self.calls == 1:
                raise ServerError(
                    503,
                    {"error": {"message": "high demand", "status": "UNAVAILABLE"}},
                )
            return SimpleNamespace(text="Revenue was $1.2 billion.")

    models = Models()
    monkeypatch.setattr(generate, "get_client", lambda: SimpleNamespace(models=models))
    sleep_calls = []
    monkeypatch.setattr(generate.time, "sleep", sleep_calls.append)

    answer = generate.generate_answer("What was revenue?", [])

    assert answer == "Revenue was $1.2 billion."
    assert sleep_calls == [2.0]
    assert models.last_config.max_output_tokens == 1000
    assert models.last_config.thinking_config.thinking_level.value == "LOW"


def test_generate_answer_does_not_sleep_on_excessive_retry_delay(monkeypatch):
    quota_error = ClientError(
        429,
        {
            "error": {
                "message": "daily quota exhausted",
                "details": [{"retryDelay": "63224s"}],
            }
        },
    )
    models = SimpleNamespace(
        generate_content=lambda **_kwargs: (_ for _ in ()).throw(quota_error)
    )
    monkeypatch.setattr(generate, "get_client", lambda: SimpleNamespace(models=models))
    sleep_calls = []
    monkeypatch.setattr(generate.time, "sleep", sleep_calls.append)

    try:
        generate.generate_answer("What was revenue?", [])
    except ClientError as exc:
        assert exc.code == 429
    else:
        raise AssertionError("expected excessive quota delay to propagate")

    assert sleep_calls == []
