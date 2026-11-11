import importlib
from types import SimpleNamespace

import pytest

from app.api.app import app
from app.generation.self_consistency import SelfConsistencyResult
from app.retrieval.retriever import RetrievedChunk, RetrievalResult


api_module = importlib.import_module("app.api.app")


@pytest.fixture
def client():
    app.config["TESTING"] = True
    with app.test_client() as test_client:
        yield test_client


def sample_retrieval():
    chunks = [
        RetrievedChunk(
            chunk_id=11,
            doc_id=7,
            section="financial_statements",
            text="Total revenue was $331,839 million.",
            page_ref="50",
            similarity=0.82,
        )
    ]
    return RetrievalResult(chunks, top1_score=0.82, spread=0.07)


def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.get_json()["status"] == "ok"


def test_ask_requires_json_question(client):
    assert client.post("/ask", json={}).status_code == 400
    assert client.post("/ask", data="text").status_code == 400


def test_research_console_uses_calibrated_endpoint(client):
    response = client.get("/")
    script = client.get("/static/app.js")

    assert response.status_code == 200
    assert b"Filing Research Console" in response.data
    assert b'fetch("/ask"' in script.data
    assert b"reviewed_demo" not in script.data


def test_source_cards_do_not_render_missing_page_placeholder(client):
    response = client.get("/static/app.js")
    assert response.status_code == 200
    assert b"Page not supplied" not in response.data


def test_query_api_returns_live_generated_answer(client, monkeypatch):
    monkeypatch.setattr(api_module, "retrieve", lambda _question: sample_retrieval())
    monkeypatch.setattr(
        api_module,
        "generate_answer",
        lambda _question, _chunks: "Microsoft reported $331,839 million.",
    )

    response = client.post("/api/query", json={"question": "What was revenue?"})

    assert response.status_code == 200
    assert response.get_json() == {
        "answer": "Microsoft reported $331,839 million.",
        "question": "What was revenue?",
        "retrieval_confidence": 0.82,
        "spread": 0.07,
        "sources": [
            {
                "chunk_id": 11,
                "document_id": 7,
                "page_ref": "50",
                "section": "financial_statements",
                "similarity": 0.82,
            }
        ],
    }


@pytest.mark.parametrize(
    ("probability", "mode", "answer_prefix"),
    [
        (0.8, "answer", "Microsoft reported"),
        (0.4, "hedge", "I'm not fully confident"),
        (0.1, "refuse", "I don't have sufficient confidence"),
    ],
)
def test_ask_runs_calibrated_routing(
    client, monkeypatch, probability, mode, answer_prefix
):
    monkeypatch.setattr(api_module, "retrieve", lambda _question: sample_retrieval())
    monkeypatch.setattr(
        api_module,
        "run_self_consistency",
        lambda _question, _chunks: SelfConsistencyResult(
            samples=["Microsoft reported $331,839 million."],
            self_consistency_score=0.9,
            extracted_numbers=[331_839_000_000.0],
        ),
    )
    model = SimpleNamespace(
        predict_proba=lambda _features: [probability],
        decide=lambda _probability: mode,
    )
    monkeypatch.setattr(api_module, "get_calibration_model", lambda: model)

    response = client.post("/ask", json={"question": "What was revenue?"})
    payload = response.get_json()

    assert response.status_code == 200
    assert payload["mode"] == mode
    assert payload["answer"].startswith(answer_prefix)
    assert payload["confidence_score"] == probability
    assert payload["retrieval_confidence"] == 0.82
    assert payload["spread"] == 0.07
    assert payload["self_consistency_score"] == 0.9
    assert payload["sources"][0]["similarity"] == 0.82


@pytest.mark.parametrize(
    ("code", "status", "message"),
    [(429, 429, "quota"), (504, 504, "timed out"),
     (503, 503, "temporarily unavailable")],
)
def test_provider_failure_is_json(client, monkeypatch, code, status, message):
    from google.genai.errors import APIError

    monkeypatch.setattr(api_module, "retrieve", lambda _question: sample_retrieval())

    def unavailable(_question, _chunks):
        raise APIError(code, {"error": {"message": "provider failure"}})

    monkeypatch.setattr(api_module, "run_self_consistency", unavailable)
    response = client.post("/ask", json={"question": "What was revenue?"})

    assert response.status_code == status
    assert response.is_json
    assert message in response.get_json()["error"]
