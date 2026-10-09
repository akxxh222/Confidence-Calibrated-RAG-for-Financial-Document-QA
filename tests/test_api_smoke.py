import importlib

import pytest
from app.api.app import app
from app.retrieval.retriever import RetrievedChunk, RetrievalResult


api_module = importlib.import_module("app.api.app")


@pytest.fixture
def client():
    app.config["TESTING"] = True
    with app.test_client() as c:
        yield c


def test_health_endpoint(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.get_json()["status"] == "ok"


def test_ask_requires_question(client):
    resp = client.post("/ask", json={})
    assert resp.status_code == 400


def test_research_console_is_served_at_root(client):
    resp = client.get("/")

    assert resp.status_code == 200
    assert b"Filing Research Console" in resp.data
    assert b'id="research-form"' in resp.data


def test_query_api_requires_nonempty_json_question(client):
    assert client.post("/api/query", json={}).status_code == 400
    assert client.post("/api/query", json={"question": "  "}).status_code == 400
    assert client.post(
        "/api/query", data="not-json", content_type="text/plain"
    ).status_code == 400


def test_query_api_returns_normal_rag_answer_and_sources(client, monkeypatch):
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
    monkeypatch.setattr(
        api_module,
        "retrieve",
        lambda _question: RetrievalResult(chunks, top1_score=0.82, spread=0.07),
    )
    monkeypatch.setattr(
        api_module,
        "generate_answer",
        lambda _question, _chunks: "Microsoft reported $331,839 million.",
    )

    resp = client.post("/api/query", json={"question": "What was revenue?"})

    assert resp.status_code == 200
    assert resp.get_json() == {
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


# NOTE: a full /ask smoke test requires a live DB + calibration model file
# and is intentionally excluded from unit CI (Section 15 scopes CI to
# lint/test/build, not integration tests against live infra). Add an
# integration test suite separately once the DB fixture/mocking strategy
# is decided, rather than hitting real Postgres/LLM APIs in CI.
