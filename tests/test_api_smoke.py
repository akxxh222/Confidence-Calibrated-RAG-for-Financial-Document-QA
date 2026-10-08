import pytest
from app.api.app import app


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


# NOTE: a full /ask smoke test requires a live DB + calibration model file
# and is intentionally excluded from unit CI (Section 15 scopes CI to
# lint/test/build, not integration tests against live infra). Add an
# integration test suite separately once the DB fixture/mocking strategy
# is decided, rather than hitting real Postgres/LLM APIs in CI.
