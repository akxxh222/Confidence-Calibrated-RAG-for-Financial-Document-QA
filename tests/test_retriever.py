from contextlib import contextmanager

from app.retrieval import retriever


class FakeCursor:
    def __init__(self):
        self.calls = []

    def execute(self, sql, params=None):
        self.calls.append((sql, params))

    def fetchall(self):
        return []


class FakeConnection:
    def __init__(self):
        self.cursor_instance = FakeCursor()

    @contextmanager
    def cursor(self):
        yield self.cursor_instance

    def close(self):
        pass


def test_retrieve_probes_all_ivfflat_lists_for_small_corpus(monkeypatch):
    conn = FakeConnection()
    monkeypatch.setattr(retriever, "get_connection", lambda: conn)
    monkeypatch.setattr(retriever, "embed_text", lambda _query: [0.0] * 768)

    retriever.retrieve("What was revenue?")

    first_sql, first_params = conn.cursor_instance.calls[0]
    assert "ivfflat.probes" in first_sql
    assert first_params == ("100",)
