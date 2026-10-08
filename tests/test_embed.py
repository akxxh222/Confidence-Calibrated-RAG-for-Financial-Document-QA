from unittest.mock import Mock

from google.genai.errors import ClientError

from app.retrieval import embed


def test_quota_retry_honors_api_retry_delay(monkeypatch):
    quota_error = ClientError(
        429,
        {"error": {"message": "quota", "details": [{"retryDelay": "2s"}]}},
    )
    request = Mock(side_effect=[quota_error, [[1.0, 0.0]]])
    sleep = Mock()
    monkeypatch.setattr(embed, "_request_embeddings", request)
    monkeypatch.setattr(embed.time, "sleep", sleep)

    assert embed._request_embeddings_with_retry(["revenue"]) == [[1.0, 0.0]]
    sleep.assert_called_once_with(3.0)


def test_quota_retry_does_not_swallow_non_quota_errors(monkeypatch):
    api_error = ClientError(400, {"error": {"message": "bad request"}})
    monkeypatch.setattr(embed, "_request_embeddings", Mock(side_effect=api_error))

    try:
        embed._request_embeddings_with_retry(["revenue"])
    except ClientError as exc:
        assert exc.code == 400
    else:
        raise AssertionError("expected the non-quota API error to propagate")
