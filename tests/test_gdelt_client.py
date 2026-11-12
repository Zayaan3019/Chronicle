"""GdeltClient's handling of GDELT's per-IP rate limit (HTTP 429)."""
from gdelt_client import GDELT_RETRIES, GdeltClient


class _Resp:
    def __init__(self, status, payload=None, headers=None):
        self.status_code = status
        self._payload = payload or {}
        self.headers = headers or {}

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def json(self):
        return self._payload


ARTICLE = {"url": "https://example.org/a", "title": "t", "seendate": "20261111T120000Z", "domain": "example.org"}


def test_429_is_retried_with_backoff_then_succeeds():
    responses = [_Resp(429), _Resp(429, headers={"Retry-After": "2"}), _Resp(200, {"articles": [ARTICLE]})]
    slept = []
    client = GdeltClient(http_get=lambda *a, **k: responses.pop(0), sleep=slept.append)

    articles = client.search("Apple")

    assert [a.url for a in articles] == ["https://example.org/a"]
    assert articles[0].seen_at == "2026-11-11T12:00:00+00:00"
    assert 6.0 in slept and 2.0 in slept  # exponential default, then the server's Retry-After


def test_persistent_429_surfaces_as_an_error():
    calls = []

    def always_429(*a, **k):
        calls.append(1)
        return _Resp(429)

    client = GdeltClient(http_get=always_429, sleep=lambda _s: None)
    try:
        client.search("Apple")
    except RuntimeError as exc:
        assert "429" in str(exc)
    else:
        raise AssertionError("a persistent 429 must not be reported as zero articles")
    assert len(calls) == GDELT_RETRIES + 1
