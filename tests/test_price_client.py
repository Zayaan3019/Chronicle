"""Unit tests for price_client.py's Yahoo chart JSON parsing (mocked HTTP,
no network) — dependency-injected via the http_get callable, same pattern
used throughout live_data.py/edgar_client.py."""

from __future__ import annotations

from price_client import PriceClient


class _FakeResponse:
    def __init__(self, payload: dict):
        self._payload = payload
        self.status_code = 200

    def raise_for_status(self):
        pass

    def json(self):
        return self._payload


def test_daily_closes_parses_timestamps_and_closes():
    payload = {
        "chart": {
            "result": [
                {
                    "timestamp": [1704067200, 1704153600],  # 2024-01-01, 2024-01-02 UTC
                    "indicators": {"quote": [{"close": [100.0, 102.5]}]},
                }
            ]
        }
    }
    client = PriceClient(http_get=lambda *a, **k: _FakeResponse(payload))
    points = client.daily_closes("NVDA", period1=0, period2=1, )
    assert [p.date for p in points] == ["2024-01-01", "2024-01-02"]
    assert [p.close for p in points] == [100.0, 102.5]


def test_daily_closes_skips_null_close_values():
    payload = {
        "chart": {
            "result": [
                {
                    "timestamp": [1704067200, 1704153600],
                    "indicators": {"quote": [{"close": [100.0, None]}]},
                }
            ]
        }
    }
    client = PriceClient(http_get=lambda *a, **k: _FakeResponse(payload))
    points = client.daily_closes("NVDA", period1=0, period2=1)
    assert len(points) == 1
    assert points[0].close == 100.0


def test_daily_closes_empty_result_returns_empty_list():
    payload = {"chart": {"result": []}}
    client = PriceClient(http_get=lambda *a, **k: _FakeResponse(payload))
    assert client.daily_closes("NVDA", period1=0, period2=1) == []
