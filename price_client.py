"""Minimal free daily-close price fetcher — used ONLY by Phase 4's event
study to compute a forward N-day return per tone-delta observation. This is
deliberately not a general market-data integration: it fetches a close-price
time series for one ticker over one date range and nothing else. Extending
this into a backtest, portfolio, or trading-rule engine is explicitly out of
scope (MODIFICATION.md's non-goals — that is Arbitron's territory).

Uses Yahoo Finance's public chart JSON endpoint (query1.finance.yahoo.com) —
free, no API key, no registration. It is an unofficial/undocumented endpoint
(the same one the widely-used `yfinance` library wraps), not a stable
contractual API; documented here plainly rather than presented as an
official data vendor integration.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable

import requests

import config

YAHOO_CHART_URL = "https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"


@dataclass(frozen=True)
class PricePoint:
    date: str    # YYYY-MM-DD, UTC trading date
    close: float


class PriceClient:
    def __init__(
        self,
        http_get: Callable[..., Any] = requests.get,
        timeout: int | None = None,
    ) -> None:
        self.http_get = http_get
        self.timeout = timeout or config.REQUEST_TIMEOUT

    def daily_closes(self, ticker: str, period1: int, period2: int) -> list[PricePoint]:
        """`period1`/`period2`: Unix epoch seconds range (UTC)."""
        url = YAHOO_CHART_URL.format(ticker=ticker)
        headers = {"User-Agent": config.LIVE_USER_AGENT}
        resp = self.http_get(
            url,
            params={"period1": period1, "period2": period2, "interval": "1d"},
            headers=headers,
            timeout=self.timeout,
        )
        resp.raise_for_status()
        data = resp.json()
        results = data.get("chart", {}).get("result") or []
        if not results:
            return []
        result = results[0]
        timestamps = result.get("timestamp") or []
        quote_blocks = result.get("indicators", {}).get("quote") or [{}]
        closes = quote_blocks[0].get("close") or []
        points: list[PricePoint] = []
        for ts, close in zip(timestamps, closes):
            if close is None:
                continue
            date = datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d")
            points.append(PricePoint(date=date, close=float(close)))
        return points


def forward_return(points: list[PricePoint], event_date: str, horizon_days: int) -> float | None:
    """Return (close on/after event_date + horizon_days) / (close on/after
    event_date) - 1, using ONLY points already present in `points` — the
    caller is responsible for having fetched a window that does not extend
    the anchor search before event_date (see event_study.py, which fetches
    from event_date forward only — this keeps the whole computation
    point-in-time correct: no price data from before the filing date is
    ever used to pick the anchor)."""
    if not points:
        return None
    ordered = sorted(points, key=lambda p: p.date)
    anchor = next((p for p in ordered if p.date >= event_date), None)
    if anchor is None:
        return None
    target_date = _add_calendar_days(event_date, horizon_days)
    horizon_point = next((p for p in ordered if p.date >= target_date), None)
    if horizon_point is None or horizon_point.date == anchor.date:
        return None
    if anchor.close == 0:
        return None
    return (horizon_point.close - anchor.close) / anchor.close


def _add_calendar_days(date_str: str, days: int) -> str:
    from datetime import timedelta
    dt = datetime.strptime(date_str, "%Y-%m-%d") + timedelta(days=days)
    return dt.strftime("%Y-%m-%d")
