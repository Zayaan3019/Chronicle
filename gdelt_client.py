"""GDELT 2.0 DOC API client — free, no key required.

GDELT returns article *metadata* only (url, title, seendate, domain) — never
body text. `seendate` is when GDELT observed/indexed the article, which is an
honest proxy for recency but is NOT the article's original publish timestamp;
callers must not treat it as point-in-time filing truth (that role belongs
only to EDGAR's `acceptanceDateTime`, see edgar_client.py).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Callable

import requests

import config


@dataclass(frozen=True)
class GdeltArticle:
    url: str
    title: str
    seen_at: str   # ISO-8601 UTC, converted from GDELT's "20260802T080000Z" format
    domain: str


def _parse_seendate(raw: str) -> str:
    try:
        dt = datetime.strptime(raw, "%Y%m%dT%H%M%SZ").replace(tzinfo=UTC)
        return dt.isoformat()
    except (ValueError, TypeError):
        return ""


class GdeltClient:
    def __init__(
        self,
        http_get: Callable[..., Any] = requests.get,
        timeout: int | None = None,
        user_agent: str | None = None,
    ) -> None:
        self.http_get = http_get
        self.timeout = timeout or config.REQUEST_TIMEOUT
        self.user_agent = user_agent or config.LIVE_USER_AGENT

    def search(
        self,
        query: str,
        max_records: int | None = None,
        timespan: str | None = None,
    ) -> list[GdeltArticle]:
        params = {
            "query": query,
            "mode": "ArtList",
            "format": "json",
            "maxrecords": str(max_records or config.GDELT_MAX_RECORDS),
            "timespan": timespan or config.GDELT_DEFAULT_TIMESPAN,
        }
        resp = self.http_get(
            config.GDELT_DOC_API_URL,
            params=params,
            headers={"User-Agent": self.user_agent},
            timeout=self.timeout,
        )
        resp.raise_for_status()
        try:
            data = resp.json()
        except ValueError:
            return []
        articles = []
        for item in data.get("articles", []) or []:
            articles.append(
                GdeltArticle(
                    url=str(item.get("url", "")),
                    title=str(item.get("title", "")),
                    seen_at=_parse_seendate(str(item.get("seendate", ""))),
                    domain=str(item.get("domain", "")),
                )
            )
        return [a for a in articles if a.url]
