"""GDELT 2.0 DOC API client — free, no key required.

GDELT returns article *metadata* only (url, title, seendate, domain) — never
body text. `seendate` is when GDELT observed/indexed the article, which is an
honest proxy for recency but is NOT the article's original publish timestamp;
callers must not treat it as point-in-time filing truth (that role belongs
only to EDGAR's `acceptanceDateTime`, see edgar_client.py).
"""

from __future__ import annotations

import threading
import time
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


# GDELT's DOC API answers 429 to more than one request per ~5 s from an IP,
# and a refresh queries one source per company back to back. Calls are paced
# process-wide (several clients may exist) and a 429 is retried with backoff.
GDELT_MIN_INTERVAL_S = 5.5
GDELT_RETRIES = 3
_pace_lock = threading.Lock()
_last_call = 0.0


class GdeltClient:
    def __init__(
        self,
        http_get: Callable[..., Any] = requests.get,
        timeout: int | None = None,
        user_agent: str | None = None,
        sleep: Callable[[float], None] | None = None,
    ) -> None:
        self.http_get = http_get
        self.timeout = timeout or config.REQUEST_TIMEOUT
        self.user_agent = user_agent or config.LIVE_USER_AGENT
        # Pacing only applies to real HTTP; an injected fake (tests) runs unpaced.
        real_http = http_get is requests.get
        self._sleep = sleep or (time.sleep if real_http else (lambda _s: None))
        self._min_interval = GDELT_MIN_INTERVAL_S if (real_http or sleep) else 0.0

    def _get(self, params: dict[str, str]) -> Any:
        global _last_call
        for attempt in range(GDELT_RETRIES + 1):
            with _pace_lock:
                wait = _last_call + self._min_interval - time.monotonic()
                if wait > 0:
                    self._sleep(wait)
                _last_call = time.monotonic()
            resp = self.http_get(
                config.GDELT_DOC_API_URL,
                params=params,
                headers={"User-Agent": self.user_agent},
                timeout=self.timeout,
            )
            status = getattr(resp, "status_code", 200)
            # GDELT also signals the limit as a 200 with a plain-text notice
            # ("Please limit requests to one every 5 seconds..."), which would
            # otherwise parse as zero articles.
            text_notice = status == 200 and "limit requests" in str(getattr(resp, "text", ""))[:300].lower()
            if not (status == 429 or text_notice):
                return resp
            if attempt == GDELT_RETRIES:
                if text_notice:
                    raise requests.HTTPError("GDELT rate limit (plain-text notice) after retries")
                return resp
            retry_after = (getattr(resp, "headers", None) or {}).get("Retry-After", "")
            self._sleep(float(retry_after) if retry_after.isdigit() else 6.0 * 2**attempt)
        return resp

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
        resp = self._get(params)
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
