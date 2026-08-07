"""Rate-limited SEC EDGAR client.

SEC's fair-access guidance (https://www.sec.gov/os/webmaster-faq#developers)
requires a declarative User-Agent identifying the requester and describes
automated blocking of non-compliant clients. This module self-imposes a
request-interval floor (`config.EDGAR_MIN_REQUEST_INTERVAL_SECONDS`, default
~3 req/s — well under SEC's stated ceiling) and always sends a real,
project-identifying User-Agent. Every method here is a thin, testable wrapper
around one HTTP concern; no method fetches an LLM-chosen or otherwise
unconfigured URL.
"""

from __future__ import annotations

from dataclasses import dataclass
import threading
import time
from typing import Any, Callable

import requests

import config


@dataclass(frozen=True)
class FilingMeta:
    """One row from a company's EDGAR submissions history."""

    cik: str
    accession_number: str
    form: str
    filing_date: str          # calendar date EDGAR recorded (YYYY-MM-DD)
    filed_at: str              # exact acceptanceDateTime, ISO-8601 UTC — the point-in-time anchor
    primary_document: str
    primary_doc_description: str
    report_date: str           # fiscal period the filing covers (YYYY-MM-DD or "")
    is_amendment: bool
    supersedes_report_date: str = ""  # convenience copy of report_date for matching originals


class RateLimiter:
    """Simple thread-safe minimum-interval limiter (self-imposed SEC rate cap)."""

    def __init__(self, min_interval_seconds: float) -> None:
        self._min_interval = max(0.0, min_interval_seconds)
        self._lock = threading.Lock()
        self._last_call = 0.0

    def wait(self) -> None:
        with self._lock:
            now = time.monotonic()
            elapsed = now - self._last_call
            remaining = self._min_interval - elapsed
            if remaining > 0:
                time.sleep(remaining)
            self._last_call = time.monotonic()


class EdgarClient:
    """Rate-limited, real-UA client for the EDGAR endpoints this project uses."""

    def __init__(
        self,
        user_agent: str | None = None,
        min_request_interval: float | None = None,
        http_get: Callable[..., Any] = requests.get,
        timeout: int | None = None,
    ) -> None:
        self.user_agent = user_agent or config.EDGAR_USER_AGENT
        self.http_get = http_get
        self.timeout = timeout or config.REQUEST_TIMEOUT
        self._limiter = RateLimiter(
            config.EDGAR_MIN_REQUEST_INTERVAL_SECONDS if min_request_interval is None else min_request_interval
        )

    def _get(self, url: str, **kwargs) -> requests.Response:
        self._limiter.wait()
        headers = {"User-Agent": self.user_agent, "Accept-Encoding": "gzip, deflate"}
        headers.update(kwargs.pop("headers", {}) or {})
        resp = self.http_get(url, headers=headers, timeout=self.timeout, **kwargs)
        resp.raise_for_status()
        return resp

    # ── Ticker → CIK resolution ──────────────────────────────────────────

    def resolve_cik(self, ticker: str) -> str | None:
        """Resolve a ticker to a zero-padded 10-digit CIK via SEC's static map."""
        resp = self._get(config.EDGAR_TICKER_MAP_URL)
        data = resp.json()
        ticker_upper = ticker.strip().upper()
        for entry in data.values():
            if str(entry.get("ticker", "")).upper() == ticker_upper:
                return f"{int(entry['cik_str']):010d}"
        return None

    # ── Filing metadata ───────────────────────────────────────────────────

    def get_submissions(self, cik: str) -> dict:
        """Fetch a company's filing history/metadata (includes acceptanceDateTime)."""
        url = config.EDGAR_SUBMISSIONS_URL.format(cik=int(cik))
        return self._get(url).json()

    def list_filings(self, cik: str, forms: tuple[str, ...] | None = None, limit: int = 40) -> list[FilingMeta]:
        """Return recent filings for `cik`, filtered to `forms`, newest first."""
        forms = tuple(f.upper() for f in (forms or config.EDGAR_TRACKED_FORMS))
        data = self.get_submissions(cik)
        recent = data.get("filings", {}).get("recent", {})
        n = len(recent.get("accessionNumber", []))
        out: list[FilingMeta] = []
        for i in range(n):
            form = str(recent["form"][i])
            if form.upper() not in forms:
                continue
            out.append(
                FilingMeta(
                    cik=str(int(cik)),
                    accession_number=recent["accessionNumber"][i],
                    form=form,
                    filing_date=recent.get("filingDate", [""] * n)[i],
                    filed_at=recent.get("acceptanceDateTime", [""] * n)[i],
                    primary_document=recent.get("primaryDocument", [""] * n)[i],
                    primary_doc_description=recent.get("primaryDocDescription", [""] * n)[i],
                    report_date=recent.get("reportDate", [""] * n)[i],
                    is_amendment=form.upper().endswith("/A"),
                )
            )
            if len(out) >= limit:
                break
        return out

    # ── Filing document / exhibit retrieval ──────────────────────────────

    @staticmethod
    def _accession_no_dashes(accession_number: str) -> str:
        return accession_number.replace("-", "")

    def filing_document_url(self, cik: str, accession_number: str, document: str) -> str:
        return f"{config.EDGAR_ARCHIVES_BASE}/{int(cik)}/{self._accession_no_dashes(accession_number)}/{document}"

    def get_filing_document(self, cik: str, accession_number: str, document: str) -> str:
        """Fetch the raw HTML/text of one document within a filing's accession folder."""
        return self._get(self.filing_document_url(cik, accession_number, document)).text

    def get_filing_index(self, cik: str, accession_number: str) -> list[dict]:
        """List every document (including exhibits) in a filing's accession folder."""
        url = f"{config.EDGAR_ARCHIVES_BASE}/{int(cik)}/{self._accession_no_dashes(accession_number)}/{accession_number}-index.json"
        data = self._get(url).json()
        return data.get("directory", {}).get("item", [])

    def get_8k_exhibits(self, cik: str, accession_number: str) -> list[dict]:
        """Return EX-99.* exhibits from an 8-K — the only transcript-adjacent source.

        No free live earnings-call transcript feed exists; an EX-99 press-release
        exhibit is the honest substitute this project uses. It is not a transcript
        and must not be presented as one (see README).
        """
        items = self.get_filing_index(cik, accession_number)
        exhibits = []
        for item in items:
            item_type = str(item.get("type", "")).upper()
            name = str(item.get("name", ""))
            if item_type.startswith("EX-99") or "EX-99" in name.upper():
                exhibits.append(item)
        return exhibits

    # ── Discovery: full text search ───────────────────────────────────────

    def full_text_search(
        self, query: str, forms: tuple[str, ...] | None = None, start_date: str = "", end_date: str = "", limit: int = 10
    ) -> list[dict]:
        """Query efts.sec.gov full-text search for discovery (not point-in-time truth)."""
        params: dict[str, Any] = {"q": query, "forms": ",".join(forms or ())}
        if start_date and end_date:
            params["dateRange"] = "custom"
            params["startdt"] = start_date
            params["enddt"] = end_date
        resp = self._get(config.EDGAR_FULLTEXT_SEARCH_URL, params=params)
        hits = resp.json().get("hits", {}).get("hits", [])
        return hits[:limit]
