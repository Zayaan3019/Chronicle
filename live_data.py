"""Incremental, provenance-preserving live-data ingestion for Chronicle.

Sources are local JSON configuration (a company/CIK watchlist), not
LLM-discovered URLs. This is an important production boundary: an LLM must
never be able to make the server fetch an arbitrary URL. Each refresh hashes
normalized content, upserts only changed chunks, and records enough
provenance — including EDGAR's `acceptanceDateTime` as `filed_at` — to audit
every answer and enforce point-in-time correctness (retrieval/hybrid.py).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from contextlib import closing
import hashlib
import ipaddress
import json
import re
import socket
import sqlite3
import uuid
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urlparse
from xml.etree import ElementTree

from bs4 import BeautifulSoup
import requests

import config
import db_postgres
import edgar_parse
from edgar_client import EdgarClient
from gdelt_client import GdeltClient
from retrieval.bm25_store import BM25Store
from retrieval.vector_store import ChunkResult, VectorStore
from utils import chunk_text, embed_batch


_SOURCE_ID_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")
_FRESHNESS_RE = re.compile(
    r"\b(latest|live|current|today|newest|recent|up[- ]?to[- ]?date|breaking)\b",
    re.IGNORECASE,
)
_WORD_RE = re.compile(r"[a-z0-9]{3,}")


@dataclass(frozen=True)
class LiveSource:
    id: str
    url: str
    kind: str = "web"  # web | rss | edgar | gdelt
    title: str = "Live source"
    publisher: str = ""
    cluster: str = "Financial live sources"
    refresh_minutes: int = 60
    max_items: int = 20
    keywords: tuple[str, ...] = ()
    # kind == "edgar": zero-padded 10-digit CIK, pre-resolved (no network call
    # at config-load time — resolve tickers offline with edgar_client.resolve_cik).
    cik: str = ""
    forms: tuple[str, ...] = ()          # e.g. ("10-K", "10-K/A", "10-Q", "8-K"); defaults to config.EDGAR_TRACKED_FORMS
    # kind == "gdelt": free-text query passed to the GDELT DOC 2.0 API.
    gdelt_query: str = ""


@dataclass
class RefreshReport:
    source_id: str
    status: str  # updated | unchanged | skipped | error
    documents_added: int = 0
    documents_changed: int = 0
    message: str = ""


def _utcnow() -> str:
    return datetime.now(UTC).isoformat()


def _normalise_text(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def _content_hash(text: str) -> str:
    return hashlib.sha256(_normalise_text(text).encode("utf-8")).hexdigest()


def is_freshness_query(query: str) -> bool:
    """Public predicate used by the supervisor to reserve live-source context."""
    return bool(_FRESHNESS_RE.search(query or ""))


def _safe_url(url: str, allow_insecure: bool, allow_private: bool) -> None:
    parsed = urlparse(url)
    if parsed.scheme not in ({"https"} if not allow_insecure else {"https", "http"}):
        raise ValueError("live source URL must use HTTPS")
    if not parsed.hostname:
        raise ValueError("live source URL requires a hostname")
    if not allow_private:
        try:
            addresses = {info[4][0] for info in socket.getaddrinfo(parsed.hostname, None)}
        except socket.gaierror as exc:
            raise ValueError(f"cannot resolve live source host: {parsed.hostname}") from exc
        for address in addresses:
            ip = ipaddress.ip_address(address)
            if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved or ip.is_multicast:
                raise ValueError("live source host resolves to a non-public address")


class LiveDataAgent:
    """Coordinates safe live-source refreshes and incremental hybrid-index writes."""

    def __init__(
        self,
        vector_store: VectorStore,
        bm25_store: BM25Store,
        state_path: str | Path | None = None,
        sources_path: str | Path | None = None,
        embedder: Callable[[list[str]], list[list[float]]] = embed_batch,
        http_get: Callable[..., Any] = requests.get,
        allow_insecure: bool | None = None,
        allow_private: bool | None = None,
        edgar_min_request_interval: float | None = None,
        dsn: str | None = None,
    ) -> None:
        self.vector_store = vector_store
        self.bm25_store = bm25_store
        self.sources_path = Path(sources_path or config.LIVE_SOURCES_PATH)
        self.embedder = embedder
        self.http_get = http_get
        self.allow_insecure = config.ALLOW_INSECURE_LIVE_SOURCES if allow_insecure is None else allow_insecure
        self.allow_private = config.ALLOW_PRIVATE_LIVE_SOURCES if allow_private is None else allow_private
        self._edgar_min_request_interval = edgar_min_request_interval
        # Same rule as memory.py/decision_log.py: an explicit state_path (every
        # existing test passes one) always forces SQLite; Postgres is used only
        # when a DSN is available and nothing forced a specific sqlite file.
        self.dsn = dsn or (db_postgres.dsn() if state_path is None else None)
        self.backend = "postgres" if self.dsn else "sqlite"
        if self.backend == "sqlite":
            self.state_path = Path(state_path or config.LIVE_STATE_DB_PATH)
            self.state_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialise()
        self._edgar_client: EdgarClient | None = None
        self._gdelt_client: GdeltClient | None = None

    def _get_edgar_client(self) -> EdgarClient:
        if self._edgar_client is None:
            self._edgar_client = EdgarClient(http_get=self.http_get, min_request_interval=self._edgar_min_request_interval)
        return self._edgar_client

    def _get_gdelt_client(self) -> GdeltClient:
        if self._gdelt_client is None:
            self._gdelt_client = GdeltClient(http_get=self.http_get)
        return self._gdelt_client

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.state_path, timeout=15, isolation_level=None)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA busy_timeout = 15000")
        return conn

    def _initialise(self) -> None:
        if self.backend == "postgres":
            self._pg_initialise()
            return
        with closing(self._connect()) as conn:
            conn.execute("PRAGMA journal_mode = WAL")
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS live_source_state (
                    source_id TEXT PRIMARY KEY,
                    etag TEXT, last_modified TEXT, last_checked TEXT NOT NULL,
                    last_success TEXT, last_error TEXT,
                    lease_owner TEXT, lease_until TEXT
                )
                """
            )
            existing_columns = {row[1] for row in conn.execute("PRAGMA table_info(live_source_state)")}
            for column in ("lease_owner", "lease_until"):
                if column not in existing_columns:
                    conn.execute(f"ALTER TABLE live_source_state ADD COLUMN {column} TEXT")
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS live_documents (
                    source_id TEXT NOT NULL, item_key TEXT NOT NULL,
                    article_id TEXT NOT NULL, title TEXT NOT NULL, url TEXT NOT NULL,
                    published TEXT NOT NULL, content_hash TEXT NOT NULL,
                    chunk_ids_json TEXT NOT NULL, ingested_at TEXT NOT NULL,
                    PRIMARY KEY(source_id, item_key)
                )
                """
            )

    def _pg_initialise(self) -> None:
        with db_postgres.connect(self.dsn) as conn, conn.cursor() as cur:
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS live_source_state (
                    source_id TEXT PRIMARY KEY,
                    etag TEXT, last_modified TEXT, last_checked TEXT NOT NULL,
                    last_success TEXT, last_error TEXT,
                    lease_owner TEXT, lease_until TEXT
                )
                """
            )
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS live_documents (
                    source_id TEXT NOT NULL, item_key TEXT NOT NULL,
                    article_id TEXT NOT NULL, title TEXT NOT NULL, url TEXT NOT NULL,
                    published TEXT NOT NULL, content_hash TEXT NOT NULL,
                    chunk_ids_json TEXT NOT NULL, ingested_at TEXT NOT NULL,
                    PRIMARY KEY(source_id, item_key)
                )
                """
            )
            conn.commit()

    @staticmethod
    def load_sources(path: str | Path) -> list[LiveSource]:
        path = Path(path)
        if not path.exists():
            return []
        raw = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(raw, list):
            raise ValueError("live source config must be a JSON list")
        sources: list[LiveSource] = []
        for entry in raw:
            if not isinstance(entry, dict):
                raise ValueError("each live source must be an object")
            source_id = str(entry.get("id", "")).strip().lower()
            if not _SOURCE_ID_RE.fullmatch(source_id):
                raise ValueError(f"invalid live source id: {source_id!r}")
            kind = str(entry.get("kind", "web")).lower()
            if kind not in {"web", "rss", "edgar", "gdelt"}:
                raise ValueError(f"unsupported source kind for {source_id}: {kind}")
            cik = str(entry.get("cik", "")).strip()
            if kind == "edgar":
                if not cik.isdigit():
                    raise ValueError(f"edgar source {source_id} requires a pre-resolved numeric cik")
                cik = f"{int(cik):010d}"
            gdelt_query = str(entry.get("gdelt_query", "")).strip()
            if kind == "gdelt" and not gdelt_query:
                raise ValueError(f"gdelt source {source_id} requires gdelt_query")
            # edgar/gdelt sources don't fetch a single feed URL; url is kept as a
            # human-readable provenance pointer (company filings/news page).
            url = str(entry.get("url", "")).strip()
            if kind in {"web", "rss"} and not url:
                raise ValueError(f"source {source_id} has no URL")
            forms = tuple(str(f).upper() for f in entry.get("forms", []) if str(f).strip()) or config.EDGAR_TRACKED_FORMS
            sources.append(
                LiveSource(
                    id=source_id,
                    url=url,
                    kind=kind,
                    title=str(entry.get("title") or source_id),
                    publisher=str(entry.get("publisher") or ""),
                    cluster=str(entry.get("cluster") or "Financial live sources"),
                    refresh_minutes=max(config.LIVE_MIN_REFRESH_MINUTES, int(entry.get("refresh_minutes", 60))),
                    max_items=max(1, min(100, int(entry.get("max_items", 20)))),
                    keywords=tuple(str(k).lower() for k in entry.get("keywords", []) if str(k).strip()),
                    cik=cik,
                    forms=forms,
                    gdelt_query=gdelt_query,
                )
            )
        return sources

    def _lookup_live_document(self, source_id: str, item_key: str) -> dict | None:
        if self.backend == "postgres":
            with db_postgres.connect(self.dsn) as conn, conn.cursor() as cur:
                cur.execute(
                    "SELECT * FROM live_documents WHERE source_id = %s AND item_key = %s",
                    (source_id, item_key),
                )
                row = cur.fetchone()
            return dict(row) if row else None
        with closing(self._connect()) as conn:
            row = conn.execute(
                "SELECT * FROM live_documents WHERE source_id = ? AND item_key = ?",
                (source_id, item_key),
            ).fetchone()
        return dict(row) if row else None

    def _state(self, source_id: str) -> dict[str, str]:
        if self.backend == "postgres":
            with db_postgres.connect(self.dsn) as conn, conn.cursor() as cur:
                cur.execute("SELECT * FROM live_source_state WHERE source_id = %s", (source_id,))
                row = cur.fetchone()
            return dict(row) if row else {}
        with closing(self._connect()) as conn:
            row = conn.execute("SELECT * FROM live_source_state WHERE source_id = ?", (source_id,)).fetchone()
        return dict(row) if row else {}

    def _is_due(self, source: LiveSource, force: bool) -> bool:
        if force:
            return True
        state = self._state(source.id)
        # A lease creates the initial state row before the first HTTP attempt;
        # it must not make a never-refreshed source look fresh.
        if not state.get("last_success") and not state.get("last_error"):
            return True
        last_success = state.get("last_success") or state.get("last_checked")
        if not last_success:
            return True
        try:
            then = datetime.fromisoformat(last_success)
        except ValueError:
            return True
        return datetime.now(UTC) >= then + timedelta(minutes=source.refresh_minutes)

    def _acquire_lease(self, source_id: str) -> str | None:
        """Acquire a short cross-process lease before a network/index mutation."""
        if self.backend == "postgres":
            return self._pg_acquire_lease(source_id)
        owner = uuid.uuid4().hex
        now = datetime.now(UTC)
        expires = (now + timedelta(minutes=max(2, config.REQUEST_TIMEOUT // 15))).isoformat()
        with closing(self._connect()) as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                "SELECT lease_owner, lease_until FROM live_source_state WHERE source_id = ?", (source_id,)
            ).fetchone()
            available = not row or not row["lease_until"]
            if row and row["lease_until"]:
                try:
                    available = datetime.fromisoformat(row["lease_until"]) <= now
                except ValueError:
                    available = True
            if available:
                conn.execute(
                    """
                    INSERT INTO live_source_state(source_id, last_checked, lease_owner, lease_until)
                    VALUES (?, ?, ?, ?)
                    ON CONFLICT(source_id) DO UPDATE SET lease_owner=excluded.lease_owner, lease_until=excluded.lease_until
                    """,
                    (source_id, now.isoformat(), owner, expires),
                )
                conn.execute("COMMIT")
                return owner
            conn.execute("COMMIT")
        return None

    def _pg_acquire_lease(self, source_id: str) -> str | None:
        """Postgres version of the lease above, but as one atomic statement
        instead of SQLite's explicit BEGIN IMMEDIATE + read-then-write: the
        WHERE clause on DO UPDATE is re-checked against the row's current
        state under Postgres's own row lock, so two concurrent callers can't
        both observe "available" before either writes -- the same guarantee
        BEGIN IMMEDIATE gave, with no separate transaction-control statements
        needed. Comparing lease_until as ::timestamptz (not raw TEXT) avoids
        a lexicographic-comparison bug: Python's isoformat() omits
        microseconds when they're exactly zero, producing variable-length
        strings that would sort incorrectly as plain text.
        """
        owner = uuid.uuid4().hex
        now = datetime.now(UTC)
        expires = now + timedelta(minutes=max(2, config.REQUEST_TIMEOUT // 15))
        with db_postgres.connect(self.dsn) as conn, conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO live_source_state(source_id, last_checked, lease_owner, lease_until)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (source_id) DO UPDATE SET
                    last_checked = EXCLUDED.last_checked,
                    lease_owner  = EXCLUDED.lease_owner,
                    lease_until  = EXCLUDED.lease_until
                WHERE live_source_state.lease_until IS NULL
                   OR live_source_state.lease_until::timestamptz <= %s::timestamptz
                RETURNING lease_owner
                """,
                (source_id, now.isoformat(), owner, expires.isoformat(), now.isoformat()),
            )
            row = cur.fetchone()
            conn.commit()
            return owner if row else None

    def _release_lease(self, source_id: str, owner: str) -> None:
        if self.backend == "postgres":
            with db_postgres.connect(self.dsn) as conn, conn.cursor() as cur:
                cur.execute(
                    "UPDATE live_source_state SET lease_owner = NULL, lease_until = NULL "
                    "WHERE source_id = %s AND lease_owner = %s",
                    (source_id, owner),
                )
                conn.commit()
            return
        with closing(self._connect()) as conn:
            conn.execute(
                "UPDATE live_source_state SET lease_owner = NULL, lease_until = NULL "
                "WHERE source_id = ? AND lease_owner = ?",
                (source_id, owner),
            )

    def _source_matches_query(self, source: LiveSource, query: str) -> bool:
        terms = set(_WORD_RE.findall(query.lower()))
        source_terms = set(_WORD_RE.findall(f"{source.id} {source.title} {source.publisher}".lower()))
        return bool(terms & source_terms) or any(keyword in query.lower() for keyword in source.keywords)

    def refresh_for_query(self, query: str, force: bool = False) -> list[RefreshReport]:
        """Refresh sources relevant to an explicitly fresh/live user request."""
        if not config.LIVE_REFRESH_ON_QUERY and not force:
            return []
        sources = self.load_sources(self.sources_path)
        fresh_intent = is_freshness_query(query)
        selected = [s for s in sources if force or fresh_intent or self._source_matches_query(s, query)]
        return [self.refresh_source(source, force=force) for source in selected]

    def _fetch(self, source: LiveSource, state: dict[str, str]) -> tuple[int, str, dict[str, str], str]:
        _safe_url(source.url, self.allow_insecure, self.allow_private)
        headers = {"User-Agent": config.LIVE_USER_AGENT, "Accept": "application/rss+xml, application/xml, text/html;q=0.9"}
        if state.get("etag"):
            headers["If-None-Match"] = state["etag"]
        if state.get("last_modified"):
            headers["If-Modified-Since"] = state["last_modified"]
        response = self.http_get(source.url, headers=headers, timeout=config.REQUEST_TIMEOUT, allow_redirects=True)
        # Validate the eventual target too; this blocks redirects to an internal host.
        _safe_url(getattr(response, "url", source.url), self.allow_insecure, self.allow_private)
        if response.status_code == 304:
            return 304, "", {}, ""
        response.raise_for_status()
        content = response.content
        if len(content) > config.LIVE_MAX_RESPONSE_BYTES:
            raise ValueError("live source response exceeds configured size limit")
        encoding = getattr(response, "encoding", None) or "utf-8"
        text = content.decode(encoding, errors="replace")
        headers_out = {
            "etag": response.headers.get("ETag", ""),
            "last_modified": response.headers.get("Last-Modified", ""),
        }
        return response.status_code, text, headers_out, response.url

    def _fetch_url(self, url: str) -> str:
        """Generic safety-checked GET for a discovered URL (e.g. a GDELT article link).

        Unlike `_fetch`, this is not tied to a configured LiveSource's ETag
        state — GDELT discovers URLs dynamically, so each is a one-off fetch.
        The same public-address/HTTPS checks and byte-size cap still apply.
        """
        _safe_url(url, self.allow_insecure, self.allow_private)
        headers = {"User-Agent": config.LIVE_USER_AGENT}
        response = self.http_get(url, headers=headers, timeout=config.REQUEST_TIMEOUT, allow_redirects=True)
        _safe_url(getattr(response, "url", url), self.allow_insecure, self.allow_private)
        response.raise_for_status()
        content = response.content
        if len(content) > config.LIVE_MAX_RESPONSE_BYTES:
            raise ValueError("fetched response exceeds configured size limit")
        encoding = getattr(response, "encoding", None) or "utf-8"
        return content.decode(encoding, errors="replace")

    def _collect_edgar_documents(self, source: LiveSource) -> list[dict]:
        """Pull tracked-form filings for one company, Item-chunked, with filed_at.

        Reuses the caller's diff/upsert loop unchanged: each returned document
        is keyed by an `item_key` the existing content-hash logic can dedupe
        against exactly like an RSS entry.
        """
        if not source.cik:
            raise ValueError(f"edgar source {source.id} has no cik configured")
        client = self._get_edgar_client()
        forms = source.forms or config.EDGAR_TRACKED_FORMS
        filings = client.list_filings(source.cik, forms=forms, limit=source.max_items)

        # An amendment (10-K/A, 10-Q/A) supersedes the most recent ORIGINAL
        # filing covering the same fiscal period; the original stays retrievable.
        originals_by_period: dict[tuple[str, str], str] = {}
        for filing in filings:
            if not filing.is_amendment:
                originals_by_period[(filing.form, filing.report_date)] = filing.accession_number

        documents: list[dict] = []
        for filing in filings:
            supersedes = ""
            if filing.is_amendment:
                base_form = filing.form.replace("/A", "")
                supersedes = originals_by_period.get((base_form, filing.report_date), "")

            docs_to_fetch = [(filing.primary_document, "primary")]
            if filing.form == "8-K":
                # 8-K exhibits are the only transcript-adjacent source this
                # project uses — no free live earnings-call transcript exists.
                try:
                    for exhibit in client.get_8k_exhibits(filing.cik, filing.accession_number):
                        name = str(exhibit.get("name", ""))
                        if name and name != filing.primary_document:
                            docs_to_fetch.append((name, "exhibit"))
                except Exception:
                    pass

            for doc_name, doc_role in docs_to_fetch:
                if not doc_name:
                    continue
                try:
                    html = client.get_filing_document(filing.cik, filing.accession_number, doc_name)
                except Exception:
                    continue
                pre_chunks = [
                    {"item": c.item, "title": c.title, "text": c.text}
                    for c in edgar_parse.chunk_filing(html)
                ]
                if not pre_chunks:
                    continue
                role_suffix = " — Exhibit (earnings-call-transcript substitute, not a transcript)" if doc_role == "exhibit" else ""
                # `text` here drives the content-hash used for change detection
                # (see _refresh_source_locked); it is NOT what gets chunked —
                # pre_chunks (Item-boundary chunked) is. Deriving it from the
                # actual parsed content, rather than leaving it empty, means a
                # future edgar_parse.py change still safely reindexes old filings
                # instead of every EDGAR document hashing identically forever.
                documents.append({
                    "item_key": f"{filing.accession_number}:{doc_name}",
                    "title": f"{source.title} {filing.form} ({filing.report_date or filing.filing_date}){role_suffix}",
                    "url": client.filing_document_url(filing.cik, filing.accession_number, doc_name),
                    "published": filing.filed_at or filing.filing_date,
                    "text": "\n".join(c["text"] for c in pre_chunks)[:20000],
                    "pre_chunks": pre_chunks,
                    "filed_at": filing.filed_at,
                    "form_type": filing.form,
                    "accession_number": filing.accession_number,
                    "is_amendment": filing.is_amendment,
                    "supersedes_accession": supersedes,
                })
        return documents

    def _collect_gdelt_documents(self, source: LiveSource) -> list[dict]:
        """Discover recent news via GDELT, then fetch+extract each article's text.

        GDELT gives metadata only; `seendate` is an honest recency proxy, not a
        point-in-time filing timestamp — these documents carry filed_at="" and
        are never eligible for the as_of filing filter (see retrieval/hybrid.py).
        """
        client = self._get_gdelt_client()
        articles = client.search(source.gdelt_query, max_records=source.max_items)
        documents: list[dict] = []
        for article in articles:
            try:
                text = self._html_to_text(self._fetch_url(article.url))
            except Exception:
                continue
            if len(text) < 80:
                continue
            documents.append({
                "item_key": article.url,
                "title": article.title or source.title,
                "url": article.url,
                "published": article.seen_at or _utcnow(),
                "text": text,
                "filed_at": "",
                "form_type": "",
                "accession_number": "",
                "is_amendment": False,
                "supersedes_accession": "",
            })
        return documents

    @staticmethod
    def _html_to_text(html: str) -> str:
        try:
            import trafilatura
            extracted = trafilatura.extract(html, favor_recall=True)
            if extracted and len(extracted.strip()) > 100:
                return _normalise_text(extracted)
        except Exception:
            pass
        soup = BeautifulSoup(html, "lxml")
        for tag in soup(["script", "style", "nav", "footer", "header", "noscript"]):
            tag.decompose()
        return _normalise_text(soup.get_text(" ", strip=True))

    def _documents_from_response(self, source: LiveSource, body: str, fetched_url: str) -> list[dict[str, str]]:
        if source.kind == "web":
            text = self._html_to_text(body)
            if len(text) < 80:
                raise ValueError("live source yielded too little extractable text")
            return [{"item_key": "root", "title": source.title, "url": fetched_url, "published": _utcnow(), "text": text}]

        root = ElementTree.fromstring(body)
        entries = root.findall(".//item") or root.findall(".//{http://www.w3.org/2005/Atom}entry")
        documents = []
        for item in entries[:source.max_items]:
            title = (item.findtext("title") or item.findtext("{http://www.w3.org/2005/Atom}title") or source.title).strip()
            link = item.findtext("link") or ""
            atom_link = item.find("{http://www.w3.org/2005/Atom}link")
            if atom_link is not None:
                link = atom_link.attrib.get("href", link)
            item_key = (item.findtext("guid") or link or title).strip()
            description = item.findtext("description") or item.findtext("{http://www.w3.org/2005/Atom}summary") or ""
            published = (item.findtext("pubDate") or item.findtext("{http://www.w3.org/2005/Atom}updated") or _utcnow()).strip()
            text = self._html_to_text(description)
            if len(text) >= 40 and item_key:
                documents.append({"item_key": item_key, "title": title, "url": link or fetched_url, "published": published, "text": text})
        if not documents:
            raise ValueError("RSS source contained no usable entries")
        return documents

    @staticmethod
    def _article_id(source_id: str, item_key: str) -> str:
        return f"live:{source_id}:{hashlib.sha256(item_key.encode('utf-8')).hexdigest()[:12]}"

    def _make_chunks(self, source: LiveSource, document: dict, article_id: str, version: str) -> list[ChunkResult]:
        chunks: list[ChunkResult] = []
        provenance = (
            f"Live source: {source.title}. Publisher: {source.publisher or source.title}. "
            f"Published: {document['published']}. "
        )
        common = dict(
            article_id=article_id,
            article_title=document["title"],
            article_source=source.publisher or source.title,
            cluster=source.cluster,
            url=document["url"],
            date=document["published"],
            filed_at=document.get("filed_at", ""),
            form_type=document.get("form_type", ""),
            accession_number=document.get("accession_number", ""),
            supersedes_accession=document.get("supersedes_accession", ""),
            is_amendment=bool(document.get("is_amendment", False)),
        )

        pre_chunks = document.get("pre_chunks")
        if pre_chunks:
            # EDGAR path: already Item/section-boundary chunked (edgar_parse.py);
            # do not re-split by word count across an Item boundary.
            for number, section_chunk in enumerate(pre_chunks):
                item_label = section_chunk.get("item", "")
                item_prefix = f"[{item_label}] " if item_label and item_label not in ("FULL_DOCUMENT", "PREAMBLE") else ""
                chunks.append(
                    ChunkResult(
                        chunk_id=f"live_{source.id}_{article_id.rsplit(':', 1)[-1]}_{version[:12]}_{number:03d}",
                        text=provenance + item_prefix + section_chunk["text"],
                        chunk_num=number,
                        item=item_label,
                        **common,
                    )
                )
            return chunks

        for number, text in enumerate(chunk_text(document["text"])):
            chunks.append(
                ChunkResult(
                    chunk_id=f"live_{source.id}_{article_id.rsplit(':', 1)[-1]}_{version[:12]}_{number:03d}",
                    # Repeating lightweight provenance makes sparse retrieval work
                    # for a source-name query (e.g. "latest earnings-related news").
                    text=provenance + text,
                    chunk_num=number,
                    **common,
                )
            )
        return chunks

    def _write_state(self, source: LiveSource, headers: dict[str, str], error: str = "") -> None:
        now = _utcnow()
        values = (source.id, headers.get("etag", ""), headers.get("last_modified", ""), now, None if error else now, error or None)
        if self.backend == "postgres":
            with db_postgres.connect(self.dsn) as conn, conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO live_source_state(source_id, etag, last_modified, last_checked, last_success, last_error)
                    VALUES (%s, %s, %s, %s, %s, %s)
                    ON CONFLICT (source_id) DO UPDATE SET etag=EXCLUDED.etag, last_modified=EXCLUDED.last_modified,
                        last_checked=EXCLUDED.last_checked,
                        last_success=CASE WHEN EXCLUDED.last_success IS NULL THEN live_source_state.last_success ELSE EXCLUDED.last_success END,
                        last_error=EXCLUDED.last_error
                    """,
                    values,
                )
                conn.commit()
            return
        with closing(self._connect()) as conn:
            conn.execute(
                """
                INSERT INTO live_source_state(source_id, etag, last_modified, last_checked, last_success, last_error)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(source_id) DO UPDATE SET etag=excluded.etag, last_modified=excluded.last_modified,
                    last_checked=excluded.last_checked,
                    last_success=CASE WHEN excluded.last_success IS NULL THEN live_source_state.last_success ELSE excluded.last_success END,
                    last_error=excluded.last_error
                """,
                values,
            )

    def refresh_source(self, source: LiveSource, force: bool = False) -> RefreshReport:
        lease_owner = self._acquire_lease(source.id)
        if not lease_owner:
            return RefreshReport(source.id, "skipped", message="Another worker is refreshing this source")
        try:
            return self._refresh_source_locked(source, force)
        finally:
            self._release_lease(source.id, lease_owner)

    def _refresh_source_locked(self, source: LiveSource, force: bool = False) -> RefreshReport:
        """Refresh implementation; caller must hold the source lease."""
        if not self._is_due(source, force):
            return RefreshReport(source.id, "skipped", message="Refresh interval has not elapsed")
        state = self._state(source.id)
        try:
            headers: dict[str, str] = {}
            if source.kind in {"web", "rss"}:
                status, body, headers, fetched_url = self._fetch(source, state)
                if status == 304:
                    self._write_state(source, state)
                    return RefreshReport(source.id, "unchanged", message="Source returned HTTP 304")
                documents = self._documents_from_response(source, body, fetched_url)
            elif source.kind == "edgar":
                documents = self._collect_edgar_documents(source)
            elif source.kind == "gdelt":
                documents = self._collect_gdelt_documents(source)
            else:
                raise ValueError(f"unsupported source kind: {source.kind}")

            changed: list[tuple[dict, str, list[str], list[ChunkResult]]] = []
            additions = 0
            replacements = 0
            for document in documents:
                # Version includes provenance-template fields, so a code/config
                # change affecting retrieval representation safely reindexes old
                # live documents instead of treating them as unchanged forever.
                digest = _content_hash(
                    f"{source.title}|{source.publisher}|{source.cluster}|{document['text']}"
                )
                row = self._lookup_live_document(source.id, document["item_key"])
                if row and row["content_hash"] == digest:
                    continue
                article_id = row["article_id"] if row else self._article_id(source.id, document["item_key"])
                old_chunk_ids = json.loads(row["chunk_ids_json"]) if row else []
                chunks = self._make_chunks(source, document, article_id, digest)
                if not chunks:
                    continue
                changed.append((document, article_id, old_chunk_ids, chunks))
                additions += 0 if row else 1
                replacements += 1 if row else 0
            if not changed:
                self._write_state(source, headers)
                return RefreshReport(source.id, "unchanged", message="Content hash unchanged")

            # Index first, then commit source-state metadata. Upserts make retrying a
            # crash between the two phases safe and idempotent.
            old_ids = [chunk_id for _, _, old, _ in changed for chunk_id in old]
            if old_ids:
                self.vector_store.delete_chunk_ids(old_ids)
            new_chunks = [chunk for _, _, _, chunks in changed for chunk in chunks]
            self.vector_store.add_chunks(new_chunks, self.embedder([chunk.text for chunk in new_chunks]))
            self.bm25_store.replace_article_chunks(
                list({article_id for _, article_id, old, _ in changed if old}), new_chunks
            )
            now = _utcnow()
            doc_rows = [
                (source.id, document["item_key"], article_id, document["title"], document["url"], document["published"],
                 _content_hash(f"{source.title}|{source.publisher}|{source.cluster}|{document['text']}"), json.dumps([c.chunk_id for c in chunks]), now)
                for document, article_id, _, chunks in changed
            ]
            if self.backend == "postgres":
                with db_postgres.connect(self.dsn) as conn, conn.cursor() as cur:
                    for row in doc_rows:
                        cur.execute(
                            """
                            INSERT INTO live_documents(source_id, item_key, article_id, title, url, published, content_hash, chunk_ids_json, ingested_at)
                            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                            ON CONFLICT (source_id, item_key) DO UPDATE SET title=EXCLUDED.title, url=EXCLUDED.url,
                                published=EXCLUDED.published, content_hash=EXCLUDED.content_hash,
                                chunk_ids_json=EXCLUDED.chunk_ids_json, ingested_at=EXCLUDED.ingested_at
                            """,
                            row,
                        )
                    conn.commit()
            else:
                with closing(self._connect()) as conn:
                    conn.execute("BEGIN IMMEDIATE")
                    for row in doc_rows:
                        conn.execute(
                            """
                            INSERT INTO live_documents(source_id, item_key, article_id, title, url, published, content_hash, chunk_ids_json, ingested_at)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                            ON CONFLICT(source_id, item_key) DO UPDATE SET title=excluded.title, url=excluded.url,
                                published=excluded.published, content_hash=excluded.content_hash,
                                chunk_ids_json=excluded.chunk_ids_json, ingested_at=excluded.ingested_at
                            """,
                            row,
                        )
                    conn.execute("COMMIT")
            self._write_state(source, headers)
            return RefreshReport(source.id, "updated", additions, replacements, f"Indexed {len(new_chunks)} chunks")
        except Exception as exc:
            self._write_state(source, {}, error=str(exc)[:500])
            return RefreshReport(source.id, "error", message=str(exc))


if __name__ == "__main__":
    # Convenient scheduler/cron entry point: `python live_data.py --force`.
    import sys
    from indexer import load_index

    vector, bm25 = load_index()
    agent = LiveDataAgent(vector, bm25)
    reports = agent.refresh_for_query("latest filing", force="--force" in sys.argv)
    for report in reports:
        print(f"{report.source_id}: {report.status} ({report.message})")
