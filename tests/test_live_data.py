import json

from agents.reasoning_agent import _build_context, _extract_citations
from live_data import LiveDataAgent
from pipeline import ChroniclePipeline
from retrieval.vector_store import ChunkResult


class FakeResponse:
    def __init__(self, text, status_code=200, headers=None, url="http://localhost/feed"):
        self.content = text.encode("utf-8")
        self.status_code = status_code
        self.headers = headers or {"ETag": '"v1"'}
        self.url = url
        self.encoding = "utf-8"

    @property
    def text(self):
        return self.content.decode("utf-8")

    def json(self):
        return json.loads(self.content.decode("utf-8"))

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


class FakeVectorStore:
    def __init__(self):
        self.added = []
        self.deleted = []

    def add_chunks(self, chunks, embeddings):
        assert len(chunks) == len(embeddings)
        self.added.extend(chunks)

    def delete_chunk_ids(self, ids):
        self.deleted.extend(ids)


class FakeBM25:
    def __init__(self):
        self.calls = []

    def replace_article_chunks(self, removed, chunks):
        self.calls.append((removed, list(chunks)))


def _source_config(path):
    path.write_text(json.dumps([{
        "id": "test-source", "kind": "web", "url": "http://localhost/feed",
        "title": "Test live feed", "publisher": "Test publisher",
        "refresh_minutes": 5, "keywords": ["test live"],
    }]), encoding="utf-8")


def test_live_refresh_is_incremental_and_provenanced(tmp_path):
    config_path = tmp_path / "sources.json"
    _source_config(config_path)
    response = {"text": "<html><body><main>" + "initial live financial news update " * 40 + "</main></body></html>"}

    def get(*args, **kwargs):
        return FakeResponse(response["text"])

    vector, bm25 = FakeVectorStore(), FakeBM25()
    agent = LiveDataAgent(
        vector, bm25, state_path=tmp_path / "state.sqlite3", sources_path=config_path,
        embedder=lambda texts: [[float(i)] for i, _ in enumerate(texts)], http_get=get,
        allow_insecure=True, allow_private=True,
    )
    first = agent.refresh_for_query("latest test live update")[0]
    assert first.status == "updated"
    assert first.documents_added == 1
    assert vector.added and vector.added[0].article_id.startswith("live:test-source:")
    assert bm25.calls[-1][0] == []

    unchanged = agent.refresh_for_query("latest test live update", force=True)[0]
    assert unchanged.status == "unchanged"

    response["text"] = "<html><body><main>" + "revised live financial news update with a new market signal " * 40 + "</main></body></html>"
    changed = agent.refresh_for_query("latest test live update", force=True)[0]
    assert changed.status == "updated"
    assert changed.documents_changed == 1
    assert vector.deleted
    assert bm25.calls[-1][0]  # old article chunks were replaced, not appended


class ProbingVectorStore(FakeVectorStore):
    """A local index that can say which chunks it holds, like VectorStore."""

    def has_chunk_ids(self, ids):
        held = {c.chunk_id for c in self.added} - set(self.deleted)
        return bool(ids) and set(ids) <= held


def test_fresh_index_is_refilled_even_when_durable_state_says_unchanged(tmp_path):
    """Source tracking survives in shared storage (Neon in production) while
    the vector index is per-container. A new container with an empty index
    must re-embed documents the shared state already records -- otherwise it
    reports 'Content hash unchanged' and serves an empty index forever."""
    config_path = tmp_path / "sources.json"
    _source_config(config_path)
    html = "<html><body><main>" + "durable state live financial update " * 40 + "</main></body></html>"
    kwargs = dict(
        state_path=tmp_path / "state.sqlite3", sources_path=config_path,
        embedder=lambda texts: [[float(i)] for i, _ in enumerate(texts)],
        http_get=lambda *a, **k: FakeResponse(html), allow_insecure=True, allow_private=True,
    )

    first_container = ProbingVectorStore()
    assert LiveDataAgent(first_container, FakeBM25(), **kwargs).refresh_for_query("x", force=True)[0].status == "updated"

    same_container = LiveDataAgent(first_container, FakeBM25(), **kwargs)
    assert same_container.refresh_for_query("x", force=True)[0].status == "unchanged"

    new_container = ProbingVectorStore()  # empty local index, same durable state
    report = LiveDataAgent(new_container, FakeBM25(), **kwargs).refresh_for_query("x", force=True)[0]
    assert report.status == "updated"
    assert new_container.added


def test_live_citations_are_exposed_to_the_reasoner():
    chunk = ChunkResult(
        chunk_id="x", text="A live safety update was published.",
        article_id="live:test-source:abc123", article_title="Test update",
        article_source="Test publisher", cluster="Live — Dynamic", url="https://example.test",
    )
    context = _build_context([chunk])
    assert "[EVIDENCE 1] [Source test-source/abc123]" in context
    assert _extract_citations("Fact [Source test-source/abc123]") == ["live:test-source/abc123"]
    answer = ChroniclePipeline._ensure_source_provenance("A grounded fact.", [chunk])
    assert "[Source test-source/abc123]" in answer


def test_live_source_lease_prevents_concurrent_refresh(tmp_path):
    vector, bm25 = FakeVectorStore(), FakeBM25()
    kwargs = dict(
        vector_store=vector, bm25_store=bm25, state_path=tmp_path / "state.sqlite3",
        sources_path=tmp_path / "missing.json", embedder=lambda texts: [[0.0] for _ in texts],
        allow_insecure=True, allow_private=True,
    )
    first = LiveDataAgent(**kwargs)
    second = LiveDataAgent(**kwargs)
    owner = first._acquire_lease("shared-source")
    assert owner
    assert second._acquire_lease("shared-source") is None
    first._release_lease("shared-source", owner)
    assert second._acquire_lease("shared-source")


# ── EDGAR / GDELT ingestion (Phase 1: domain conversion) ─────────────────────

_10K_HTML = (
    "<html><body>"
    "<p>Item 1. Business</p><p>We design, manufacture, and sell widgets and related accessories worldwide "
    "through our retail and online channels, competing on quality and price.</p>"
    "<p>Item 7. Management's Discussion and Analysis</p>"
    "<p>Revenue grew year over year, driven by higher unit volumes across all reportable segments.</p>"
    "<table><tr><td>Net sales</td><td>$9,999</td></tr></table>"
    "</body></html>"
)


def _edgar_submissions_json(filings: list[dict]) -> str:
    keys = ["form", "filingDate", "acceptanceDateTime", "accessionNumber", "primaryDocument", "primaryDocDescription", "reportDate"]
    recent = {k: [f[k] for f in filings] for k in keys}
    return json.dumps({"cik": "1", "name": "Test Co", "filings": {"recent": recent}})


def _edgar_fake_get(filings: list[dict], doc_html_by_filename: dict[str, str], exhibits_by_accession: dict[str, list[dict]] | None = None):
    """Fake EDGAR HTTP transport. `doc_html_by_filename` keys on the document's
    own filename (e.g. "test10k.htm"), not the accession, since multiple
    documents can share an accession folder and a naive accession-substring
    match would ambiguously return the wrong document (e.g. an 8-K's primary
    document instead of its exhibit)."""
    exhibits_by_accession = exhibits_by_accession or {}

    def get(url, headers=None, timeout=None, params=None, **kwargs):
        if "data.sec.gov/submissions" in url:
            return FakeResponse(_edgar_submissions_json(filings))
        if url.endswith("-index.json"):
            accession = url.rsplit("/", 1)[-1].replace("-index.json", "")
            return FakeResponse(json.dumps({"directory": {"item": exhibits_by_accession.get(accession, [])}}))
        for filename, html in doc_html_by_filename.items():
            if url.endswith(filename):
                return FakeResponse(html)
        raise AssertionError(f"unexpected EDGAR URL in test: {url}")

    return get


def test_edgar_ingestion_stamps_filed_at_from_acceptance_datetime_and_item_chunks(tmp_path):
    accession = "0000000001-24-000010"
    filings = [{
        "form": "10-K", "filingDate": "2024-02-10", "acceptanceDateTime": "2024-02-10T16:30:00.000Z",
        "accessionNumber": accession, "primaryDocument": "test10k.htm", "primaryDocDescription": "10-K", "reportDate": "2023-12-31",
    }]
    fake_get = _edgar_fake_get(filings, {"test10k.htm": _10K_HTML})

    config_path = tmp_path / "sources.json"
    config_path.write_text(json.dumps([{
        "id": "test-edgar", "kind": "edgar", "cik": "1", "url": "https://example.test",
        "title": "Test Co", "publisher": "SEC EDGAR", "forms": ["10-K"], "refresh_minutes": 5,
    }]), encoding="utf-8")

    vector, bm25 = FakeVectorStore(), FakeBM25()
    agent = LiveDataAgent(
        vector, bm25, state_path=tmp_path / "state.sqlite3", sources_path=config_path,
        embedder=lambda texts: [[float(i)] for i, _ in enumerate(texts)], http_get=fake_get,
        allow_insecure=True, allow_private=True, edgar_min_request_interval=0.0,
    )
    report = agent.refresh_for_query("latest filing", force=True)[0]
    assert report.status == "updated"
    assert vector.added, "expected chunks to be indexed"
    for chunk in vector.added:
        assert chunk.filed_at == "2024-02-10T16:30:00.000Z"
        assert chunk.form_type == "10-K"
        assert chunk.accession_number == accession
        assert chunk.is_amendment is False
    items = {c.item for c in vector.added}
    assert "Item 1" in items and "Item 7" in items
    revenue_chunk = next(c for c in vector.added if c.item == "Item 7")
    assert "$9,999" in revenue_chunk.text  # table number survived verbatim


def test_edgar_amendment_links_to_superseded_original_and_original_stays_retrievable(tmp_path):
    original_accession = "0000000001-24-000001"
    amendment_accession = "0000000001-24-000005"
    filings = [
        {  # amendment listed first (EDGAR returns newest-first)
            "form": "10-K/A", "filingDate": "2024-03-01", "acceptanceDateTime": "2024-03-01T12:00:00.000Z",
            "accessionNumber": amendment_accession, "primaryDocument": "amend10k.htm", "primaryDocDescription": "10-K/A", "reportDate": "2023-12-31",
        },
        {
            "form": "10-K", "filingDate": "2024-02-10", "acceptanceDateTime": "2024-02-10T16:30:00.000Z",
            "accessionNumber": original_accession, "primaryDocument": "orig10k.htm", "primaryDocDescription": "10-K", "reportDate": "2023-12-31",
        },
    ]
    fake_get = _edgar_fake_get(filings, {"orig10k.htm": _10K_HTML, "amend10k.htm": _10K_HTML})

    config_path = tmp_path / "sources.json"
    config_path.write_text(json.dumps([{
        "id": "test-edgar", "kind": "edgar", "cik": "1", "url": "https://example.test",
        "title": "Test Co", "publisher": "SEC EDGAR", "forms": ["10-K", "10-K/A"], "refresh_minutes": 5,
    }]), encoding="utf-8")

    vector, bm25 = FakeVectorStore(), FakeBM25()
    agent = LiveDataAgent(
        vector, bm25, state_path=tmp_path / "state.sqlite3", sources_path=config_path,
        embedder=lambda texts: [[float(i)] for i, _ in enumerate(texts)], http_get=fake_get,
        allow_insecure=True, allow_private=True, edgar_min_request_interval=0.0,
    )
    agent.refresh_for_query("latest filing", force=True)

    original_chunks = [c for c in vector.added if c.accession_number == original_accession]
    amendment_chunks = [c for c in vector.added if c.accession_number == amendment_accession]
    assert original_chunks, "original filing must remain indexed/retrievable, not vanish"
    assert amendment_chunks
    assert all(not c.is_amendment for c in original_chunks)
    assert all(c.is_amendment for c in amendment_chunks)
    assert all(c.supersedes_accession == original_accession for c in amendment_chunks)


def test_edgar_8k_exhibit_is_ingested_and_labelled_as_transcript_substitute(tmp_path):
    accession = "0000000001-24-000020"
    exhibit_name = "ex991pressrelease.htm"
    filings = [{
        "form": "8-K", "filingDate": "2024-04-01", "acceptanceDateTime": "2024-04-01T20:05:00.000Z",
        "accessionNumber": accession, "primaryDocument": "test8k.htm", "primaryDocDescription": "8-K", "reportDate": "",
    }]
    exhibit_items = [
        {"name": exhibit_name, "type": "EX-99.1"},
        {"name": "test8k.htm", "type": "8-K"},
    ]
    fake_get = _edgar_fake_get(
        filings,
        {
            "test8k.htm": "<html><body><p>Item 1. Business</p><p>Q1 operational update for the quarter.</p></body></html>",
            exhibit_name: "<html><body><p>Item 1. Business</p><p>This press release announces quarterly results.</p></body></html>",
        },
        exhibits_by_accession={accession: exhibit_items},
    )

    config_path = tmp_path / "sources.json"
    config_path.write_text(json.dumps([{
        "id": "test-edgar", "kind": "edgar", "cik": "1", "url": "https://example.test",
        "title": "Test Co", "publisher": "SEC EDGAR", "forms": ["8-K"], "refresh_minutes": 5,
    }]), encoding="utf-8")

    vector, bm25 = FakeVectorStore(), FakeBM25()
    agent = LiveDataAgent(
        vector, bm25, state_path=tmp_path / "state.sqlite3", sources_path=config_path,
        embedder=lambda texts: [[float(i)] for i, _ in enumerate(texts)], http_get=fake_get,
        allow_insecure=True, allow_private=True, edgar_min_request_interval=0.0,
    )
    agent.refresh_for_query("latest filing", force=True)
    titles = {c.article_title for c in vector.added}
    assert any("Exhibit" in t and "not a transcript" in t for t in titles)


def test_gdelt_ingestion_fetches_and_extracts_article_text_with_no_filed_at(tmp_path):
    article_url = "https://news.example.test/apple-earnings"
    article_html = "<html><body><main>" + "Apple reported strong quarterly results today. " * 20 + "</main></body></html>"
    gdelt_response = json.dumps({"articles": [{
        "url": article_url, "title": "Apple earnings beat", "seendate": "20240115T120000Z", "domain": "news.example.test",
    }]})

    def get(url, headers=None, timeout=None, params=None, allow_redirects=None, **kwargs):
        if "gdeltproject.org" in url:
            return FakeResponse(gdelt_response)
        if url == article_url:
            return FakeResponse(article_html, url=article_url)
        raise AssertionError(f"unexpected URL in test: {url}")

    config_path = tmp_path / "sources.json"
    config_path.write_text(json.dumps([{
        "id": "test-gdelt", "kind": "gdelt", "gdelt_query": "Apple earnings", "url": "https://example.test",
        "title": "Apple news", "publisher": "GDELT Project", "refresh_minutes": 5,
    }]), encoding="utf-8")

    vector, bm25 = FakeVectorStore(), FakeBM25()
    agent = LiveDataAgent(
        vector, bm25, state_path=tmp_path / "state.sqlite3", sources_path=config_path,
        embedder=lambda texts: [[float(i)] for i, _ in enumerate(texts)], http_get=get,
        allow_insecure=True, allow_private=True,
    )
    report = agent.refresh_for_query("latest filing", force=True)[0]
    assert report.status == "updated"
    assert vector.added
    assert all(c.filed_at == "" for c in vector.added)  # GDELT is never a point-in-time filing source
    assert any("Apple reported strong quarterly results" in c.text for c in vector.added)
