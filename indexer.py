"""Dynamic-index lifecycle helpers.

There is intentionally no fixed corpus in this application. The index starts
empty and is populated only by `LiveDataAgent` from the local trusted-source
registry (`live_sources.json`: SEC EDGAR filing trackers + GDELT news
trackers). This module keeps a small compatibility surface for the pipeline
and CLI users that previously imported `load_index`.

The filing-structure-aware chunking (Item/section boundaries, table-preserving
text, sub-window sizing) lives in `edgar_parse.py`, not here — this module has
never held chunking logic; `live_data.LiveDataAgent._collect_edgar_documents`
calls `edgar_parse.chunk_filing()` directly during ingestion.
"""

from __future__ import annotations

from retrieval.bm25_store import BM25Store
from retrieval.vector_store import VectorStore


def load_index() -> tuple[VectorStore, BM25Store]:
    vector_store = VectorStore()
    bm25_store = BM25Store()
    bm25_store.load()  # An empty first-run index is valid.
    return vector_store, bm25_store


def get_index_stats() -> dict[str, int | bool]:
    vector_store, bm25_store = load_index()
    return {
        "vector_chunks": vector_store.count(),
        "bm25_ready": bm25_store.is_populated,
    }


def refresh_sources(force: bool = False):
    """Refresh configured live sources; use this from a scheduler or deployment job."""
    from live_data import LiveDataAgent

    vector_store, bm25_store = load_index()
    return LiveDataAgent(vector_store, bm25_store).refresh_for_query(
        "latest SEC filings and financial news", force=force
    )


if __name__ == "__main__":
    import sys

    for report in refresh_sources(force="--force" in sys.argv):
        print(f"{report.source_id}: {report.status} ({report.message})")
