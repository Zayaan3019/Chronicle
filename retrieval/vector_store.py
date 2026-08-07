"""
ChromaDB persistent vector store wrapper.
Embeddings are pre-computed with Ollama and stored directly in Chroma.
"""

from __future__ import annotations
import os
from dataclasses import dataclass, field
from typing import Optional

import chromadb

import config
import pit


@dataclass
class ChunkResult:
    chunk_id:       str
    text:           str
    article_id:     str
    article_title:  str
    article_source: str
    cluster:        str
    url:            str
    date:           str   = ""    # article publication date (e.g. "Apr 2025")
    chunk_num:      int   = 0     # zero-based chunk index within the article
    score:          float = 0.0   # distance (lower = better for cosine)
    rrf_score:      float = 0.0   # filled in by hybrid search / reranker
    rank:           int   = 0

    # ── Point-in-time / financial-filing provenance ───────────────────────
    filed_at:            str = ""   # EDGAR acceptanceDateTime (ISO-8601 UTC) — the point-in-time anchor. "" for non-filing sources (e.g. GDELT news).
    form_type:            str = ""  # "10-K", "10-K/A", "10-Q", "8-K", "" for non-EDGAR
    accession_number:     str = ""  # EDGAR accession number, "" for non-EDGAR
    item:                  str = ""  # e.g. "Item 1A" — the Item/section this chunk came from
    supersedes_accession:  str = ""  # accession number of the original filing this amendment supersedes, "" if not an amendment
    is_amendment:          bool = False

    # ── Runtime-only supersession annotation (Phase 2) ─────────────────────
    # Populated by HybridRetriever at query time when a visible amendment
    # supersedes this chunk's filing; never persisted to Chroma (add_chunks
    # does not write these fields to metadata — see search() below).
    superseded_by_accession: str = ""
    superseded_by_form:      str = ""
    superseded_by_filed_at:  str = ""

    @property
    def citation_label(self) -> str:
        """Stable human-readable provenance label for configured source documents."""
        if self.article_id.startswith("live:"):
            _, source_id, item_id = self.article_id.split(":", 2)
            return f"Source {source_id}/{item_id}"
        return f"Source {self.article_id}"


def _point_in_time_where(as_of_epoch: Optional[float]) -> Optional[dict]:
    """Hard pre-ranking filter: filed_at_ts <= as_of_epoch, OR no filed_at at all.

    This is a real Chroma `where` clause evaluated by the query engine before
    any similarity ranking happens — not a post-hoc filter applied to results
    after the fact (Phase 2's binding requirement).
    """
    if as_of_epoch is None:
        return None
    return {
        "$or": [
            {"filed_at_ts": {"$lte": as_of_epoch}},
            {"filed_at_ts": {"$eq": pit.NO_FILED_AT_SENTINEL}},
        ]
    }


def _combine_where(a: Optional[dict], b: Optional[dict]) -> Optional[dict]:
    if a and b:
        return {"$and": [a, b]}
    return a or b


class VectorStore:
    COLLECTION_NAME = "chronicle_filings_v1"

    def __init__(self):
        os.makedirs(config.CHROMA_PERSIST_DIR, exist_ok=True)
        self._client = chromadb.PersistentClient(path=config.CHROMA_PERSIST_DIR)
        self._col = self._client.get_or_create_collection(
            name=self.COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )

    # ── Write ──────────────────────────────────────────────────────────────────

    def add_chunks(
        self,
        chunks: list[ChunkResult],
        embeddings: list[list[float]],
    ) -> None:
        if not chunks:
            return
        # Upsert makes a live refresh idempotent and also protects a restarted
        # refresh worker from duplicate-ID failures.
        self._col.upsert(
            ids        = [c.chunk_id for c in chunks],
            embeddings = embeddings,
            documents  = [c.text for c in chunks],
            metadatas  = [
                {
                    "article_id":            c.article_id,
                    "article_title":         c.article_title,
                    "article_source":        c.article_source,
                    "cluster":               c.cluster,
                    "url":                   c.url,
                    "date":                  c.date,
                    "chunk_num":             c.chunk_num,
                    "filed_at":              c.filed_at,
                    "form_type":             c.form_type,
                    "accession_number":      c.accession_number,
                    "item":                  c.item,
                    "supersedes_accession":  c.supersedes_accession,
                    "is_amendment":          c.is_amendment,
                    # Numeric mirror of filed_at for Chroma's $lte range filter
                    # (Chroma comparison operators require numeric metadata;
                    # string-compare on ISO-8601 is not a documented guarantee).
                    # NO_FILED_AT_SENTINEL marks chunks with no point-in-time
                    # anchor (e.g. GDELT news) so the as_of filter can exempt them.
                    "filed_at_ts": (
                        pit.parse_iso_to_epoch(c.filed_at)
                        if pit.parse_iso_to_epoch(c.filed_at) is not None
                        else pit.NO_FILED_AT_SENTINEL
                    ),
                }
                for c in chunks
            ],
        )

    def delete_chunk_ids(self, chunk_ids: list[str]) -> None:
        """Delete exactly the supplied chunks (used when a live document changes)."""
        if chunk_ids:
            self._col.delete(ids=chunk_ids)

    def clear(self) -> None:
        self._client.delete_collection(self.COLLECTION_NAME)
        self._col = self._client.get_or_create_collection(
            name=self.COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )

    # ── Read ───────────────────────────────────────────────────────────────────

    def search(
        self,
        query_embedding: list[float],
        n_results: int = config.TOP_K_DENSE,
        where: Optional[dict] = None,
        as_of_epoch: Optional[float] = None,
    ) -> list[ChunkResult]:
        total = self.count()
        if total == 0:
            return []
        safe_n = min(n_results, total)
        final_where = _combine_where(where, _point_in_time_where(as_of_epoch))
        kwargs: dict = dict(
            query_embeddings = [query_embedding],
            n_results        = safe_n,
            include          = ["documents", "metadatas", "distances"],
        )
        if final_where:
            kwargs["where"] = final_where

        res = self._col.query(**kwargs)

        results = []
        for chunk_id, doc, meta, dist in zip(
            res["ids"][0],
            res["documents"][0],
            res["metadatas"][0],
            res["distances"][0],
        ):
            results.append(
                ChunkResult(
                    chunk_id             = chunk_id,
                    text                 = doc,
                    article_id           = meta["article_id"],
                    article_title        = meta["article_title"],
                    article_source       = meta["article_source"],
                    cluster              = meta["cluster"],
                    url                  = meta["url"],
                    date                 = meta.get("date", ""),
                    chunk_num            = int(meta.get("chunk_num", 0)),
                    score                = dist,
                    filed_at             = meta.get("filed_at", ""),
                    form_type            = meta.get("form_type", ""),
                    accession_number     = meta.get("accession_number", ""),
                    item                 = meta.get("item", ""),
                    supersedes_accession = meta.get("supersedes_accession", ""),
                    is_amendment         = bool(meta.get("is_amendment", False)),
                )
            )
        return results

    def count(self) -> int:
        return self._col.count()

    @property
    def is_populated(self) -> bool:
        return self.count() > 0
