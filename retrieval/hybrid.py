"""
Hybrid retrieval:  dense (ChromaDB cosine) + sparse (BM25) fused via RRF,
then optional cross-encoder re-ranking.

Bonus feature: demonstrates hybrid > dense-only on hard multi-hop questions.

Speed optimisations (Session 4):
  search():
    - Parallel embed_text + BM25 via ThreadPoolExecutor  → saves ~150 ms per call.
    - Accepts pre-computed query_vector to skip embed_text when called in a loop.
    - simple_query flag caps cross-encoder candidates at TOP_K_HYBRID_SIMPLE (8 vs 16)
      → saves ~1.5 s on single-hop queries.
    - skip_rerank flag defers CE to the caller's single combined pass.

  search_multi_query():
    - Batch embed: all sub-queries embedded in ONE Ollama call (embed_batch)
      → saves (N-1) × ~1.5 s (e.g. 4 × 1.5 s = 6 s for 5 sub-queries).
    - Parallel retrieval: per-query dense + BM25 run concurrently
      → saves ~0.5 s.
    - Single batched CE pass: all (query × chunk) pairs scored in one predict()
      call; MAX score per chunk preserves the multi-hop invariant and eliminates
      the double-reranking distortion of the earlier design.
"""

from __future__ import annotations
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import os
from typing import Optional

import numpy as np

import config
import pit
from retrieval.vector_store import VectorStore, ChunkResult
from retrieval.bm25_store   import BM25Store
from utils import embed_text, embed_batch


@dataclass
class RetrievalStats:
    dense_hits:  int  = 0
    sparse_hits: int  = 0
    union_size:  int  = 0
    final_size:  int  = 0
    reranked:    bool = False


class HybridRetriever:
    def __init__(self, vector_store: VectorStore, bm25_store: BM25Store):
        self._vs   = vector_store
        self._bm25 = bm25_store
        self._reranker = None
        if config.USE_RERANKER:
            self._load_reranker()

    # ── Reranker ───────────────────────────────────────────────────────────────

    def _load_reranker(self) -> None:
        try:
            # This application uses the PyTorch CrossEncoder. Prevent Transformers
            # from importing an unrelated TensorFlow installation, whose protobuf
            # ABI can differ from Streamlit's supported protobuf range.
            os.environ.setdefault("USE_TF", "0")
            os.environ.setdefault("USE_FLAX", "0")
            from sentence_transformers import CrossEncoder
            self._reranker = CrossEncoder(
                config.RERANK_MODEL, max_length=512
            )
            print(f"[hybrid] Cross-encoder loaded: {config.RERANK_MODEL}")
        except Exception as e:
            print(f"[hybrid] Cross-encoder not loaded ({e}); skipping reranking.")
            self._reranker = None

    # ── Main search ────────────────────────────────────────────────────────────

    def search(
        self,
        query: str,
        n_results: int = config.TOP_K_RERANK,
        article_filter: Optional[list[str]] = None,
        cluster_filter: Optional[str] = None,
        query_vector: Optional[list[float]] = None,
        simple_query: bool = False,
        skip_rerank: bool = False,
        as_of: Optional[str] = None,
        as_of_epoch: Optional[float] = None,
    ) -> tuple[list[ChunkResult], RetrievalStats]:
        """
        Single-query hybrid search.  Returns (ranked_chunks, stats).

        query_vector:  pre-computed embedding — skips the embed_text HTTP call.
                       Pass this when calling search() in a loop with the same
                       logical query (e.g. force-include, multi-query loops).
        simple_query:  use TOP_K_HYBRID_SIMPLE (8) candidates before cross-encoder
                       instead of TOP_K_HYBRID (16) — saves ~1.5 s for factual
                       single-hop queries that never need the wider candidate pool.
        skip_rerank:   omit the cross-encoder pass entirely; the caller is
                       responsible for a single combined rerank of the merged pool
                       (used by search_multi_query to avoid double-reranking).
        as_of:         optional ISO-8601 timestamp (point-in-time query). Defaults
                       to now. Hard-filters BOTH dense and sparse candidates to
                       filed_at <= as_of before ranking (retrieval/vector_store.py,
                       retrieval/bm25_store.py) — not a post-hoc check.
        as_of_epoch:   pre-resolved epoch, used internally by search_multi_query
                       so every sub-query in one logical request shares exactly
                       the same as_of instant instead of each resolving "now"
                       independently.
        """
        resolved_epoch = as_of_epoch if as_of_epoch is not None else pit.resolve_as_of_epoch(as_of)
        stats = RetrievalStats()
        if article_filter:
            where = {"article_id": {"$in": article_filter}}
        elif cluster_filter:
            where = {"cluster": cluster_filter}
        else:
            where = None

        if query_vector is not None:
            # Pre-computed vector provided: dense + sparse run sequentially (both fast)
            dense_results  = self._vs.search(query_vector, n_results=config.TOP_K_DENSE, where=where, as_of_epoch=resolved_epoch)
            sparse_results = self._bm25.search(query, n_results=config.TOP_K_BM25, as_of_epoch=resolved_epoch)
        else:
            # Fresh query: embed_text (slow HTTP, ~1–2 s) + BM25 (fast CPU, ~50 ms) in parallel.
            # BM25 overlaps with the Ollama round-trip → saves ~100–150 ms per call.
            with ThreadPoolExecutor(max_workers=2) as _ex:
                _fut_vec  = _ex.submit(embed_text, query)
                _fut_bm25 = _ex.submit(self._bm25.search, query, config.TOP_K_BM25, resolved_epoch)
                query_vector   = _fut_vec.result()
                sparse_results = _fut_bm25.result()
            dense_results = self._vs.search(query_vector, n_results=config.TOP_K_DENSE, where=where, as_of_epoch=resolved_epoch)

        if article_filter:
            sparse_results = [r for r in sparse_results if r.article_id in article_filter]
        elif cluster_filter:
            sparse_results = [r for r in sparse_results if r.cluster == cluster_filter]

        stats.dense_hits  = len(dense_results)
        stats.sparse_hits = len(sparse_results)

        # RRF fusion
        fused = _rrf_fuse(dense_results, sparse_results, k=config.RRF_K)
        stats.union_size = len(fused)

        # Candidate cap before cross-encoder.
        # Simple queries use a tighter budget → halves CE pairs → ~1.5 s saved.
        hybrid_cap   = config.TOP_K_HYBRID_SIMPLE if simple_query else config.TOP_K_HYBRID
        n_candidates = max(hybrid_cap, n_results * 2)
        candidates   = fused[:n_candidates]

        # Cross-encoder reranking (skipped when called from search_multi_query)
        if self._reranker and len(candidates) > 1 and not skip_rerank:
            pairs     = [(query, c.text) for c in candidates]
            ce_scores = self._reranker.predict(pairs)
            for chunk, score in zip(candidates, ce_scores):
                chunk.rrf_score = float(score)
            candidates.sort(key=lambda c: c.rrf_score, reverse=True)
            stats.reranked = True

        final = candidates[:n_results]
        for i, c in enumerate(final):
            c.rank = i + 1
        stats.final_size = len(final)
        self._annotate_supersession(final, resolved_epoch)
        return final, stats

    def _annotate_supersession(self, chunks: list[ChunkResult], as_of_epoch: float) -> None:
        """Flag any original filing in the result set that a currently-visible
        amendment supersedes, rather than silently returning it unmarked.

        This is a bounded, in-memory scan (BM25's full chunk set already lives
        in RAM — no extra network/DB call) over amendment chunks only, keyed
        by accession number, so it stays cheap even as the corpus grows.
        """
        amendments = [
            c for c in self._bm25.all_chunks()
            if c.is_amendment and c.supersedes_accession and pit.is_visible(c.filed_at, as_of_epoch)
        ]
        if not amendments:
            return
        superseding_by_accession = {c.supersedes_accession: c for c in amendments}
        for chunk in chunks:
            if not chunk.accession_number or chunk.is_amendment:
                continue
            amendment = superseding_by_accession.get(chunk.accession_number)
            if amendment:
                chunk.superseded_by_accession = amendment.accession_number
                chunk.superseded_by_form = amendment.form_type
                chunk.superseded_by_filed_at = amendment.filed_at

    def search_multi_query(
        self,
        queries: list[str],
        n_per_query: int = 5,
        final_top_k: int = None,
        as_of: Optional[str] = None,
    ) -> tuple[list[ChunkResult], RetrievalStats]:
        """
        Run hybrid search for each sub-query, deduplicate, then rerank.

        The cross-encoder scores each candidate against EVERY sub-query and takes
        the maximum score, so evidence for a later leg of a multi-hop question is
        not buried by a score against only the first sub-query.

        SPEED improvements vs original design:
          1. embed_batch()  — all sub-query embeddings in ONE Ollama call.
          2. ThreadPoolExecutor — per-query dense + BM25 retrieval in parallel.
          3. skip_rerank=True on inner calls — inner CE skipped; one batched outer
             CE pass replaces N serial calls and eliminates double-reranking.
          4. Single predict() call — all (query × chunk) pairs in one batch;
             PyTorch processes them more efficiently than N small batches.
        """
        if final_top_k is None:
            final_top_k = config.MULTI_HOP_TOP_K

        # Resolved ONCE for the whole multi-query request so every sub-query
        # shares exactly the same as_of instant (no drift between legs).
        resolved_epoch = pit.resolve_as_of_epoch(as_of)
        combined_stats = RetrievalStats()

        # ── 1. Batch embed all sub-queries in one Ollama call ─────────────────
        # Saves (N-1) × ~1.5 s compared to N serial embed_text calls.
        query_vectors = embed_batch(queries)

        # ── 2. Parallel per-query dense + BM25 retrieval ──────────────────────
        # skip_rerank=True: inner CE skipped; single outer CE pass is better.
        # n_per_query from RRF gives diverse candidates for the outer CE.
        def _search_one(args: tuple[str, list[float]]) -> tuple[list[ChunkResult], RetrievalStats]:
            q, vec = args
            return self.search(
                q,
                n_results    = n_per_query,
                query_vector = vec,
                skip_rerank  = True,   # outer CE pass handles all candidates
                as_of_epoch  = resolved_epoch,
            )

        with ThreadPoolExecutor(max_workers=min(len(queries), 4)) as _ex:
            all_results = list(_ex.map(_search_one, zip(queries, query_vectors)))

        all_chunks: list[ChunkResult] = []
        for per_q_chunks, stats in all_results:
            all_chunks.extend(per_q_chunks)
            combined_stats.dense_hits  += stats.dense_hits
            combined_stats.sparse_hits += stats.sparse_hits

        # ── 3. Deduplicate by text fingerprint (first 120 chars) ──────────────
        seen:   set[str]          = set()
        unique: list[ChunkResult] = []
        for c in all_chunks:
            key = c.text[:120]
            if key not in seen:
                seen.add(key)
                unique.append(c)

        combined_stats.union_size = len(unique)

        # ── 4. Single batched multi-query CE pass ─────────────────────────────
        # Flatten all (query, chunk) pairs into ONE predict() call.
        # PyTorch's batched inference is more efficient than N separate calls.
        # MAX score across queries per chunk preserves the multi-hop invariant.
        if self._reranker and len(unique) > 1:
            all_pairs  = [(q, c.text) for q in queries for c in unique]
            all_scores = self._reranker.predict(all_pairs)
            score_matrix = np.array(all_scores, dtype=np.float32).reshape(
                len(queries), len(unique)
            )
            max_scores = score_matrix.max(axis=0)
            for chunk, score in zip(unique, max_scores):
                chunk.rrf_score = float(score)
            unique.sort(key=lambda c: c.rrf_score, reverse=True)
            combined_stats.reranked = True

        final = unique[:final_top_k]
        for i, c in enumerate(final):
            c.rank = i + 1
        combined_stats.final_size = len(final)
        return final, combined_stats


# ── RRF helper ─────────────────────────────────────────────────────────────────

def _rrf_fuse(
    dense_results:  list[ChunkResult],
    sparse_results: list[ChunkResult],
    k: int = 60,
) -> list[ChunkResult]:
    """
    Reciprocal Rank Fusion:
        score(d) = sum_i  1 / (k + rank_i)
    Combines ranked lists from dense and sparse retrievers.
    Chunks identified by first 120 chars of text as fingerprint.
    """
    scores:    dict[str, float]       = {}
    chunk_map: dict[str, ChunkResult] = {}

    def _fp(c: ChunkResult) -> str:
        return c.text[:120]

    for rank, chunk in enumerate(dense_results, start=1):
        fp = _fp(chunk)
        scores[fp]    = scores.get(fp, 0.0) + 1.0 / (k + rank)
        chunk_map[fp] = chunk

    for rank, chunk in enumerate(sparse_results, start=1):
        fp = _fp(chunk)
        scores[fp]    = scores.get(fp, 0.0) + 1.0 / (k + rank)
        if fp not in chunk_map:
            chunk_map[fp] = chunk

    ranked_fps = sorted(scores, key=lambda fp: scores[fp], reverse=True)
    result = []
    for fp in ranked_fps:
        c = chunk_map[fp]
        c.rrf_score = scores[fp]
        result.append(c)
    return result
