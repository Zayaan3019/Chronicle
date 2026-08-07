"""
BM25 sparse retrieval store backed by rank_bm25.
The index is serialized to disk so it survives app restarts.
"""

from __future__ import annotations
import os
import pickle
import re
import tempfile
from typing import Optional

from rank_bm25 import BM25Okapi

import config
import pit
from retrieval.vector_store import ChunkResult


# Simple stopword set (avoids NLTK dependency)
_STOPWORDS = {
    "a","an","the","and","or","but","in","on","at","to","for","of","with",
    "by","from","is","are","was","were","be","been","being","have","has",
    "had","do","does","did","will","would","could","should","may","might",
    "that","this","these","those","it","its","as","if","not","no","nor",
}


def _tokenize(text: str) -> list[str]:
    tokens = re.findall(r"[a-z0-9]+(?:'[a-z]+)?", text.lower())
    return [t for t in tokens if t not in _STOPWORDS and len(t) > 1]


class BM25Store:
    def __init__(self):
        self._chunks: list[ChunkResult] = []
        self._bm25:   Optional[BM25Okapi] = None

    # ── Build ──────────────────────────────────────────────────────────────────

    def build(self, chunks: list[ChunkResult]) -> None:
        self._chunks = chunks
        tokenized = [_tokenize(c.text) for c in chunks]
        self._bm25 = BM25Okapi(tokenized)

    # ── Save / Load ────────────────────────────────────────────────────────────

    def save(self, path: str = None) -> None:
        path = path or config.BM25_INDEX_PATH
        # A process crash must not leave a partially-written sparse index.
        directory = os.path.dirname(os.path.abspath(path))
        fd, tmp_path = tempfile.mkstemp(prefix=".bm25-", suffix=".tmp", dir=directory)
        with os.fdopen(fd, "wb") as f:
            pickle.dump({"bm25": self._bm25, "chunks": self._chunks}, f)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, path)

    def load(self, path: str = None) -> bool:
        path = path or config.BM25_INDEX_PATH
        if not os.path.exists(path):
            return False
        with open(path, "rb") as f:
            data = pickle.load(f)
        self._bm25  = data["bm25"]
        self._chunks = data["chunks"]
        return True

    # ── Search ─────────────────────────────────────────────────────────────────

    def search(
        self,
        query: str,
        n_results: int = config.TOP_K_BM25,
        as_of_epoch: Optional[float] = None,
    ) -> list[ChunkResult]:
        if not self._bm25 or not self._chunks:
            return []
        tokens = _tokenize(query)
        if not tokens:
            return []
        scores = self._bm25.get_scores(tokens)

        # Hard point-in-time filter BEFORE ranking: documents filed after
        # as_of are removed from the candidate set entirely, so no score can
        # ever put them in the top-n regardless of relevance. This mirrors
        # the Chroma $lte pre-ranking filter in vector_store.search().
        if as_of_epoch is None:
            candidate_idx = range(len(self._chunks))
        else:
            candidate_idx = [
                i for i in range(len(self._chunks))
                if pit.is_visible(self._chunks[i].filed_at, as_of_epoch)
            ]

        ranked = sorted(candidate_idx, key=lambda i: scores[i], reverse=True)[:n_results]

        results = []
        for idx in ranked:
            score = scores[idx]
            if score <= 0:
                continue
            chunk = self._chunks[idx]
            chunk.score = float(score)
            results.append(chunk)
        return results

    def all_chunks(self) -> list[ChunkResult]:
        """Full in-memory chunk set — used for supersession lookups (no I/O)."""
        return self._chunks

    def replace_article_chunks(
        self,
        removed_article_ids: list[str],
        replacement_chunks: list[ChunkResult],
    ) -> None:
        """Atomically rebuild the in-memory BM25 document set after source upserts."""
        removed = set(removed_article_ids)
        self._chunks = [c for c in self._chunks if c.article_id not in removed]
        self._chunks.extend(replacement_chunks)
        self.build(self._chunks)
        self.save()

    @property
    def is_populated(self) -> bool:
        return self._bm25 is not None and len(self._chunks) > 0
