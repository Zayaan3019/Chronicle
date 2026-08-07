"""Phase 5 — retrieval pipeline evaluation.

Builds a labelled question set by generating ONE factual question per
sampled, already-ingested real chunk via the LLM — the source chunk_id is
recorded as ground truth BY CONSTRUCTION (the question is generated FROM
that chunk's text). Said plainly: this is an LLM-assisted, chunk-grounded
synthetic query set, not a hand-curated benchmark reviewed by a financial
analyst — the honest way to describe how these labels were produced.

Reports recall@k and nDCG@k for four retrieval configurations, all built
from the SAME production code paths (no reimplemented ranking logic):
  - bm25       : retrieval/bm25_store.py  BM25Store.search directly
  - dense      : retrieval/vector_store.py VectorStore.search directly
  - hybrid_rrf : retrieval/hybrid.py HybridRetriever.search(skip_rerank=True)
  - hybrid_ce  : retrieval/hybrid.py HybridRetriever.search(skip_rerank=False)

nDCG@k here has exactly one relevant document per query by construction, so
it reduces to 1/log2(rank+1) if the ground-truth chunk is found in the top-k
(rank is 1-indexed), else 0 — implemented generally rather than special-cased,
so it stays correct if the label set is ever extended to multiple relevants.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import random

import config
from retrieval.bm25_store import BM25Store
from retrieval.hybrid import HybridRetriever
from retrieval.vector_store import ChunkResult, VectorStore
from utils import call_llm, embed_text, extract_json

_QUESTION_SYSTEM = """You write evaluation questions for a financial-research
retrieval system. Given ONE excerpt from a company's SEC filing, write ONE
specific, self-contained factual question whose answer requires information
from THIS excerpt specifically (not general knowledge). Do not say "the
excerpt" or "the filing" in the question itself — write it as a standalone
question a user might actually ask.

Return ONLY JSON: {"question": "..."}"""


@dataclass(frozen=True)
class LabelledQuestion:
    question: str
    ground_truth_chunk_id: str
    ticker: str
    item: str
    form_type: str


@dataclass(frozen=True)
class AblationResult:
    config_name: str
    k: int
    n_questions: int
    recall_at_k: float
    ndcg_at_k: float


def generate_question(chunk: ChunkResult, model: str | None = None) -> str | None:
    raw = call_llm(
        messages=[
            {"role": "system", "content": _QUESTION_SYSTEM},
            {"role": "user", "content": chunk.text[:1500]},
        ],
        model=model or config.FAST_LLM,
        temperature=0.3,
        json_mode=True,
    )
    data = extract_json(raw)
    question = str(data.get("question") or "").strip()
    return question or None


def build_labelled_set(
    chunks: list[ChunkResult], n: int, seed: int = 20260807, model: str | None = None,
) -> list[LabelledQuestion]:
    rng = random.Random(seed)
    candidates = [c for c in chunks if len(c.text.split()) >= 40]
    rng.shuffle(candidates)
    labelled: list[LabelledQuestion] = []
    for chunk in candidates:
        if len(labelled) >= n:
            break
        question = generate_question(chunk, model=model)
        if not question:
            continue
        labelled.append(
            LabelledQuestion(
                question=question,
                ground_truth_chunk_id=chunk.chunk_id,
                ticker=chunk.article_source,
                item=chunk.item,
                form_type=chunk.form_type,
            )
        )
    return labelled


def recall_and_ndcg(ranked_chunk_ids: list[str], relevant_ids: set[str], k: int) -> tuple[int, float]:
    """Standard recall@k (0/1 hit for a single-relevant query) and nDCG@k
    (binary relevance, ideal ranking places all relevant docs first)."""
    top_k = ranked_chunk_ids[:k]
    hits = [1 if cid in relevant_ids else 0 for cid in top_k]
    if not any(hits):
        return 0, 0.0
    dcg = sum(rel / math.log2(idx + 2) for idx, rel in enumerate(hits))
    ideal_hits = min(len(relevant_ids), k)
    idcg = sum(1 / math.log2(idx + 2) for idx in range(ideal_hits))
    return 1, (dcg / idcg if idcg > 0 else 0.0)


def _bm25_only(bm25: BM25Store, question: str, k: int) -> list[str]:
    return [c.chunk_id for c in bm25.search(question, n_results=k)]


def _dense_only(vs: VectorStore, question: str, k: int) -> list[str]:
    vector = embed_text(question)
    return [c.chunk_id for c in vs.search(vector, n_results=k)]


def _hybrid(retriever: HybridRetriever, question: str, k: int, use_reranker: bool) -> list[str]:
    chunks, _ = retriever.search(question, n_results=k, skip_rerank=not use_reranker)
    return [c.chunk_id for c in chunks]


def run_ablation(
    vector_store: VectorStore,
    bm25_store: BM25Store,
    retriever: HybridRetriever,
    questions: list[LabelledQuestion],
    k: int = 5,
) -> list[AblationResult]:
    configs = {
        "bm25": lambda q: _bm25_only(bm25_store, q, k),
        "dense": lambda q: _dense_only(vector_store, q, k),
        "hybrid_rrf": lambda q: _hybrid(retriever, q, k, use_reranker=False),
        "hybrid_rrf_ce": lambda q: _hybrid(retriever, q, k, use_reranker=True),
    }
    results = []
    for name, run_fn in configs.items():
        recalls = []
        ndcgs = []
        for lq in questions:
            ranked = run_fn(lq.question)
            recall, ndcg = recall_and_ndcg(ranked, {lq.ground_truth_chunk_id}, k)
            recalls.append(recall)
            ndcgs.append(ndcg)
        n = len(questions)
        results.append(
            AblationResult(
                config_name=name, k=k, n_questions=n,
                recall_at_k=(sum(recalls) / n) if n else 0.0,
                ndcg_at_k=(sum(ndcgs) / n) if n else 0.0,
            )
        )
    return results
