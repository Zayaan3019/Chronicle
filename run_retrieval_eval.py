"""Phase 5 CLI — real end-to-end retrieval evaluation + ingestion-latency benchmark.

Two things this script measures for real, against a dedicated eval index
(never the default production paths — see --eval-dir):

1. Ingestion latency (item #16): for each EDGAR document ingested in this
   run, `wall_clock_queryable_time - filed_at`. Reported as p50/p99.
   HONESTLY CAVEATED: these are real historical 10-K/10-Q/8-K filings, not
   brand-new same-day filings, so this number is dominated by "how old the
   filing already was when this script ran" — not by this system's
   processing speed. A second, more meaningful number is reported alongside
   it: `source_refresh_wall_seconds` (fetch-start to queryable, per source
   refresh call) — THAT is the number that reflects actual system
   throughput. Both are printed; neither is presented as the other.

2. Retrieval ablation (item #15): recall@k / nDCG@k for BM25-only,
   dense-only, hybrid+RRF, and hybrid+RRF+cross-encoder, using a labelled
   question set built from the just-ingested corpus (eval_retrieval.py).

Usage:
    python run_retrieval_eval.py [--n-questions 60] [--k 5] [--eval-dir data/eval_run]
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

import config
import pit
from eval_retrieval import build_labelled_set, run_ablation
from live_data import LiveDataAgent, LiveSource
from retrieval.bm25_store import BM25Store
from retrieval.hybrid import HybridRetriever
from retrieval.vector_store import VectorStore

EDGAR_SOURCES = [
    LiveSource(id="aapl-edgar-eval", url="https://www.sec.gov/cgi-bin/browse-edgar?CIK=AAPL", kind="edgar",
               title="Apple Inc.", publisher="SEC EDGAR", cik="0000320193",
               forms=("10-K", "10-Q"), max_items=2, refresh_minutes=1),
    LiveSource(id="msft-edgar-eval", url="https://www.sec.gov/cgi-bin/browse-edgar?CIK=MSFT", kind="edgar",
               title="Microsoft Corporation", publisher="SEC EDGAR", cik="0000789019",
               forms=("10-K", "10-Q"), max_items=2, refresh_minutes=1),
    LiveSource(id="nvda-edgar-eval", url="https://www.sec.gov/cgi-bin/browse-edgar?CIK=NVDA", kind="edgar",
               title="NVIDIA Corporation", publisher="SEC EDGAR", cik="0001045810",
               forms=("10-K", "10-Q"), max_items=2, refresh_minutes=1),
]


def _percentile(values: list[float], pct: float) -> float | None:
    return float(np.percentile(values, pct)) if values else None


def _load_latency_log(eval_dir: Path) -> dict:
    path = eval_dir / "latency_samples.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {"filed_to_queryable": [], "source_refresh": []}


def _save_latency_log(eval_dir: Path, log: dict) -> None:
    (eval_dir / "latency_samples.json").write_text(json.dumps(log), encoding="utf-8")


def ingest_and_measure_latency(eval_dir: Path, tickers: list[str] | None = None) -> tuple[VectorStore, BM25Store, dict]:
    """Ingests EDGAR_SOURCES into a persistent index at `eval_dir`. Safe to call
    multiple times (e.g. one ticker per invocation, to fit a bounded wall-clock
    budget per call): already-ingested, unchanged filings are skipped by the
    same content-hash diffing live_data.py always uses — this is not a special
    resume mode, it is the normal incremental-refresh behavior reused as-is."""
    config.CHROMA_PERSIST_DIR = str(eval_dir / "chroma")
    config.BM25_INDEX_PATH = str(eval_dir / "bm25.pkl")
    state_path = str(eval_dir / "state.sqlite3")

    vector_store = VectorStore()
    bm25_store = BM25Store()
    if Path(config.BM25_INDEX_PATH).exists():
        bm25_store.load()
    agent = LiveDataAgent(vector_store, bm25_store, state_path=state_path)

    filed_to_queryable: list[float] = []
    source_refresh_seconds: list[float] = []

    sources = EDGAR_SOURCES
    if tickers:
        wanted = {t.upper() for t in tickers}
        sources = [s for s in EDGAR_SOURCES if s.id.split("-")[0].upper() in wanted]

    for source in sources:
        print(f"Ingesting {source.title} ({source.id}) ...")
        t_start = time.time()
        report = agent.refresh_source(source, force=True)
        t_end = time.time()
        source_refresh_seconds.append(t_end - t_start)
        print(f"  status={report.status} added={report.documents_added} "
              f"changed={report.documents_changed} msg={report.message} "
              f"wall_seconds={t_end - t_start:.2f}")

        prefix = f"live:{source.id}:"
        for chunk in bm25_store.all_chunks():
            if not chunk.article_id.startswith(prefix) or not chunk.filed_at:
                continue
            filed_epoch = pit.parse_iso_to_epoch(chunk.filed_at)
            if filed_epoch is not None:
                filed_to_queryable.append(t_end - filed_epoch)

    # Persist samples across invocations — each ingestion call (e.g. one
    # ticker at a time, to fit a bounded wall-clock budget) contributes its
    # own samples; the final --eval-only pass reports over the accumulated set.
    log = _load_latency_log(eval_dir)
    log["filed_to_queryable"].extend(filed_to_queryable)
    log["source_refresh"].extend(source_refresh_seconds)
    _save_latency_log(eval_dir, log)

    latency_stats = {
        "filed_to_queryable_p50_seconds": _percentile(log["filed_to_queryable"], 50),
        "filed_to_queryable_p99_seconds": _percentile(log["filed_to_queryable"], 99),
        "filed_to_queryable_n": len(log["filed_to_queryable"]),
        "source_refresh_p50_seconds": _percentile(log["source_refresh"], 50),
        "source_refresh_p99_seconds": _percentile(log["source_refresh"], 99),
        "source_refresh_n": len(log["source_refresh"]),
    }
    return vector_store, bm25_store, latency_stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-questions", type=int, default=60)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--eval-dir", default=str(Path(config.BASE_DIR) / "data" / "eval_run"))
    parser.add_argument("--print-samples", type=int, default=5, help="print this many labelled questions for spot-checking")
    parser.add_argument("--tickers", default=None,
                         help="comma-separated subset to ingest this call (e.g. one ticker at a time to fit a time budget); default: all configured EDGAR_SOURCES")
    parser.add_argument("--ingest-only", action="store_true", help="ingest and report latency, then stop (no question generation/ablation)")
    parser.add_argument("--eval-only", action="store_true", help="skip ingestion entirely; run question generation + ablation over the already-persisted index")
    args = parser.parse_args()

    eval_dir = Path(args.eval_dir)
    eval_dir.mkdir(parents=True, exist_ok=True)
    tickers = [t.strip() for t in args.tickers.split(",")] if args.tickers else None

    if args.eval_only:
        config.CHROMA_PERSIST_DIR = str(eval_dir / "chroma")
        config.BM25_INDEX_PATH = str(eval_dir / "bm25.pkl")
        vector_store = VectorStore()
        bm25_store = BM25Store()
        bm25_store.load()
        log = _load_latency_log(eval_dir)
        latency_stats = {
            "filed_to_queryable_p50_seconds": _percentile(log["filed_to_queryable"], 50),
            "filed_to_queryable_p99_seconds": _percentile(log["filed_to_queryable"], 99),
            "filed_to_queryable_n": len(log["filed_to_queryable"]),
            "source_refresh_p50_seconds": _percentile(log["source_refresh"], 50),
            "source_refresh_p99_seconds": _percentile(log["source_refresh"], 99),
            "source_refresh_n": len(log["source_refresh"]),
        }
    else:
        print(f"=== Ingestion + latency measurement (tickers={tickers or 'all'}) ===")
        vector_store, bm25_store, latency_stats = ingest_and_measure_latency(eval_dir, tickers=tickers)

    print("\nIngestion latency (EDGAR filed_at -> queryable), n documents = "
          f"{latency_stats['filed_to_queryable_n']}:")
    p50 = latency_stats["filed_to_queryable_p50_seconds"]
    p99 = latency_stats["filed_to_queryable_p99_seconds"]
    print(f"  p50 = {p50:,.0f}s ({p50/86400:,.1f} days)" if p50 is not None else "  p50 = n/a")
    print(f"  p99 = {p99:,.0f}s ({p99/86400:,.1f} days)" if p99 is not None else "  p99 = n/a")
    print("  CAVEAT: these are real historical filings, so this number is dominated by how old")
    print("  each filing already was when this script ran, not by system processing speed.")
    print(f"\nSource refresh wall time (fetch-start -> queryable), n source-refresh calls = "
          f"{latency_stats['source_refresh_n']}:")
    rp50 = latency_stats["source_refresh_p50_seconds"]
    rp99 = latency_stats["source_refresh_p99_seconds"]
    print(f"  p50 = {rp50:.2f}s" if rp50 is not None else "  p50 = n/a")
    print(f"  p99 = {rp99:.2f}s" if rp99 is not None else "  p99 = n/a")
    print("  This IS a real measure of this system's ingest-to-queryable processing speed.")

    total_chunks = vector_store.count()
    print(f"\nCorpus: {total_chunks} chunks across {len(EDGAR_SOURCES)} configured EDGAR sources")
    if args.ingest_only:
        print("\n--ingest-only set: stopping before question generation/ablation.")
        return
    if total_chunks == 0:
        print("No chunks ingested — cannot build a labelled question set or run the ablation. Stopping.")
        return

    print(f"\n=== Building labelled question set (target n={args.n_questions}) ===")
    print("Each question is LLM-generated FROM one specific real chunk; that chunk's id is the")
    print("ground-truth relevant document by construction (see eval_retrieval.py docstring).")
    all_chunks = bm25_store.all_chunks()
    questions = build_labelled_set(all_chunks, n=args.n_questions)
    print(f"Built {len(questions)} labelled question(s).")
    for lq in questions[: args.print_samples]:
        print(f"  [{lq.ticker} {lq.form_type} {lq.item}] Q: {lq.question}  -> ground_truth={lq.ground_truth_chunk_id}")

    if len(questions) < 5:
        print("\nFewer than 5 labelled questions were generated — too few for a meaningful ablation. Stopping.")
        return

    print(f"\n=== Retrieval ablation (k={args.k}, n={len(questions)} questions) ===")
    retriever = HybridRetriever(vector_store, bm25_store)
    results = run_ablation(vector_store, bm25_store, retriever, questions, k=args.k)
    print(f"\n{'config':<15}{'recall@' + str(args.k):<14}{'nDCG@' + str(args.k):<14}n")
    for r in results:
        print(f"{r.config_name:<15}{r.recall_at_k:<14.3f}{r.ndcg_at_k:<14.3f}{r.n_questions}")


if __name__ == "__main__":
    main()
