# Chronicle Architecture

## Data model

There is no bundled corpus, article map, answer key, cache, or static index.
`live_sources.json` is the only data-entry point: a locally controlled
allow-list of SEC EDGAR CIK trackers and GDELT news queries, one entry per
company in the watchlist.

```text
Opaque authenticated session ID
  -> Memory Agent (SQLite/WAL, scoped retrieval, TTL, delete)
  -> Live Data Agent (EDGAR/GDELT allow-list, HTTP validators, lease, provenance)
  -> Research Planner (query_agent.py — generic decomposition)
  -> Hybrid Retriever (Chroma dense + BM25 sparse + RRF + cross-encoder,
     hard as_of point-in-time filter before ranking)
  -> Evidence Synthesizer (reasoning_agent.py)
  -> Evidence Critic (critic_agent.py — LLM citation check + deterministic
     numeric-claim verification)
  -> Guardrail (guardrail.py — deny-by-default policy layer, independent
     second point-in-time check, prompt-injection scan, schema validation)
  -> cited answer + append-only decision log + durable final-turn memory
```

## Ingestion: SEC EDGAR + GDELT

- `edgar_client.py`: rate-limited (`EDGAR_MIN_REQUEST_INTERVAL_SECONDS`,
  default 0.34s ≈ 3 req/s — well under SEC's ~10 req/s fair-access ceiling)
  client for `data.sec.gov/submissions` (filing metadata incl. the exact
  `acceptanceDateTime`), `www.sec.gov/Archives/edgar/data` (filing
  documents/exhibits), and `efts.sec.gov` (full-text search, discovery
  only — never the point-in-time source of truth). Always sends a real,
  concise User-Agent (`EDGAR_USER_AGENT`) — SEC's bot detection was observed
  empirically to reject verbose/instructional User-Agent strings.
- `edgar_parse.py`: Item/section-boundary chunking. Tables are flattened to
  pipe-delimited text *before* rendering, so a numeric claim inside a table
  survives verbatim and in position; a sub-chunk never crosses an Item
  boundary.
- `gdelt_client.py`: GDELT 2.0 DOC API, free, no key, metadata-only —
  `live_data.py` does a secondary fetch+extract of each article's body text.
  GDELT chunks carry no `filed_at` (no authoritative publish-timestamp
  source exists for them) and are exempt from the point-in-time hard filter
  by design, not by oversight (`pit.py::is_visible`).
- `live_data.py`'s incremental refresh/lease/diff/upsert core is reused
  unchanged for both source kinds — `_refresh_source_locked` only dispatches
  the *fetch* strategy by `source.kind`; the content-hash diff and
  Chroma/BM25 upsert loop is the same code path RSS/web sources already used.

## Point-in-time correctness

- `pit.py` is the single definition of "visible at `as_of`" — every other
  module (retrieval, guardrail) imports it rather than re-deriving timestamp
  comparison.
- `retrieval/vector_store.py` stores a numeric `filed_at_ts` mirror of
  `filed_at` (Chroma's `$lte` comparison operator needs numeric metadata) and
  applies a real `where` clause — `filed_at_ts <= as_of OR no filed_at at
  all` — evaluated by Chroma before similarity ranking.
- `retrieval/bm25_store.py` filters the BM25 candidate index by the same
  rule *before* `sorted()`/truncation — an excluded document can never enter
  the ranked list regardless of score.
- `retrieval/hybrid.py::_annotate_supersession` flags an original filing's
  chunk with `superseded_by_*` only when a 10-K/A that supersedes it is
  itself visible at the query's `as_of` — surfacing the conflict rather than
  silently returning the newer filing with no indication anything changed.
- Proven zero-violation by a 200-trial randomized adversarial sweep,
  `tests/test_point_in_time.py`.

## Numeric-claim verification + guardrail

- `numeric_verify.py`: deterministic, no LLM call. Extracts
  financially-decorated numbers ($/%/scale-word/comma-grouped; bare
  undecorated integers like item numbers or years are excluded) from a
  generated answer and checks each against its cited source chunk, with
  unit/scale normalization and a 1% relative tolerance for rounding noise
  only.
- `agents/critic_agent.py` runs this check unconditionally (even if the
  LLM's own JSON output is invalid) and forces `NEEDS_CORRECTION` on any
  unsupported number, regardless of the LLM's own verdict.
- `guardrail.py`: pluggable (depends only on `str`/`list[ChunkResult]`, never
  on pipeline internals), deny-by-default — every check must explicitly pass
  for `ALLOW`. `pipeline.py::run_guardrail` wraps the call in try/except: any
  exception (a killed/crashed guardrail) is converted to a hard `BLOCK` via
  `guardrail.fail_closed_verdict`, never a silent pass-through.
- `decision_log.py`: append-only SQLite log of every verdict — no
  update/delete function is exposed by the module.
- `.github/workflows/tests.yml` runs the full suite (including the
  guardrail's adversarial fixture suite, which reports both false-positive
  and false-negative rates every run) as a CI gate.

## Text-derived signal (Phase 4) — explicitly bounded scope

- `tone_signal.py`: Loughran-McDonald word-list tone-delta (genuine LM
  Master Dictionary Negative/Uncertainty categories — see
  `data/lm_lexicon/SOURCE.md`) plus an independent LLM-based tone-direction
  classification, compared and reported honestly (agreement measured, not
  assumed).
- `price_client.py` + `event_study.py`: forward-return Information
  Coefficient (Spearman) with a Newey-West HAC-adjusted significance test.
  **This stops at the IC and its significance test** — no backtest,
  portfolio optimizer, or trading rule is built on top of it (see
  MODIFICATION.md's non-goals; that is explicitly out of this project's
  scope).

## Retrieval evaluation (Phase 5)

- `eval_retrieval.py` / `run_retrieval_eval.py`: recall@k and nDCG@k across
  BM25-only, dense-only, hybrid+RRF, and hybrid+RRF+cross-encoder, all
  exercised through the same production code paths
  (`retrieval/bm25_store.py`, `retrieval/vector_store.py`,
  `retrieval/hybrid.py` — no ranking logic is reimplemented for the eval).
  The labelled question set is LLM-generated from real ingested chunks
  (ground truth by construction), stated as such rather than presented as a
  hand-curated benchmark.
- Ingestion latency (EDGAR `filed_at` -> queryable) is measured over a real
  ingestion run and reported as p50/p99, alongside a second, honestly
  distinguished metric (source-refresh wall time) that actually reflects
  system throughput rather than "how old was the filing already."

## Live data safeguards

- The model cannot choose URLs. It only queries already-configured source
  IDs (CIKs / GDELT queries) from `live_sources.json`.
- HTTPS and public-address checks are enforced by default for web/rss
  sources, including redirect targets. Responses are byte-limited.
- ETag/Last-Modified headers (web/rss), content-hash diffing (all sources),
  and source leases make refresh idempotent and prevent duplicate concurrent
  source updates.
- Changed documents delete old vector chunks, upsert fresh chunks, and
  replace the BM25 document set. Every chunk carries `filed_at`, form type,
  accession number, Item/section, and a stable `[Source source-id/item-id]`
  citation label.

Run `python indexer.py --force` on deployment and schedule it subsequently.
The Streamlit sidebar can also force a refresh. A production deployment
should run refreshing as a separate worker rather than inside web request
processes.

## Memory and safety

Memory is session-scoped, retention-bounded, and deletable. It stores only
final user/assistant turns, never hidden reasoning or retrieved documents.
In a real deployment, replace the Streamlit-generated ID with a
tenant-authorized opaque identifier and store SQLite data on encrypted,
access-controlled volumes.

Retrieved source text is treated as untrusted evidence, not instructions.
The synthesizer, critic, and guardrail all reject prompt-injection-like
instructions in sources — the guardrail's `prompt_injection_scan` is the
independently testable enforcement point for this.

## Required production infrastructure

Before serving multiple nodes or external users, add identity/authorization,
encryption and KMS-backed backups, monitoring, a managed relational/vector
store, distributed worker locking, an SEC data-license review (this project
uses only EDGAR's free public endpoints and GDELT's free API — verify terms
before any commercial redistribution), and human review of any signal before
it informs a real decision.

## Verification

```powershell
python -m compileall -q .
python -m pytest -q
python indexer.py --force
python run_event_study.py --tickers AAPL,MSFT,NVDA --limit 8
python run_retrieval_eval.py --n-questions 50 --k 5
```

On a machine with local embedding inference under CPU contention (observed
during this project's own verification run: ~1.4s/chunk with a competing
CPU-bound process running), `run_retrieval_eval.py` supports splitting
ingestion into bounded, resumable per-ticker calls that reuse the same
persisted index — this is not a special mode, it's the normal content-hash
incremental-refresh behavior `live_data.py` always has, just invoked once
per ticker:

```powershell
python run_retrieval_eval.py --ingest-only --tickers AAPL
python run_retrieval_eval.py --ingest-only --tickers MSFT
python run_retrieval_eval.py --ingest-only --tickers NVDA
python run_retrieval_eval.py --eval-only --n-questions 50 --k 5
```

The test suite verifies durable-memory isolation/retention/deletion,
incremental source updates, source provenance, cross-process source-refresh
leasing, zero point-in-time violations (200-trial adversarial sweep),
numeric-claim verification (including a deliberately corrupted source
chunk), and the guardrail's fail-closed behavior + adversarial fixture
suite.
