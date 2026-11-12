# Chronicle

A live, point-in-time-correct financial research RAG system. It builds its
evidence index only from SEC EDGAR filings and GDELT news for a configured
company watchlist (`live_sources.json`) — no bundled static knowledge base,
no LLM-discovered URLs.

**Live:** https://52-6-3-88.sslip.io. It runs always-on on AWS EC2 behind
Caddy (HTTPS), with the index over Apple, Microsoft and NVIDIA's latest
10-K/10-Q/8-K filings, refreshed daily. Try: *"What did Apple report about
iPhone net sales in its most recent 10-K?"* Deployment details:
[DEPLOYMENT.md](DEPLOYMENT.md).

```text
session memory -> live-source refresh (EDGAR/GDELT) -> research planner
-> hybrid retrieval (as_of hard-filtered) -> evidence synthesizer
-> evidence critic (+ numeric-claim verification) -> deny-by-default guardrail
-> cited, point-in-time-correct response
```

**Scope, stated plainly**: this project extracts signals from filing text
(tone delta) and tests them against forward returns with a proper
significance test. It does **not** do personal transaction
categorization/budgeting, macro/price forecasting, or portfolio/backtest
construction — those are explicitly out of scope.

## Constraints, stated plainly

- **SEC rate limit**: this project self-imposes a request floor of
  `EDGAR_MIN_REQUEST_INTERVAL_SECONDS` (default 0.34s, ≈3 req/s), well under
  SEC's fair-access ceiling (~10 req/s), and always sends a real,
  project-identifying User-Agent
  (https://www.sec.gov/os/webmaster-faq#developers). A verbose/instructional
  User-Agent string was empirically observed to get HTTP 403 from SEC's bot
  detection during development — keep `EDGAR_USER_AGENT` short:
  `AppName/Version (contact-email)`.
- **No free earnings-call transcript source exists.** 8-K EX-99 press-release
  exhibits are the only transcript-adjacent source this project uses, and are
  never presented as a transcript.
- **GDELT news chunks carry no `filed_at`.** They are exempt from the
  point-in-time hard filter by design (there is no authoritative
  publish-timestamp source for them) — see `pit.py`.

## Point-in-time correctness (the core claim)

Every EDGAR-sourced chunk carries `filed_at` from EDGAR's actual
`acceptanceDateTime` — never a scrape timestamp. Every retrieval call takes
an optional `as_of` (default: now) and hard-filters `filed_at <= as_of`
**before ranking**, at both the dense (Chroma numeric `where` filter) and
sparse (BM25 candidate-set filter) layers — not a post-hoc check
(`retrieval/vector_store.py`, `retrieval/bm25_store.py`, `retrieval/hybrid.py`).

A 10-K/A amendment gets its own `filed_at`; the original stays retrievable
and is annotated `superseded_by_*` only once the amendment is itself visible
at the query's `as_of` (`retrieval/hybrid.py::_annotate_supersession`).

**Adversarial test result** (`tests/test_point_in_time.py`, real Chroma +
real BM25Okapi, no mocks): a 200-trial randomized sweep of `as_of` instants
spanning before/between/after a filing and its amendment —

```
[point-in-time sweep] trials=200 violations=0
```

## Numeric-claim verification + guardrail

`agents/critic_agent.py` extends the existing LLM citation critic with a
deterministic numeric-claim check (`numeric_verify.py`): every number in a
generated answer is checked against its cited source chunk, with
unit/scale normalization (`"$1.2B"` ≡ `"1,200,000,000"`). An unsupported
number forces the verdict to `NEEDS_CORRECTION` regardless of what the LLM
concluded — proven by test with a deliberately corrupted source chunk
(`tests/test_critic_agent.py`).

`guardrail.py` is a pluggable, deny-by-default policy layer sitting after
synthesis: numeric-claim match, a **second, independent** point-in-time
recheck on the final answer's actual citations, a prompt-injection pattern
scan over retrieved evidence, and output-schema validation. Any check
failure — or the guardrail itself raising an exception — blocks the answer;
nothing is served unchecked (`pipeline.py::run_guardrail`). Every verdict is
written to an append-only decision log (`decision_log.py`).

```
[guardrail adversarial suite] false_positive_rate=0.00% false_negative_rate=0.00% (good_n=3, bad_n=5)
```

## Tone-delta signal + event study (Phase 4)

`tone_signal.py` scores Item 1A (Risk Factors) + Item 7 (MD&A) with the
genuine Loughran-McDonald Negative/Uncertainty word lists (`data/lm_lexicon/`
— provenance and verification in `data/lm_lexicon/SOURCE.md`), and computes
the DELTA between consecutive 10-Ks (the literature-supported measure, not
the raw level). An independent LLM-based tone-direction classification is
run on the same text; 9 real comparisons completed before an environment
timeout interrupted a longer run showed **5/9 (56%) agreement** between the
two methods (`run_event_study.py --llm-compare`) — disagreements were
methodologically real (e.g. the LM count moving slightly one way while the
LLM judged newly-added risk-factor detail as a qualitative worsening), not
bugs.

`event_study.py` computes the Information Coefficient (Spearman) between
tone-delta and a point-in-time-correct forward-10-day return
(`price_client.py`, Yahoo Finance daily closes, free/no-key), with a
Newey-West HAC-adjusted significance test (filing dates cluster in earnings
season, so a naive iid p-value would overstate significance).

**Real result** (`python run_event_study.py --tickers AAPL,MSFT,NVDA --limit 8`,
n=17 point-in-time-correct observations across 3 companies' full available
10-K history):

```
Information Coefficient (Spearman rho): -0.1299
  naive iid p-value (comparison only): 0.6192
  Newey-West HAC t-stat: -0.5942  p-value: 0.5524  (maxlags=2)
```

**Reported honestly**: this relationship is **not distinguishable from
zero** on this sample (|t| < 1). n=17 is small — three companies' available
10-K history — so this is a real finding from a real run, not a claim the
signal has no value at larger scale. This project stops at the IC and its
significance test; it does not extend into a backtest, portfolio, or
trading rule.

## Retrieval evaluation (Phase 5)

`eval_retrieval.py` builds a labelled question set by generating one
factual question per sampled real ingested chunk via the LLM — the source
chunk's id is the ground-truth relevant document by construction (not a
hand-curated benchmark; stated plainly). `run_retrieval_eval.py` runs the
resulting ablation for real against BM25-only, dense-only, hybrid+RRF, and
hybrid+RRF+cross-encoder, plus measures real EDGAR-ingestion latency.

**Real result** (`python run_retrieval_eval.py --n-questions 30 --k 5`, 661 real
chunks ingested from AAPL/MSFT/NVDA 10-K/10-Q filings, 30 LLM-generated
labelled questions):

```
config         recall@5      nDCG@5        n
bm25           0.867         0.664         30
dense          0.633         0.486         30
hybrid_rrf     0.867         0.698         30
hybrid_rrf_ce  0.900         0.781         30
```

Each retrieval stage earns its place on this run: hybrid+RRF ties BM25 on
recall but raises nDCG (better ranking of the same hits), and adding the
cross-encoder raises both further. Dense-only trails the others here — a
real result on this question set, not a claim that dense embeddings are
worthless in general.

**Ingestion latency**, measured over the same real run (3 EDGAR source
refreshes, 661 chunks):

```
filed_at -> queryable:      p50 = 97.8 days   p99 = 162.4 days   (n=661 documents)
source-refresh wall time:   p50 = 1010.55s    p99 = 1162.91s     (n=3 source-refresh calls)
```

The first number is real but dominated by filing age (these are historical
10-Ks/10-Qs, not same-day filings) — it is NOT a system-latency claim. The
second number — fetch-start to queryable, per source-refresh call — is the
one that reflects this system's actual processing throughput; both are
reported so neither is mistaken for the other.

## Run

```powershell
python -m pip install -r requirements.txt
python indexer.py --force                 # initialize from configured live sources
streamlit run app.py
python -m pytest -q                        # 79 tests
python run_event_study.py --tickers AAPL,MSFT,NVDA --limit 8 [--llm-compare]
python run_retrieval_eval.py --n-questions 50 --k 5
```

For operations, security controls, data lifecycle, source configuration, and
deployment boundaries, see [PRODUCTION_ARCHITECTURE.md](PRODUCTION_ARCHITECTURE.md).

This is research support — signal extraction and evidence synthesis, not
investment advice, a price target, or a trading recommendation.
