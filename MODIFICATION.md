# Convert healthcare_rag into a live, point-in-time-correct financial research
# agent — working title "Chronicle" (rename freely)

You are converting a working repository to a new domain, not rebuilding it.
Preserve and extend the generic infrastructure; replace only the domain-specific
layers.

## Rules (non-negotiable)
- Read the existing repo FIRST: agents/{critic,query,reasoning}_agent.py,
  retrieval/{hybrid,bm25_store,vector_store}.py, live_data.py, indexer.py,
  memory.py, pipeline.py, tests/. Report what each does before changing it.
- Cite file:line for every claim. Run tests and paste real output.
- Never invent a metric. If something isn't measured, say so.
- STOP after each phase and show me a summary before continuing.

## Explicit non-goals — do not rebuild these
- No personal transaction categorization, budgeting, or anomaly detection on
  bank data. That is FinGuru's territory and FinGuru is retired from the
  resume specifically to avoid this overlap — do not reintroduce it here.
- No time-series forecasting of macro or price targets, no ARDL, no next-value
  prediction. That is the Neuro-Econometric Engine's territory.
- No quintile long/short backtest, no portfolio optimizer, no multi-factor
  research lab. That is Arbitron's territory. This project's quantitative
  component stops at signal extraction + significance testing.
- Remove all medical/clinical-specific scaffolding (prompts, entity types,
  citation formats, source configs) rather than leaving it alongside the new
  financial domain.

## PHASE 1 — Domain conversion of ingestion
1. Repoint the live-source configuration from its current (medical) sources to:
   - SEC EDGAR: `data.sec.gov/submissions/CIK{cik}.json` for filing metadata
     (INCLUDES exact filing timestamps — this is what Phase 3 depends on),
     and `efts.sec.gov` full-text search for discovery. Set a real User-Agent
     identifying this project and yourself; self-impose a rate cap (SEC blocks
     non-compliant clients). Document this constraint plainly in the README.
   - GDELT for real-time news and event context (free, no key required).
   - 8-K exhibits as the ONLY transcript-adjacent source. State clearly that
     no free live earnings-call transcript source exists and this is the
     honest substitute — do not imply broader coverage than this provides.
2. Extend `indexer.py` for real filing structure: 10-K/10-Q are Item-sectioned,
   mix HTML tables with prose, and are large. Chunk by Item/section boundary,
   not by naive token count. Preserve table structure well enough that a
   numeric claim inside a table survives into a retrievable chunk verbatim.
3. Reuse `live_data.py`'s incremental refresh and source-lease pattern as-is —
   it already does what a "live" system needs. Verify its existing tests still
   pass against the new sources.

## PHASE 2 — Point-in-time correctness (this is the project's core claim)
4. Every ingested document gets a `filed_at` timestamp from EDGAR's actual
   submission metadata — never a scrape/ingestion timestamp, which would be
   wrong. Amendments (10-K/A) get their own `filed_at` and a link to what they
   supersede; the original stays retrievable, it does not vanish.
5. Every query takes an optional `as_of` parameter (default: now). The
   retrieval layer MUST filter `filed_at <= as_of` before ranking — implement
   this as a hard filter in the query, not a post-hoc check.
6. Write the adversarial test suite for this: query `as_of` a date before a
   known filing's `filed_at` and assert it is NOT retrievable; query after and
   assert it is; query between an original filing and its amendment and assert
   only the original is visible. **Report a violation count — it must be
   zero, and say how you know.**
7. When a 10-K/A exists and is visible under the current `as_of`, and the
   query would otherwise return the superseded original as top result, the
   system must surface the conflict/supersession rather than silently
   returning the newer one with no indication anything changed.

## PHASE 3 — Numeric-claim verification (extend, do not replace, the existing critic)
8. Read `agents/critic_agent.py` to understand its current citation-verification
   approach. Extend it: any number appearing in a generated answer must be
   checked against the exact number present in the cited source chunk (string
   or normalized-numeric match, handling units/scale like "$1.2B" vs
   "1,200,000,000"). A number with no matching source number fails
   verification and the answer must flag it, not present it as grounded.
9. Test with a deliberately corrupted source chunk (a number changed) and
   confirm the verifier catches the mismatch in the generated answer.\

## PHASE 3B — Guardrail layer (deny-by-default runtime policy enforcement)

Structure the checks already planned in Phase 3 as an explicit, independently
testable policy layer, following the same fail-closed pattern as AMME's
risk-svc — deny by default, veto power separate from the generator, no silent
pass-through on error.

1. Build the guardrail as a pluggable module (not hard-wired into the RAG
   pipeline internals), so it could later front AgentForge's reward
   verification or the eval platform's custom evaluators without a rewrite.
2. Checks, each independently unit-testable with a pass/fail verdict:
   - Numeric-claim match against source (already scoped, Phase 3)
   - Point-in-time violation, checked a SECOND time on the final answer's
     citations, independent from the retrieval-time filter
   - Prompt-injection pattern in retrieved document content, with retrieved
     text always treated as delimited data, never as instruction
   - Output schema/format validation — malformed structured output is
     BLOCKED, not passed through with a shrug
3. Fail-closed test: kill the guardrail process mid-request; assert the
   caller blocks or escalates, never silently serves an unchecked answer.
4. Immutable decision log: verdict, rule triggered, input/output hash,
   timestamp, append-only.
5. Adversarial fixture suite: known-bad inputs (injection attempts, corrupted
   numbers, stale-document citations) each with an expected verdict. Wire as
   a CI gate — non-zero exit on regression. Report BOTH false-positive rate
   and false-negative rate on every run, never one without the other.
6. If the eval platform exists by the time this is built, route this fixture
   suite through its existing regression-gate CI integration rather than
   building a second one. If not, a standalone GitHub Action is sufficient.

Definition of done: fail-closed proven by test, both error rates reported on
every CI run, and the decision log format matches what the eval platform's
trace-to-eval promotion would expect to ingest later.

## PHASE 4 — Text-derived signal (kept deliberately light — see non-goals)
10. Implement Loughran-McDonald negative/uncertainty word-list scoring over
    the Risk Factors and MD&A sections of consecutive 10-Ks for a given
    company; compute the tone DELTA between filings (the literature-supported
    version, not the raw level).
11. As a comparison, extract the same tone signal via LLM-based classification
    of the same sections. Report where the two agree and disagree — this
    comparison is itself a finding, report it honestly either way.
12. Event study: for each point-in-time-correct tone-delta observation, compute
    forward N-day return using ONLY data that would have been available at
    that time. Compute the Information Coefficient (Spearman) between
    tone-delta and forward return, with a **Newey-West-adjusted significance
    test** — filing dates cluster in earnings season, so naive iid SEs will
    overstate significance, exactly the mistake already fixed in Arbitron.
13. **Report the IC and its significance honestly.** If it is not
    distinguishable from zero, that is a legitimate, publishable-shaped
    finding — say so, the way the Neuro-Econometric Engine's own evaluation
    report says "needs significant improvement" rather than hiding it.
14. Do not extend this into a backtest, a portfolio, or a trading rule. Stop
    at the IC and its significance test.

## PHASE 5 — Evaluation of the retrieval pipeline itself
15. Build a labelled set of 50-100 finance questions with known relevant
    filing sections. Report recall@k and nDCG@k for BM25-only, dense-only,
    hybrid+RRF, and hybrid+RRF+cross-encoder — the same ablation structure
    used to hardened Clinical RAG, now on financial documents. This proves
    each retrieval stage earns its place, on the new domain.
16. Measure end-to-end ingestion latency: time from EDGAR's `filed_at` to the
    document being queryable in your index. This is your "live" metric —
    report p50/p99 over a real ingestion run, not an estimate.

## PHASE 6 — Tests and honesty
17. Bring the test suite to reflect the new domain and new claims: point-in-time
    filtering (Phase 2), numeric verification (Phase 3), signal computation and
    its statistical test (Phase 4), retrieval ablation reproducibility (Phase 5).
18. Rewrite the README for the new domain. State the SEC rate-limit constraint,
    the earnings-transcript limitation, and the IC finding (whatever it turned
    out to be) as plainly as the rest of this portfolio states its limitations.

## Definition of done
- Zero point-in-time violations, proven by an adversarial test, not asserted
- Numeric-claim verification catches a deliberately corrupted number, by test
- Retrieval ablation table (recall@k, nDCG@k) across all four configurations
- IC and its Newey-West-adjusted significance reported, whatever the result
- Ingestion latency (EDGAR filed_at -> queryable) measured with p50/p99
- README contains zero claims not backed by a number produced in this repo
- All medical-domain scaffolding removed, not left dormant alongside the new code
- At the end, push all the required 7 appropriate code files to the repo - https://github.com/Zayaan3019/Chronicle