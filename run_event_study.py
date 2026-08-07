"""Phase 4 CLI — real, end-to-end tone-delta event study.

Ingests each configured company's real 10-K filing history directly via
edgar_client (a standalone research script, not the RAG index), computes
Loughran-McDonald tone deltas between consecutive 10-Ks (Item 1A + Item 7),
optionally runs the independent LLM-based tone-direction comparison, fetches
real forward returns via price_client (Yahoo Finance, free/no-key), and
reports the pooled Information Coefficient with its Newey-West-adjusted
significance test — honestly, including when n is too small to claim
significance at all.

This script does not build a backtest, portfolio, or trading rule — it stops
at printing the IC and its significance test (MODIFICATION.md's non-goals).

Usage:
    python run_event_study.py [--horizon-days 10] [--tickers AAPL,MSFT,NVDA] [--llm-compare]
"""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone

import edgar_parse
import tone_signal
from edgar_client import EdgarClient
from event_study import EventObservation, compute_event_study
from price_client import PriceClient, forward_return

# CIKs already verified against SEC's own company_tickers.json (see live_sources.json).
WATCHLIST = {
    "AAPL": "0000320193",
    "MSFT": "0000789019",
    "NVDA": "0001045810",
}


def _fetch_tone_deltas(edgar: EdgarClient, ticker: str, cik: str, limit: int):
    filings = [f for f in edgar.list_filings(cik, forms=("10-K",), limit=limit) if not f.is_amendment]
    filings.sort(key=lambda f: f.filed_at)  # oldest first — must pair consecutively in time
    parsed: list[tuple] = []
    for filing in filings:
        try:
            html = edgar.get_filing_document(filing.cik, filing.accession_number, filing.primary_document)
        except Exception as exc:
            print(f"  [skip] {ticker} {filing.accession_number}: fetch failed ({exc})")
            continue
        chunks = edgar_parse.chunk_filing(html)
        text = tone_signal.extract_target_section_text(chunks)
        if len(text.split()) < 200:
            print(f"  [skip] {ticker} {filing.accession_number}: only {len(text.split())} words of Item 1A/7 text extracted")
            continue
        parsed.append((filing, text))

    results = []
    for (prev_filing, prev_text), (curr_filing, curr_text) in zip(parsed, parsed[1:]):
        delta = tone_signal.tone_delta(
            cik, prev_filing.accession_number, prev_filing.filed_at, prev_text,
            curr_filing.accession_number, curr_filing.filed_at, curr_text,
        )
        results.append((delta, curr_filing.filed_at, prev_text, curr_text))
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--horizon-days", type=int, default=10, help="forward-return horizon in calendar days")
    parser.add_argument("--tickers", default=",".join(WATCHLIST), help="comma-separated tickers from the watchlist")
    parser.add_argument("--limit", type=int, default=20, help="max 10-K filings to pull per company")
    parser.add_argument("--llm-compare", action="store_true", help="also run the LLM-based tone comparison (needs Ollama reachable)")
    args = parser.parse_args()

    tickers = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]
    edgar = EdgarClient()
    price = PriceClient()

    observations: list[EventObservation] = []
    agreements = 0
    llm_compared = 0

    print(f"Horizon: {args.horizon_days} calendar day(s)\n")
    for ticker in tickers:
        cik = WATCHLIST.get(ticker)
        if not cik:
            print(f"[skip] unknown ticker {ticker!r} — not in WATCHLIST")
            continue
        print(f"=== {ticker} (CIK {cik}) ===")
        deltas = _fetch_tone_deltas(edgar, ticker, cik, args.limit)
        print(f"  {len(deltas)} consecutive-10-K tone-delta observation(s)")

        for delta, event_filed_at, prev_text, curr_text in deltas:
            event_date = event_filed_at[:10]
            window_start = datetime.strptime(event_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
            period1 = int(window_start.timestamp())
            period2 = int((window_start + timedelta(days=args.horizon_days + 10)).timestamp())
            try:
                points = price.daily_closes(ticker, period1=period1, period2=period2)
                fwd = forward_return(points, event_date=event_date, horizon_days=args.horizon_days)
            except Exception as exc:
                print(f"    [skip] price fetch failed for {event_date}: {exc}")
                continue
            if fwd is None:
                print(f"    [skip] no forward-return price data available for {event_date}")
                continue

            print(
                f"    event_date={event_date} negative_delta={delta.negative_delta:+.5f} "
                f"uncertainty_delta={delta.uncertainty_delta:+.5f} "
                f"forward_return({args.horizon_days}d)={fwd:+.4f}"
            )
            observations.append(EventObservation(delta, event_date, fwd, ticker))

            if args.llm_compare:
                try:
                    llm_result = tone_signal.llm_tone_direction(prev_text, curr_text)
                    comparison = tone_signal.compare_methods(delta, llm_result)
                    llm_compared += 1
                    if comparison["llm_available"]:
                        agreements += int(comparison["agree"])
                        print(
                            f"      LM direction={comparison['lm_direction']}  "
                            f"LLM direction={comparison['llm_direction']}  "
                            f"agree={comparison['agree']}  rationale={llm_result.get('rationale', '')!r}"
                        )
                    else:
                        print("      LLM comparison unavailable (invalid/empty model output)")
                except Exception as exc:
                    print(f"      [llm-compare skipped: {exc}]")

    if args.llm_compare and llm_compared:
        print(f"\nLM vs LLM tone-direction agreement: {agreements}/{llm_compared} ({agreements / llm_compared:.0%})")
    elif args.llm_compare:
        print("\nLM vs LLM tone-direction comparison: no pairs were compared (see skip messages above)")

    print(f"\n=== Pooled event study: n={len(observations)} observation(s) across {len(tickers)} ticker(s) ===")
    result = compute_event_study(observations)
    if result.insufficient_data:
        print(result.note)
    else:
        print(f"Information Coefficient (Spearman rho): {result.information_coefficient:+.4f}")
        print(f"  naive iid p-value (reported for comparison only, NOT the headline number): {result.ic_pvalue_iid:.4f}")
        print(
            f"  Newey-West HAC t-stat: {result.newey_west_tstat:+.4f}  "
            f"p-value: {result.newey_west_pvalue:.4f}  (maxlags={result.newey_west_lags})"
        )
        print(f"  {result.note}")


if __name__ == "__main__":
    main()
