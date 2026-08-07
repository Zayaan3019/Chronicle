"""Unit tests for the Phase 4 event study's statistical machinery
(event_study.py) and the point-in-time-safe forward-return helper
(price_client.py). All synthetic/deterministic — no network calls.
"""

from __future__ import annotations

import pytest

from event_study import EventObservation, compute_event_study, MIN_OBSERVATIONS_FOR_SIGNIFICANCE
from price_client import PricePoint, forward_return
from tone_signal import ToneDelta, ToneScore


def _tone_delta(cik: str, prev_neg: float, curr_neg: float, filed_at: str) -> ToneDelta:
    prev_score = ToneScore(total_words=1000, negative_count=int(prev_neg * 1000),
                            uncertainty_count=0, negative_proportion=prev_neg, uncertainty_proportion=0.0)
    curr_score = ToneScore(total_words=1000, negative_count=int(curr_neg * 1000),
                            uncertainty_count=0, negative_proportion=curr_neg, uncertainty_proportion=0.0)
    return ToneDelta(
        company_cik=cik, previous_accession="prev", current_accession="curr",
        previous_filed_at="2023-01-01T00:00:00.000Z", current_filed_at=filed_at,
        previous_score=prev_score, current_score=curr_score,
    )


# ── price_client.forward_return: point-in-time safety ──────────────────────

def test_forward_return_uses_only_points_on_or_after_event_date():
    points = [
        PricePoint("2024-01-01", 100.0),   # before event_date — must never be the anchor
        PricePoint("2024-03-01", 100.0),   # anchor (>= event_date)
        PricePoint("2024-03-11", 110.0),   # horizon
    ]
    result = forward_return(points, event_date="2024-03-01", horizon_days=10)
    assert result == pytest.approx(0.10)


def test_forward_return_none_when_no_horizon_point_available():
    points = [PricePoint("2024-03-01", 100.0)]
    assert forward_return(points, event_date="2024-03-01", horizon_days=10) is None


def test_forward_return_none_when_empty():
    assert forward_return([], event_date="2024-03-01", horizon_days=10) is None


# ── compute_event_study: honest small-N handling ────────────────────────────

def test_insufficient_observations_makes_no_significance_claim():
    observations = [
        EventObservation(_tone_delta("cik1", 0.01, 0.02, "2024-01-01T00:00:00.000Z"), "2024-01-01", 0.01, "NVDA"),
        EventObservation(_tone_delta("cik1", 0.02, 0.03, "2024-02-01T00:00:00.000Z"), "2024-02-01", 0.02, "NVDA"),
    ]
    assert len(observations) < MIN_OBSERVATIONS_FOR_SIGNIFICANCE
    result = compute_event_study(observations)
    assert result.insufficient_data is True
    assert result.information_coefficient is None
    assert result.newey_west_pvalue is None
    assert result.n_observations == 2


def test_perfectly_correlated_synthetic_signal_gives_ic_near_one():
    """Validates the statistical machinery itself: if tone_delta and forward
    return are constructed to be monotonically related, the IC should
    recover that relationship (not a claim about real markets)."""
    observations = []
    for i in range(8):
        delta_val = 0.001 * i
        observations.append(
            EventObservation(
                _tone_delta("cik1", 0.01, 0.01 + delta_val, f"2024-{i+1:02d}-01T00:00:00.000Z"),
                f"2024-{i+1:02d}-01", 0.01 * i, "NVDA",
            )
        )
    result = compute_event_study(observations)
    assert result.insufficient_data is False
    assert result.information_coefficient == pytest.approx(1.0, abs=1e-6)
    assert result.n_observations == 8
    assert result.newey_west_lags is not None
    assert result.newey_west_tstat is not None


def test_uncorrelated_signal_is_reported_as_not_significant():
    """Alternating signs with no relationship to forward return — IC should
    be far from +/-1 and the honest result is a small/insignificant number,
    not a manufactured strong signal."""
    tone_vals = [0.001, -0.001, 0.002, -0.002, 0.0015, -0.0015, 0.0005, -0.0005]
    fwd_vals = [0.01, 0.01, -0.02, -0.02, 0.015, 0.015, -0.005, -0.005]
    observations = [
        EventObservation(
            _tone_delta("cik1", 0.01, 0.01 + t, f"2024-{i+1:02d}-01T00:00:00.000Z"),
            f"2024-{i+1:02d}-01", f_, "NVDA",
        )
        for i, (t, f_) in enumerate(zip(tone_vals, fwd_vals))
    ]
    result = compute_event_study(observations)
    assert result.insufficient_data is False
    assert result.information_coefficient is not None
    assert abs(result.information_coefficient) < 1.0
