"""Phase 4 — event study: forward-return Information Coefficient for the
tone-delta signal, with a Newey-West-adjusted significance test.

Stops exactly where the spec requires: computes the Information Coefficient
(Spearman rank correlation) between a tone-delta observation and its
point-in-time-correct forward return, and a Newey-West HAC-adjusted
significance test on that relationship. Does NOT build a backtest,
portfolio, or trading rule (see MODIFICATION.md's non-goals) — this module
stops at the IC and its significance test, full stop.

Every `EventObservation` fed in here must already be point-in-time correct:
`event_date` is the CURRENT filing's real `filed_at` (never a scrape time),
and `forward_return` must have been computed using ONLY price data from
`event_date` forward (price_client.forward_return enforces this by
construction — it never looks at a price before the anchor date).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats as scipy_stats
# Import the narrow regression submodule directly rather than
# `statsmodels.api` — the latter eagerly imports statsmodels.graphics, which
# pulls in a Cython lowess extension observed to be binary-incompatible with
# this environment's numpy 2.x (ValueError: numpy.dtype size changed). Only
# OLS + add_constant are actually needed for the Newey-West HAC regression.
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant

from tone_signal import ToneDelta

# Below this many observations, a correlation/significance number is not a
# meaningful claim — report that plainly instead of manufacturing a p-value
# out of a handful of points.
MIN_OBSERVATIONS_FOR_SIGNIFICANCE = 5


@dataclass(frozen=True)
class EventObservation:
    tone_delta: ToneDelta
    event_date: str          # the CURRENT filing's filed_at — the point-in-time anchor
    forward_return: float    # must already be point-in-time correct (price_client.forward_return)
    ticker: str


@dataclass
class EventStudyResult:
    n_observations: int
    information_coefficient: float | None = None   # Spearman rho
    ic_pvalue_iid: float | None = None              # scipy's naive iid p-value — reported for comparison only, not the headline claim
    newey_west_tstat: float | None = None
    newey_west_pvalue: float | None = None
    newey_west_lags: int | None = None
    insufficient_data: bool = False
    note: str = ""


def _newey_west_lags(n: int) -> int:
    """Newey & West's (1994) rule-of-thumb lag, floor(4*(n/100)^(2/9)),
    floored at 1 and capped so the HAC estimator stays identified on a small
    sample. An explicit, documented choice — not a hidden default."""
    lag = int(4 * (n / 100) ** (2 / 9))
    return max(1, min(lag, max(1, n // 3)))


def compute_event_study(observations: list[EventObservation]) -> EventStudyResult:
    n = len(observations)
    if n < MIN_OBSERVATIONS_FOR_SIGNIFICANCE:
        return EventStudyResult(
            n_observations=n,
            insufficient_data=True,
            note=(
                f"Only {n} point-in-time-correct observation(s) available — below the "
                f"{MIN_OBSERVATIONS_FOR_SIGNIFICANCE} needed for a non-degenerate "
                "significance test. No IC or significance claim is made for this run."
            ),
        )

    # Sort chronologically by event_date BEFORE the HAC regression — Newey-West
    # assumes the lag structure follows the observations' natural order, and
    # filing dates cluster in earnings season (the exact autocorrelation this
    # test exists to correct for, per MODIFICATION.md).
    ordered = sorted(observations, key=lambda o: o.event_date)
    tone = np.array([o.tone_delta.negative_delta for o in ordered], dtype=float)
    fwd = np.array([o.forward_return for o in ordered], dtype=float)

    rho, p_iid = scipy_stats.spearmanr(tone, fwd)

    # Newey-West HAC significance test on the SAME rank relationship: OLS of
    # rank(forward_return) on rank(tone_delta) recovers Spearman's rho as a
    # (rescaled) regression slope, so its HAC-adjusted t-stat is a legitimate
    # significance test for the same correlation that accounts for
    # autocorrelated observations — unlike scipy's iid p-value above.
    rank_tone = scipy_stats.rankdata(tone)
    rank_fwd = scipy_stats.rankdata(fwd)
    design = add_constant(rank_tone)
    lags = _newey_west_lags(n)
    ols = OLS(rank_fwd, design).fit(cov_type="HAC", cov_kwds={"maxlags": lags})
    slope_idx = 1
    hac_tstat = float(ols.tvalues[slope_idx])
    hac_pvalue = float(ols.pvalues[slope_idx])

    return EventStudyResult(
        n_observations=n,
        information_coefficient=float(rho) if np.isfinite(rho) else None,
        ic_pvalue_iid=float(p_iid) if np.isfinite(p_iid) else None,
        newey_west_tstat=hac_tstat,
        newey_west_pvalue=hac_pvalue,
        newey_west_lags=lags,
        insufficient_data=False,
        note=(
            f"IC computed on n={n} point-in-time-correct (tone_delta, forward_return) pairs. "
            f"Newey-West HAC maxlags={lags} (Newey & West 1994 rule-of-thumb)."
        ),
    )
