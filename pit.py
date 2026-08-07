"""Point-in-time correctness helpers shared by the whole retrieval layer.

This is the single place that parses/compares `filed_at`. Every component that
needs a point-in-time decision — the vector store's numeric Chroma filter, the
BM25 pre-ranking filter, and (Phase 3B) the guardrail's independent second
check — imports this module rather than re-implementing timestamp comparison,
so there is exactly one definition of "visible at as_of" in the codebase.

`filed_at` values are ISO-8601 UTC strings taken verbatim from EDGAR's
`acceptanceDateTime` (e.g. "2026-07-02T13:23:16.000Z") — never a scrape or
ingestion timestamp. Non-filing sources (GDELT news) carry no authoritative
point-in-time anchor and are stored with `filed_at = ""`; see `is_visible`
for how that is handled honestly rather than silently.
"""

from __future__ import annotations

from datetime import datetime, timezone

# Sentinel stored in Chroma metadata for chunks with no filed_at (e.g. GDELT
# news). Chroma metadata cannot hold None, and we still need a numeric value
# so a single $lte/$or where-clause can express "either before as_of, or has
# no point-in-time anchor at all".
NO_FILED_AT_SENTINEL = -1.0


def parse_iso_to_epoch(value: str | None) -> float | None:
    """Parse an ISO-8601 UTC timestamp to a Unix epoch float, or None if unparseable/empty."""
    if not value:
        return None
    text = value.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(text)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.timestamp()


def resolve_as_of_epoch(as_of: str | None) -> float:
    """Resolve the optional `as_of` query parameter to an epoch. Defaults to now (UTC)."""
    if as_of is None:
        return datetime.now(timezone.utc).timestamp()
    epoch = parse_iso_to_epoch(as_of)
    if epoch is None:
        raise ValueError(f"as_of is not a valid ISO-8601 timestamp: {as_of!r}")
    return epoch


def is_visible(filed_at: str, as_of_epoch: float) -> bool:
    """True if a document with this `filed_at` (possibly "") is visible at `as_of_epoch`.

    Documents with no filed_at (non-EDGAR sources, e.g. GDELT news) carry no
    point-in-time guarantee at all and are always considered visible — the
    hard as_of filter is a filing-level guarantee, not a blanket claim about
    every source in the corpus. This is a documented limitation (README),
    not a silent gap: `filed_at == ""` is explicit, not a default that could
    be confused with "filed at the epoch".
    """
    epoch = parse_iso_to_epoch(filed_at)
    if epoch is None:
        return True
    return epoch <= as_of_epoch
