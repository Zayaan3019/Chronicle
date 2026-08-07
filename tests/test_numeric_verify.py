"""Unit tests for the deterministic numeric-claim verifier (Phase 3)."""

from __future__ import annotations

import pytest

from numeric_verify import extract_numbers, verify_numeric_claims
from retrieval.vector_store import ChunkResult


def _chunk(chunk_id: str, text: str) -> ChunkResult:
    return ChunkResult(
        chunk_id=chunk_id,
        text=text,
        article_id=f"live:nvda:{chunk_id}",
        article_title="NVIDIA 10-K",
        article_source="SEC EDGAR",
        cluster="Financial live sources",
        url=f"https://example.test/{chunk_id}",
    )


# ── Normalization: format-invariant equivalence ──────────────────────────────

def test_dollar_billion_matches_expanded_digits():
    answer = "Full-year revenue guidance was $1.2B [Source nvda/abc123]."
    source = _chunk("abc123", "The company guided to full-year revenue of $1,200,000,000.")
    result = verify_numeric_claims(answer, [source])
    assert result.unsupported == []
    assert result.checked_against_chunks == 1


def test_percent_matches_percent_word():
    answer = "Gross margin was 45.2% [Source nvda/abc123]."
    source = _chunk("abc123", "Gross margin came in at 45.2 percent for the quarter.")
    result = verify_numeric_claims(answer, [source])
    assert result.unsupported == []


def test_million_suffix_matches_full_digits():
    answer = "R&D spend was $1.5M [Source nvda/abc123]."
    source = _chunk("abc123", "Research and development expense totaled $1,500,000.")
    result = verify_numeric_claims(answer, [source])
    assert result.unsupported == []


def test_percent_does_not_match_plain_number_of_equal_magnitude():
    """45% and 45 are not the same claim — percent-ness must be preserved."""
    answer = "Growth was 45% [Source nvda/abc123]."
    source = _chunk("abc123", "The backlog reached 45 units in the quarter.")
    result = verify_numeric_claims(answer, [source])
    assert result.unsupported == ["45%"]


# ── Corrupted source chunk: the required Phase 3 test ────────────────────────

def test_corrupted_source_number_is_flagged_unsupported():
    """The exact scenario the spec requires: deliberately change a number in
    the source chunk and confirm the verifier catches the mismatch."""
    answer = "Full-year revenue guidance was $1.2B [Source nvda/abc123]."
    corrupted_source = _chunk("abc123", "The company guided to full-year revenue of $1,500,000,000.")
    result = verify_numeric_claims(answer, [corrupted_source])
    assert result.unsupported == ["$1.2B"]
    assert result.checked_against_chunks == 1


def test_uncited_numeric_claim_is_reported_separately():
    answer = "Revenue was $1.2B, though no source was given for that figure."
    source = _chunk("abc123", "Unrelated commentary with no matching number.")
    result = verify_numeric_claims(answer, [source])
    assert result.unsupported == []  # not checked against a source at all
    assert result.uncited == ["$1.2B"]


def test_claim_supported_by_any_of_multiple_cited_sources():
    answer = "Revenue was $1.2B [Source nvda/abc123] [Source nvda/def456]."
    wrong_source = _chunk("abc123", "Unrelated text with a different number, $9.9B.")
    right_source = _chunk("def456", "Revenue of $1,200,000,000 was reported.")
    result = verify_numeric_claims(answer, [wrong_source, right_source])
    assert result.unsupported == []


# ── extract_numbers: bare numbers excluded, financial numbers included ──────

@pytest.mark.parametrize(
    "text,expected_raws",
    [
        ("Filed under Item 7 for fiscal 2024.", []),                 # bare ints: item/year, not financial
        ("Revenue was $1.2B this quarter.", ["$1.2B"]),
        ("Margin improved to 45.2%.", ["45.2%"]),
        ("Total assets were 1,234,000,000.", ["1,234,000,000"]),
        ("Headcount grew to 12,000 employees.", ["12,000"]),         # comma-grouped -> counted even w/o $/%
    ],
)
def test_extract_numbers_filters_bare_integers(text, expected_raws):
    claims = extract_numbers(text)
    assert [c.raw for c in claims] == expected_raws


def test_no_chunks_returns_empty_result():
    result = verify_numeric_claims("Revenue was $1.2B [Source nvda/abc123].", [])
    assert result.unsupported == []
    assert result.total_claims == 0
