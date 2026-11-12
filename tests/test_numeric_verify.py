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


# ── Filing tables: declared units and pipe-separated cells ───────────────────
# The chunk below is the real shape of an Apple 10-K net-sales table as it
# reaches the index (caption states the unit; cells are pipe-separated).

_TEN_K_TABLE = (
    "The following table shows net sales by category for 2026, 2025 and 2024 "
    "(dollars in millions): [TABLE] 2026 | Change | 2025 | Change | 2024 "
    "iPhone | $ | 254,098 | 21 | % | $ | 209,586 | 4 | % | $ | 201,183"
)


def test_declared_unit_and_table_cells_support_a_correct_answer():
    answer = (
        "iPhone net sales were $254,098 million in 2026, up 21% from $209,586 million "
        "[Source nvda/t1]. That is about $254.1 billion [Source nvda/t1]."
    )
    result = verify_numeric_claims(answer, [_chunk("t1", _TEN_K_TABLE)])
    assert result.unsupported == []
    assert result.checked_against_chunks == 4


def test_declared_unit_does_not_excuse_a_wrong_figure():
    answer = "iPhone net sales were $264,098 million, up 31% [Source nvda/t1]."
    result = verify_numeric_claims(answer, [_chunk("t1", _TEN_K_TABLE)])
    assert sorted(result.unsupported) == ["$264,098 million", "31%"]


def test_unscaled_figure_without_declared_unit_still_fails_at_scale():
    answer = "Revenue was $254,098 million [Source nvda/t2]."
    source = _chunk("t2", "Revenue | $ | 254,098 | 2025 | 209,586")  # no unit caption
    assert verify_numeric_claims(answer, [source]).unsupported == ["$254,098 million"]
