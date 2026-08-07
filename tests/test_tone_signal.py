"""Unit tests for the Loughran-McDonald tone-delta scorer (Phase 4)."""

from __future__ import annotations

import tone_signal


def test_lexicons_loaded_with_published_category_sizes():
    """Sanity check that the bundled word lists are genuinely the LM Master
    Dictionary's Negative/Uncertainty categories (see data/lm_lexicon/SOURCE.md)
    and not an accidental truncation — verified against the officially
    published category sizes for this dictionary vintage."""
    assert len(tone_signal.NEGATIVE_WORDS) == 2355
    assert len(tone_signal.UNCERTAINTY_WORDS) == 297
    assert "litigation" in tone_signal.NEGATIVE_WORDS
    assert "uncertainty" in tone_signal.UNCERTAINTY_WORDS


def test_score_text_counts_known_negative_words():
    text = "The company faced significant litigation and impairment losses this year."
    score = tone_signal.score_text(text)
    assert score.negative_count >= 2  # litigation, impairment, losses (at least some subset)
    assert score.total_words == len(text.split())
    assert 0.0 < score.negative_proportion < 1.0


def test_score_text_empty_string_is_zero_not_divide_by_zero():
    score = tone_signal.score_text("")
    assert score.total_words == 0
    assert score.negative_proportion == 0.0
    assert score.uncertainty_proportion == 0.0


def test_tone_delta_worsening_when_negative_words_increase():
    previous = "Business conditions were stable and results were solid."
    current = "Business faced litigation, impairment, and significant losses amid uncertainty."
    delta = tone_signal.tone_delta(
        "0000000001",
        "acc-prev", "2023-03-01T00:00:00.000Z", previous,
        "acc-curr", "2024-03-01T00:00:00.000Z", current,
    )
    assert delta.negative_delta > 0
    assert tone_signal.lm_direction(delta.negative_delta) == "WORSENED"


def test_tone_delta_improving_when_negative_words_decrease():
    previous = "Litigation, impairment, and significant losses affected results."
    current = "Business conditions were stable, favorable, and consistent."
    delta = tone_signal.tone_delta(
        "0000000001",
        "acc-prev", "2023-03-01T00:00:00.000Z", previous,
        "acc-curr", "2024-03-01T00:00:00.000Z", current,
    )
    assert delta.negative_delta < 0
    assert tone_signal.lm_direction(delta.negative_delta) == "IMPROVED"


def test_lm_direction_threshold_treats_tiny_delta_as_unchanged():
    assert tone_signal.lm_direction(0.0001) == "UNCHANGED"
    assert tone_signal.lm_direction(-0.0001) == "UNCHANGED"
    assert tone_signal.lm_direction(0.01) == "WORSENED"
    assert tone_signal.lm_direction(-0.01) == "IMPROVED"


def test_extract_target_section_text_orders_by_item_then_chunk_num():
    class _Chunk:
        def __init__(self, item, chunk_num, text):
            self.item, self.chunk_num, self.text = item, chunk_num, text

    chunks = [
        _Chunk("Item 7", 1, "second-mda-chunk"),
        _Chunk("Item 1A", 0, "first-risk-chunk"),
        _Chunk("PREAMBLE", 0, "should be excluded"),
        _Chunk("Item 7", 0, "first-mda-chunk"),
    ]
    text = tone_signal.extract_target_section_text(chunks)
    assert "should be excluded" not in text
    assert text.index("first-risk-chunk") < text.index("first-mda-chunk") < text.index("second-mda-chunk")


def test_compare_methods_reports_disagreement_honestly():
    previous = "Business conditions were stable and results were solid."
    current = "Business faced litigation, impairment, and significant losses amid uncertainty."
    delta = tone_signal.tone_delta(
        "0000000001", "acc-prev", "2023-03-01T00:00:00.000Z", previous,
        "acc-curr", "2024-03-01T00:00:00.000Z", current,
    )
    comparison = tone_signal.compare_methods(delta, {"direction": "IMPROVED", "rationale": "test"})
    assert comparison["lm_direction"] == "WORSENED"
    assert comparison["llm_direction"] == "IMPROVED"
    assert comparison["agree"] is False


def test_compare_methods_llm_unavailable_is_reported_not_hidden():
    previous = "Business conditions were stable."
    current = "Business faced litigation and losses."
    delta = tone_signal.tone_delta(
        "0000000001", "acc-prev", "2023-03-01T00:00:00.000Z", previous,
        "acc-curr", "2024-03-01T00:00:00.000Z", current,
    )
    comparison = tone_signal.compare_methods(delta, {"direction": None, "rationale": ""})
    assert comparison["llm_available"] is False
    assert comparison["agree"] is False
