"""Integration tests for agents/critic_agent.py's numeric-claim extension (Phase 3).

The LLM call is monkeypatched (no Ollama needed) so these tests isolate the
behavior Phase 3 actually added: the deterministic numeric check must be able
to override an LLM verdict of APPROVED when a cited number doesn't match its
source, and must never be silently skipped when the LLM output is malformed.
"""

from __future__ import annotations

import json

import agents.critic_agent as critic_agent
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


def _fake_llm_approves(**kwargs) -> str:
    return json.dumps({
        "verdict": "APPROVED",
        "unsupported_claims": [],
        "wrong_citations": [],
        "missing_info": [],
        "corrected_answer": None,
        "confidence": 0.95,
        "verification_notes": "Answer is well grounded.",
    })


def test_corrupted_source_number_overrides_llm_approval(monkeypatch):
    """The required Phase 3 test: corrupt a source chunk's number, confirm the
    numeric layer catches the mismatch EVEN when the LLM critic approves."""
    monkeypatch.setattr(critic_agent, "call_llm", _fake_llm_approves)
    answer = "Full-year revenue guidance was $1.2B [Source nvda/abc123]."
    corrupted_chunk = _chunk("abc123", "The company guided to full-year revenue of $1,500,000,000.")

    result = critic_agent.run("What was revenue guidance?", [corrupted_chunk], answer)

    assert result.verdict == "NEEDS_CORRECTION"
    assert result.unsupported_numeric_claims == ["$1.2B"]
    assert result.numeric_claims_checked == 1
    assert result.confidence <= 0.3
    assert "Numeric verification" in result.verification_notes


def test_correct_source_number_keeps_llm_approval(monkeypatch):
    monkeypatch.setattr(critic_agent, "call_llm", _fake_llm_approves)
    answer = "Full-year revenue guidance was $1.2B [Source nvda/abc123]."
    correct_chunk = _chunk("abc123", "The company guided to full-year revenue of $1,200,000,000.")

    result = critic_agent.run("What was revenue guidance?", [correct_chunk], answer)

    assert result.verdict == "APPROVED"
    assert result.unsupported_numeric_claims == []
    assert result.numeric_claims_checked == 1


def test_numeric_check_still_runs_when_llm_output_is_invalid(monkeypatch):
    monkeypatch.setattr(critic_agent, "call_llm", lambda **kwargs: "not valid json at all")
    answer = "Full-year revenue guidance was $1.2B [Source nvda/abc123]."
    corrupted_chunk = _chunk("abc123", "The company guided to full-year revenue of $1,500,000,000.")

    result = critic_agent.run("What was revenue guidance?", [corrupted_chunk], answer)

    assert result.verdict == "NEEDS_CORRECTION"
    assert result.unsupported_numeric_claims == ["$1.2B"]
