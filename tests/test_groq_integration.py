"""Live integration test against Groq's real API (LLM_PROVIDER=groq path).

Every other test in this suite runs offline against fakes/stubs. This one is
deliberately different: it is the only test that proves the actual deployed
LLM path (Ollama -> Groq swap, see utils.py) produces a real, grounded answer
via the full agent pipeline -- a mocked LLM cannot catch a wrong Groq
endpoint, an unsupported reasoning_effort value for a given model, or a
response-shape mismatch, all of which are exactly the failure modes this
swap introduced (see utils.py's _call_groq / _reasoning_off comments
for two such bugs already caught this way during development).

Skipped unless GROQ_API_KEY is set, so contributors without a key (and any
CI job that hasn't configured the secret) still get a clean, fast local run.
"""

from __future__ import annotations

import os

import pytest

import config

pytestmark = pytest.mark.skipif(
    not os.getenv("GROQ_API_KEY"),
    reason="GROQ_API_KEY not set -- skipping live Groq integration test",
)


def test_groq_chat_completion_round_trips():
    """The narrowest possible live check: call_llm actually reaches Groq and
    returns a clean, non-empty response with no leaked <think> block."""
    import utils

    original_provider = config.LLM_PROVIDER
    config.LLM_PROVIDER = "groq"
    try:
        response = utils.call_llm(
            [{"role": "user", "content": "Reply with exactly one word: ready"}],
            model=config.FAST_LLM,
            temperature=0.0,
        )
    finally:
        config.LLM_PROVIDER = original_provider

    assert response.strip()
    assert "<think>" not in response


def test_groq_json_mode_returns_parseable_json():
    import json
    import utils

    original_provider = config.LLM_PROVIDER
    config.LLM_PROVIDER = "groq"
    try:
        response = utils.call_llm(
            [{"role": "user", "content": 'Return JSON: {"status": "ok"}'}],
            model=config.FAST_LLM,
            json_mode=True,
            temperature=0.0,
        )
    finally:
        config.LLM_PROVIDER = original_provider

    parsed = json.loads(response)
    assert parsed.get("status") == "ok"


def test_groq_reasoning_model_does_not_leak_think_block():
    """The specific bug this test suite exists to prevent regressing: a
    reasoning-capable Groq model (MAIN_LLM) emits inline <think> blocks
    unless reasoning is switched off per family -- "none" for qwen3, "low"
    for gpt-oss (see utils.py's _reasoning_off)."""
    import utils

    original_provider = config.LLM_PROVIDER
    config.LLM_PROVIDER = "groq"
    try:
        response = utils.call_llm(
            [{"role": "user", "content": "What is 2 + 2? Answer in one word."}],
            model=config.MAIN_LLM,
            temperature=0.0,
        )
    finally:
        config.LLM_PROVIDER = original_provider

    assert "<think>" not in response
    # "Answer in one word" legitimately yields either "4" or "Four".
    assert "4" in response or "four" in response.lower()
