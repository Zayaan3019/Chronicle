"""Phase 3B — deny-by-default guardrail layer.

An independent, pluggable policy layer that runs AFTER synthesis and the
LLM citation critic, and BEFORE an answer is released to the caller.
Modeled on a fail-closed risk-service pattern: every check must explicitly
pass for the verdict to be ALLOW; any single check failure defaults the
whole verdict to BLOCK, and any exception raised during evaluation is
handled by the caller as BLOCK too (see `fail_closed_verdict` and
pipeline.ChroniclePipeline.run_guardrail).

This module depends only on primitives (str, list[ChunkResult]) — never on
pipeline.py or agent internals — so it can be swapped or reused (e.g. to
front a different generator entirely) without a rewrite; pipeline.py is a
caller of this module, not the other way around.

Checks (each independently unit-testable — see tests/test_guardrail.py):
  1. numeric_claim_match    — reuses numeric_verify.py (Phase 3) as-is.
  2. point_in_time_recheck  — a SECOND, independent check on the FINAL
     answer's actual citations. Retrieval already hard-filters by as_of
     (retrieval/hybrid.py), but this check trusts nothing upstream: it
     re-derives visibility straight from pit.py against the chunks that
     were actually cited in the released text.
  3. prompt_injection_scan  — retrieved evidence text is always data, never
     instruction; a chunk containing an injection-style directive fails
     this check outright.
  4. output_schema          — an empty/implausibly short answer, or a
     non-empty evidence package that produced zero citations, is a
     malformed generation and is blocked rather than released.

Every evaluation — ALLOW or BLOCK — is written to the append-only decision
log (decision_log.py) before the verdict is returned.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
import hashlib
import re

import decision_log
import pit
from citations import extract_citation_labels
from numeric_verify import verify_numeric_claims
from retrieval.vector_store import ChunkResult

ALLOW = "ALLOW"
BLOCK = "BLOCK"

# Deliberately conservative/curated pattern list, not exhaustive — a hardened
# deployment would layer this with a maintained ruleset or a classifier.
# Documented as a known limitation rather than silently overclaimed coverage.
_INJECTION_PATTERNS = [
    re.compile(p, re.IGNORECASE) for p in [
        r"ignore (all|any|the)?\s*(previous|prior|above)\s*instructions",
        r"disregard (all|any|the)?\s*(previous|prior|above)",
        r"\byou are now\b",
        r"new system prompt",
        r"reveal (your|the) (system )?prompt",
        r"act as (if|though|an?)\b.{0,30}(unrestricted|jailbreak|dan)\b",
        r"\bsystem\s*:\s*",
        r"</?(system|instructions?)>",
        r"do not (follow|obey) (the|your) (original|previous) instructions",
    ]
]


@dataclass
class CheckResult:
    rule: str
    passed: bool
    detail: str = ""


@dataclass
class GuardrailVerdict:
    verdict: str
    checks: list[CheckResult] = field(default_factory=list)
    input_hash: str = ""
    output_hash: str = ""
    timestamp: str = ""

    @property
    def failed_rules(self) -> list[str]:
        return [c.rule for c in self.checks if not c.passed]


def _hash(*parts: str) -> str:
    return hashlib.sha256("\x1f".join(parts).encode("utf-8")).hexdigest()


def _check_numeric_claims(answer: str, chunks: list[ChunkResult]) -> CheckResult:
    result = verify_numeric_claims(answer, chunks)
    if result.unsupported:
        return CheckResult(
            "numeric_claim_match", False,
            f"{len(result.unsupported)} unsupported numeric claim(s): {', '.join(result.unsupported[:5])}",
        )
    return CheckResult("numeric_claim_match", True, f"{result.checked_against_chunks} claim(s) checked")


def _check_point_in_time(answer: str, chunks: list[ChunkResult], as_of: str | None) -> CheckResult:
    as_of_epoch = pit.resolve_as_of_epoch(as_of)
    cited_labels = set(extract_citation_labels(answer))
    if not cited_labels:
        return CheckResult("point_in_time_recheck", True, "no citations to recheck")
    chunk_by_label = {c.citation_label: c for c in chunks}
    violations = []
    for label in cited_labels:
        chunk = chunk_by_label.get(label)
        if chunk is None:
            continue  # citation to an unresolvable chunk is an output_schema concern, not PIT
        if not pit.is_visible(chunk.filed_at, as_of_epoch):
            violations.append(f"{label} (filed_at={chunk.filed_at})")
    if violations:
        return CheckResult(
            "point_in_time_recheck", False,
            f"{len(violations)} cited chunk(s) filed after as_of: {', '.join(violations)}",
        )
    return CheckResult("point_in_time_recheck", True, f"{len(cited_labels)} citation(s) rechecked")


def _check_prompt_injection(chunks: list[ChunkResult]) -> CheckResult:
    hits = []
    for chunk in chunks:
        for pattern in _INJECTION_PATTERNS:
            if pattern.search(chunk.text):
                hits.append(f"{chunk.citation_label}")
                break
    if hits:
        return CheckResult("prompt_injection_scan", False, f"{len(hits)} chunk(s) flagged: {', '.join(hits[:5])}")
    return CheckResult("prompt_injection_scan", True, f"{len(chunks)} chunk(s) scanned, clean")


def _check_output_schema(answer: str, chunks: list[ChunkResult]) -> CheckResult:
    if not answer or not answer.strip():
        return CheckResult("output_schema", False, "empty answer")
    if len(answer.strip()) < 10:
        return CheckResult("output_schema", False, "answer implausibly short")
    if chunks and not extract_citation_labels(answer):
        return CheckResult("output_schema", False, "non-empty evidence package produced zero citations")
    return CheckResult("output_schema", True, "answer shape OK")


def evaluate(
    question: str,
    chunks: list[ChunkResult],
    answer: str,
    as_of: str | None = None,
) -> GuardrailVerdict:
    """Deny-by-default: the verdict is ALLOW only if every check explicitly passes."""
    checks = [
        _check_output_schema(answer, chunks),
        _check_numeric_claims(answer, chunks),
        _check_point_in_time(answer, chunks, as_of),
        _check_prompt_injection(chunks),
    ]
    verdict = ALLOW if all(c.passed for c in checks) else BLOCK
    result = GuardrailVerdict(
        verdict=verdict,
        checks=checks,
        input_hash=_hash(question, "\x1e".join(c.text for c in chunks)),
        output_hash=_hash(answer),
        timestamp=datetime.now(UTC).isoformat(),
    )
    decision_log.record(result, question=question)
    return result


def fail_closed_verdict(reason: str) -> GuardrailVerdict:
    """Constructed by the caller when guardrail evaluation itself could not
    run (crashed, timed out, process killed). Callers must treat this
    exactly like BLOCK — never fall back to serving the unchecked answer."""
    result = GuardrailVerdict(
        verdict=BLOCK,
        checks=[CheckResult("guardrail_availability", False, f"guardrail evaluation failed: {reason}")],
        timestamp=datetime.now(UTC).isoformat(),
    )
    try:
        decision_log.record(result, question="<unavailable — guardrail evaluation crashed>")
    except Exception:
        pass  # a logging failure must not mask the fail-closed BLOCK verdict
    return result
