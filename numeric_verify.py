"""Deterministic numeric-claim verification (Phase 3).

Independent of the LLM-based citation critic in `agents/critic_agent.py` —
this module never calls a model. It extracts every number-shaped claim from
a generated answer, resolves which cited source chunk(s) back the sentence
it appears in, and checks whether an equivalent number (after unit/scale
normalization) actually appears in that chunk's text. A number with no
match is reported as unsupported; the caller decides how to act on that
(critic_agent.py forces the verdict to NEEDS_CORRECTION — see Phase 3).

Also reused as-is by the Phase 3B guardrail layer's numeric-claim check, so
this module intentionally has no dependency on the critic or the LLM.

Normalization handles the concrete case the spec calls out: "$1.2B" and
"1,200,000,000" are the same claim in two formats and must match. A number
that only matches after ignoring more than TOLERANCE is NOT considered a
match — that is what lets the verifier catch a genuinely corrupted source
number instead of being fooled by "close enough" values from an unrelated
figure.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import re

from citations import extract_citation_labels
from retrieval.vector_store import ChunkResult

# Relative tolerance for float-rounding noise only (e.g. a value re-typed
# with one fewer digit of precision) — NOT intended to paper over a real
# discrepancy between two different figures.
TOLERANCE = 0.01

_SCALE = {
    "trillion": 1e12, "tn": 1e12, "t": 1e12,
    "billion": 1e9, "bn": 1e9, "b": 1e9,
    "million": 1e6, "mm": 1e6, "m": 1e6,
    "thousand": 1e3, "k": 1e3,
}

_NUMBER_CORE = r"\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+\.\d+|\d+"

_CLAIM_RE = re.compile(
    rf"""
    (?P<currency>[$€£])?\s?
    (?P<number>{_NUMBER_CORE})
    \s?
    (?P<scale>trillion|billion|million|thousand|tn|bn|mm|[kKmMbBtT])?(?![A-Za-z])
    \s?
    (?P<percent>%|percent)?(?![A-Za-z])
    """,
    re.VERBOSE,
)


@dataclass
class NumericClaim:
    raw: str
    value: float
    is_percent: bool
    span: tuple[int, int]


def extract_numbers(text: str) -> list[NumericClaim]:
    """Extract financially-decorated numeric claims from free text.

    Deliberately excludes bare, undecorated integers/decimals (no $, %,
    scale word, or thousands-comma grouping) — those are far more likely to
    be a section/item number, a calendar year, or an unrelated count than a
    financial figure, and including them would flood the verifier with
    claims that were never meant to be checked against a source number.
    """
    claims: list[NumericClaim] = []
    for m in _CLAIM_RE.finditer(text):
        number = m.group("number")
        if not number:
            continue
        currency = m.group("currency")
        scale = m.group("scale")
        percent = m.group("percent")
        has_commas = "," in number
        if not (currency or scale or percent or has_commas):
            continue
        value = _parse_value(number, scale)
        if value is None:
            continue
        claims.append(
            NumericClaim(
                raw=m.group(0).strip(),
                value=value,
                is_percent=bool(percent),
                span=m.span(),
            )
        )
    return claims


def _parse_value(number_text: str, scale: str | None) -> float | None:
    try:
        value = float(number_text.replace(",", ""))
    except ValueError:
        return None
    if scale:
        factor = _SCALE.get(scale.lower())
        if factor:
            value *= factor
    return value


def _values_match(a: NumericClaim, b: NumericClaim) -> bool:
    if a.is_percent != b.is_percent:
        return False
    if a.value == 0 and b.value == 0:
        return True
    denom = max(abs(a.value), abs(b.value), 1e-9)
    return abs(a.value - b.value) / denom <= TOLERANCE


_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z\[])")


@dataclass
class NumericVerificationResult:
    total_claims: int = 0
    unsupported: list[str] = field(default_factory=list)   # raw claim text with no matching source number
    uncited: list[str] = field(default_factory=list)       # numeric claims in a sentence with no citation at all
    checked_against_chunks: int = 0


def verify_numeric_claims(answer: str, chunks: list[ChunkResult]) -> NumericVerificationResult:
    """Check every numeric claim in `answer` against its cited source chunk(s).

    A sentence's citation labels are resolved via the same `[Source id/id]`
    format reasoning_agent.py emits and vector_store.ChunkResult.citation_label
    produces, so this stays in sync with the actual citation contract rather
    than a separately-invented one.
    """
    result = NumericVerificationResult()
    if not answer.strip() or not chunks:
        return result

    chunk_by_label = {c.citation_label: c for c in chunks}
    chunk_numbers_cache: dict[str, list[NumericClaim]] = {}

    def numbers_for(label: str) -> list[NumericClaim]:
        if label not in chunk_numbers_cache:
            chunk = chunk_by_label.get(label)
            chunk_numbers_cache[label] = extract_numbers(chunk.text) if chunk else []
        return chunk_numbers_cache[label]

    for sentence in _SENTENCE_SPLIT_RE.split(answer):
        sentence = sentence.strip()
        if not sentence:
            continue
        claims = extract_numbers(sentence)
        if not claims:
            continue
        cited_labels = extract_citation_labels(sentence)
        result.total_claims += len(claims)
        if not cited_labels:
            result.uncited.extend(c.raw for c in claims)
            continue
        for claim in claims:
            result.checked_against_chunks += 1
            supported = any(
                any(_values_match(claim, source_num) for source_num in numbers_for(label))
                for label in cited_labels
            )
            if not supported:
                result.unsupported.append(claim.raw)

    return result
