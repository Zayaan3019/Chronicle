"""Single definition of the inter-agent citation format: `[Source id/id]`.

agents/reasoning_agent.py emits this format; numeric_verify.py (Phase 3) and
guardrail.py (Phase 3B) both need to resolve a citation label back to the
chunk it points to. The guardrail's point-in-time recheck is only a
meaningful second, INDEPENDENT check if it parses the exact same format
reasoning_agent actually emits — so the regex lives in exactly one place
rather than being redefined per module.
"""

from __future__ import annotations

import re

CITATION_RE = re.compile(r"\[Source\s+([A-Za-z0-9_-]+/[A-Za-z0-9_-]+)\]", re.IGNORECASE)


def extract_citation_labels(text: str) -> list[str]:
    """Return `Source id/id` labels — matching ChunkResult.citation_label — found in text."""
    return [f"Source {m}" for m in CITATION_RE.findall(text)]
