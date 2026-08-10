"""Agent 1 — financial research planner.

The planner knows no fixed corpus, article IDs, or answer keys. It decides whether
a question benefits from decomposition and produces short retrieval queries for
whatever trusted SEC EDGAR/GDELT sources are currently configured and refreshed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import re

import config
from utils import call_llm, extract_json


_COMPLEX_TERMS = {
    "compare", "contrast", "difference", "versus", "mechanism", "cause", "why",
    "synthesize", "tradeoff", "implication", "framework", "evidence", "guideline",
    "risk", "benefit", "relationship", "across", "and", "or",
}
_SIMPLE_PATTERN = re.compile(
    r"^(what|which|when|who|where|how many|is there|are there|list)\b", re.IGNORECASE
)


@dataclass
class QueryAnalysis:
    raw_query: str
    resolved_query: str
    complexity: str  # simple | complex
    sub_queries: list[str] = field(default_factory=list)
    target_domains: list[str] = field(default_factory=list)
    needs_multihop: bool = False
    reasoning: str = ""


def classify_complexity_heuristic(query: str) -> str:
    words = set(re.findall(r"[a-z]+", query.lower()))
    if len(words & _COMPLEX_TERMS) >= 2 or any(term in query.lower() for term in ("how does", "what are the risks")):
        return "complex"
    return "simple" if _SIMPLE_PATTERN.match(query.strip()) else "complex"


_SYSTEM = """You are the planning agent in Chronicle, a live, point-in-time-correct
financial research RAG system.

There is no fixed corpus and no hidden answer key. Available evidence comes only
from trusted, dynamically refreshed sources: SEC EDGAR filings (10-K, 10-K/A,
10-Q, 10-Q/A, 8-K) and GDELT news. Do not invent source names, documents,
statistics, or financial facts.

Classify the request as simple (one focused lookup) or complex (comparison,
multi-part synthesis, causal reasoning, or several evidence needs). Resolve
references from the supplied conversation history and create short, source-neutral
retrieval queries. This system provides research information — signal extraction
and evidence synthesis, not investment advice, price forecasts, or trading
recommendations.

Return ONLY this JSON object:
{
  "complexity": "simple" or "complex",
  "resolved_query": "...",
  "sub_queries": ["..."],
  "target_domains": ["risk factors", "financial statements", "MD&A", "earnings/news", "governance", "other"],
  "needs_multihop": true or false,
  "reasoning": "brief retrieval strategy"
}"""


def run(query: str, history: list[dict] | None = None) -> QueryAnalysis:
    """Create a bounded plan; deterministic fallback keeps retrieval available."""
    history = history or []
    heuristic = classify_complexity_heuristic(query)
    if heuristic == "simple" and not history:
        return QueryAnalysis(
            raw_query=query, resolved_query=query, complexity="simple", sub_queries=[query],
            reasoning="Focused first-turn lookup; retrieve from currently refreshed sources.",
        )

    turns = []
    for turn in history[-config.MAX_HISTORY_TURNS:]:
        turns.append(f"{turn.get('role', '')}: {str(turn.get('content', ''))[:350]}")
    # Backslash-in-f-string-expression (PEP 701) needs Python 3.12+; this
    # deployment's runtime is not guaranteed to be that new (e.g. Streamlit
    # Community Cloud), so the join happens outside the f-string.
    history_block = "\n".join(turns) or "None"
    raw = call_llm(
        messages=[
            {"role": "system", "content": _SYSTEM},
            {"role": "user", "content": f"History:\n{history_block}\n\nQuestion: {query}"},
        ],
        model=config.FAST_LLM,
        temperature=config.DECOMPOSER_TEMP,
        json_mode=True,
    )
    data = extract_json(raw)
    complexity = "complex" if heuristic == "complex" else data.get("complexity", "simple")
    if complexity not in {"simple", "complex"}:
        complexity = heuristic
    sub_queries = data.get("sub_queries")
    if not isinstance(sub_queries, list):
        sub_queries = []
    sub_queries = [str(item).strip() for item in sub_queries if str(item).strip()][:5] or [query]
    domains = data.get("target_domains")
    if not isinstance(domains, list):
        domains = []
    return QueryAnalysis(
        raw_query=query,
        resolved_query=str(data.get("resolved_query") or query),
        complexity=complexity,
        sub_queries=sub_queries,
        target_domains=[str(item)[:80] for item in domains][:5],
        needs_multihop=bool(data.get("needs_multihop", complexity == "complex")),
        reasoning=str(data.get("reasoning") or "Retrieve and synthesize only refreshed source evidence.")[:500],
    )
