"""Chronicle — live, point-in-time-correct financial research multi-agent RAG.

Workflow: Memory Agent -> Live Data Agent (EDGAR/GDELT) -> Planning Agent ->
Hybrid Retrieval (point-in-time filtered) -> Synthesis Agent -> Evidence
Critic -> durable memory write. No fixed articles, article IDs,
source-specific answer rules, or static corpus are used anywhere in this
pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Generator

import config
import guardrail
from agents import critic_agent, query_agent, reasoning_agent
from agents.critic_agent import CriticResult
from agents.query_agent import QueryAnalysis
from agents.reasoning_agent import ReasoningResult
from guardrail import GuardrailVerdict
from indexer import load_index
from live_data import LiveDataAgent, RefreshReport, is_freshness_query
from memory import MemoryAgent
from retrieval.hybrid import HybridRetriever, RetrievalStats
from retrieval.vector_store import ChunkResult


@dataclass
class PipelineResult:
    query: str
    route: str
    query_analysis: QueryAnalysis
    retrieved_chunks: list[ChunkResult]
    retrieval_stats: RetrievalStats | None
    reasoning: ReasoningResult
    critique: CriticResult | None
    final_answer: str
    memory_turns_used: int = 0
    live_refresh: list[RefreshReport] = field(default_factory=list)
    as_of: str | None = None
    guardrail_verdict: GuardrailVerdict | None = None

    @property
    def cited_articles(self) -> list[str]:
        return reasoning_agent.extract_citations(self.final_answer + "\n" + self.reasoning.chain_of_thought)


class ChroniclePipeline:
    def __init__(self) -> None:
        vector_store, bm25_store = load_index()
        self.retriever = HybridRetriever(vector_store, bm25_store)
        self.memory_agent = MemoryAgent()
        self.live_data_agent = LiveDataAgent(vector_store, bm25_store)

    def run_memory_agent(self, session_id: str | None, query: str, history: list[dict] | None = None) -> list[dict]:
        return self.memory_agent.context_for_query(session_id, query, history)

    def run_live_data_agent(self, query: str, force: bool = False) -> list[RefreshReport]:
        return self.live_data_agent.refresh_for_query(query, force=force)

    def forget_session(self, session_id: str) -> None:
        self.memory_agent.store.forget_session(session_id)

    def run_agent1(self, query: str, history: list[dict] | None = None) -> QueryAnalysis:
        return query_agent.run(query, history)

    def run_retrieval(
        self, analysis: QueryAnalysis, query: str, as_of: str | None = None,
    ) -> tuple[list[ChunkResult], RetrievalStats]:
        retrieval_query = analysis.resolved_query or query
        if analysis.complexity == "complex":
            queries = list(dict.fromkeys([retrieval_query, *analysis.sub_queries]))[:5]
            chunks, stats = self.retriever.search_multi_query(
                queries, n_per_query=8, final_top_k=config.MULTI_HOP_TOP_K, as_of=as_of,
            )
        else:
            chunks, stats = self.retriever.search(
                retrieval_query, n_results=config.TOP_K_RERANK, simple_query=True, as_of=as_of,
            )

        # Explicit freshness requests reserve evidence from dynamically refreshed
        # sources, preventing broad historical matches from obscuring it.
        if is_freshness_query(query):
            live_chunks, _ = self.retriever.search(
                retrieval_query,
                n_results=2,
                cluster_filter="Financial live sources",
                simple_query=True,
                as_of=as_of,
            )
            seen = {chunk.chunk_id for chunk in live_chunks}
            max_chunks = config.MULTI_HOP_TOP_K if analysis.complexity == "complex" else config.TOP_K_RERANK
            chunks = (live_chunks + [chunk for chunk in chunks if chunk.chunk_id not in seen])[:max_chunks]
            for rank, chunk in enumerate(chunks, 1):
                chunk.rank = rank
        return chunks, stats

    def run_agent2(
        self, query: str, chunks: list[ChunkResult], analysis: QueryAnalysis,
        route: str = "simple", model: str | None = None, history: list[dict] | None = None,
    ) -> ReasoningResult:
        selected_model = model or (config.FAST_LLM if route == "simple" and config.SIMPLE_USE_FAST_LLM else config.MAIN_LLM)
        return reasoning_agent.run(query, chunks, analysis.reasoning, history or [], selected_model)

    def run_agent2_stream(
        self, query: str, chunks: list[ChunkResult], analysis: QueryAnalysis,
        route: str = "simple", model: str | None = None, history: list[dict] | None = None,
    ) -> Generator[str, None, None]:
        selected_model = model or (config.FAST_LLM if route == "simple" and config.SIMPLE_USE_FAST_LLM else config.MAIN_LLM)
        yield from reasoning_agent.run_stream(query, chunks, analysis.reasoning, history or [], selected_model)

    @staticmethod
    def should_run_critic(route: str, chunks: list[ChunkResult]) -> bool:
        # The critic is always useful for a live, non-empty evidence package.
        return bool(chunks)

    def run_critic(self, query: str, chunks: list[ChunkResult], answer: str) -> CriticResult | None:
        return critic_agent.run(query, chunks, answer) if chunks else None

    @staticmethod
    def should_apply_correction(critique: CriticResult | None, chunks: list[ChunkResult]) -> bool:
        return bool(
            critique and critique.verdict == "NEEDS_CORRECTION" and critique.corrected_answer
            and len(critique.corrected_answer.strip()) >= 30 and critique.confidence >= 0.5
        )

    @staticmethod
    def _ensure_source_provenance(answer: str, chunks: list[ChunkResult]) -> str:
        labels = list(dict.fromkeys(chunk.citation_label for chunk in chunks))
        if not labels or any(f"[{label}]" in answer for label in labels):
            return answer
        return answer.rstrip() + "\n\nSources used: " + ", ".join(f"[{label}]" for label in labels[:3])

    def post_process_answer(self, answer: str, chunks: list[ChunkResult], critique: CriticResult | None = None) -> str:
        return self._ensure_source_provenance(answer, chunks)

    def run_guardrail(
        self, query: str, chunks: list[ChunkResult], answer: str, as_of: str | None = None,
    ) -> GuardrailVerdict:
        """Fail-closed call site: if guardrail evaluation itself raises for any
        reason (bug, crash, killed dependency), that is treated exactly like a
        BLOCK verdict — this method never lets an exception fall through and
        silently release the unchecked answer to the caller."""
        try:
            return guardrail.evaluate(query, chunks, answer, as_of=as_of)
        except Exception as exc:
            return guardrail.fail_closed_verdict(str(exc))

    @staticmethod
    def _blocked_message(verdict: GuardrailVerdict) -> str:
        reasons = "; ".join(f"{c.rule}: {c.detail}" for c in verdict.checks if not c.passed)
        return (
            "This answer was BLOCKED by the guardrail policy layer and cannot be released as-is. "
            f"Reason(s): {reasons or 'guardrail evaluation unavailable'}."
        )

    def query(
        self, user_query: str, history: list[dict] | None = None,
        on_step: Callable[[str, str], None] | None = None,
        session_id: str | None = None, refresh_live: bool = False,
        as_of: str | None = None,
    ) -> PipelineResult:
        """`as_of`: optional ISO-8601 timestamp. Retrieval hard-filters every
        candidate to filed_at <= as_of before ranking (retrieval/hybrid.py);
        default None means "now" (see pit.resolve_as_of_epoch)."""
        history = history or []
        effective_history = self.run_memory_agent(session_id, user_query, history)
        if on_step:
            on_step("memory", f"Loaded {len(effective_history)} session-memory turns")
            on_step("live", "Refreshing due trusted SEC EDGAR/GDELT sources")
        refresh = self.run_live_data_agent(user_query, force=refresh_live)
        if on_step:
            on_step("planner", "Planning retrieval")
        analysis = self.run_agent1(user_query, effective_history)
        if on_step:
            on_step("retrieval", "Running hybrid retrieval")
        chunks, stats = self.run_retrieval(analysis, user_query, as_of=as_of)
        if on_step:
            on_step("synthesis", "Synthesizing source-grounded answer")
        reasoning = self.run_agent2(user_query, chunks, analysis, analysis.complexity, history=effective_history)
        critique = None
        if self.should_run_critic(analysis.complexity, chunks):
            if on_step:
                on_step("critic", "Verifying evidence and citations")
            critique = self.run_critic(user_query, chunks, reasoning.final_answer)
        final = critique.corrected_answer if self.should_apply_correction(critique, chunks) else reasoning.final_answer
        final = self.post_process_answer(final, chunks, critique)
        if on_step:
            on_step("guardrail", "Running deny-by-default policy guardrail")
        guardrail_verdict = self.run_guardrail(user_query, chunks, final, as_of=as_of)
        if guardrail_verdict.verdict != guardrail.ALLOW:
            final = self._blocked_message(guardrail_verdict)
        self.memory_agent.remember_interaction(session_id, user_query, final)
        if on_step:
            on_step("done", "Complete")
        return PipelineResult(
            query=user_query, route=analysis.complexity, query_analysis=analysis,
            retrieved_chunks=chunks, retrieval_stats=stats, reasoning=reasoning,
            critique=critique, final_answer=final, memory_turns_used=len(effective_history),
            live_refresh=refresh, as_of=as_of, guardrail_verdict=guardrail_verdict,
        )
