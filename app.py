"""Streamlit interface for Chronicle — a live, point-in-time-correct financial research RAG system."""

from __future__ import annotations

import time
import uuid

import streamlit as st

import config
from live_data import LiveDataAgent
from utils import check_ollama


st.set_page_config(page_title="Chronicle — Financial Research RAG", page_icon="📈", layout="wide")

if "messages" not in st.session_state:
    st.session_state.messages = []
if "pipeline" not in st.session_state:
    st.session_state.pipeline = None
if "memory_session_id" not in st.session_state:
    st.session_state.memory_session_id = uuid.uuid4().hex


def pipeline():
    if st.session_state.pipeline is None:
        from pipeline import ChroniclePipeline
        st.session_state.pipeline = ChroniclePipeline()
    return st.session_state.pipeline


@st.cache_resource(show_spinner=False)
def _ensure_index_populated() -> list[str]:
    """Self-healing cold start: on a fresh container, live_chroma_db/ is
    empty (it's local disk, not part of the Neon Postgres migration -- see
    DEPLOYMENT.md's "Known limitation" section for why), so the first
    visitor after any redeploy would otherwise get zero retrieval results
    with no indication why. @st.cache_resource runs this exactly once per
    running process, shared across every session that hits this container
    (not once per user) -- the second and subsequent visitors pay nothing.
    Returns human-readable report lines for the sidebar; empty list if the
    index was already populated (nothing to do).
    """
    from indexer import get_index_stats
    from pipeline import ChroniclePipeline

    if get_index_stats()["vector_chunks"] > 0:
        return []
    reports = ChroniclePipeline().run_live_data_agent("latest SEC filings and financial news", force=True)
    return [f"{r.source_id}: {r.status} — {r.message}" for r in reports]


def source_label(chunk) -> str:
    return chunk.citation_label if hasattr(chunk, "citation_label") else chunk.get("citation_label", "Source")


def render_sources(chunks) -> None:
    for chunk in chunks:
        if isinstance(chunk, dict):
            label = chunk["citation_label"]
            title = chunk["title"]
            publisher = chunk["publisher"]
            date = chunk["date"]
            url = chunk["url"]
            snippet = chunk["text"]
        else:
            label = chunk.citation_label
            title = chunk.article_title
            publisher = chunk.article_source
            date = chunk.date
            url = chunk.url
            snippet = chunk.text
        st.markdown(f"**[{label}]** — [{title}]({url})")
        st.caption(f"{publisher} · {date}")
        st.write(snippet[:450] + ("…" if len(snippet) > 450 else ""))


def serialise_chunks(chunks) -> list[dict]:
    return [
        {
            "citation_label": chunk.citation_label,
            "title": chunk.article_title,
            "publisher": chunk.article_source,
            "date": chunk.date,
            "url": chunk.url,
            "text": chunk.text,
        }
        for chunk in chunks
    ]


with st.spinner("First run on this container: populating the retrieval index from live sources…"):
    _cold_start_reports = _ensure_index_populated()

with st.sidebar:
    st.header("Live source operations")
    if _cold_start_reports:
        with st.expander("Cold-start index population (ran once for this container)", expanded=False):
            for line in _cold_start_reports:
                st.caption(line)
    configured_sources = LiveDataAgent.load_sources(config.LIVE_SOURCES_PATH)
    if configured_sources:
        for source in configured_sources:
            st.caption(f"• {source.title} ({source.refresh_minutes} min)")
    else:
        st.warning("No live sources configured. Add trusted sources to live_sources.json.")

    ok, status = check_ollama()
    st.caption(f"LLM ({config.LLM_PROVIDER}): {'ready' if ok else status}")
    if st.button("Refresh all sources", use_container_width=True):
        reports = pipeline().run_live_data_agent("latest SEC filings and financial news", force=True)
        for report in reports:
            st.write(f"{report.source_id}: {report.status} — {report.message}")
    if st.button("Forget durable memory", use_container_width=True):
        pipeline().forget_session(st.session_state.memory_session_id)
        st.session_state.messages = []
        st.rerun()
    if st.button("Clear visible chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.divider()
    st.header("Point-in-time query")
    st.caption(
        "Retrieval hard-filters every candidate to filed_at <= as_of before ranking "
        "(not a post-hoc check). Leave unset to query as of now."
    )
    use_as_of = st.checkbox("Query as of a specific date", value=False)
    as_of_date = st.date_input("as_of date (UTC)", disabled=not use_as_of) if use_as_of else None


st.title("Chronicle")
st.caption(
    "Live, point-in-time-correct financial research over SEC EDGAR filings and GDELT news. "
    "Trusted sources → planner → hybrid retrieval (as_of filtered) → synthesis → evidence critic "
    "→ deny-by-default guardrail. Signal extraction and evidence synthesis, not investment advice."
)

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        details = message.get("details")
        if details:
            with st.expander("Evidence and agent trace", expanded=False):
                st.caption(f"Route: {details['route']} · Memory turns: {details['memory_turns']}")
                st.write(details["planner"])
                if details["refresh"]:
                    st.caption("Refresh: " + "; ".join(details["refresh"]))
                render_sources(details["chunks"])
                if details["critic"]:
                    st.caption("Critic: " + details["critic"])
                if details.get("guardrail"):
                    st.caption("Guardrail: " + details["guardrail"])


question = st.chat_input("Ask a financial research question about a tracked company's filings…")
if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        progress = st.status("Running multi-agent research workflow…", expanded=True)
        events: list[str] = []

        def on_step(name: str, message: str) -> None:
            events.append(f"{name}: {message}")
            progress.write(events[-1])

        history = [
            {"role": item["role"], "content": item["content"]}
            for item in st.session_state.messages[:-1]
        ]
        started = time.time()
        as_of_param = as_of_date.isoformat() if use_as_of and as_of_date else None
        try:
            result = pipeline().query(
                question, history=history, session_id=st.session_state.memory_session_id,
                on_step=on_step, as_of=as_of_param,
            )
            progress.update(label="Research complete", state="complete", expanded=False)
            st.markdown(result.final_answer)
            st.caption(f"Completed in {time.time() - started:.1f}s" + (f" · as_of={as_of_param}" if as_of_param else ""))
            details = {
                "route": result.route,
                "memory_turns": result.memory_turns_used,
                "planner": result.query_analysis.reasoning,
                "refresh": [f"{item.source_id}: {item.status}" for item in result.live_refresh],
                "chunks": serialise_chunks(result.retrieved_chunks),
                "critic": result.critique.verdict if result.critique else "No evidence package to verify",
                "guardrail": result.guardrail_verdict.verdict if result.guardrail_verdict else "",
            }
            st.session_state.messages.append({"role": "assistant", "content": result.final_answer, "details": details})
        except Exception as exc:
            progress.update(label="Research workflow failed", state="error")
            st.error(f"Unable to complete the research workflow: {exc}")
