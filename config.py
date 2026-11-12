import os

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv is optional locally; deployment targets (Streamlit
          # Community Cloud, etc.) inject real env vars directly.

try:
    # Streamlit Community Cloud's documented behavior is that root-level
    # secrets.toml keys "will be available as an environment variable" --
    # but the exact timing of that relative to this module's own import-time
    # os.getenv() calls below is undocumented, and empirically it did not
    # take effect on a real deploy (LLM_PROVIDER stayed at its "ollama"
    # default despite being set to "groq" in the dashboard). Bridging
    # st.secrets into os.environ explicitly here removes the dependency on
    # that undocumented timing. setdefault so a real OS env var (local .env,
    # AWS Lambda/EC2 env vars) always wins over a Streamlit secret, never the
    # reverse. Broad except: st.secrets raises when no secrets.toml exists
    # and there's no active Streamlit runtime context at all (pytest, other
    # deployment targets where `streamlit` may not even be installed).
    import streamlit as st
    for _key, _value in st.secrets.items():
        os.environ.setdefault(_key, str(_value))
except Exception:
    pass

# ── LLM provider ──────────────────────────────────────────────────────
# "ollama" (default): the original local-only path, unchanged.
# "groq": hosted inference via Groq's free tier -- see utils.py's call_llm.
# Ollama-only tuning knobs (NUM_CTX, NUM_THREADS, num_ctx per-call overrides)
# have no Groq equivalent and are silently ignored on that path rather than
# left to imply a control that no longer does anything.
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama").lower()

# ── Ollama ─────────────────────────────────────────────────────────────
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

# ── Groq (used when LLM_PROVIDER=groq) ──────────────────────────────────
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_BASE_URL = os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1")

if LLM_PROVIDER == "groq":
    # Verified live against Groq's /v1/models on 2026-11-12 (the 2026-08-10
    # defaults, qwen3.6-27b / llama-3.1-8b-instant / llama-3.3-70b-versatile,
    # had since been retired and returned 404). Groq's free-tier
    # roster rotates -- if a model here 404s, check console.groq.com/docs/models
    # and update the default (or override via env var without a code change).
    _DEFAULT_MAIN_LLM = "qwen/qwen3.8-27b"    # reasoning agent; same family as the qwen3:8b this replaces
    _DEFAULT_FAST_LLM = "openai/gpt-oss-20b"  # decomposer; fastest model on the roster
    # Deliberately a DIFFERENT model from FAST_LLM, not the same one -- Groq's
    # rate limits are per-model, independent token/request pools (verified
    # empirically 2026-08-10 by comparing live x-ratelimit-* response headers
    # across models, which directly contradicted Groq's own rate-limits doc
    # page summary claiming org-wide shared limits). query_agent (decomposer),
    # critic_agent, and simple-query reasoning can all call FAST_LLM within
    # seconds of one another for a single question -- a real deploy hit
    # FAST_LLM's 6,000 TPM pool exactly this way ("Rate limit reached...
    # Used 3141, Requested 3141"). Moving the critic to its own pool
    # (llama-3.3-70b-versatile: 12,000 TPM, double FAST_LLM's) cuts peak
    # same-pool pressure from 3 concurrent calls to 2 in the worst case, with
    # no new provider or dependency. Verified this model produces clean,
    # think-block-free JSON matching critic_agent's exact call shape
    # (json_mode=True) before wiring it in here.
    _DEFAULT_CRITIC_LLM = "openai/gpt-oss-120b"
else:
    # Primary reasoning model — qwen3:8b for Agent 2 (Reasoner).
    # qwen3:8b is a newer generation Qwen model with built-in thinking mode; its CoT output
    # significantly improves factual accuracy and citation quality for complex synthesis tasks.
    _DEFAULT_MAIN_LLM = "qwen3:8b"
    # Fast model for structured JSON tasks (Agent 1 decomposer, Agent 3 critic).
    # qwen2.5:3b is fast, reliable for JSON output, and sufficient for routing/verification.
    # Ollama's format=json automatically disables qwen3 thinking mode for JSON calls too.
    _DEFAULT_FAST_LLM = "qwen2.5:3b"
    # Local Ollama has no per-model rate-limit pools to split load across --
    # the CRITIC_LLM/FAST_LLM split exists specifically for Groq's per-model
    # accounting, so the critic just uses the same local model here.
    _DEFAULT_CRITIC_LLM = "qwen2.5:3b"

MAIN_LLM = os.getenv("MAIN_LLM", _DEFAULT_MAIN_LLM)
FAST_LLM = os.getenv("FAST_LLM", _DEFAULT_FAST_LLM)
CRITIC_LLM = os.getenv("CRITIC_LLM", _DEFAULT_CRITIC_LLM)

# ── Embeddings ────────────────────────────────────────────────────────
# Always local via sentence-transformers, regardless of LLM_PROVIDER -- avoids
# a second external API dependency, and sentence-transformers is already a
# pinned requirement (it backs the cross-encoder reranker in retrieval/hybrid.py).
# Switching this value changes the vector space: live_chroma_db/ must be fully
# rebuilt (see README's reindex runbook), never partially/incrementally.
EMBED_MODEL = os.getenv("EMBED_MODEL", "BAAI/bge-small-en-v1.5")

# ── Chunking ───────────────────────────────────────────────────────────
CHUNK_SIZE    = 420   # approximate words per chunk
CHUNK_OVERLAP = 80    # word overlap between consecutive chunks

# ── Retrieval ──────────────────────────────────────────────────────────
TOP_K_DENSE      = 8    # candidates from dense (ChromaDB)
TOP_K_BM25       = 8    # candidates from sparse (BM25)
TOP_K_RERANK     = 5    # final chunks after cross-encoder reranking (simple queries)
TOP_K_HYBRID     = 16   # CRITICAL: max candidates entering cross-encoder before final cut
MULTI_HOP_TOP_K  = 7    # final chunks kept for complex/multi-hop queries
RRF_K            = 60   # RRF smoothing constant (standard)
RERANK_MODEL     = "cross-encoder/ms-marco-MiniLM-L-6-v2"
USE_RERANKER     = True  # set False to skip cross-encoder (faster, lower quality)
EMBED_BATCH_SIZE = 20    # chunks per embed_batch call

# ── Storage ────────────────────────────────────────────────────────────
BASE_DIR           = os.path.dirname(os.path.abspath(__file__))
# A new namespace ensures no legacy/static vectors or sparse artifacts can ever
# be read by the live-only application.
# Overridable so a container can keep both indexes on a persistent volume
# (the EC2 deployment mounts one at /state); defaults are unchanged locally.
CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", os.path.join(BASE_DIR, "live_chroma_db"))
BM25_INDEX_PATH    = os.getenv("BM25_INDEX_PATH", os.path.join(BASE_DIR, "data", "live_bm25_index.pkl"))

# ── SEC EDGAR ──────────────────────────────────────────────────────────
# SEC's fair-access policy (https://www.sec.gov/os/webmaster-faq#developers)
# requests a declarative User-Agent identifying the requester and caps abusive
# clients around 10 req/s. We self-impose a materially lower ceiling and a real
# contact identity — required reading before pointing this at data.sec.gov.
# SEC's bot-detection rejects long/instructional User-Agent strings (verified
# empirically: a verbose placeholder here returned HTTP 403 while a concise
# "AppName/Version (contact-email)" string returned 200 for the same URL) —
# keep this short. Operators MUST override via env var with their own contact
# identity before running this against SEC EDGAR in their own deployment.
EDGAR_USER_AGENT = os.getenv("EDGAR_USER_AGENT", "Chronicle-Research/1.0 (research@chronicle-project.org)")
EDGAR_MIN_REQUEST_INTERVAL_SECONDS = float(os.getenv("EDGAR_MIN_REQUEST_INTERVAL_SECONDS", "0.34"))  # ~3 req/s, well under SEC's stated ceiling
EDGAR_SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik:010d}.json"
EDGAR_FULLTEXT_SEARCH_URL = "https://efts.sec.gov/LATEST/search-index"
EDGAR_ARCHIVES_BASE = "https://www.sec.gov/Archives/edgar/data"
EDGAR_TICKER_MAP_URL = "https://www.sec.gov/files/company_tickers.json"
# Forms this project ingests. 10-K/10-Q/their amendments are the primary
# point-in-time corpus; 8-K exhibits are the only transcript-adjacent source
# (see README — no free live earnings-call transcript source exists).
EDGAR_TRACKED_FORMS = ("10-K", "10-K/A", "10-Q", "10-Q/A", "8-K")
EDGAR_ITEM_CHUNK_WORDS = 380     # sub-chunk size *within* an Item/Part boundary
EDGAR_ITEM_CHUNK_OVERLAP = 60

# ── GDELT ──────────────────────────────────────────────────────────────
# GDELT 2.0 DOC API — free, no key. Returns article metadata only (url, title,
# seendate, domain), never body text, so ingestion still does a secondary
# fetch+extract of the article URL itself.
GDELT_DOC_API_URL = "https://api.gdeltproject.org/api/v2/doc/doc"
GDELT_DEFAULT_TIMESPAN = os.getenv("GDELT_DEFAULT_TIMESPAN", "3d")
GDELT_MAX_RECORDS = int(os.getenv("GDELT_MAX_RECORDS", "20"))

# ── Company watchlist ──────────────────────────────────────────────────
COMPANY_WATCHLIST_PATH = os.getenv("COMPANY_WATCHLIST_PATH", os.path.join(BASE_DIR, "live_sources.json"))

# ── HTTP ───────────────────────────────────────────────────────────────
REQUEST_TIMEOUT = 30
REQUEST_DELAY   = 1.2   # seconds between article fetches
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/123.0 Safari/537.36"
)

# ── Agent temperature ──────────────────────────────────────────────────
DECOMPOSER_TEMP = 0.1   # Agent 1 – needs deterministic JSON
REASONER_TEMP   = 0.2   # Agent 2 – allow some creativity in synthesis
CRITIC_TEMP     = 0.05  # Agent 3 – purely analytical

# ── Speed optimisations ────────────────────────────────────────────────
# Candidate cap for cross-encoder on simple (single-hop) queries.
# Halves CE pairs vs the complex path cap (8 vs 16) → saves ~1.5s.
TOP_K_HYBRID_SIMPLE = 8

# When True, simple (single-hop) queries use FAST_LLM for Agent 2 reasoning.
# Safe because simple queries are single-article factual lookups — FAST_LLM
# handles citation formatting correctly at ~5-8s lower latency than MAIN_LLM.
SIMPLE_USE_FAST_LLM = True

# ── Conversation ───────────────────────────────────────────────────────

# Ollama context window (tokens). 6144 gives >= 800 tok output headroom while
# fitting 5-7 chunks (~4500 tok context + system prompt).
NUM_CTX = 6144

# Number of CPU threads for Ollama inference (0 = Ollama default).
# Intel Core Ultra 7 155H: 16 cores - 12 is optimal (avoids hyperthreading overhead).
NUM_THREADS = 12
MAX_HISTORY_TURNS = 3   # 3 turns covers 95 % of conversations; saves ~200 ms/query

# ── Postgres (Neon) ──────────────────────────────────────────────────────
# Opt-in durable-state backend for memory.py / decision_log.py / live_data.py
# -- used automatically when set AND the caller doesn't force a specific
# sqlite path (see each module's docstring). Empty string means "use SQLite",
# matching every existing test's explicit sqlite path construction.
NEON_DATABASE_URL = os.getenv("NEON_DATABASE_URL", "")

# ── Live / dynamic data ────────────────────────────────────────────────────
# Dynamic documents are incrementally upserted and retain their own provenance.
LIVE_SOURCES_PATH = os.getenv("LIVE_SOURCES_PATH", os.path.join(BASE_DIR, "live_sources.json"))
LIVE_STATE_DB_PATH = os.getenv("LIVE_STATE_DB_PATH", os.path.join(BASE_DIR, "data", "live_state.sqlite3"))
LIVE_REFRESH_ON_QUERY = os.getenv("LIVE_REFRESH_ON_QUERY", "true").lower() == "true"
LIVE_MAX_RESPONSE_BYTES = int(os.getenv("LIVE_MAX_RESPONSE_BYTES", str(5 * 1024 * 1024)))
LIVE_MIN_REFRESH_MINUTES = int(os.getenv("LIVE_MIN_REFRESH_MINUTES", "5"))
# Use a transparent identifier for live sources. Some public-sector endpoints
# reject browser-spoofing user agents; set a monitored contact address in prod.
LIVE_USER_AGENT = os.getenv("LIVE_USER_AGENT", "Chronicle/1.0 (financial research; contact@example.org)")
# Keep these false in deployment. They are available only for local integration
# tests or a deliberately isolated internal source.
ALLOW_INSECURE_LIVE_SOURCES = os.getenv("ALLOW_INSECURE_LIVE_SOURCES", "false").lower() == "true"
ALLOW_PRIVATE_LIVE_SOURCES = os.getenv("ALLOW_PRIVATE_LIVE_SOURCES", "false").lower() == "true"

# ── Durable conversational memory ──────────────────────────────────────────
# Memory is session-scoped; a real deployment must supply an authenticated,
# opaque tenant/session identifier and place this database on encrypted storage.
MEMORY_ENABLED = os.getenv("MEMORY_ENABLED", "true").lower() == "true"
MEMORY_DB_PATH = os.getenv("MEMORY_DB_PATH", os.path.join(BASE_DIR, "data", "memory.sqlite3"))
MEMORY_RETENTION_DAYS = int(os.getenv("MEMORY_RETENTION_DAYS", "30"))
MEMORY_MAX_TURNS = int(os.getenv("MEMORY_MAX_TURNS", "8"))
MEMORY_MAX_CONTENT_CHARS = int(os.getenv("MEMORY_MAX_CONTENT_CHARS", "6000"))

# ── Guardrail (Phase 3B) ──────────────────────────────────────────────────
# Append-only decision log for every guardrail evaluation (guardrail.py).
GUARDRAIL_DECISION_LOG_PATH = os.getenv(
    "GUARDRAIL_DECISION_LOG_PATH", os.path.join(BASE_DIR, "data", "guardrail_decisions.sqlite3")
)
