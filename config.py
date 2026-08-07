import os

# ── Ollama ─────────────────────────────────────────────────────────────
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

# Primary reasoning model — qwen3:8b for Agent 2 (Reasoner).
# qwen3:8b is a newer generation Qwen model with built-in thinking mode; its CoT output
# significantly improves factual accuracy and citation quality for complex synthesis tasks.
MAIN_LLM  = os.getenv("MAIN_LLM",  "qwen3:8b")
# Fast model for structured JSON tasks (Agent 1 decomposer, Agent 3 critic).
# qwen2.5:3b is fast, reliable for JSON output, and sufficient for routing/verification.
# Ollama's format=json automatically disables qwen3 thinking mode for JSON calls too.
FAST_LLM  = os.getenv("FAST_LLM",  "qwen2.5:3b")
EMBED_MODEL = os.getenv("EMBED_MODEL", "nomic-embed-text")

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
CHROMA_PERSIST_DIR = os.path.join(BASE_DIR, "live_chroma_db")
BM25_INDEX_PATH    = os.path.join(BASE_DIR, "data", "live_bm25_index.pkl")

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
