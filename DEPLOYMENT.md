# Deploying Chronicle

Implements Phase 3 of the approved zero-cost deployment plan. Summary of
what changed to make this possible: the LLM/embedding stack no longer
requires a local Ollama server (swapped to Groq's free hosted API + a local
`sentence-transformers` embedding model — see `utils.py` and `config.py`),
and durable state (conversation memory, the guardrail audit log, live-source
tracking) now runs on Neon Postgres instead of local SQLite files, since the
hosting target's disk isn't guaranteed to survive every redeploy.

## Primary target: Streamlit Community Cloud

Chosen over an AWS-hosted option because `app.py` is a genuine Streamlit
app — it needs a real WebSocket connection for its reactivity, which AWS
Lambda Function URLs don't support, and the only AWS-native alternative
(another EC2 box) would add a second time-boxed 6-month free-tier cliff for
the least AWS-critical of this plan's three projects. Streamlit Community
Cloud is purpose-built for this, genuinely free forever (not time-boxed),
and auto-redeploys from GitHub on every push — a real form of CD with zero
extra CI wiring needed for the deploy step itself.

**Steps:**
1. Push this repo to GitHub (public is simplest; private repos work too via
   Streamlit's GitHub app permission grant, still free).
2. Go to https://share.streamlit.io, sign in with GitHub, "New app".
3. Repository: this repo. Branch: `main`. Main file path: `app.py`.
4. Before first deploy, open **Advanced settings > Secrets** and paste:
   ```toml
   LLM_PROVIDER = "groq"
   GROQ_API_KEY = "gsk_..."
   NEON_DATABASE_URL = "postgresql://...neon.tech/neondb?sslmode=require"
   EDGAR_USER_AGENT = "Chronicle-Research/1.0 (your-real-contact@example.org)"
   ```
   Streamlit injects everything in this TOML block as real OS environment
   variables for the app process, so `config.py`'s existing `os.getenv(...)`
   calls pick these up with no code changes — same mechanism as the local
   `.env` file, different delivery method. **Use your own Groq key and Neon
   DSN, not ones that have been pasted into any chat transcript** — rotate
   any credential that was ever shared outside this deployment path.
5. Deploy. First build installs `requirements.txt` (a few minutes — includes
   downloading the `sentence-transformers` embedding model and the
   cross-encoder reranker on first run, cached afterward).
6. Nothing to run by hand after deploy: the first visit to a fresh
   container populates the retrieval index from the live sources (see
   below).

## Cold start: the vector index is rebuilt per container

`live_chroma_db/` (ChromaDB's on-disk index) is local disk. Only the
*relational* state (memory, guardrail log, live-source tracking) lives in
Neon Postgres, because Chroma's persistence layer doesn't speak Postgres
(moving it would need `pgvector` and a retrieval-layer rewrite). So every
fresh Streamlit Cloud container starts with an empty vector index.

`app.py`'s `_ensure_index_populated()` handles that. On startup it checks
`indexer.get_index_stats()` and, if `vector_chunks == 0`, runs one forced
refresh of the configured live sources (SEC EDGAR filings and GDELT news).
It is wrapped in `@st.cache_resource`, so it runs once per container, not
once per visitor. The first visitor after a redeploy waits a minute or two
behind a spinner, and the sidebar lists what each source returned. Every
chunk in the index comes from those live sources; nothing is bundled or
generated. "Refresh all sources" in the sidebar forces another refresh at
any time.

## Verifying a successful deploy

1. Open the live URL. Sidebar should show `LLM (groq): ready` (not an
   Ollama-not-reachable error — confirms `check_ollama()`'s Groq branch, see
   `utils.py`, is wired correctly).
2. On the first visit, wait for the cold-start spinner to finish, then
   check the sidebar's "Cold-start index population" lines for each source.
3. Ask a real question, e.g. "What did Apple report about iPhone revenue in
   its most recent 10-K?" — should return a cited, grounded answer.
4. Restart the app (Streamlit Cloud's "Reboot app") and confirm a prior
   conversation's memory persists via Neon (proves the Postgres migration
   actually took effect, not just that it compiles) — ask a follow-up
   referencing the earlier question in a **new** session with the same
   session ID, or check `guardrail_decisions` row count in Neon before/after
   the reboot to confirm the audit log survived.

## Fallback: AWS EC2, if you want a pure-AWS story instead

`.dockerignore` in this repo is ready for it, but no Dockerfile/Terraform
project exists yet for Chronicle specifically (unlike Neuro-Econ and
AI-Evals) — would follow the same `ec2-docker-host` module pattern already
used for AI-Evals (see `infra/projects/ai-evals/`), with the same month-5
Free-Plan decommission checkpoint. Not built by default because Streamlit
Community Cloud is strictly better for this specific app (WebSocket support,
free forever vs. time-boxed, zero infra to maintain) — only worth the extra
EC2 complexity if "everything is AWS" is a harder requirement than "this
works and stays free."
