"""
Shared utilities: LLM calls (via Ollama or Groq, config.LLM_PROVIDER-selected),
local embeddings (sentence-transformers), robust JSON parsing.
LLM calls use direct HTTP calls to avoid Python library version conflicts.
"""

import json
import re
import time
import requests
from typing import Any

import config


# ── LLM calls ─────────────────────────────────────────────────────────────────

def _strip_think_blocks(text: str) -> str:
    """
    Remove Qwen3 <think>...</think> blocks from non-JSON responses.
    qwen3:8b emits an optional reasoning preamble in <think> tags before its
    actual answer.  We discard those blocks so downstream parsers see clean output.
    When json_mode=True, Ollama automatically suppresses thinking, so this only
    applies to Agent 2 (reasoning_agent) free-text responses.
    """
    # Remove full <think>...</think> blocks (possibly multiline)
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    return cleaned.strip()


_GROQ_MAX_ATTEMPTS = 3  # Groq's free tier is shared per-org, not per-key -- a
                        # 429 here is routine under real multi-agent traffic
                        # (query_agent + reasoning_agent + critic_agent can
                        # each call the same FAST_LLM within seconds of each
                        # other), not exceptional; worth retrying, not failing fast.


def _groq_retry_delay(resp: "requests.Response", attempt: int) -> float:
    """
    Seconds to wait before retrying a Groq 429. Verified empirically (2026-08-10)
    that Groq sets a real Retry-After header on 429 responses -- prefer that
    over a guessed backoff, since it reflects the server's own accounting of
    when the token/request budget actually resets. +0.5s buffer for clock
    skew; capped at 30s so a malformed or unexpectedly large header value
    can't hang the app on what should be a short wait.
    """
    retry_after = resp.headers.get("Retry-After")
    if retry_after:
        try:
            return min(float(retry_after) + 0.5, 30.0)
        except ValueError:
            pass
    return min(2.0 * (attempt + 1), 10.0)


def _is_reasoning_model(model: str) -> bool:
    """
    Whether `model` accepts reasoning_effort="none" specifically (not just
    "accepts the reasoning_effort parameter at all" -- verified empirically
    2026-08-10 that gpt-oss models DO accept the parameter but reject "none",
    only "low"/"medium"/"high", a different contract from qwen3's. Only
    qwen3-family models are confirmed to accept "none"; extend this by
    testing each new model added, not by assuming a shared contract across
    Groq's reasoning models -- see console.groq.com/docs/models.
    """
    return "qwen3" in model.lower()


def _ollama_options(temperature: float, num_ctx: int | None = None) -> dict:
    """
    Build the Ollama options dict.
    num_thread: Ollama defaults to half the CPU threads; setting it explicitly
    to the number of physical cores improves throughput on modern CPUs
    (Intel Core Ultra 7 155H has 16 cores / 22 threads — 16 is a good default).
    num_ctx: context window in tokens. Larger = better quality but slower.
    """
    ctx = num_ctx or getattr(config, "NUM_CTX", 6144)
    opts: dict[str, Any] = {
        "temperature": temperature,
        "num_ctx": ctx,
    }
    num_thread = getattr(config, "NUM_THREADS", 0)
    if num_thread > 0:
        opts["num_thread"] = num_thread
    return opts


def call_llm(
    messages: list[dict],
    model: str = None,
    temperature: float = 0.1,
    json_mode: bool = False,
    num_ctx: int | None = None,
    think: bool = False,
) -> str:
    """
    Call the configured LLM provider's chat endpoint (config.LLM_PROVIDER).
    Returns the assistant message content as a string. Strips <think> blocks
    from free-text (non-JSON) responses automatically -- both Ollama's local
    qwen3 and Groq's hosted qwen3.6 emit these the same way.
    """
    model = model or config.MAIN_LLM
    if config.LLM_PROVIDER == "groq":
        return _call_groq(messages, model, temperature, json_mode, think)
    return _call_ollama(messages, model, temperature, json_mode, num_ctx, think)


def _call_ollama(
    messages: list[dict],
    model: str,
    temperature: float,
    json_mode: bool,
    num_ctx: int | None,
    think: bool,
) -> str:
    """
    Call Ollama chat endpoint. Retries once on connection error.

    think=False (default): passes "think": false to Ollama for qwen3 models.
    This suppresses internal <think>...</think> token generation entirely, saving
    3–5× inference time.  We already strip these blocks, so disabling them is
    pure latency savings with no accuracy loss.  json_mode calls are unaffected
    (Ollama already suppresses thinking when format=json).
    """
    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": _ollama_options(temperature, num_ctx),
    }
    if json_mode:
        payload["format"] = "json"
    # Suppress qwen3 internal reasoning tokens when think=False.
    # Supported by Ollama ≥ 0.6.4; silently ignored on older versions.
    if not think and "qwen3" in model.lower():
        payload["think"] = False

    for attempt in range(2):
        try:
            resp = requests.post(
                f"{config.OLLAMA_BASE_URL}/api/chat",
                json=payload,
                timeout=300,
            )
            resp.raise_for_status()
            content = resp.json()["message"]["content"]
            # Strip any residual qwen3 thinking blocks (safety net for older Ollama).
            if not json_mode:
                content = _strip_think_blocks(content)
            return content
        except requests.exceptions.ConnectionError:
            if attempt == 0:
                time.sleep(2)
                continue
            raise RuntimeError(
                "Cannot connect to Ollama. "
                "Make sure Ollama is running: `ollama serve`"
            )
        except Exception as e:
            raise RuntimeError(f"LLM call failed: {e}") from e


def _call_groq(
    messages: list[dict],
    model: str,
    temperature: float,
    json_mode: bool,
    think: bool,
) -> str:
    """
    Call Groq's OpenAI-compatible chat completions endpoint.

    Verified live (2026-08-10) against qwen/qwen3.6-27b: it emits inline
    <think>...</think> blocks exactly like local Ollama qwen3 UNLESS
    reasoning_effort="none" is passed, which suppresses them at the API level
    (Groq's equivalent of Ollama's think=false) -- confirmed empirically, this
    also works cleanly together with response_format=json_object to return
    directly-parseable JSON with no thinking preamble at all.
    """
    if not config.GROQ_API_KEY:
        raise RuntimeError(
            "LLM_PROVIDER=groq but GROQ_API_KEY is not set. "
            "Get a free key at https://console.groq.com/keys"
        )
    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
    }
    if json_mode:
        payload["response_format"] = {"type": "json_object"}
    # reasoning_effort is only accepted by reasoning-capable models -- verified
    # empirically that Groq 400s ("reasoning_effort is not supported with this
    # model") when sent to a plain model like llama-3.1-8b-instant (FAST_LLM).
    # Same guard shape as the Ollama path's `"think": False` gate below it.
    if not think and _is_reasoning_model(model):
        payload["reasoning_effort"] = "none"
    # num_ctx / num_thread (Ollama server-tuning knobs) have no Groq equivalent
    # and are intentionally dropped rather than mapped to something misleading.

    for attempt in range(_GROQ_MAX_ATTEMPTS):
        try:
            resp = requests.post(
                f"{config.GROQ_BASE_URL}/chat/completions",
                headers={"Authorization": f"Bearer {config.GROQ_API_KEY}"},
                json=payload,
                timeout=300,
            )
        except requests.exceptions.ConnectionError:
            if attempt < _GROQ_MAX_ATTEMPTS - 1:
                time.sleep(2)
                continue
            raise RuntimeError("Cannot reach Groq's API (network error).")

        # Checked before raise_for_status() specifically so a 429 gets a
        # retry-with-backoff instead of falling into the generic HTTPError
        # handler below, which fails immediately -- verified this was the
        # actual failure mode live: a real deploy hit "429 ... Rate limit
        # reached ... tokens per minute" and surfaced it straight to the user
        # with no retry at all.
        if resp.status_code == 429:
            if attempt < _GROQ_MAX_ATTEMPTS - 1:
                time.sleep(_groq_retry_delay(resp, attempt))
                continue
            raise RuntimeError(
                f"Groq rate limit exceeded after {_GROQ_MAX_ATTEMPTS} attempts: {resp.text[:500]}"
            )

        try:
            resp.raise_for_status()
        except requests.exceptions.HTTPError as e:
            raise RuntimeError(f"Groq API error: {e} — {resp.text[:500]}") from e

        content = resp.json()["choices"][0]["message"]["content"]
        # Safety net: reasoning_effort="none" suppresses thinking on models
        # that accept it, but strip defensively regardless (e.g. a plain
        # model that emits think-like text unprompted, or a future Groq
        # roster change) rather than assume it always will.
        return _strip_think_blocks(content)


def call_llm_stream(
    messages: list[dict],
    model: str = None,
    temperature: float = 0.2,
    num_ctx: int | None = None,
    think: bool = False,
):
    """Generator yielding text chunks from a streaming chat response, from
    whichever provider config.LLM_PROVIDER selects. Useful for Streamlit
    progressive display."""
    model = model or config.MAIN_LLM
    if config.LLM_PROVIDER == "groq":
        yield from _call_groq_stream(messages, model, temperature, think)
    else:
        yield from _call_ollama_stream(messages, model, temperature, num_ctx, think)


def _call_ollama_stream(messages, model, temperature, num_ctx, think):
    """
    think=False (default): disables qwen3 internal <think> token generation at
    the model level (Ollama ≥ 0.6.4), so no tokens need to be filtered out.
    The in_think state-machine below is kept as a safety net for older Ollama.
    """
    payload = {
        "model": model,
        "messages": messages,
        "stream": True,
        "options": _ollama_options(temperature, num_ctx),
    }
    if not think and "qwen3" in model.lower():
        payload["think"] = False
    in_think = False
    with requests.post(
        f"{config.OLLAMA_BASE_URL}/api/chat",
        json=payload,
        stream=True,
        timeout=600,
    ) as resp:
        resp.raise_for_status()
        for line in resp.iter_lines():
            if line:
                data = json.loads(line)
                if not data.get("done"):
                    chunk = data["message"]["content"]
                    # Safety-net: suppress any residual <think> blocks
                    if "<think>" in chunk:
                        in_think = True
                    if in_think:
                        if "</think>" in chunk:
                            in_think = False
                            # Yield any content that comes AFTER </think> in this chunk
                            after = chunk.split("</think>", 1)[1]
                            if after:
                                yield after
                        continue
                    yield chunk


def _call_groq_stream(messages, model, temperature, think):
    """Groq streams OpenAI-style SSE (`data: {...}` lines, `data: [DONE]`
    sentinel) rather than Ollama's raw NDJSON -- the parsing differs even
    though the <think>-stripping state machine below is identical."""
    if not config.GROQ_API_KEY:
        raise RuntimeError(
            "LLM_PROVIDER=groq but GROQ_API_KEY is not set. "
            "Get a free key at https://console.groq.com/keys"
        )
    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "stream": True,
    }
    if not think and _is_reasoning_model(model):
        payload["reasoning_effort"] = "none"

    # Retry-with-backoff on 429 happens BEFORE entering the streaming/yield
    # phase -- safe to retry the whole request here since no chunks have been
    # yielded yet, unlike a mid-stream failure which can't be transparently
    # retried without the caller re-requesting. Same rate-limit reality as
    # _call_groq: verified live that a real deploy's multi-agent traffic
    # (query_agent + reasoning_agent + critic_agent sharing FAST_LLM) can
    # exceed Groq's free-tier tokens-per-minute budget within seconds.
    resp = None
    for attempt in range(_GROQ_MAX_ATTEMPTS):
        try:
            resp = requests.post(
                f"{config.GROQ_BASE_URL}/chat/completions",
                headers={"Authorization": f"Bearer {config.GROQ_API_KEY}"},
                json=payload,
                stream=True,
                timeout=600,
            )
        except requests.exceptions.ConnectionError:
            if attempt < _GROQ_MAX_ATTEMPTS - 1:
                time.sleep(2)
                continue
            raise RuntimeError("Cannot reach Groq's API (network error).")

        if resp.status_code == 429:
            if attempt < _GROQ_MAX_ATTEMPTS - 1:
                delay = _groq_retry_delay(resp, attempt)
                resp.close()
                time.sleep(delay)
                continue
            raise RuntimeError(
                f"Groq rate limit exceeded after {_GROQ_MAX_ATTEMPTS} attempts: {resp.text[:500]}"
            )

        try:
            resp.raise_for_status()
        except requests.exceptions.HTTPError as e:
            raise RuntimeError(f"Groq API error: {e} — {resp.text[:500]}") from e
        break

    in_think = False
    with resp:
        for line in resp.iter_lines():
            if not line:
                continue
            line = line.decode("utf-8") if isinstance(line, bytes) else line
            if not line.startswith("data: "):
                continue
            data_str = line[len("data: "):]
            if data_str.strip() == "[DONE]":
                break
            data = json.loads(data_str)
            delta = data["choices"][0]["delta"]
            chunk = delta.get("content")
            if not chunk:
                continue
            if "<think>" in chunk:
                in_think = True
            if in_think:
                if "</think>" in chunk:
                    in_think = False
                    after = chunk.split("</think>", 1)[1]
                    if after:
                        yield after
                continue
            yield chunk


# ── Embeddings ────────────────────────────────────────────────────────────────
# Local via sentence-transformers regardless of LLM_PROVIDER (see config.py) --
# avoids a second external API dependency and keeps embedding cost at $0
# regardless of traffic. Same public signatures as the old Ollama-backed
# versions (str -> list[float], list[str] -> list[list[float]]), so every
# caller (retrieval/hybrid.py, live_data.py, indexer.py) is unchanged.

_EMBED_MAX_CHARS = 5000  # matches the old Ollama-path truncation; bge-small's
                          # 512-token context is comparably sized in practice.

_embed_model = None  # lazy-loaded singleton -- imported at call time, not
                      # module import time, so importing utils.py doesn't
                      # force a model download/load for callers that never embed.


def _get_embed_model():
    global _embed_model
    if _embed_model is None:
        from sentence_transformers import SentenceTransformer
        _embed_model = SentenceTransformer(config.EMBED_MODEL)
    return _embed_model


def embed_text(text: str) -> list[float]:
    """Return a float embedding vector for a single text string."""
    if len(text) > _EMBED_MAX_CHARS:
        text = text[:_EMBED_MAX_CHARS]
    vec = _get_embed_model().encode(text, convert_to_numpy=True, normalize_embeddings=True)
    return vec.tolist()


def embed_batch(texts: list[str]) -> list[list[float]]:
    """Embed multiple texts in one batched forward pass (CPU-friendly at this model size)."""
    if not texts:
        return []
    truncated = [t[:_EMBED_MAX_CHARS] if len(t) > _EMBED_MAX_CHARS else t for t in texts]
    vecs = _get_embed_model().encode(truncated, convert_to_numpy=True, normalize_embeddings=True)
    return vecs.tolist()


# ── JSON parsing ──────────────────────────────────────────────────────────────

def extract_json(text: str) -> dict:
    """
    Robustly extract a JSON object from model output that may contain prose,
    markdown fences, or extra whitespace.
    Returns empty dict if nothing parseable found.
    """
    # 1. Direct parse
    try:
        return json.loads(text.strip())
    except json.JSONDecodeError:
        pass

    # 2. JSON inside ```json ... ``` fences
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass

    # 3. Largest {...} block in the text
    brace_start = text.find("{")
    brace_end   = text.rfind("}")
    if brace_start != -1 and brace_end > brace_start:
        candidate = text[brace_start : brace_end + 1]
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass

    return {}


# ── Connectivity check ────────────────────────────────────────────────────────

def check_ollama() -> tuple[bool, str]:
    """
    Return (ok, message) checking the configured LLM provider is reachable.
    Named check_ollama for backward compatibility with existing callers
    (app.py); dispatches to Groq's check when LLM_PROVIDER=groq.
    """
    if config.LLM_PROVIDER == "groq":
        return _check_groq()
    try:
        resp = requests.get(f"{config.OLLAMA_BASE_URL}/api/tags", timeout=5)
        resp.raise_for_status()
        models = {m["name"].split(":")[0] for m in resp.json().get("models", [])}
        # also allow full-name match
        full_names = {m["name"] for m in resp.json().get("models", [])}
        all_names = models | full_names

        needed = {config.MAIN_LLM, config.EMBED_MODEL}
        missing = []
        for n in needed:
            base = n.split(":")[0]
            if n not in all_names and base not in models:
                missing.append(n)

        if missing:
            return False, f"Missing Ollama models: {missing}. Run: ollama pull " + " && ollama pull ".join(missing)
        return True, "Ollama OK"
    except Exception as e:
        return False, f"Ollama not reachable: {e}. Start with: ollama serve"


def _check_groq() -> tuple[bool, str]:
    if not config.GROQ_API_KEY:
        return False, "GROQ_API_KEY not set. Get a free key at https://console.groq.com/keys"
    try:
        resp = requests.get(
            f"{config.GROQ_BASE_URL}/models",
            headers={"Authorization": f"Bearer {config.GROQ_API_KEY}"},
            timeout=10,
        )
        resp.raise_for_status()
        available = {m["id"] for m in resp.json().get("data", [])}
        missing = [m for m in (config.MAIN_LLM, config.FAST_LLM, config.CRITIC_LLM) if m not in available]
        if missing:
            return False, f"Configured model(s) not on Groq's current roster: {missing}. See console.groq.com/docs/models"
        return True, "Groq OK"
    except Exception as e:
        return False, f"Groq not reachable: {e}"


# ── Text utilities ────────────────────────────────────────────────────────────

def chunk_text(text: str, chunk_size: int = None, overlap: int = None) -> list[str]:
    """
    Split text into overlapping word-level chunks.
    Paragraph-aware: tries to break at double newlines first.
    """
    chunk_size = chunk_size or config.CHUNK_SIZE
    overlap    = overlap    or config.CHUNK_OVERLAP

    # Split into paragraphs first
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]

    words: list[str] = []
    for para in paragraphs:
        words.extend(para.split())
        words.append("\n\n")   # paragraph boundary marker

    chunks: list[str] = []
    i = 0
    while i < len(words):
        window = words[i : i + chunk_size]
        chunk  = " ".join(w for w in window if w != "\n\n").strip()
        if len(chunk) > 80:   # skip tiny fragments
            chunks.append(chunk)
        i += chunk_size - overlap

    return chunks
