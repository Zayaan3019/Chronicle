"""
Shared utilities: LLM calls (via Ollama REST API), embeddings, robust JSON parsing.
Uses direct HTTP calls to avoid Python library version conflicts.
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
    Call Ollama chat endpoint. Returns the assistant message content as a string.
    Retries once on connection error.
    Strips qwen3 <think> blocks from free-text (non-JSON) responses automatically.

    think=False (default): passes "think": false to Ollama for qwen3 models.
    This suppresses internal <think>...</think> token generation entirely, saving
    3–5× inference time.  We already strip these blocks, so disabling them is
    pure latency savings with no accuracy loss.  json_mode calls are unaffected
    (Ollama already suppresses thinking when format=json).
    """
    model = model or config.MAIN_LLM
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


def call_llm_stream(
    messages: list[dict],
    model: str = None,
    temperature: float = 0.2,
    num_ctx: int | None = None,
    think: bool = False,
):
    """
    Generator that yields text chunks from a streaming Ollama chat response.
    Useful for Streamlit progressive display.

    think=False (default): disables qwen3 internal <think> token generation at
    the model level (Ollama ≥ 0.6.4), so no tokens need to be filtered out.
    The in_think state-machine below is kept as a safety net for older Ollama.
    """
    model = model or config.MAIN_LLM
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


# ── Embeddings ────────────────────────────────────────────────────────────────

_EMBED_MAX_CHARS = 5000   # nomic-embed-text default context ≈ 2048 tokens; 5000 chars is safe


def embed_text(text: str) -> list[float]:
    """
    Return a float embedding vector for a single text string.
    Truncates to _EMBED_MAX_CHARS to stay within the model's context window
    (long URL fragments in scraped text can otherwise exceed token limits).
    """
    if len(text) > _EMBED_MAX_CHARS:
        text = text[:_EMBED_MAX_CHARS]
    resp = requests.post(
        f"{config.OLLAMA_BASE_URL}/api/embed",
        json={"model": config.EMBED_MODEL, "input": text},
        timeout=120,   # 120s: Ollama may queue embed behind an in-flight LLM call
    )
    resp.raise_for_status()
    data = resp.json()
    # Ollama returns {"embeddings": [[...]]} for /api/embed
    embeddings = data.get("embeddings") or data.get("embedding")
    if isinstance(embeddings[0], list):
        return embeddings[0]
    return embeddings


def embed_batch(texts: list[str]) -> list[list[float]]:
    """
    Embed multiple texts. Tries a single batch Ollama call first (fast path).
    Falls back to individual embed_text calls if the batch API returns an error
    or an unexpected response (handles older Ollama versions that reject list input).
    """
    if not texts:
        return []

    # Fast path: batch call
    try:
        resp = requests.post(
            f"{config.OLLAMA_BASE_URL}/api/embed",
            json={"model": config.EMBED_MODEL, "input": texts},
            timeout=120,
        )
        if resp.status_code == 200:
            data = resp.json()
            embeddings = data.get("embeddings") or data.get("embedding")
            if embeddings:
                if not isinstance(embeddings[0], list):
                    embeddings = [embeddings]
                if len(embeddings) == len(texts):
                    return embeddings
    except Exception:
        pass

    # Slow-path fallback: one call per text (reliable on all Ollama versions)
    return [embed_text(t) for t in texts]


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
    """Return (ok, message) checking Ollama is reachable and required models exist."""
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
