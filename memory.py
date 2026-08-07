"""Durable, session-scoped conversational memory for the RAG supervisor.

This module intentionally stores conversation turns rather than extracting hidden
facts about the user with an LLM.  That makes memory auditable, prevents accidental
construction of a persistent profile beyond the conversation itself, and lets
callers delete an entire session exactly. SQLite/WAL is appropriate for a
single-node Streamlit or API deployment; move the same interface behind a
managed encrypted database for a multi-node deployment.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from contextlib import closing
import re
import sqlite3
from pathlib import Path
from typing import Iterable

import config


_SESSION_ID_RE = re.compile(r"^[A-Za-z0-9_.-]{1,128}$")
_WORD_RE = re.compile(r"[A-Za-z0-9]{2,}")


@dataclass(frozen=True)
class MemoryTurn:
    role: str
    content: str
    created_at: str


class MemoryStore:
    """SQLite-backed turn storage with strict session isolation and retention."""

    def __init__(
        self,
        path: str | Path | None = None,
        retention_days: int | None = None,
        enabled: bool | None = None,
    ) -> None:
        self.path = Path(path or config.MEMORY_DB_PATH)
        self.retention_days = retention_days if retention_days is not None else config.MEMORY_RETENTION_DAYS
        self.enabled = config.MEMORY_ENABLED if enabled is None else enabled
        if self.enabled:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._initialise()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path, timeout=10, isolation_level=None)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("PRAGMA busy_timeout = 10000")
        conn.execute("PRAGMA secure_delete = ON")
        return conn

    def _initialise(self) -> None:
        with closing(self._connect()) as conn:
            conn.execute("PRAGMA journal_mode = WAL")
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS memory_turns (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    role TEXT NOT NULL CHECK(role IN ('user', 'assistant', 'system')),
                    content TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_memory_session_created "
                "ON memory_turns(session_id, created_at DESC)"
            )
            # FTS gives bounded, relevant recall without sending all historical
            # content to an LLM. session_id is UNINDEXED but still filtered in SQL.
            conn.execute(
                """
                CREATE VIRTUAL TABLE IF NOT EXISTS memory_fts USING fts5(
                    content, session_id UNINDEXED, tokenize='unicode61'
                )
                """
            )

    @staticmethod
    def _validate_session(session_id: str) -> str:
        if not _SESSION_ID_RE.fullmatch(session_id or ""):
            raise ValueError("session_id must be an opaque 1-128 character identifier")
        return session_id

    @staticmethod
    def _validate_role(role: str) -> str:
        if role not in {"user", "assistant", "system"}:
            raise ValueError("role must be user, assistant, or system")
        return role

    def _purge_expired(self, conn: sqlite3.Connection) -> None:
        cutoff = (datetime.now(UTC) - timedelta(days=max(0, self.retention_days))).isoformat()
        ids = [r[0] for r in conn.execute("SELECT id FROM memory_turns WHERE created_at < ?", (cutoff,))]
        if ids:
            placeholders = ",".join("?" for _ in ids)
            conn.execute(f"DELETE FROM memory_fts WHERE rowid IN ({placeholders})", ids)
            conn.execute(f"DELETE FROM memory_turns WHERE id IN ({placeholders})", ids)

    def record_turn(self, session_id: str, role: str, content: str) -> None:
        """Persist one bounded turn. Disabled memory is a deliberate no-op."""
        if not self.enabled:
            return
        session_id = self._validate_session(session_id)
        role = self._validate_role(role)
        content = (content or "").strip()
        if not content:
            return
        content = content[:config.MEMORY_MAX_CONTENT_CHARS]
        now = datetime.now(UTC).isoformat()
        with closing(self._connect()) as conn:
            conn.execute("BEGIN IMMEDIATE")
            self._purge_expired(conn)
            cur = conn.execute(
                "INSERT INTO memory_turns(session_id, role, content, created_at) VALUES (?, ?, ?, ?)",
                (session_id, role, content, now),
            )
            conn.execute(
                "INSERT INTO memory_fts(rowid, content, session_id) VALUES (?, ?, ?)",
                (cur.lastrowid, content, session_id),
            )
            conn.execute("COMMIT")

    def recent(self, session_id: str, limit: int | None = None) -> list[MemoryTurn]:
        if not self.enabled:
            return []
        session_id = self._validate_session(session_id)
        limit = max(1, limit or config.MEMORY_MAX_TURNS)
        with closing(self._connect()) as conn:
            self._purge_expired(conn)
            rows = conn.execute(
                "SELECT role, content, created_at FROM memory_turns "
                "WHERE session_id = ? ORDER BY id DESC LIMIT ?",
                (session_id, limit),
            ).fetchall()
        return [MemoryTurn(**dict(row)) for row in reversed(rows)]

    def relevant(self, session_id: str, query: str, limit: int = 4) -> list[MemoryTurn]:
        """Return relevant earlier turns, never crossing session boundaries."""
        if not self.enabled:
            return []
        session_id = self._validate_session(session_id)
        terms = list(dict.fromkeys(_WORD_RE.findall((query or "").lower())))[:12]
        if not terms:
            return []
        # FTS syntax is built only from regex-tokenised alphanumerics.
        match = " OR ".join(terms)
        with closing(self._connect()) as conn:
            self._purge_expired(conn)
            rows = conn.execute(
                """
                SELECT t.role, t.content, t.created_at
                FROM memory_fts f JOIN memory_turns t ON t.id = f.rowid
                WHERE memory_fts MATCH ? AND t.session_id = ?
                ORDER BY bm25(memory_fts), t.id DESC LIMIT ?
                """,
                (match, session_id, max(1, limit)),
            ).fetchall()
        return [MemoryTurn(**dict(row)) for row in rows]

    def forget_session(self, session_id: str) -> None:
        """Permanently delete all durable state belonging to one session."""
        if not self.enabled:
            return
        session_id = self._validate_session(session_id)
        with closing(self._connect()) as conn:
            conn.execute("BEGIN IMMEDIATE")
            ids = [r[0] for r in conn.execute("SELECT id FROM memory_turns WHERE session_id = ?", (session_id,))]
            if ids:
                placeholders = ",".join("?" for _ in ids)
                conn.execute(f"DELETE FROM memory_fts WHERE rowid IN ({placeholders})", ids)
            conn.execute("DELETE FROM memory_turns WHERE session_id = ?", (session_id,))
            conn.execute("COMMIT")


class MemoryAgent:
    """Small deterministic agent that selects and persists conversation memory."""

    def __init__(self, store: MemoryStore | None = None) -> None:
        self.store = store or MemoryStore()

    def context_for_query(
        self,
        session_id: str | None,
        query: str,
        transient_history: Iterable[dict] | None = None,
    ) -> list[dict]:
        """Merge relevant persistent memory with current process history, deduplicated."""
        transient = list(transient_history or [])[-config.MEMORY_MAX_TURNS:]
        if not session_id or not self.store.enabled:
            return transient
        # Recent turns preserve conversational continuity; relevant recall restores
        # older facts when a user returns to an earlier topic.
        selected = self.store.relevant(session_id, query, limit=4) + self.store.recent(session_id)
        merged: list[dict] = []
        seen: set[tuple[str, str]] = set()
        for turn in selected:
            item = {"role": turn.role, "content": turn.content}
            key = (item["role"], item["content"])
            if key not in seen:
                merged.append(item)
                seen.add(key)
        for item in transient:
            normalised = {"role": item.get("role", "user"), "content": str(item.get("content", ""))}
            key = (normalised["role"], normalised["content"])
            if normalised["content"] and key not in seen:
                merged.append(normalised)
                seen.add(key)
        return merged[-config.MEMORY_MAX_TURNS:]

    def remember_interaction(self, session_id: str | None, question: str, answer: str) -> None:
        if session_id and self.store.enabled:
            self.store.record_turn(session_id, "user", question)
            self.store.record_turn(session_id, "assistant", answer)
