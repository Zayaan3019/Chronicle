"""Append-only decision log for the Phase 3B guardrail.

Every guardrail evaluation — ALLOW or BLOCK — is recorded here before the
verdict reaches the caller (guardrail.py:evaluate). Only INSERT is exposed
by this module's API; there is no update/delete function. That enforces
append-only behavior at the application layer. True tamper-evidence in a
production deployment would additionally need OS-level file permissions or
a write-once storage backend (e.g. object storage with object-lock) — that
hardening step is a documented next step, not something this module claims
to provide on its own.
"""

from __future__ import annotations

from contextlib import closing
import json
import sqlite3
from pathlib import Path
from typing import TYPE_CHECKING

import config

if TYPE_CHECKING:
    from guardrail import GuardrailVerdict


def _connect(db_path: str | Path) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path, timeout=15, isolation_level=None)
    conn.execute("PRAGMA busy_timeout = 15000")
    return conn


def _initialise(db_path: str | Path) -> None:
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    with closing(_connect(db_path)) as conn:
        conn.execute("PRAGMA journal_mode = WAL")
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS guardrail_decisions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                verdict TEXT NOT NULL,
                rules_triggered TEXT NOT NULL,
                input_hash TEXT NOT NULL,
                output_hash TEXT NOT NULL,
                question_preview TEXT NOT NULL,
                checks_json TEXT NOT NULL
            )
            """
        )


def record(verdict: "GuardrailVerdict", question: str, db_path: str | Path | None = None) -> int:
    """Append one immutable decision row. Returns the new row id."""
    db_path = db_path or config.GUARDRAIL_DECISION_LOG_PATH
    _initialise(db_path)
    with closing(_connect(db_path)) as conn:
        cur = conn.execute(
            """
            INSERT INTO guardrail_decisions
                (timestamp, verdict, rules_triggered, input_hash, output_hash, question_preview, checks_json)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                verdict.timestamp,
                verdict.verdict,
                ",".join(verdict.failed_rules),
                verdict.input_hash,
                verdict.output_hash,
                question[:200],
                json.dumps([{"rule": c.rule, "passed": c.passed, "detail": c.detail} for c in verdict.checks]),
            ),
        )
        return cur.lastrowid


def read_all(db_path: str | Path | None = None) -> list[dict]:
    """Read-only accessor for audit/inspection — never used to mutate rows."""
    db_path = db_path or config.GUARDRAIL_DECISION_LOG_PATH
    if not Path(db_path).exists():
        return []
    with closing(_connect(db_path)) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute("SELECT * FROM guardrail_decisions ORDER BY id ASC").fetchall()
    return [dict(row) for row in rows]
