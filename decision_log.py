"""Append-only decision log for the Phase 3B guardrail.

Every guardrail evaluation — ALLOW or BLOCK — is recorded here before the
verdict reaches the caller (guardrail.py:evaluate). Only INSERT is exposed
by this module's API; there is no update/delete function. That enforces
append-only behavior at the application layer. True tamper-evidence in a
production deployment would additionally need OS-level file permissions or
a write-once storage backend (e.g. object storage with object-lock) — that
hardening step is a documented next step, not something this module claims
to provide on its own.

Two backends behind the same functions, same selection rule as memory.py: an
explicit `db_path` (every caller in this codebase passes one, or falls back
to config.GUARDRAIL_DECISION_LOG_PATH, itself always a concrete sqlite path)
uses SQLite; passing `dsn=` explicitly switches to Postgres. Callers that
want the deployment default get it via record(dsn=config.NEON_DATABASE_URL
or None) -- see guardrail.py for how this is wired.
"""

from __future__ import annotations

from contextlib import closing
import json
import sqlite3
from pathlib import Path
from typing import TYPE_CHECKING

import config
import db_postgres

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


def _pg_initialise(dsn: str) -> None:
    with db_postgres.connect(dsn) as conn, conn.cursor() as cur:
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS guardrail_decisions (
                id BIGSERIAL PRIMARY KEY,
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
        conn.commit()


def record(
    verdict: "GuardrailVerdict",
    question: str,
    db_path: str | Path | None = None,
    dsn: str | None = None,
) -> int:
    """Append one immutable decision row. Returns the new row id.

    Same backend-selection rule as memory.py/live_data.py: an explicit
    db_path forces SQLite; otherwise NEON_DATABASE_URL is picked up
    automatically if set, so guardrail.py's call sites need no change to
    start writing to Postgres in a deployment that sets that env var.
    """
    dsn = dsn or (db_postgres.dsn() if db_path is None else None)
    values = (
        verdict.timestamp,
        verdict.verdict,
        ",".join(verdict.failed_rules),
        verdict.input_hash,
        verdict.output_hash,
        question[:200],
        json.dumps([{"rule": c.rule, "passed": c.passed, "detail": c.detail} for c in verdict.checks]),
    )

    if dsn:
        _pg_initialise(dsn)
        with db_postgres.connect(dsn) as conn, conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO guardrail_decisions
                    (timestamp, verdict, rules_triggered, input_hash, output_hash, question_preview, checks_json)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                RETURNING id
                """,
                values,
            )
            new_id = cur.fetchone()["id"]
            conn.commit()
        return new_id

    db_path = db_path or config.GUARDRAIL_DECISION_LOG_PATH
    _initialise(db_path)
    with closing(_connect(db_path)) as conn:
        cur = conn.execute(
            """
            INSERT INTO guardrail_decisions
                (timestamp, verdict, rules_triggered, input_hash, output_hash, question_preview, checks_json)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            values,
        )
        return cur.lastrowid


def read_all(db_path: str | Path | None = None, dsn: str | None = None) -> list[dict]:
    """Read-only accessor for audit/inspection — never used to mutate rows."""
    dsn = dsn or (db_postgres.dsn() if db_path is None else None)
    if dsn:
        _pg_initialise(dsn)
        with db_postgres.connect(dsn) as conn, conn.cursor() as cur:
            cur.execute("SELECT * FROM guardrail_decisions ORDER BY id ASC")
            return list(cur.fetchall())

    db_path = db_path or config.GUARDRAIL_DECISION_LOG_PATH
    if not Path(db_path).exists():
        return []
    with closing(_connect(db_path)) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute("SELECT * FROM guardrail_decisions ORDER BY id ASC").fetchall()
    return [dict(row) for row in rows]
