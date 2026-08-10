"""Shared Postgres (Neon) connection helper for memory.py, decision_log.py,
and live_data.py's durable state.

Each of those modules keeps its SQLite path as the default backend
(unchanged local-dev/test behavior -- existing tests construct stores with
an explicit sqlite path and stay on that path deterministically, regardless
of environment). Postgres is used only when a DSN is available AND the
caller did not force a specific sqlite file, mirroring how config.py's
LLM_PROVIDER switches the LLM backend: opt-in via configuration, invisible
to code that doesn't ask for it.

Why a real migration and not "just leave it as SQLite": the deployment
target's local disk is not guaranteed to survive every redeploy (a fresh
container on each `git push` to Streamlit Community Cloud, or a Lambda
execution environment elsewhere in this portfolio's stack) -- losing
guardrail_decisions on every redeploy would undermine the one part of this
project's audit trail that is supposed to be durable.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

import psycopg
from psycopg.rows import dict_row

import config


def dsn() -> str | None:
    return getattr(config, "NEON_DATABASE_URL", "") or None


@contextmanager
def connect(db_dsn: str | None = None) -> Iterator[psycopg.Connection]:
    """Yields a connection with dict-like row access (mirrors sqlite3.Row's
    `dict(row)` usage in the modules this backs)."""
    conn = psycopg.connect(db_dsn or dsn(), row_factory=dict_row, autocommit=False)
    try:
        yield conn
    finally:
        conn.close()
