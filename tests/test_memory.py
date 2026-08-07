from datetime import UTC, datetime, timedelta
import sqlite3

from memory import MemoryAgent, MemoryStore


def test_memory_is_scoped_relevant_and_deletable(tmp_path):
    store = MemoryStore(tmp_path / "memory.sqlite3", retention_days=30, enabled=True)
    store.record_turn("session-a", "user", "Please compare FDA AI device guidance")
    store.record_turn("session-a", "assistant", "I will compare the FDA guidance.")
    store.record_turn("session-b", "user", "This must never appear in another session")

    relevant = store.relevant("session-a", "What does FDA guidance say about AI devices?")
    assert relevant
    assert all("another session" not in turn.content for turn in relevant)

    agent = MemoryAgent(store)
    context = agent.context_for_query("session-a", "FDA device guidance", [{"role": "user", "content": "new turn"}])
    assert any("FDA" in item["content"] for item in context)
    assert context[-1]["content"] == "new turn"

    store.forget_session("session-a")
    assert store.recent("session-a") == []
    assert len(store.recent("session-b")) == 1


def test_memory_retention_purges_expired_turns(tmp_path):
    store = MemoryStore(tmp_path / "memory.sqlite3", retention_days=1, enabled=True)
    store.record_turn("session-a", "user", "fresh information")
    old = (datetime.now(UTC) - timedelta(days=3)).isoformat()
    with sqlite3.connect(store.path) as conn:
        conn.execute(
            "INSERT INTO memory_turns(session_id, role, content, created_at) VALUES (?, ?, ?, ?)",
            ("session-a", "user", "expired information", old),
        )
    turns = store.recent("session-a")
    assert [turn.content for turn in turns] == ["fresh information"]
