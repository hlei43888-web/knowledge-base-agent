"""SQLite trace logger for request logging."""

import json
import sqlite3
import uuid
import logging
from datetime import datetime, timezone
from contextlib import contextmanager

from app.config import SQLITE_DB_PATH

logger = logging.getLogger(__name__)

_CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS trace_logs (
    request_id TEXT PRIMARY KEY,
    timestamp TEXT NOT NULL,
    user_query TEXT NOT NULL,
    intent TEXT NOT NULL,
    retrieved_chunks TEXT NOT NULL,
    llm_prompt TEXT NOT NULL,
    llm_response TEXT NOT NULL,
    output TEXT NOT NULL,
    latency_ms INTEGER NOT NULL,
    error TEXT
)
"""

_INSERT_SQL = """
INSERT INTO trace_logs
    (request_id, timestamp, user_query, intent, retrieved_chunks,
     llm_prompt, llm_response, output, latency_ms, error)
VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
"""

_initialized = False


@contextmanager
def _get_conn():
    conn = sqlite3.connect(SQLITE_DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def _ensure_table():
    global _initialized
    if _initialized:
        return
    with _get_conn() as conn:
        conn.execute(_CREATE_TABLE_SQL)
    _initialized = True


def log_trace(
    user_query: str,
    intent: str,
    retrieved_chunks: list[str],
    llm_prompt: str,
    llm_response: str,
    output: dict,
    latency_ms: int,
    error: str | None = None,
) -> str:
    """Write a trace record to SQLite. Returns the request_id."""
    _ensure_table()
    request_id = str(uuid.uuid4())
    timestamp = datetime.now(timezone.utc).isoformat()

    try:
        with _get_conn() as conn:
            conn.execute(_INSERT_SQL, (
                request_id,
                timestamp,
                user_query,
                intent,
                json.dumps(retrieved_chunks, ensure_ascii=False),
                llm_prompt,
                llm_response,
                json.dumps(output, ensure_ascii=False),
                latency_ms,
                error,
            ))
        logger.info(f"Trace logged: {request_id} intent={intent} latency={latency_ms}ms")
    except Exception:
        logger.exception(f"Failed to log trace for query: {user_query[:50]}")

    return request_id


def get_trace(request_id: str) -> dict | None:
    """Get a single trace record by request_id."""
    _ensure_table()
    with _get_conn() as conn:
        row = conn.execute(
            "SELECT * FROM trace_logs WHERE request_id = ?", (request_id,)
        ).fetchone()
    if row is None:
        return None
    return _row_to_dict(row)


def list_traces(limit: int = 50, offset: int = 0) -> list[dict]:
    """List recent trace records, newest first."""
    _ensure_table()
    with _get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM trace_logs ORDER BY timestamp DESC LIMIT ? OFFSET ?",
            (limit, offset),
        ).fetchall()
    return [_row_to_dict(r) for r in rows]


def count_traces() -> int:
    """Count total trace records."""
    _ensure_table()
    with _get_conn() as conn:
        row = conn.execute("SELECT COUNT(*) as cnt FROM trace_logs").fetchone()
    return row["cnt"]


def _row_to_dict(row: sqlite3.Row) -> dict:
    d = dict(row)
    d["retrieved_chunks"] = json.loads(d["retrieved_chunks"])
    d["output"] = json.loads(d["output"])
    return d
