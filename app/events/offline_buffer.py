"""
SQLite offline buffer — lưu sự kiện vi phạm khi mất kết nối tới Backend/Kafka.
Khi kết nối phục hồi, dùng http_publisher.publish_batch() để nạp lại.
"""

import json
import sqlite3
import logging
from pathlib import Path
from app.constants import OFFLINE_BUFFER_DB, OFFLINE_BUFFER_BATCH_SIZE
from app.schemas.violation_event import ViolationEvent

logger = logging.getLogger(__name__)

DB_PATH = Path(OFFLINE_BUFFER_DB)

_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS offline_events (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    payload     TEXT    NOT NULL,
    created_at  TEXT    NOT NULL DEFAULT (datetime('now')),
    retried     INTEGER NOT NULL DEFAULT 0
);
"""


def _get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.execute(_CREATE_TABLE)
    conn.commit()
    return conn


def save_event(event: ViolationEvent) -> None:
    """Persist a violation event to the local SQLite buffer."""
    payload_json = event.model_dump_json(by_alias=True)
    with _get_conn() as conn:
        conn.execute(
            "INSERT INTO offline_events (payload) VALUES (?)", (payload_json,)
        )
    logger.warning("Offline buffer: saved event for camera %s", event.cameraCode)


def load_batch() -> tuple[list[int], list[ViolationEvent]]:
    """
    Load up to OFFLINE_BUFFER_BATCH_SIZE pending events.
    Returns (ids, events) — caller must call delete_events(ids) after successful publish.
    """
    with _get_conn() as conn:
        rows = conn.execute(
            "SELECT id, payload FROM offline_events ORDER BY id LIMIT ?",
            (OFFLINE_BUFFER_BATCH_SIZE,),
        ).fetchall()
    ids = [r[0] for r in rows]
    events = [ViolationEvent.model_validate_json(r[1]) for r in rows]
    return ids, events


def delete_events(ids: list[int]) -> None:
    """Remove successfully published events from the buffer."""
    if not ids:
        return
    placeholders = ",".join("?" * len(ids))
    with _get_conn() as conn:
        conn.execute(f"DELETE FROM offline_events WHERE id IN ({placeholders})", ids)
    logger.info("Offline buffer: deleted %d flushed events", len(ids))


def pending_count() -> int:
    with _get_conn() as conn:
        return conn.execute("SELECT COUNT(*) FROM offline_events").fetchone()[0]
