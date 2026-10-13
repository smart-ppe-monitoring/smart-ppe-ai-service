"""
Tests for event dispatching and offline buffer (app.events).
Verifies:
1. Dispatch via Kafka
2. Fallback to HTTP when Kafka fails
3. Fallback to SQLite offline buffer when HTTP fails
4. Flush pending offline buffer via batch endpoint
"""

import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from app.schemas.violation_event import (
    ViolationEvent,
    Evidence,
    Detection,
    ViolationItem,
    BoundingBox,
    FrameResolution,
)
from app.constants import (
    VIOLATION_NO_HELMET,
    SEVERITY_HIGH,
    LABEL_PERSON,
)
from app.events import dispatch_event, flush_offline_buffer
import app.events.offline_buffer as offline_buffer_mod


def _sample_event() -> ViolationEvent:
    return ViolationEvent(
        cameraCode="CAM-01",
        timestamp="2026-10-13T10:00:00.000Z",
        evidence=Evidence(
            snapshotUrl="http://localhost:9000/ppe-evidences/test.jpg",
            frameResolution=FrameResolution(width=1920, height=1080),
        ),
        detections=[
            Detection(
                classLabel=LABEL_PERSON,
                confidence=0.95,
                box=BoundingBox(x=100.0, y=100.0, width=50.0, height=150.0),
                trackId=1,
            )
        ],
        violations=[
            ViolationItem(
                violationType=VIOLATION_NO_HELMET,
                severity=SEVERITY_HIGH,
                description="Không đội mũ",
                confidence=0.92,
            )
        ],
    )


def test_dispatch_via_kafka_success():
    event = _sample_event()
    with patch("app.events.publish_violation") as mock_pub:
        result = dispatch_event(event)
        assert result == "kafka"
        mock_pub.assert_called_once_with(event)


def test_dispatch_fallback_to_http():
    event = _sample_event()
    with patch("app.events.publish_violation", side_effect=RuntimeError("Kafka down")), \
         patch("app.events.publish_event_sync", return_value={"status": "ok"}) as mock_http:
        result = dispatch_event(event)
        assert result == "http"
        mock_http.assert_called_once_with(event, sync=False)


def test_dispatch_fallback_to_offline_buffer(tmp_path):
    test_db = tmp_path / "test_buffer.db"
    event = _sample_event()

    with patch.object(offline_buffer_mod, "DB_PATH", test_db), \
         patch("app.events.save_event", side_effect=lambda e: offline_buffer_mod.save_event(e)), \
         patch("app.events.publish_violation", side_effect=RuntimeError("Kafka down")), \
         patch("app.events.publish_event_sync", side_effect=ConnectionRefusedError("HTTP down")):
        result = dispatch_event(event)
        assert result == "offline_buffer"
        assert offline_buffer_mod.pending_count() == 1


def test_flush_offline_buffer(tmp_path):
    test_db = tmp_path / "test_buffer.db"
    event = _sample_event()

    with patch.object(offline_buffer_mod, "DB_PATH", test_db), \
         patch("app.events.pending_count", side_effect=lambda: offline_buffer_mod.pending_count()), \
         patch("app.events.load_batch", side_effect=lambda: offline_buffer_mod.load_batch()), \
         patch("app.events.delete_events", side_effect=lambda ids: offline_buffer_mod.delete_events(ids)):
        offline_buffer_mod.save_event(event)
        assert offline_buffer_mod.pending_count() == 1

        with patch("app.events.publish_batch_sync", return_value={"flushed": 1}):
            flushed = flush_offline_buffer(sync=True)
            assert flushed == 1
            assert offline_buffer_mod.pending_count() == 0
