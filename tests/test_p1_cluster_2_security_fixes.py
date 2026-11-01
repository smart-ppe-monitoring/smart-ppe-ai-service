"""Security regression tests for P1 Vulnerability Cluster 2:
- SEC-09: Silent Kafka Async Event Loss Recovery to SQLite Buffer
- SEC-10: Spatial Track Collision & Cooldown Suppression Prevention
- SEC-11: Unbounded File Upload Read DoS & 500MB Limit
- SEC-12: Cleartext RTSP Credentials Masking in Logs
"""
import io
import json
import uuid
import pytest
from unittest.mock import MagicMock
from fastapi.testclient import TestClient
import importlib

from app.schemas.violation_event import (
    ViolationEvent,
    Evidence,
    FrameResolution,
    BoundingBox,
    Detection as SchemaDetection,
    ViolationItem,
)
from app.events.kafka_publisher import _delivery_report
from app.events.offline_buffer import load_batch, delete_events
from app.rules.deduplication import ViolationDeduplicator
from app.detection.adapter import Detection
from app.rules.confirmation import ConfirmedViolation
from app.main import mask_credentials

test_ui_server = importlib.import_module("test-ui.server")


@pytest.fixture
def test_event():
    return ViolationEvent(
        cameraCode="CAM-01",
        timestamp="2026-10-30T10:00:00.000Z",
        evidence=Evidence(
            snapshotUrl="http://localhost:9000/evidence/snap.jpg",
            frameResolution=FrameResolution(width=1920, height=1080),
        ),
        detections=[
            SchemaDetection(
                classLabel="PERSON",
                confidence=0.95,
                box=BoundingBox(x=100.0, y=100.0, width=100.0, height=300.0),
                trackId=1,
            )
        ],
        violations=[
            ViolationItem(
                violationType="NO_HELMET",
                severity="HIGH",
                confidence=0.95,
            )
        ],
    )


def test_sec_09_kafka_async_failure_recovery(test_event):
    """SEC-09: Verify that when Kafka async delivery fails, the event is recovered into offline SQLite buffer."""
    # Ensure offline buffer is empty before test
    ids, _ = load_batch()
    if ids:
        delete_events(ids)

    # Construct mock undelivered Kafka message
    envelope = {
        "topic": "ppe.violation.detected",
        "producer": "ai-worker",
        "payload": test_event.model_dump(by_alias=True),
    }
    msg_bytes = json.dumps(envelope).encode("utf-8")
    mock_msg = MagicMock()
    mock_msg.value.return_value = msg_bytes

    # Trigger delivery report with an error (simulating broker timeout / disconnect)
    mock_err = "BrokerTransportFailure: Connection reset by peer"
    _delivery_report(mock_err, mock_msg)

    # Verify event was captured into offline buffer
    recovered_ids, recovered_events = load_batch()
    assert len(recovered_events) >= 1
    assert recovered_events[-1].cameraCode == "CAM-01"
    assert recovered_events[-1].violations[0].violationType == "NO_HELMET"

    # Cleanup
    delete_events(recovered_ids)


def test_sec_10_spatial_track_collision_prevention():
    """SEC-10: Verify that persons with distinct aspect ratio / height are not falsely linked across grace period."""
    dedup = ViolationDeduplicator(camera_id="CAM-01", cooldown_s=30.0)

    # Person 1 (tall worker, aspect ratio 100/300 = 0.33) commits violation at t=3.0
    p1_det = Detection(class_name="person", confidence=0.9, bbox=(10, 10, 110, 310), track_id=1)
    c1 = ConfirmedViolation(person_track_id=1, violation_type="no_helmet", start_t=0.0, confirmed_t=3.0, duration_s=3.0)
    out1 = dedup.filter([c1], [p1_det], 3.0)
    assert len(out1) == 1

    # Person 1 disappears at t=3.5
    dedup.filter([], [], 3.5)

    # Person 2 (short/wide object or distinct person, width=200, height=150, aspect=1.33) appears near same center at t=4.0
    # Center of Person 1: (60, 160). Center of Person 2: (60, 160). Distance = 0!
    p2_det = Detection(class_name="person", confidence=0.9, bbox=(-40, 85, 160, 235), track_id=2)
    dedup.filter([], [p2_det], 4.0)

    # Person 2 commits violation at t=5.0
    c2 = ConfirmedViolation(person_track_id=2, violation_type="no_helmet", start_t=4.0, confirmed_t=5.0, duration_s=1.0)
    out2 = dedup.filter([c2], [p2_det], 5.0)

    # Because aspect ratio and height ratio differ significantly, Person 2 must NOT inherit Person 1's canonical ID
    assert len(out2) == 1, "Person 2 violation must NOT be suppressed by Person 1 cooldown"
    assert out2[0].person_track_id == 2
    assert out2[0].canonical_person_id != out1[0].canonical_person_id


def test_sec_11_upload_file_size_limit_and_streaming():
    """SEC-11: Verify 500MB upload limit and chunked streaming protection."""
    client = TestClient(test_ui_server.app)

    # Test small valid video upload succeeds
    fake_video = io.BytesIO(b"\x00\x00\x00\x20ftypisom" + b"A" * 1024)
    res = client.post("/api/upload", files={"file": ("test.mp4", fake_video, "video/mp4")})
    assert res.status_code == 200
    assert "job_id" in res.json()

    # Test upload exceeding 500MB is rejected with 413
    original_max = test_ui_server.MAX_UPLOAD_SIZE_BYTES
    try:
        # Temporarily set limit to 10KB to test limit enforcement without allocating 500MB in test
        test_ui_server.MAX_UPLOAD_SIZE_BYTES = 10 * 1024
        large_video = io.BytesIO(b"\x00\x00\x00\x20ftypisom" + b"B" * (20 * 1024))
        res_large = client.post("/api/upload", files={"file": ("large.mp4", large_video, "video/mp4")})
        assert res_large.status_code == 413
        assert "vượt quá giới hạn tối đa" in res_large.json().get("detail", "")
    finally:
        test_ui_server.MAX_UPLOAD_SIZE_BYTES = original_max


def test_sec_12_rtsp_credentials_masking():
    """SEC-12: Verify credentials in RTSP and HTTP URLs are properly masked for safe logging."""
    rtsp_url = "rtsp://admin:superSecretP@ss123@192.168.1.100:554/stream1"
    masked_rtsp = mask_credentials(rtsp_url)
    assert "superSecretP@ss123" not in masked_rtsp
    assert masked_rtsp == "rtsp://admin:***@192.168.1.100:554/stream1"

    http_url = "http://user:P@ssword!@camera.local:8080/mjpg"
    masked_http = mask_credentials(http_url)
    assert "P@ssword!" not in masked_http
    assert masked_http == "http://user:***@camera.local:8080/mjpg"

    # Non-credential URLs or file paths remain intact
    assert mask_credentials("rtsp://192.168.1.100:554/live") == "rtsp://192.168.1.100:554/live"
    assert mask_credentials("sample_video.mp4") == "sample_video.mp4"
    assert mask_credentials(None) == ""
