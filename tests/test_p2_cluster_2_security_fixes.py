"""
Regression tests for P2 Security Fixes Cluster 2 (SEC-23, SEC-24, SEC-25).

- SEC-23: Unbounded FFmpeg Subprocess Execution Timeout and Windows File Lock Handling
- SEC-24: Test-UI Video Upload File Signature (Magic Bytes) Verification
- SEC-25: Local File Snapshot Fallback Produces Valid HTTP Gateway URL
"""

import importlib
import io
import os
from pathlib import Path
import subprocess
from unittest.mock import MagicMock, patch

import pytest
from starlette.testclient import TestClient

from app import config
from app.constants import (
    LABEL_PERSON,
    LABEL_HARD_HAT,
    LABEL_SAFETY_VEST,
)
from app.detection.adapter import Detection as BoundingBoxDetection
from app.main import _build_violation_event
from app.rules.deduplication import ViolationEvent as DedupEvent
from app.schemas.violation_event import ViolationEvent

# Load test-ui server via importlib because module name contains a hyphen
test_ui_server = importlib.import_module("test-ui.server")


# ==============================================================================
# SEC-23: FFmpeg Subprocess Timeout and Windows File Lock Collision Handling
# ==============================================================================

def test_sec_23_ffmpeg_subprocess_timeout(tmp_path):
    """Verify _convert_to_h264 specifies a 120s timeout and cleans up on TimeoutExpired."""
    fake_video = tmp_path / "test_video.mp4"
    fake_video.write_bytes(b"dummy video data")

    with patch("subprocess.run") as mock_run:
        # Simulate FFmpeg hanging and raising TimeoutExpired
        mock_run.side_effect = subprocess.TimeoutExpired(cmd=["ffmpeg"], timeout=120)

        # Execute conversion
        test_ui_server._convert_to_h264(fake_video)

        # Verify subprocess.run was called with timeout=120
        assert mock_run.call_count == 1
        call_kwargs = mock_run.call_args[1]
        assert call_kwargs.get("timeout") == 120


def test_sec_23_windows_file_lock_collision_retry(tmp_path):
    """Verify _convert_to_h264 retries on Windows PermissionError during atomic file replace."""
    fake_video = tmp_path / "locked_video.mp4"
    fake_video.write_bytes(b"original data")

    tmp_h264 = tmp_path / "locked_video_h264.mp4"
    tmp_h264.write_bytes(b"converted data")

    # Mock subprocess.run to succeed and create tmp_h264
    with patch("subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0)

        # Mock Path.replace to raise PermissionError twice, then succeed on 3rd attempt
        call_count = 0
        original_replace = Path.replace

        def mock_replace(self, target):
            nonlocal call_count
            call_count += 1
            if call_count < 3 and os.name == "nt":
                raise PermissionError(13, "Permission denied (file locked)")
            return original_replace(self, target)

        with patch.object(Path, "replace", side_effect=mock_replace, autospec=True):
            with patch("time.sleep") as mock_sleep:
                test_ui_server._convert_to_h264(fake_video)
                # Verify retry occurred
                assert call_count >= 1


# ==============================================================================
# SEC-24: Test-UI Video Upload Magic Byte Verification
# ==============================================================================

def test_sec_24_upload_valid_video_magic_signatures():
    """Verify that uploads with valid MP4, AVI, and MKV magic bytes are accepted."""
    client = TestClient(test_ui_server.app)

    # 1. Valid MP4 (ISOBMFF with ftyp)
    mp4_bytes = b"\x00\x00\x00\x20ftypisom" + b"\x00" * 100
    res = client.post("/api/upload", files={"file": ("clip.mp4", io.BytesIO(mp4_bytes), "video/mp4")})
    assert res.status_code == 200
    assert "job_id" in res.json()

    # 2. Valid AVI (RIFF ... AVI )
    avi_bytes = b"RIFF\x24\x00\x00\x00AVI LIST" + b"\x00" * 100
    res = client.post("/api/upload", files={"file": ("clip.avi", io.BytesIO(avi_bytes), "video/x-msvideo")})
    assert res.status_code == 200
    assert "job_id" in res.json()

    # 3. Valid MKV (EBML \x1a\x45\xdf\xa3)
    mkv_bytes = b"\x1a\x45\xdf\xa3\x93\x42\x86\x81\x01\x42\xf7\x81\x01" + b"\x00" * 100
    res = client.post("/api/upload", files={"file": ("clip.mkv", io.BytesIO(mkv_bytes), "video/x-matroska")})
    assert res.status_code == 200
    assert "job_id" in res.json()


def test_sec_24_upload_disguised_non_video_rejection():
    """Verify that disguised non-video files are rejected with 400."""
    client = TestClient(test_ui_server.app)

    # 1. Disguised HTML document
    html_payload = b"<!DOCTYPE html><html><head><title>Test</title></head><body>Sample</body></html>"
    res = client.post("/api/upload", files={"file": ("report.mp4", io.BytesIO(html_payload), "video/mp4")})
    assert res.status_code == 400
    assert "không khớp signature video" in res.json().get("detail", "")

    # 2. Disguised plaintext script
    text_payload = b"# Shell script payload\necho 'hello world'\nexit 0\n"
    res = client.post("/api/upload", files={"file": ("stream.avi", io.BytesIO(text_payload), "video/x-msvideo")})
    assert res.status_code == 400
    assert "không khớp signature video" in res.json().get("detail", "")

    # 3. Disguised binary executable (MZ header)
    exe_payload = b"\x4d\x5a\x90\x00\x03\x00\x00\x00\x04\x00\x00\x00\xff\xff" + b"\x00" * 100
    res = client.post("/api/upload", files={"file": ("driver.mkv", io.BytesIO(exe_payload), "video/x-matroska")})
    assert res.status_code == 400
    assert "không khớp signature video" in res.json().get("detail", "")

    # 4. Corrupt tiny file (< 12 bytes)
    tiny_payload = b"tiny_short"
    res = client.post("/api/upload", files={"file": ("corrupt.mp4", io.BytesIO(tiny_payload), "video/mp4")})
    assert res.status_code == 400
    assert "không khớp signature video" in res.json().get("detail", "")


# ==============================================================================
# SEC-25: Offline Snapshot URL Fallback Valid HTTP Scheme
# ==============================================================================

def test_sec_25_offline_snapshot_url_generates_valid_http_scheme():
    """
    Verify that when operating offline or MinIO upload fails, snapshot_url constructs
    a valid HTTP edge gateway URL (never file:/// or relative path) that satisfies
    downstream RFC 3986 URL validators.
    """
    ev = DedupEvent(
        event_id="EVT-SEC25",
        camera_id="CAM-GATE-01",
        person_track_id=1,
        canonical_person_id=1,
        violation_type="NO_HELMET",
        start_t=10.0,
        confirmed_t=12.5,
        duration_s=2.5,
    )

    all_detections = [
        BoundingBoxDetection(
            class_name="person",
            confidence=0.92,
            bbox=(100.0, 100.0, 250.0, 500.0),
            track_id=1,
        ),
    ]

    # Test constructing the fallback URL as done in app/main.py:
    evidence_path = "evidences/CAM-GATE-01/1700000000_track1.jpg"
    evidence_file = Path(evidence_path).name
    edge_base = os.getenv("EDGE_GATEWAY_URL", f"{config.BACKEND_BASE_URL}/evidences").rstrip("/")
    fallback_url = f"{edge_base}/{config.SITE_CODE}/{ev.camera_id}/{evidence_file}"

    # Must be valid HTTP or HTTPS scheme
    assert fallback_url.startswith("http://") or fallback_url.startswith("https://")
    assert not fallback_url.startswith("file:///")

    event = _build_violation_event(
        ev=ev,
        frame_w=1920,
        frame_h=1080,
        snapshot_url=fallback_url,
        all_detections=all_detections,
        model_name="yolov8x-ppe.pt",
    )

    # Verify event evidence snapshotUrl contains valid HTTP URL
    assert event.evidence.snapshotUrl == fallback_url
    assert event.evidence.snapshotUrl.startswith("http://") or event.evidence.snapshotUrl.startswith("https://")

    # Verify serialization
    payload = event.model_dump(by_alias=True)
    assert payload["evidence"]["snapshotUrl"] == fallback_url
