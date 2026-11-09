"""
Tests for Live Stream AI Engine (app/live_engine.py).

Verifies:
1. YouTube URL recognition, whitelist enforcement, and yt-dlp resolution.
2. Foot-point based Polygon ROI Restricted Area checking (D08.pdf Section 2.6 & Example 5).
3. Integration of Restricted Area in Safety Rule Engine (evaluate_safety).
4. Visualizer rendering of ROI polygons and Restricted Area violations.
5. LiveStreamSession frame generation, FPS & Sampling rate pacing, and realtime metrics.
6. Test-UI FastAPI live endpoints (/api/live/*).
"""

import importlib
from unittest.mock import MagicMock, patch
import numpy as np
import pytest
from starlette.testclient import TestClient

from app.constants import (
    VIOLATION_NO_HELMET,
    VIOLATION_NO_VEST,
    VIOLATION_RESTRICTED_AREA,
    STATE_COMPLIANT,
    STATE_CANDIDATE,
)
from app.detection.adapter import Detection
from app.association.associator import PersonPPE
from app.video.video_reader import (
    is_youtube_url,
    resolve_youtube_url,
    VideoSourceError,
    VideoReader,
)
from app.rules.safety import is_inside_roi, evaluate as evaluate_safety
from app.video.visualizer import draw_roi_polygons, draw_detections
from app.live_engine import LiveStreamSession, LiveEngineManager, live_engine_manager

test_ui_server = importlib.import_module("test-ui.server")
app = test_ui_server.app


# ==============================================================================
# 1. YouTube URL Whitelist & Resolution Tests
# ==============================================================================

def test_is_youtube_url_whitelist():
    """Verify is_youtube_url only recognizes valid and safe YouTube domains."""
    valid_urls = [
        "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
        "https://youtube.com/watch?v=xyz123",
        "https://m.youtube.com/watch?v=abc",
        "https://youtu.be/dQw4w9WgXcQ",
        "http://www.youtube.com/live/stream1",
    ]
    for url in valid_urls:
        assert is_youtube_url(url) is True, f"Failed to identify valid YouTube URL: {url}"

    invalid_urls = [
        "https://evil.com/video.mp4",
        "http://youtube.com.attacker.com/watch",
        "rtsp://192.168.1.100:554/live",
        "http://localhost:8000/sample.mp4",
        "not_a_url",
        123,
        None,
    ]
    for url in invalid_urls:
        assert is_youtube_url(url) is False, f"Erroneously matched invalid YouTube URL: {url}"


def test_resolve_youtube_url_security_rejection():
    """Verify resolve_youtube_url rejects untrusted or malicious non-YouTube URLs."""
    with pytest.raises(VideoSourceError, match="không an toàn"):
        resolve_youtube_url("https://malicious-site.com/video.mp4")


def test_resolve_youtube_url_success():
    """Verify resolve_youtube_url invokes yt-dlp correctly and returns stream URL."""
    fake_stream_url = "https://rr3---sn-4g5ednks.googlevideo.com/videoplayback?id=123"
    with patch("yt_dlp.YoutubeDL") as mock_ydl_cls:
        mock_ydl = MagicMock()
        mock_ydl.extract_info.return_value = {"url": fake_stream_url}
        mock_ydl_cls.return_value.__enter__.return_value = mock_ydl

        res = resolve_youtube_url("https://www.youtube.com/watch?v=dQw4w9WgXcQ")
        assert res == fake_stream_url
        mock_ydl.extract_info.assert_called_once_with("https://www.youtube.com/watch?v=dQw4w9WgXcQ", download=False)


# ==============================================================================
# 2. Foot-point Polygon ROI Checking (D08.pdf Section 2.6 & Example 5)
# ==============================================================================

def test_is_inside_roi_foot_point():
    """Verify is_inside_roi checks bottom-center (foot point) against polygon ROI."""
    # Polygon 200x200 square at (100, 100) -> (300, 300)
    poly = [(100, 100), (300, 100), (300, 300), (100, 300)]

    # Person 1: foot at (200, 250) -> inside
    p1_bbox = (150, 100, 250, 250)
    assert is_inside_roi(p1_bbox, poly) is True

    # Person 2: foot at (200, 350) -> outside below polygon
    p2_bbox = (150, 100, 250, 350)
    assert is_inside_roi(p2_bbox, poly) is False

    # Person 3: foot at (50, 200) -> outside to the left
    p3_bbox = (0, 100, 100, 200)
    assert is_inside_roi(p3_bbox, poly) is False

    # Dict format: {"points": [...]}
    assert is_inside_roi(p1_bbox, {"points": poly}) is True

    # Empty / None ROI returns False
    assert is_inside_roi(p1_bbox, None) is False
    assert is_inside_roi(p1_bbox, []) is False


# ==============================================================================
# 3. Safety Rule Engine Restricted Area Integration
# ==============================================================================

def test_safety_rules_with_restricted_area():
    """Verify evaluate_safety flags RESTRICTED_AREA violation when person steps into ROI."""
    poly = [(100, 100), (300, 100), (300, 300), (100, 300)]

    # Person with compliant PPE (has helmet + vest)
    p_inside = Detection(class_name="person", confidence=0.95, bbox=(150, 100, 250, 250), track_id=1)
    hm = Detection(class_name="helmet", confidence=0.90, bbox=(160, 100, 240, 140), track_id=2)
    vt = Detection(class_name="safety_vest", confidence=0.90, bbox=(160, 150, 240, 240), track_id=3)
    asso_inside = [PersonPPE(1, p_inside.bbox, helmet_track_id=2, vest_track_id=3)]

    # 1. Inside ROI with restricted area enabled -> VIOLATION_CANDIDATE with RESTRICTED_AREA
    states = evaluate_safety(
        detections=[p_inside, hm, vt],
        associations=asso_inside,
        frame_w=1000,
        frame_h=1000,
        roi_polygons=[poly],
        enable_restricted_area=True,
    )
    assert len(states) == 1
    assert states[0].status == STATE_CANDIDATE
    assert VIOLATION_RESTRICTED_AREA in states[0].violations

    # 2. Outside ROI -> COMPLIANT
    p_outside = Detection(class_name="person", confidence=0.95, bbox=(400, 100, 500, 250), track_id=4)
    hm_out = Detection(class_name="helmet", confidence=0.90, bbox=(410, 100, 490, 140), track_id=5)
    vt_out = Detection(class_name="safety_vest", confidence=0.90, bbox=(410, 150, 490, 240), track_id=6)
    asso_outside = [PersonPPE(4, p_outside.bbox, helmet_track_id=5, vest_track_id=6)]

    states_out = evaluate_safety(
        detections=[p_outside, hm_out, vt_out],
        associations=asso_outside,
        frame_w=1000,
        frame_h=1000,
        roi_polygons=[poly],
        enable_restricted_area=True,
    )
    assert states_out[0].status == STATE_COMPLIANT
    assert len(states_out[0].violations) == 0

    # 3. Inside ROI but enable_restricted_area=False -> COMPLIANT
    states_disabled = evaluate_safety(
        detections=[p_inside, hm, vt],
        associations=asso_inside,
        frame_w=1000,
        frame_h=1000,
        roi_polygons=[poly],
        enable_restricted_area=False,
    )
    assert states_disabled[0].status == STATE_COMPLIANT


# ==============================================================================
# 4. Visualizer Rendering Tests
# ==============================================================================

def test_visualizer_roi_and_restricted_violation():
    """Verify visualizer renders ROI polygons and restricted area violations properly."""
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    poly = [(50, 50), (250, 50), (250, 250), (50, 250)]

    # Draw ROI
    out_frame = draw_roi_polygons(frame, [poly])
    assert out_frame.shape == (480, 640, 3)

    # Draw detections with Restricted Area violation
    p_det = Detection(class_name="person", confidence=0.9, bbox=(60, 60, 200, 240), track_id=1)
    fake_state = MagicMock(person_track_id=1, violations=[VIOLATION_RESTRICTED_AREA])
    drawn = draw_detections(out_frame, [p_det], person_states=[fake_state], roi_polygons=[poly])
    assert drawn.shape == (480, 640, 3)


# ==============================================================================
# 5. LiveStreamSession & Realtime Metrics Tests
# ==============================================================================

def test_live_stream_session_config_and_metrics():
    """Verify LiveStreamSession hot-updating and realtime metric tracking."""
    session = LiveStreamSession(
        camera_id="CAM-TEST",
        source=0,
        target_fps=30.0,
        ai_fps=10.0,
        roi_polygons=[[(0, 0), (100, 0), (100, 100), (0, 100)]],
    )
    assert session.target_fps == 30.0
    assert session.ai_fps == 10.0
    assert session.metrics["target_fps"] == 30.0

    # Test hot-update config
    session.update_config(target_fps=20.0, ai_fps=10.0, enable_no_helmet=False)
    assert session.target_fps == 20.0
    assert session.ai_fps == 10.0
    assert session.enable_no_helmet is False
    assert session.metrics["target_fps"] == 20.0


def test_live_stream_session_frame_loop_mock():
    """Verify LiveStreamSession frame loop produces MJPEG bytes and updates metrics."""
    fake_frame = np.zeros((240, 320, 3), dtype=np.uint8)

    with patch("app.live_engine.VideoReader") as mock_vr_cls, \
         patch("app.live_engine.Detector") as mock_det_cls:

        # Mock VideoReader returning 3 frames then stopping
        mock_vr = MagicMock()
        mock_vr.is_stream = True
        mock_vr.fps = 25.0
        mock_vr.__enter__.return_value = mock_vr
        mock_vr.__iter__.return_value = [(1, fake_frame), (2, fake_frame), (3, fake_frame)]
        mock_vr_cls.return_value = mock_vr

        # Mock Detector
        mock_det = MagicMock()
        mock_det.load.return_value = mock_det
        mock_det.detect.return_value = []
        mock_det_cls.return_value = mock_det

        session = LiveStreamSession(
            camera_id="CAM-MOCK",
            source="mock_stream",
            target_fps=25.0,
            ai_fps=25.0,
        )
        session.start()
        # Wait briefly for thread to complete the 3 frames
        session._thread.join(timeout=2.0)

        assert session.frame_number == 3
        assert session.latest_jpeg_bytes is not None
        assert session.status == "stopped"

        # Check MJPEG generator produces at least one frame block
        gen = session.generate_mjpeg()
        # Because session is stopped, generator should terminate cleanly
        items = list(gen)
        # Should finish cleanly
        session.stop()


# ==============================================================================
# 6. Test-UI FastAPI Live Endpoints Tests
# ==============================================================================

def test_api_live_endpoints_flow():
    """Verify FastAPI /api/live endpoints for starting, configuring, querying, and stopping live streams."""
    client = TestClient(app)

    with patch("app.live_engine.VideoReader"), patch("app.live_engine.Detector"):
        # 1. Start live stream
        start_payload = {
            "camera_id": "CAM-E2E-TEST",
            "source": "0",
            "target_fps": 25.0,
            "ai_fps": 15.0,
            "roi_polygons": [[(10, 10), (100, 10), (100, 100), (10, 100)]],
            "enable_no_helmet": True,
            "enable_no_vest": True,
            "enable_restricted_area": True,
        }
        res = client.post("/api/live/start", json=start_payload)
        assert res.status_code == 200
        data = res.json()
        assert data["status"] == "started"
        assert data["camera_id"] == "CAM-E2E-TEST"

        # 2. Query stats
        res_stats = client.get("/api/live/CAM-E2E-TEST/stats")
        assert res_stats.status_code == 200
        stats_data = res_stats.json()
        assert stats_data["camera_id"] == "CAM-E2E-TEST"
        assert "metrics" in stats_data
        assert stats_data["metrics"]["target_fps"] == 25.0
        assert stats_data["metrics"]["ai_fps"] == 15.0

        # 3. Hot-update config
        patch_payload = {
            "target_fps": 20.0,
            "ai_fps": 10.0,
        }
        res_patch = client.patch("/api/live/CAM-E2E-TEST/config", json=patch_payload)
        assert res_patch.status_code == 200
        patch_data = res_patch.json()
        assert patch_data["target_fps"] == 20.0
        assert patch_data["ai_fps"] == 10.0

        # 4. List active sessions
        res_list = client.get("/api/live/sessions")
        assert res_list.status_code == 200
        sessions = res_list.json()
        assert any(s["camera_id"] == "CAM-E2E-TEST" for s in sessions)

        # 5. Stop session
        res_stop = client.post("/api/live/CAM-E2E-TEST/stop")
        assert res_stop.status_code == 200
        assert res_stop.json()["status"] == "stopped"

        # 6. Stop already stopped session returns 404
        res_stop_404 = client.post("/api/live/CAM-E2E-TEST/stop")
        assert res_stop_404.status_code == 404


def test_live_before_after_evidence_urls_and_endpoint(tmp_path, monkeypatch):
    """Verify live violations record both snapshot_url and reference_snapshot_url and serve via /api/live/evidence."""
    from app import config
    monkeypatch.setattr(config, "EVIDENCE_DIR", str(tmp_path))

    session = LiveStreamSession(camera_id="CAM-EV", source=0, dispatch_enabled=False)
    frame_viol = np.full((120, 160, 3), 50, dtype=np.uint8)
    frame_ref = np.full((120, 160, 3), 120, dtype=np.uint8)
    p_det = Detection(class_name="person", confidence=0.9, bbox=(20, 20, 100, 110), track_id=7)

    ev = MagicMock(
        event_id="a1b2c3d4e5f6",
        camera_id="CAM-EV",
        person_track_id=7,
        canonical_person_id=7,
        violation_type=VIOLATION_NO_VEST,
        start_t=1.0,
        confirmed_t=3.0,
        duration_s=2.0,
        reference_frame=frame_ref,
        reference_bbox=(20, 20, 100, 110),
        reference_type="COMPLIANT",
    )

    session._dispatch_evidence_async(frame_viol, [p_det], ev, 160, 120, frame_num=75)
    assert len(session.recent_violations) == 1
    rec = session.recent_violations[0]
    assert rec["snapshot_url"] == "/api/live/evidence/a1b2c3d4e5f6.jpg"
    assert rec["reference_snapshot_url"] == "/api/live/evidence/a1b2c3d4e5f6_ref.jpg"
    assert rec["has_reference"] is True
    assert rec["frame"] == 75

    client = TestClient(app)
    res_viol = client.get("/api/live/evidence/a1b2c3d4e5f6.jpg")
    assert res_viol.status_code == 200
    assert res_viol.headers["content-type"] == "image/jpeg"

    res_ref = client.get("/api/live/evidence/a1b2c3d4e5f6_ref.jpg")
    assert res_ref.status_code == 200
    assert res_ref.headers["content-type"] == "image/jpeg"

    res_bad = client.get("/api/live/evidence/evil_name.jpg")
    assert res_bad.status_code == 400
