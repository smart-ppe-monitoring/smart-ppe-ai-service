"""Regression tests cho các lỗ hổng mới ở Live Stream endpoints (/api/live/*).

1. Source allowlist: chặn đọc file cục bộ tùy ý / SSRF qua FFmpeg / glob injection.
2. site_id không còn nhận từ client; dispatch MinIO/Incident tắt mặc định trong Test-UI.
3. Giới hạn tài nguyên: số phiên, ROI, FPS, confirmation_duration_s.
4. TrustedHostMiddleware chống DNS rebinding.
"""
import importlib
from unittest.mock import MagicMock, patch

import pytest
from starlette.testclient import TestClient

from app import live_engine as le
from app.live_engine import LiveEngineManager, LiveSessionLimitError, LiveStreamSession

test_ui_server = importlib.import_module("test-ui.server")
app = test_ui_server.app


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture(autouse=True)
def _cleanup_sessions():
    yield
    le.live_engine_manager.stop_all()


# ─── 1. Source allowlist ──────────────────────────────────────────────────────

@pytest.mark.parametrize("source", [
    "C:/Windows/win.ini",
    "/etc/passwd",
    "../uploads/x.mp4",
    "..\\..\\app\\config.py",
    "file:///etc/passwd",
    "http://169.254.169.254/latest/meta-data/",
    "https://evil.example.com/stream.m3u8",
    "http://youtube.com.evil.com/watch?v=x",
    "concat:/etc/passwd",
    "*",
    "**/*",
    "99",
    "rtsp://",
    "abc\r\ndef",
])
def test_live_start_rejects_untrusted_sources(client, source):
    res = client.post("/api/live/start", json={"camera_id": "CAM-SEC", "source": source})
    assert res.status_code == 400, (source, res.text)
    assert le.live_engine_manager.get_session("CAM-SEC") is None


def test_live_start_accepts_uploaded_job_id(client, tmp_path):
    upload = test_ui_server.UPLOAD_DIR / "secjob123abc.mp4"
    upload.write_bytes(b"\x00" * 16)
    try:
        assert test_ui_server._resolve_live_source("secjob123abc") == str(upload.resolve())
        assert test_ui_server._resolve_live_source("secjob123abc.mp4") == str(upload.resolve())
    finally:
        upload.unlink(missing_ok=True)


def test_live_source_allows_webcam_youtube_rtsp():
    assert test_ui_server._resolve_live_source("0") == 0
    yt = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
    assert test_ui_server._resolve_live_source(yt) == yt
    rtsp = "rtsp://user:pass@10.0.0.5:554/stream1"
    assert test_ui_server._resolve_live_source(rtsp) == rtsp


def test_live_source_rtsp_host_allowlist(monkeypatch):
    monkeypatch.setattr(test_ui_server, "LIVE_ALLOWED_RTSP_HOSTS", {"10.0.0.5"})
    assert test_ui_server._resolve_live_source("rtsp://10.0.0.5/s") == "rtsp://10.0.0.5/s"
    with pytest.raises(test_ui_server.HTTPException) as exc:
        test_ui_server._resolve_live_source("rtsp://192.168.1.1/s")
    assert exc.value.status_code == 400


# ─── 2. site_id & dispatch ───────────────────────────────────────────────────

def test_live_start_rejects_client_site_id(client):
    res = client.post("/api/live/start", json={"source": "0", "site_id": "OTHER-SITE"})
    assert res.status_code == 422


def test_session_site_id_defaults_to_config_and_validates():
    s = LiveStreamSession(camera_id="CAM-1", source=0)
    assert s.site_id == le.config.SITE_CODE
    with pytest.raises(ValueError):
        LiveStreamSession(camera_id="CAM-1", source=0, site_id="../../other")
    with pytest.raises(ValueError):
        LiveStreamSession(camera_id="CAM/../x", source=0)


def test_test_ui_dispatch_disabled_by_default(client):
    assert test_ui_server.LIVE_DISPATCH_ENABLED is False
    with patch("app.live_engine.VideoReader"), patch("app.live_engine.Detector"):
        res = client.post("/api/live/start", json={"camera_id": "CAM-DISP", "source": "0"})
        assert res.status_code == 200
        assert le.live_engine_manager.get_session("CAM-DISP").dispatch_enabled is False


def test_dispatch_disabled_skips_minio_and_events():
    s = LiveStreamSession(camera_id="CAM-ND", source=0, dispatch_enabled=False)
    ev = MagicMock(event_id="e1", person_track_id=1, violation_type="NO_HELMET", camera_id="CAM-ND")
    evidence = MagicMock(status="saved", path="x.jpg", reference_path=None)
    with patch("app.live_engine.save_evidence", return_value=evidence), \
         patch("app.live_engine.dispatch_event") as mock_dispatch, \
         patch("app.storage.minio_client.upload_snapshot") as mock_upload:
        s._dispatch_evidence_async(None, [], ev, 640, 480)
    mock_dispatch.assert_not_called()
    mock_upload.assert_not_called()
    assert s.recent_violations[-1]["dispatched_via"] == "disabled"


# ─── 3. Resource limits ──────────────────────────────────────────────────────

def test_session_limit_enforced():
    mgr = LiveEngineManager(max_sessions=1)
    busy = MagicMock(status="running", source=0)
    mgr._sessions["CAM-A"] = busy
    with pytest.raises(LiveSessionLimitError):
        mgr.start_session(camera_id="CAM-B", source=0)


def test_api_session_limit_returns_429(client, monkeypatch):
    monkeypatch.setattr(le.live_engine_manager, "max_sessions", 1)
    le.live_engine_manager._sessions["CAM-BUSY"] = MagicMock(status="running", source=0)
    res = client.post("/api/live/start", json={"camera_id": "CAM-NEW", "source": "0"})
    assert res.status_code == 429


@pytest.mark.parametrize("payload", [
    {"source": "0", "target_fps": 1000},
    {"source": "0", "ai_fps": 0},
    {"source": "0", "confirmation_duration_s": 0},
    {"source": "0", "confirmation_duration_s": -5},
    {"source": "0", "camera_id": "CAM 01;rm"},
    {"source": "0", "roi_polygons": [[[0, 0], [1, 1]]]},
    {"source": "0", "roi_polygons": [[[i, i] for i in range(65)]]},
    {"source": "0", "roi_polygons": [[[0, 0], [1, 0], [1, 1]]] * 17},
    {"source": "0", "roi_polygons": [[[0, 0], [1, 0], [1e9, 1]]]},
    {"source": "x" * 3000},
])
def test_live_start_rejects_out_of_bounds_payloads(client, payload):
    res = client.post("/api/live/start", json=payload)
    assert res.status_code == 422, (payload, res.text)


def test_session_clamps_confirmation_duration():
    s = LiveStreamSession(camera_id="CAM-C", source=0, confirmation_duration_s=0)
    assert s.confirmation_duration_s >= 0.5


# ─── 4. DNS rebinding ────────────────────────────────────────────────────────

def test_untrusted_host_header_rejected(client):
    res = client.get("/api/live/sessions", headers={"Host": "attacker.example.com"})
    assert res.status_code == 400


def test_default_allowed_hosts_are_loopback_only():
    # conftest bổ sung "testserver"; mặc định production chỉ là loopback
    assert "127.0.0.1" in test_ui_server.TEST_UI_ALLOWED_HOSTS
    assert "*" not in test_ui_server.TEST_UI_ALLOWED_HOSTS

