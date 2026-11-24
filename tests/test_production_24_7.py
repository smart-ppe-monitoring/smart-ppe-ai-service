"""
Tests cho cơ chế Production Server 24/7, Dynamic Max Sessions, M2M Sync và Auto-Reconnect.
"""

from __future__ import annotations

import base64
import importlib
import json
from pathlib import Path
import time
from unittest.mock import MagicMock, patch

from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding, rsa
import httpx
import numpy as np
import pytest
from starlette.testclient import TestClient

from app import auth as auth_mod
from app import config
from app import live_engine as le
from app.events import flush_offline_buffer, http_publisher
import app.events.offline_buffer as offline_buffer_mod
from app.live_api import resolve_live_source
from app.live_engine import LiveEngineManager, LiveStreamSession, resolve_roi_polygons_for_frame
from app.live_sync import CameraSyncManager
from app.rules.deduplication import ViolationEvent as DedupEvent
from app.schemas.violation_event import (
    BoundingBox,
    Detection,
    Evidence,
    FrameResolution,
    ViolationEvent,
    ViolationItem,
)
from app.video.video_reader import VideoSourceError

prod_server = importlib.import_module("app.server")
prod_app = prod_server.app

_TEST_PRIVATE_KEY = rsa.generate_private_key(public_exponent=65537, key_size=2048)
_TEST_PUBLIC_PEM = _TEST_PRIVATE_KEY.public_key().public_bytes(
    encoding=serialization.Encoding.PEM,
    format=serialization.PublicFormat.SubjectPublicKeyInfo,
).decode("utf-8")


def _b64u(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii")


def _make_rs256_token(
    *,
    user_id: str = "0199a001-0000-7000-8000-999999999999",
    role: str = "ADMIN",
    permissions: list[str] | None = None,
    site_id: str | None = None,
    site_ids: list[str] | None = None,
    purpose: str | None = "ACCESS",
    must_change_password: bool = False,
    exp_offset: int = 3600,
) -> str:
    header = {"alg": "RS256", "typ": "JWT", "kid": "test-key-1"}
    payload: dict = {
        "sub": user_id,
        "id": user_id,
        "email": f"{role.lower()}@smart-ppe.test",
        "role": role,
        "exp": int(time.time()) + exp_offset,
        "mustChangePassword": must_change_password,
    }
    if permissions is not None:
        payload["permissions"] = permissions
    if site_id is not None:
        payload["siteId"] = site_id
    if site_ids is not None:
        payload["siteIds"] = site_ids
    if purpose is not None:
        payload["purpose"] = purpose

    h_b64 = _b64u(json.dumps(header, separators=(",", ":")).encode("utf-8"))
    p_b64 = _b64u(json.dumps(payload, separators=(",", ":")).encode("utf-8"))
    signing_input = f"{h_b64}.{p_b64}".encode("ascii")
    sig = _TEST_PRIVATE_KEY.sign(signing_input, padding.PKCS1v15(), hashes.SHA256())
    return f"{h_b64}.{p_b64}.{_b64u(sig)}"


def _auth_headers(**kwargs) -> dict[str, str]:
    return {"Authorization": f"Bearer {_make_rs256_token(**kwargs)}"}


@pytest.fixture
def prod_client():
    return TestClient(prod_app, base_url="http://localhost")


@pytest.fixture(autouse=True)
def _reset_live_state():
    original_max = le.live_engine_manager.get_max_sessions()
    original_pub = config.JWT_PUBLIC_KEY
    config.JWT_PUBLIC_KEY = _TEST_PUBLIC_PEM
    auth_mod.clear_auth_caches()
    yield
    le.live_engine_manager.stop_all()
    le.live_engine_manager.set_max_sessions(original_max)
    config.JWT_PUBLIC_KEY = original_pub
    auth_mod.clear_auth_caches()


def test_production_server_defaults_and_health(prod_client):
    assert prod_server.LIVE_DISPATCH_ENABLED is True
    res = prod_client.get("/health")
    assert res.status_code == 200
    body = res.json()
    assert body["status"] == "ok"
    assert body["mode"] == "production"
    assert body["dispatch_enabled"] is True
    assert body["max_sessions"] >= 16


def test_dynamic_max_sessions_config_and_scaling(prod_client):
    admin_hdrs = _auth_headers(role="ADMIN")
    res = prod_client.patch("/api/live/config", json={"max_sessions": 20}, headers=admin_hdrs)
    assert res.status_code == 200
    assert res.json()["max_sessions"] == 20
    assert le.live_engine_manager.get_max_sessions() == 20

    cfg_res = prod_client.get("/api/live/config", headers=admin_hdrs)
    assert cfg_res.status_code == 200
    assert cfg_res.json()["max_sessions"] == 20

    sess_res = prod_client.get("/api/live/sessions", headers=admin_hdrs)
    assert sess_res.status_code == 200
    assert sess_res.headers.get("X-Live-Max-Sessions") == "20"


def test_roi_normalization_dict_and_pixel_conversion():
    normalized_dict = [
        {"x": 0.1, "y": 0.2},
        {"x": 0.9, "y": 0.2},
        {"x": 0.9, "y": 0.8},
    ]
    norm_poly = le._normalize_roi_input(normalized_dict)
    pixels = resolve_roi_polygons_for_frame(norm_poly, frame_w=1000, frame_h=500)
    assert pixels == [[(100, 100), (900, 100), (900, 400)]]

    already_px = [[(50, 60), (200, 60), (200, 300)]]
    px_out = resolve_roi_polygons_for_frame(already_px, frame_w=1000, frame_h=500)
    assert px_out == [[(50, 60), (200, 60), (200, 300)]]


def test_camera_sync_manager_starts_more_than_4_cameras_with_dynamic_limit():
    mgr = LiveEngineManager(max_sessions=4)
    sync_mgr = CameraSyncManager(manager=mgr, dispatch_enabled=True, sync_interval_s=15.0)

    site_uuid = "0199a001-0000-7000-8000-0000000000aa"
    fake_cameras = [
        {
            "id": f"0199a001-0000-7000-8000-00000000000{i}",
            "siteId": site_uuid,
            "code": f"CAM-0{i}",
            "name": f"Camera {i}",
            "sourceType": "RTSP",
            "streamUrl": f"rtsp://admin:SecretPass{i}@192.168.1.{10 + i}:554/stream1",
            "status": "ONLINE",
            "config": {
                "enabled": True,
                "captureFps": 12,
                "processingEnabled": True,
                "roiPolygon": [{"x": 0.1, "y": 0.1}, {"x": 0.8, "y": 0.1}, {"x": 0.8, "y": 0.8}],
            },
        }
        for i in range(1, 7)  # 6 cameras (> old hardcoded limit of 4)
    ]

    with patch("app.live_sync.fetch_dynamic_live_max_sessions_sync", return_value=10), \
         patch("app.live_sync.fetch_internal_active_streams_sync", return_value=fake_cameras), \
         patch("app.live_sync.fetch_effective_rules_sync", return_value=None), \
         patch("app.live_engine.VideoReader"), \
         patch("app.live_engine.Detector"):
        status = sync_mgr.sync_once()

        assert mgr.get_max_sessions() == 10
        assert status["last_sync_ok"] is True
        assert status["last_synced_count"] == 6
        assert status["skipped_over_limit"] == []
        sessions = mgr.list_sessions()
        assert len(sessions) == 6
        # Raw RTSP credentials MUST remain masked in list_sessions output
        for s in sessions:
            assert s["managed"] is True
            assert s["dispatch_enabled"] is True
            assert "***" in s["source"]
            assert "SecretPass" not in s["source"]

        # Turn off CAM-06 (no longer returned as ONLINE), should stop CAM-06 cleanly
        mgr.start_session(camera_id="MANUAL-CAM", source="0", managed=False)
        with patch("app.live_sync.fetch_internal_active_streams_sync", return_value=fake_cameras[:5]), \
             patch("app.live_sync.fetch_effective_rules_sync", return_value=None):
            sync_mgr.sync_once()

        remaining_ids = {s["camera_id"] for s in mgr.list_sessions()}
        assert "CAM-06" not in remaining_ids
        assert "MANUAL-CAM" in remaining_ids
        assert len(remaining_ids) == 6
        mgr.stop_all()


def test_production_start_resolves_masked_rtsp_server_side(prod_client):
    site_uuid = "0199a001-0000-7000-8000-0000000000aa"
    cam_uuid = "0199a001-0000-7000-8000-000000000001"
    raw_url = "rtsp://admin:TopSecret99@10.10.0.5:554/cam/real"

    fake_cam = {
        "id": cam_uuid,
        "siteId": site_uuid,
        "code": "CAM-MASKED",
        "streamUrl": raw_url,
        "config": {"captureFps": 15, "roiPolygon": None},
    }

    with patch("app.server.lookup_internal_camera_sync", return_value=fake_cam) as mock_lookup, \
         patch("app.live_engine.VideoReader"), \
         patch("app.live_engine.Detector"):
        res = prod_client.post(
            "/api/live/start",
            headers=_auth_headers(role="SAFETY_MANAGER", site_id=site_uuid),
            json={
                "camera_id": "CAM-MASKED",
                "camera_uuid": cam_uuid,
                "source": "rtsp://admin:***@10.10.0.5:554/cam/real",
            },
        )
        assert res.status_code == 200
        body = res.json()
        assert body["camera_id"] == "CAM-MASKED"
        assert body["site_id"] == site_uuid
        assert body["dispatch_enabled"] is True
        mock_lookup.assert_called_once_with(cam_uuid, "CAM-MASKED")

        sess = le.live_engine_manager.get_session("CAM-MASKED")
        assert sess is not None
        assert sess.source == raw_url
        assert sess.auto_reconnect is True
        assert sess.loop_on_eof is True
        le.live_engine_manager.stop_all()


def test_live_session_auto_reconnect_and_loop_on_eof(monkeypatch):
    monkeypatch.setattr(le, "RECONNECT_BASE_DELAY_S", 0.05)
    monkeypatch.setattr(le, "RECONNECT_MAX_DELAY_S", 0.05)

    fake_frame = np.zeros((120, 160, 3), dtype=np.uint8)
    attempts = {"count": 0}

    def fake_reader_factory(src):
        attempts["count"] += 1
        if attempts["count"] == 1:
            raise VideoSourceError("Transient RTSP connection error")
        vr = MagicMock()
        vr.is_stream = True
        vr.fps = 25.0
        vr.original_source = src
        vr.__enter__.return_value = vr
        vr.__iter__.return_value = [(1, fake_frame), (2, fake_frame)]
        return vr

    with patch("app.live_engine.VideoReader", side_effect=fake_reader_factory), \
         patch("app.live_engine.Detector") as mock_det_cls:
        mock_det = MagicMock()
        mock_det.load.return_value = mock_det
        mock_det.detect.return_value = []
        mock_det_cls.return_value = mock_det

        sess = LiveStreamSession(
            camera_id="CAM-RETRY",
            source="rtsp://admin:pass@10.0.0.2/live",
            auto_reconnect=True,
            loop_on_eof=False,
            dispatch_enabled=False,
        )
        sess.start()

        import time
        for _ in range(40):
            if sess.frame_number >= 2 and sess.reconnect_count >= 1:
                break
            time.sleep(0.05)

        assert sess.reconnect_count >= 1
        assert sess.frame_number >= 2
        sess.stop()
        assert sess.status == "stopped"


def test_dispatch_payload_includes_uuid_object_key_and_internal_headers(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "EVIDENCE_DIR", str(tmp_path))
    monkeypatch.setattr(config, "INTERNAL_SERVICE_TOKEN", "m2m_test_token_12345")
    monkeypatch.setattr(config, "GATEWAY_INTERNAL_SECRET", "gw_test_secret_12345")
    monkeypatch.setattr(config, "KAFKA_ENABLED", False)

    headers = http_publisher._build_internal_headers()
    assert headers["x-internal-service-token"] == "m2m_test_token_12345"
    assert headers["Authorization"] == "Bearer m2m_test_token_12345"
    assert headers["x-gateway-secret"] == "gw_test_secret_12345"

    site_uuid = "0199a001-0000-7000-8000-0000000000aa"
    cam_uuid = "0199a001-0000-7000-8000-000000000001"
    session = LiveStreamSession(
        camera_id="CAM-DISP1",
        camera_uuid=cam_uuid,
        site_id=site_uuid,
        source="rtsp://u:p@host/s",
        dispatch_enabled=True,
    )

    frame_viol = np.full((120, 160, 3), 80, dtype=np.uint8)
    ev = DedupEvent(
        event_id="abcd1234ef56",
        camera_id="CAM-DISP1",
        person_track_id=5,
        canonical_person_id=5,
        violation_type="NO_HELMET",
        start_t=1.0,
        confirmed_t=3.5,
        duration_s=2.5,
    )

    minio_url = f"http://localhost:9000/ppe-evidences/evidences/{site_uuid}/CAM-DISP1/123_track5_abcd1234ef56.jpg"
    captured = {}

    def fake_dispatch(contract_event, sync=False):
        captured["event"] = contract_event.model_dump(by_alias=True)
        captured["sync"] = sync
        return "http"

    with patch("app.storage.minio_client.upload_snapshot", return_value=minio_url), \
         patch("app.live_engine.dispatch_event", side_effect=fake_dispatch):
        session._dispatch_evidence_async(frame_viol, [], ev, 160, 120, frame_num=50)

    assert captured["sync"] is True
    payload = captured["event"]
    assert payload["eventId"] == "abcd1234ef56"
    assert payload["siteId"] == site_uuid
    assert payload["cameraId"] == cam_uuid
    assert payload["evidence"]["objectKey"] == f"evidences/{site_uuid}/CAM-DISP1/123_track5_abcd1234ef56.jpg"
    assert len(payload["evidence"]["checksum"]) == 64


def test_trusted_host_rejects_untrusted_host_header(prod_client):
    res = prod_client.get("/health", headers={"Host": "evil.attacker.com"})
    assert res.status_code == 400


def test_production_auth_cookie_bearer_rbac_and_query_token_rejection(prod_client):
    assert prod_client.get("/api/live/sessions").status_code == 401
    assert prod_client.get("/api/live/config").status_code == 401
    assert prod_client.get("/api/models").status_code == 401
    assert prod_client.post("/api/live/start", json={"camera_id": "CAM-1"}).status_code == 401

    # CWE-598: reject query string token/ticket even if header missing
    assert prod_client.get("/api/live/sessions?token=abc").status_code == 400
    assert prod_client.get("/api/live/sessions?ticket=xyz").status_code == 400

    # Reject 2FA temp token & mustChangePassword
    temp_hdr = _auth_headers(role="ADMIN", purpose="2FA_VERIFICATION")
    assert prod_client.get("/api/live/sessions", headers=temp_hdr).status_code == 401
    mcp_hdr = _auth_headers(role="ADMIN", must_change_password=True)
    assert prod_client.get("/api/live/sessions", headers=mcp_hdr).status_code == 403

    # Reject alg=none
    none_hdr = _b64u(b'{"alg":"none","typ":"JWT"}')
    none_pay = _b64u(json.dumps({"sub": "1", "role": "ADMIN", "exp": int(time.time()) + 3600}).encode())
    assert prod_client.get("/api/live/sessions", headers={"Authorization": f"Bearer {none_hdr}.{none_pay}.x"}).status_code == 401

    # HttpOnly cookie works for SAFETY_OFFICER read, but mutate is 403
    officer_tok = _make_rs256_token(role="SAFETY_OFFICER", site_id="0199a001-0000-7000-8000-0000000000aa")
    prod_client.cookies.set("accessToken", officer_tok)
    try:
        assert prod_client.get("/api/live/sessions").status_code == 200
        assert prod_client.post("/api/live/start", json={"camera_id": "CAM-1", "source": "0"}).status_code == 403
        assert prod_client.patch("/api/live/config", json={"max_sessions": 10}).status_code == 403
    finally:
        prod_client.cookies.clear()

    # SAFETY_MANAGER cannot change global max_sessions
    mgr_hdr = _auth_headers(role="SAFETY_MANAGER", site_id="0199a001-0000-7000-8000-0000000000aa")
    assert prod_client.patch("/api/live/config", json={"max_sessions": 10}, headers=mgr_hdr).status_code == 403


def test_production_site_isolation_bola_and_evidence_access(prod_client, tmp_path, monkeypatch):
    monkeypatch.setattr(config, "EVIDENCE_DIR", str(tmp_path))
    site_a = "0199a001-0000-7000-8000-0000000000aa"
    site_b = "0199a001-0000-7000-8000-0000000000bb"

    with patch("app.live_engine.VideoReader"), patch("app.live_engine.Detector"):
        try:
            le.live_engine_manager.start_session(
                camera_id="CAM-SITE-A",
                site_id=site_a,
                source="rtsp://admin:pw@10.0.0.10/s1",
                dispatch_enabled=True,
                managed=True,
            )

            mgr_a = _auth_headers(role="SAFETY_MANAGER", site_id=site_a)
            mgr_b = _auth_headers(role="SAFETY_MANAGER", site_id=site_b)

            # Site A manager sees CAM-SITE-A; Site B manager does not
            list_a = prod_client.get("/api/live/sessions", headers=mgr_a).json()
            list_b = prod_client.get("/api/live/sessions", headers=mgr_b).json()
            assert any(s["camera_id"] == "CAM-SITE-A" for s in list_a)
            assert not any(s["camera_id"] == "CAM-SITE-A" for s in list_b)

            # Site B manager gets 403 on stats, config, stop for Site A's camera
            assert prod_client.get("/api/live/CAM-SITE-A/stats", headers=mgr_b).status_code == 403
            assert prod_client.patch("/api/live/CAM-SITE-A/config", json={"ai_fps": 5}, headers=mgr_b).status_code == 403
            assert prod_client.post("/api/live/CAM-SITE-A/stop", headers=mgr_b).status_code == 403

            # Site B manager cannot start M2M camera belonging to Site A
            fake_cam_a = {
                "id": "0199a001-0000-7000-8000-000000000010",
                "siteId": site_a,
                "code": "CAM-SITE-A",
                "streamUrl": "rtsp://admin:pw@10.0.0.10/s1",
            }
            with patch("app.server.lookup_internal_camera_sync", return_value=fake_cam_a):
                assert (
                    prod_client.post(
                        "/api/live/start",
                        headers=mgr_b,
                        json={"camera_id": "CAM-SITE-A"},
                    ).status_code
                    == 403
                )
        finally:
            le.live_engine_manager.stop_all()

    # Evidence file isolation by site
    ev_name = "0123456789ab.jpg"
    (tmp_path / ev_name).write_bytes(b"\xff\xd8\xff\xe0test-jpeg")
    le.register_evidence_owner(ev_name, site_id=site_a, managed=True)

    assert prod_client.get(f"/api/live/evidence/{ev_name}", headers=mgr_b).status_code == 403
    assert prod_client.get(f"/api/live/evidence/{ev_name}", headers=mgr_a).status_code == 200


def test_override_source_forces_dispatch_disabled_and_blocks_ssrf(prod_client):
    site_a = "0199a001-0000-7000-8000-0000000000aa"
    mgr_a = _auth_headers(role="SAFETY_MANAGER", site_id=site_a)

    with patch("app.live_engine.VideoReader"), patch("app.live_engine.Detector"):
        try:
            res = prod_client.post(
                "/api/live/start",
                headers=mgr_a,
                json={
                    "camera_id": "CAM-TEST",
                    "camera_uuid": "0199a001-0000-7000-8000-000000000099",
                    "source": "rtsp://cam:pw@10.20.30.40:554/live",
                },
            )
            assert res.status_code == 200
            body = res.json()
            assert body["dispatch_enabled"] is False
            assert body["camera_uuid"] is None
        finally:
            le.live_engine_manager.stop_all()

    # SSRF internal/loopback/metadata hosts must be blocked
    for bad_url in (
        "rtsp://127.0.0.1:554/live",
        "rtsp://localhost:8554/cam",
        "rtsp://169.254.169.254/latest/meta-data",
        "rtsp://identity-service:9999/internal",
    ):
        with pytest.raises(Exception) as exc_info:
            resolve_live_source(bad_url, upload_dir=Path("."))
        assert getattr(exc_info.value, "status_code", 400) == 400


def test_dispatch_evidence_draws_pixel_roi_coordinates(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "EVIDENCE_DIR", str(tmp_path))
    session = LiveStreamSession(
        camera_id="CAM-ROI",
        site_id="0199a001-0000-7000-8000-0000000000aa",
        source="0",
        roi_polygons=[[(0.1, 0.2), (0.5, 0.2), (0.5, 0.8)]],
        dispatch_enabled=False,
    )
    frame_viol = np.zeros((100, 200, 3), dtype=np.uint8)
    ev = DedupEvent(
        event_id="deadbeef1234",
        camera_id="CAM-ROI",
        person_track_id=1,
        canonical_person_id=1,
        violation_type="NO_HELMET",
        start_t=0.0,
        confirmed_t=2.0,
        duration_s=2.0,
    )
    drawn = {}

    def fake_draw(img, dets, person_states=None, roi_polygons=None, **_kwargs):
        drawn["roi"] = roi_polygons
        return img

    with patch("app.live_engine.draw_detections", side_effect=fake_draw):
        session._dispatch_evidence_async(frame_viol, [], ev, 200, 100, frame_num=10)

    assert drawn["roi"] == [[(20, 20), (100, 20), (100, 80)]]


def test_offline_buffer_preserves_events_on_transient_backend_outage(tmp_path):
    test_db = tmp_path / "transient_outage.db"
    event = ViolationEvent(
        cameraCode="CAM-OUTAGE",
        timestamp="2026-10-13T10:00:00.000Z",
        evidence=Evidence(
            snapshotUrl="http://localhost:9000/ppe-evidences/test.jpg",
            frameResolution=FrameResolution(width=1920, height=1080),
        ),
        detections=[
            Detection(
                classLabel="person",
                confidence=0.95,
                box=BoundingBox(x=10, y=10, width=50, height=100),
                trackId=1,
            )
        ],
        violations=[
            ViolationItem(
                violationType="NO_HELMET",
                severity="HIGH",
                description="Missing helmet",
                confidence=0.9,
            )
        ],
    )
    with patch.object(offline_buffer_mod, "DB_PATH", test_db):
        offline_buffer_mod.save_event(event)
        with patch(
            "app.events.publish_batch_sync",
            side_effect=httpx.ConnectError("Backend restarting"),
        ):
            for _ in range(5):
                assert flush_offline_buffer(sync=True) == 0

        # Still pending after 5 transient failures (retries not burned!)
        assert offline_buffer_mod.pending_count() == 1


def test_sync_once_trims_excess_managed_sessions_when_limit_shrinks():
    mgr = LiveEngineManager(max_sessions=4)
    sync_mgr = CameraSyncManager(manager=mgr, dispatch_enabled=False, sync_interval_s=15.0)
    site_uuid = "0199a001-0000-7000-8000-0000000000aa"
    fake_cameras = [
        {
            "id": f"0199a001-0000-7000-8000-00000000000{i}",
            "siteId": site_uuid,
            "code": f"CAM-TRIM-{i}",
            "streamUrl": f"rtsp://u:p@10.0.0.1{i}/s",
            "config": {"captureFps": 10},
        }
        for i in range(1, 5)
    ]
    with patch("app.live_engine.VideoReader"), patch("app.live_engine.Detector"):
        try:
            with patch("app.live_sync.fetch_dynamic_live_max_sessions_sync", return_value=4), \
                 patch("app.live_sync.fetch_internal_active_streams_sync", return_value=fake_cameras), \
                 patch("app.live_sync.fetch_effective_rules_sync", return_value=None), \
                 patch("app.live_sync.pending_count", return_value=0):
                st1 = sync_mgr.sync_once()
                assert st1["last_synced_count"] == 4

            with patch("app.live_sync.fetch_dynamic_live_max_sessions_sync", return_value=2), \
                 patch("app.live_sync.fetch_internal_active_streams_sync", return_value=fake_cameras), \
                 patch("app.live_sync.fetch_effective_rules_sync", return_value=None), \
                 patch("app.live_sync.pending_count", return_value=0):
                st2 = sync_mgr.sync_once()
                assert st2["max_sessions"] == 2
                assert len(mgr.list_sessions()) == 2
                assert len(st2["skipped_over_limit"]) == 2
        finally:
            mgr.stop_all()


def test_live_dispatch_default_ignores_test_ui_env(monkeypatch):
    import runpy

    config_path = str(Path(config.__file__).resolve())
    with patch("dotenv.load_dotenv"):
        monkeypatch.delenv("LIVE_DISPATCH_ENABLED", raising=False)
        monkeypatch.setenv("TEST_UI_LIVE_DISPATCH", "false")
        ns_default = runpy.run_path(config_path)
        assert ns_default["LIVE_DISPATCH_DEFAULT"] is True

        monkeypatch.setenv("LIVE_DISPATCH_ENABLED", "")
        ns_empty = runpy.run_path(config_path)
        assert ns_empty["LIVE_DISPATCH_DEFAULT"] is True

        monkeypatch.setenv("LIVE_DISPATCH_ENABLED", "false")
        ns_disabled = runpy.run_path(config_path)
        assert ns_disabled["LIVE_DISPATCH_DEFAULT"] is False


def test_sync_once_applies_dynamic_safety_rules_and_preserves_on_outage():
    from app.clients.backend_client import SafetyRulesConfig

    mgr = LiveEngineManager(max_sessions=4)
    sync_mgr = CameraSyncManager(manager=mgr, dispatch_enabled=False, sync_interval_s=15.0)
    site_uuid = "0199a001-0000-7000-8000-0000000000aa"
    cam_uuid = "0199a001-0000-7000-8000-0000000000bb"
    fake_cameras = [
        {
            "id": cam_uuid,
            "siteId": site_uuid,
            "code": "CAM-RULES-1",
            "streamUrl": "rtsp://u:p@10.0.0.50/s",
            "config": {"captureFps": 10},
        }
    ]
    custom_rules = SafetyRulesConfig(
        confidence=0.65,
        confirmation_s=10.0,
        cooldown_s=120,
        enable_no_helmet=True,
        enable_no_vest=False,
        enable_restricted_area=False,
    )

    with patch("app.live_engine.VideoReader"), patch("app.live_engine.Detector"):
        try:
            with patch("app.live_sync.fetch_dynamic_live_max_sessions_sync", return_value=4), \
                 patch("app.live_sync.fetch_internal_active_streams_sync", return_value=fake_cameras), \
                 patch("app.live_sync.fetch_effective_rules_sync", return_value=custom_rules), \
                 patch("app.live_sync.pending_count", return_value=0):
                sync_mgr.sync_once()

            sess = mgr.get_session("CAM-RULES-1")
            assert sess is not None
            assert sess.enable_no_helmet is True
            assert sess.enable_no_vest is False
            assert sess.enable_restricted_area is False
            assert sess.confidence_threshold == 0.65
            assert sess.confirmation_duration_s == 10.0
            assert sess.confirmation_duration_vest_s == 10.0
            assert sess.cooldown_s == 120.0

            # Site Service temporarily unreachable (returns None) -> MUST keep existing config, NOT overwrite with defaults
            with patch("app.live_sync.fetch_dynamic_live_max_sessions_sync", return_value=4), \
                 patch("app.live_sync.fetch_internal_active_streams_sync", return_value=fake_cameras), \
                 patch("app.live_sync.fetch_effective_rules_sync", return_value=None), \
                 patch("app.live_sync.pending_count", return_value=0):
                sync_mgr.sync_once()

            sess_after = mgr.get_session("CAM-RULES-1")
            assert sess_after is not None
            assert sess_after.enable_no_vest is False
            assert sess_after.enable_restricted_area is False
            assert sess_after.confidence_threshold == 0.65
            assert sess_after.confirmation_duration_s == 10.0
            assert sess_after.cooldown_s == 120.0
        finally:
            mgr.stop_all()


def test_loop_on_eof_uses_monotonic_timestamps_and_suppresses_duplicate_spam():
    from types import SimpleNamespace
    from app.rules.deduplication import ViolationDeduplicator
    from app.constants import VIOLATION_NO_HELMET
    from app.detection.adapter import PERSON

    # 1. Verify ViolationDeduplicator camera-level cooldown across EOF loops & track ID changes
    dedup = ViolationDeduplicator(camera_id="CAM-LOOP", cooldown_s=120.0, camera_level_cooldown=False)
    confirmed = [
        SimpleNamespace(
            person_track_id=1,
            violation_type=VIOLATION_NO_HELMET,
            start_t=0.0,
            confirmed_t=2.0,
            duration_s=2.0,
        )
    ]
    det_loop1 = [SimpleNamespace(class_name=PERSON, track_id=1, bbox=(10, 10, 80, 200))]
    ev1 = dedup.filter(confirmed, det_loop1, t=2.0)
    assert len(ev1) == 1

    # Simulate EOF loop restart: new track_id=99 at different bbox, time advanced monotonically to 14.0s
    dedup.enable_camera_level_cooldown()
    confirmed_loop2 = [
        SimpleNamespace(
            person_track_id=99,
            violation_type=VIOLATION_NO_HELMET,
            start_t=12.0,
            confirmed_t=14.0,
            duration_s=2.0,
        )
    ]
    det_loop2 = [SimpleNamespace(class_name=PERSON, track_id=99, bbox=(400, 10, 470, 200))]
    ev2 = dedup.filter(confirmed_loop2, det_loop2, t=14.0)
    assert len(ev2) == 0, "Duplicate violation within cooldown across EOF loop must be suppressed"

    # After cooldown expires (t=125.0s > 2.0 + 120.0), alert is allowed again
    ev3 = dedup.filter(confirmed_loop2, det_loop2, t=125.0)
    assert len(ev3) == 1

    # 2. Verify LiveStreamSession._run_loop preserves monotonic t_frame_s across VideoReader EOF restarts
    class FakeLoopReader:
        calls = 0

        def __init__(self, _src):
            FakeLoopReader.calls += 1
            self.run_idx = FakeLoopReader.calls
            self.fps = 10.0
            self.original_source = "sample.mp4"

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def release(self):
            pass

        def __iter__(self):
            for idx in (1, 2, 3):
                yield idx, np.zeros((100, 100, 3), dtype=np.uint8)

    recorded_t_values: list[float] = []
    sess = LiveStreamSession(
        camera_id="CAM-MONO",
        source="sample.mp4",
        target_fps=10.0,
        ai_fps=10.0,
        cooldown_s=120.0,
        enable_scene_cut=False,
        dispatch_enabled=False,
        loop_on_eof=True,
    )

    def fake_step(frame_num, frame, t_frame_s, detector, confirmation, dedup_obj, ai_fps_window):
        recorded_t_values.append(t_frame_s)
        if len(recorded_t_values) >= 6:
            sess._stop_requested.set()

    with patch("app.live_engine.VideoReader", FakeLoopReader), \
         patch("app.live_engine.Detector"), \
         patch.object(sess, "_process_ai_step", side_effect=fake_step):
        sess._run_loop()

    assert len(recorded_t_values) == 6
    assert recorded_t_values[:3] == [0.0, 0.1, 0.2]
    assert all(recorded_t_values[i] > recorded_t_values[i - 1] for i in range(1, len(recorded_t_values)))


def test_sync_once_stops_orphaned_unmanaged_session_when_deleted_from_db():
    mgr = LiveEngineManager(max_sessions=6)
    sync_mgr = CameraSyncManager(manager=mgr, dispatch_enabled=False, sync_interval_s=15.0)
    deleted_uuid = "0199a001-0000-7000-8000-0000000000dd"
    network_uuid = "0199a001-0000-7000-8000-0000000000ee"

    with patch("app.live_engine.VideoReader"), patch("app.live_engine.Detector"):
        try:
            # 1. Session started manually (managed=False) linked to a DB camera via _sync_camera_uuid
            orphan = mgr.start_session(
                camera_id="CAM-DELETED",
                source="0",
                managed=False,
                loop_on_eof=True,
            )
            orphan._sync_camera_uuid = deleted_uuid

            # 2. Session whose DB check hits transient network error (returns None) -> MUST NOT stop
            transient = mgr.start_session(
                camera_id="CAM-TRANSIENT",
                source="0",
                managed=False,
                loop_on_eof=True,
            )
            transient._sync_camera_uuid = network_uuid

            # 3. Pure sandbox session with no DB camera link -> MUST NOT check or stop
            mgr.start_session(
                camera_id="SANDBOX-LOCAL",
                source="0",
                managed=False,
                loop_on_eof=True,
            )

            def fake_exists(camera_id=None, code=None):
                if camera_id == deleted_uuid:
                    return False
                if camera_id == network_uuid:
                    return None
                return True

            with patch("app.live_sync.fetch_dynamic_live_max_sessions_sync", return_value=6), \
                 patch("app.live_sync.fetch_internal_active_streams_sync", return_value=[]), \
                 patch("app.live_sync.check_internal_camera_exists_sync", side_effect=fake_exists) as mock_exists, \
                 patch("app.live_sync.pending_count", return_value=0):
                sync_mgr.sync_once()

            remaining_ids = {s["camera_id"] for s in mgr.list_sessions()}
            assert "CAM-DELETED" not in remaining_ids
            assert "CAM-TRANSIENT" in remaining_ids
            assert "SANDBOX-LOCAL" in remaining_ids
            assert mock_exists.call_count == 2
        finally:
            mgr.stop_all()
