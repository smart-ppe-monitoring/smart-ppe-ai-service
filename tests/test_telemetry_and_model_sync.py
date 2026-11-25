"""
Tests cho cơ chế đẩy Telemetry AI (p95LatencyMs, e2eAlertLatencyMs) định kỳ sang Reporting Service
và nhận/áp dụng sự kiện deploy model qua POST /api/live/sync trên smart-ppe-ai-service.
"""

from __future__ import annotations

import base64
from http.server import BaseHTTPRequestHandler, HTTPServer
import importlib
import json
from pathlib import Path
import threading
import time
from unittest.mock import MagicMock, patch

from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding, rsa
import numpy as np
import pytest
from starlette.testclient import TestClient

from app import auth as auth_mod
from app import config
from app import live_engine as le
from app.clients.backend_client import (
    fetch_internal_active_model_version_sync,
    push_ai_performance_telemetry_sync,
)
from app.constants import (
    EVENT_AI_MODEL_DEPLOYED,
    HTTP_REPORTING_INTERNAL_ACTIVE_MODEL,
    HTTP_REPORTING_INTERNAL_TELEMETRY,
)
from app.live_engine import LiveEngineManager, LiveStreamSession
from app.live_sync import CameraSyncManager, _resolve_safe_local_model_path
from app.rules.confirmation import ViolationConfirmation
from app.rules.deduplication import ViolationDeduplicator, ViolationEvent as DedupEvent

prod_server = importlib.import_module("app.server")
prod_app = prod_server.app

_TEST_PRIVATE_KEY = rsa.generate_private_key(public_exponent=65537, key_size=2048)
_TEST_PUBLIC_PEM = _TEST_PRIVATE_KEY.public_key().public_bytes(
    encoding=serialization.Encoding.PEM,
    format=serialization.PublicFormat.SubjectPublicKeyInfo,
).decode("utf-8")


def _b64u(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii")


def _make_admin_rs256_token() -> str:
    header = {"alg": "RS256", "typ": "JWT", "kid": "telemetry-test-key"}
    payload = {
        "sub": "0199a001-0000-7000-8000-999999999999",
        "id": "0199a001-0000-7000-8000-999999999999",
        "email": "admin@smart-ppe.test",
        "role": "ADMIN",
        "exp": int(time.time()) + 3600,
        "mustChangePassword": False,
        "purpose": "ACCESS",
    }
    h_b64 = _b64u(json.dumps(header, separators=(",", ":")).encode("utf-8"))
    p_b64 = _b64u(json.dumps(payload, separators=(",", ":")).encode("utf-8"))
    sig = _TEST_PRIVATE_KEY.sign(
        f"{h_b64}.{p_b64}".encode("ascii"),
        padding.PKCS1v15(),
        hashes.SHA256(),
    )
    return f"{h_b64}.{p_b64}.{_b64u(sig)}"


def _start_mock_reporting_server():
    captured_requests: list[dict] = []
    server_state = {
        "telemetry_status": 201,
        "active_model_status": 200,
        "active_model_payload": {
            "id": "0199b001-0000-7000-8000-000000000777",
            "version": "2.4.0",
            "modelPath": "models/yolo11m-best.pt",
        },
    }

    class _Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(length) if length > 0 else b""
            body = json.loads(raw.decode("utf-8")) if raw else None
            captured_requests.append({
                "method": "POST",
                "path": self.path,
                "headers": {k.lower(): v for k, v in self.headers.items()},
                "body": body,
            })
            status = server_state["telemetry_status"]
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"success": 200 <= status < 300, "data": body}).encode("utf-8"))

        def do_GET(self):
            captured_requests.append({
                "method": "GET",
                "path": self.path,
                "headers": {k.lower(): v for k, v in self.headers.items()},
                "body": None,
            })
            status = server_state["active_model_status"]
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            if status == 200:
                resp = {"success": True, "data": server_state["active_model_payload"]}
            else:
                resp = {"success": False, "message": "Not found"}
            self.wfile.write(json.dumps(resp).encode("utf-8"))

        def log_message(self, format, *args):
            return

    httpd = HTTPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    port = httpd.server_address[1]
    return httpd, f"http://127.0.0.1:{port}", captured_requests, server_state


@pytest.fixture(autouse=True)
def _isolate_env():
    orig_reporting_url = config.REPORTING_SERVICE_URL
    orig_token = config.INTERNAL_SERVICE_TOKEN
    orig_gw = config.GATEWAY_INTERNAL_SECRET
    orig_pub = config.JWT_PUBLIC_KEY
    orig_max = le.live_engine_manager.get_max_sessions()

    sync_mgr = prod_server.camera_sync_manager
    orig_mv_id = sync_mgr.active_model_version_id
    orig_mv = sync_mgr.active_model_version
    orig_mpath = sync_mgr.active_model_path
    orig_expected_path = sync_mgr.active_model_expected_path
    orig_sync_status = sync_mgr.active_model_sync_status
    orig_sync_msg = sync_mgr.active_model_sync_message

    yield

    le.live_engine_manager.stop_all()
    le.live_engine_manager.set_max_sessions(orig_max)
    config.REPORTING_SERVICE_URL = orig_reporting_url
    config.INTERNAL_SERVICE_TOKEN = orig_token
    config.GATEWAY_INTERNAL_SECRET = orig_gw
    config.JWT_PUBLIC_KEY = orig_pub
    sync_mgr.active_model_version_id = orig_mv_id
    sync_mgr.active_model_version = orig_mv
    sync_mgr.active_model_path = orig_mpath
    sync_mgr.active_model_expected_path = orig_expected_path
    sync_mgr.active_model_sync_status = orig_sync_status
    sync_mgr.active_model_sync_message = orig_sync_msg
    auth_mod.clear_auth_caches()


def test_session_computes_p95_e2e_telemetry_and_restores_on_rollback():
    cam_uuid = "0199a001-0000-7000-8000-000000000101"
    mv_uuid = "0199b001-0000-7000-8000-000000000202"
    sess = LiveStreamSession(
        camera_id="CAM-TELEM-01",
        camera_uuid=cam_uuid,
        source="0",
        model_version_id=mv_uuid,
        dispatch_enabled=False,
    )

    dummy_detector = MagicMock()
    dummy_detector.detect.return_value = []
    confirmation = ViolationConfirmation(duration_s=2.0)
    dedup = ViolationDeduplicator(camera_id="CAM-TELEM-01", cooldown_s=30.0)
    frame = np.zeros((120, 160, 3), dtype=np.uint8)
    ai_window: list[float] = []

    lat_sequence_s = iter([
        # Each _process_ai_step calls time.perf_counter() 4 times: infer_start, infer_end, rule_start, rule_end
        1.000, 1.010, 1.010, 1.011,  # 10ms infer
        2.000, 2.020, 2.020, 2.021,  # 20ms infer
        3.000, 3.030, 3.030, 3.031,  # 30ms infer
        4.000, 4.040, 4.040, 4.041,  # 40ms infer
        5.000, 5.100, 5.100, 5.101,  # 100ms infer
    ])
    with patch("app.live_engine.time.perf_counter", side_effect=lambda: next(lat_sequence_s)):
        for idx in range(5):
            sess._process_ai_step(
                frame_num=idx + 1,
                frame=frame,
                t_frame_s=float(idx) * 0.2,
                detector=dummy_detector,
                confirmation=confirmation,
                dedup=dedup,
                ai_fps_window=ai_window,
            )

    dummy_ev = DedupEvent(
        event_id="evt-telemetry-1",
        camera_id="CAM-TELEM-01",
        person_track_id=1,
        canonical_person_id=1,
        violation_type="NO_HELMET",
        start_t=0.0,
        confirmed_t=2.0,
        duration_s=2.0,
    )
    sess._record_violation(ev=dummy_ev, snapshot_url=None, method="disabled", e2e_latency=210.0)
    sess._record_violation(ev=dummy_ev, snapshot_url=None, method="disabled", e2e_latency=390.0)

    assert sess.metrics["p95_latency_ms"] >= 80.0
    assert sess.metrics["e2e_alert_latency_ms"] == 390.0

    drained = sess.drain_telemetry_batch()
    assert drained is not None
    payload, rollback = drained

    assert payload["cameraId"] == cam_uuid
    assert payload["modelVersionId"] == mv_uuid
    assert payload["frameCount"] == 5
    assert 0.01 <= payload["fps"] <= 240.0
    assert payload["latencyMs"] == pytest.approx(40.0, abs=1.5)
    assert payload["p95LatencyMs"] == pytest.approx(88.0, abs=1.5)
    assert payload["e2eAlertLatencyMs"] == 300.0

    # Buffer is empty immediately after drain
    assert sess.drain_telemetry_batch() is None

    # Restore on simulated network failure and re-drain cleanly
    sess.restore_telemetry_batch(rollback)
    re_drained = sess.drain_telemetry_batch()
    assert re_drained is not None
    assert re_drained[0]["frameCount"] == 5
    assert re_drained[0]["e2eAlertLatencyMs"] == 300.0

    # Session without valid UUID never sends invalid cameraId payload
    non_uuid_sess = LiveStreamSession(camera_id="CAM-LOCAL", source="0", dispatch_enabled=False)
    non_uuid_sess._telemetry_frames_delta = 3
    assert non_uuid_sess.drain_telemetry_batch() is None


def test_backend_client_reporting_telemetry_and_active_model_http_roundtrip():
    httpd, url, captured, state = _start_mock_reporting_server()
    try:
        config.REPORTING_SERVICE_URL = url
        config.INTERNAL_SERVICE_TOKEN = "m2m-secret-token-for-pytest-32chr!"
        config.GATEWAY_INTERNAL_SECRET = "gw-secret-for-pytest"

        ok = push_ai_performance_telemetry_sync({
            "cameraId": "0199a001-0000-7000-8000-000000000333",
            "modelVersionId": "0199b001-0000-7000-8000-000000000444",
            "fps": 27.5,
            "latencyMs": 33.4,
            "p95LatencyMs": 58.2,
            "e2eAlertLatencyMs": 345.6,
            "frameCount": 120,
            "statDate": None,
        })
        assert ok is True
        assert len(captured) == 1
        post_req = captured[0]
        assert post_req["method"] == "POST"
        assert post_req["path"] == HTTP_REPORTING_INTERNAL_TELEMETRY
        assert post_req["headers"]["x-internal-service-token"] == "m2m-secret-token-for-pytest-32chr!"
        assert post_req["headers"]["authorization"] == "Bearer m2m-secret-token-for-pytest-32chr!"
        assert post_req["headers"]["x-gateway-secret"] == "gw-secret-for-pytest"
        assert "statDate" not in post_req["body"]
        assert post_req["body"]["p95LatencyMs"] == 58.2
        assert post_req["body"]["e2eAlertLatencyMs"] == 345.6

        # HTTP 500 returns False without raising
        state["telemetry_status"] = 500
        assert push_ai_performance_telemetry_sync({
            "cameraId": "0199a001-0000-7000-8000-000000000333",
            "fps": 25.0,
            "latencyMs": 30.0,
            "frameCount": 10,
        }) is False

        # Active model fetch
        active = fetch_internal_active_model_version_sync()
        assert active is not None
        assert active["id"] == "0199b001-0000-7000-8000-000000000777"
        assert captured[-1]["path"].startswith(HTTP_REPORTING_INTERNAL_ACTIVE_MODEL)
    finally:
        httpd.shutdown()
        httpd.server_close()


def test_camera_sync_manager_flush_telemetry_rollback_and_model_reload(tmp_path: Path):
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    v1_path = models_dir / "yolo-v1.pt"
    v2_path = models_dir / "yolo-v2.pt"
    v1_path.write_bytes(b"pt-v1-dummy")
    v2_path.write_bytes(b"pt-v2-dummy")

    # Security check for weight path resolution
    assert _resolve_safe_local_model_path("models/yolo-v2.pt", models_dir) == str(v2_path.resolve())
    assert _resolve_safe_local_model_path("../../etc/passwd.pt", models_dir) is None
    assert _resolve_safe_local_model_path("https://evil.com/yolo.pt", models_dir) is None
    assert _resolve_safe_local_model_path("models/missing.pt", models_dir) is None
    assert _resolve_safe_local_model_path("models/script.py", models_dir) is None

    mgr = LiveEngineManager(max_sessions=4)
    sync_mgr = CameraSyncManager(
        manager=mgr,
        dispatch_enabled=False,
        sync_interval_s=15.0,
        telemetry_enabled=True,
        telemetry_push_interval_s=60.0,
        models_dir=models_dir,
        enable_reporting_sync=True,
    )

    cam_uuid = "0199a001-0000-7000-8000-000000000501"
    v1_uuid = "0199b001-0000-7000-8000-000000000601"
    v2_uuid = "0199b001-0000-7000-8000-000000000602"

    class _BlockingMockVideoReader:
        def __init__(self, *args, **kwargs):
            self.fps = 25.0
            self.original_source = args[0] if args else "0"
            self._stop = threading.Event()

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            self._stop.set()
            return False

        def release(self):
            self._stop.set()

        def __iter__(self):
            # Keep session alive in "running" state briefly so apply_deployed_model_sync observes an active session
            for idx in range(1, 200):
                if self._stop.is_set():
                    break
                yield idx, np.zeros((64, 64, 3), dtype=np.uint8)
                time.sleep(0.02)

    try:
        with patch("app.live_engine.VideoReader", _BlockingMockVideoReader), \
             patch("app.live_engine.Detector"), \
             patch.object(LiveStreamSession, "_process_ai_step"):
            sess = mgr.start_session(
                camera_id="CAM-SYNC-01",
                camera_uuid=cam_uuid,
                site_id="0199c001-0000-7000-8000-000000000701",
                source="rtsp://cam.example.com/stream",
                model_path=str(v1_path),
                model_version_id=v1_uuid,
                managed=True,
                dispatch_enabled=False,
            )
            deadline = time.monotonic() + 2.0
            while sess.status != "running" and time.monotonic() < deadline:
                time.sleep(0.01)

            # Apply deployed model v2 -> triggers hot-reload of CAM-SYNC-01
            reload_res = sync_mgr.apply_deployed_model_sync(
                version_id=v2_uuid,
                version="2.0.0",
                model_path="models/yolo-v2.pt",
                trigger=EVENT_AI_MODEL_DEPLOYED,
            )
            assert reload_res["applied"] is True
            assert reload_res["reloaded_sessions"] == 1
            reloaded_sess = mgr.get_session("CAM-SYNC-01")
            assert reloaded_sess is not None
            assert reloaded_sess.model_version_id == v2_uuid
            assert Path(reloaded_sess.model_path).resolve() == v2_path.resolve()

            # Simulate telemetry samples accumulated in session
            reloaded_sess._telemetry_frames_delta = 42
            reloaded_sess._infer_latencies_ms = [25.0, 35.0, 50.0]

            # Failed push -> restores batch; second forced flush succeeds and uses v2_uuid
            with patch(
                "app.live_sync.push_ai_performance_telemetry_sync",
                side_effect=[False, True],
            ) as mock_push:
                fail_summary = sync_mgr.flush_telemetry_once(force=True)
                assert fail_summary["pushed"] == 0
                assert fail_summary["failed"] == 1
                assert reloaded_sess._telemetry_frames_delta == 42

                # Within 60s interval, non-forced flush is skipped
                skipped_summary = sync_mgr.flush_telemetry_once(force=False)
                assert skipped_summary["skipped"] is True

                ok_summary = sync_mgr.flush_telemetry_once(force=True)
                assert ok_summary["pushed"] == 1
                assert ok_summary["failed"] == 0
                assert mock_push.call_count == 2
                sent_payload = mock_push.call_args_list[1].args[0]
                assert sent_payload["cameraId"] == cam_uuid
                assert sent_payload["modelVersionId"] == v2_uuid
                assert sent_payload["frameCount"] == 42
    finally:
        mgr.stop_all()


def test_live_sync_endpoint_handles_empty_body_and_deploy_notification():
    config.JWT_PUBLIC_KEY = _TEST_PUBLIC_PEM
    config.INTERNAL_SERVICE_TOKEN = "sync-endpoint-m2m-token-32-chars!"
    auth_mod.clear_auth_caches()

    client = TestClient(prod_app, base_url="http://localhost")
    v_uuid = "0199b001-0000-7000-8000-000000000888"

    # Ensure at least one .pt exists in MODELS_DIR
    model_files = sorted(prod_server.MODELS_DIR.glob("*.pt"))
    assert len(model_files) > 0, "Expected at least one .pt file in models/"
    sample_rel_path = f"models/{model_files[0].name}"

    # 1. Unauthorized without token
    unauth = client.post("/api/live/sync")
    assert unauth.status_code == 401

    with patch("app.live_sync.fetch_dynamic_live_max_sessions_sync", return_value=None), \
         patch("app.live_sync.fetch_internal_active_streams_sync", return_value=[]):
        # 2. Legacy empty body with ADMIN JWT
        res_empty = client.post(
            "/api/live/sync",
            headers={"Authorization": f"Bearer {_make_admin_rs256_token()}"},
        )
        assert res_empty.status_code == 200
        body_empty = res_empty.json()
        assert body_empty["status"] == "synced"
        assert body_empty["model_sync"] is None

        # 3. Deploy Model notification from smart-ppe-reporting-service via M2M token
        res_deploy = client.post(
            "/api/live/sync",
            headers={"x-internal-service-token": "sync-endpoint-m2m-token-32-chars!"},
            json={
                "trigger": EVENT_AI_MODEL_DEPLOYED,
                "modelId": "0199b001-0000-7000-8000-000000000800",
                "modelCode": "YOLO_PPE_V11",
                "modelName": "YOLOv11 PPE Detector",
                "modelType": "OBJECT_DETECTION",
                "versionId": v_uuid,
                "version": "2.5.0",
                "modelPath": sample_rel_path,
                "framework": "PyTorch",
                "inputSize": "640x640",
                "deployedAt": "2026-11-25T11:30:00.000Z",
            },
        )
        assert res_deploy.status_code == 200
        body_deploy = res_deploy.json()
        assert body_deploy["status"] == "synced"
        assert body_deploy["active_model_version_id"] == v_uuid
        assert body_deploy["active_model_version"] == "2.5.0"
        assert body_deploy["model_sync"]["applied"] is True
        assert body_deploy["model_sync"]["status"] == "synced"
        assert body_deploy["model_sync"]["version_id"] == v_uuid
        assert body_deploy["model_sync"]["reason"] == "model_reloaded"

        # 4. Missing local weights reports explicit weights_missing status instead of silent drift
        v_missing_uuid = "0199b001-0000-7000-8000-000000000889"
        res_missing = client.post(
            "/api/live/sync",
            headers={"x-internal-service-token": "sync-endpoint-m2m-token-32-chars!"},
            json={
                "trigger": EVENT_AI_MODEL_DEPLOYED,
                "versionId": v_missing_uuid,
                "version": "9.9.9",
                "modelPath": "models/missing-weights-v9.pt",
            },
        )
        assert res_missing.status_code == 200
        body_missing = res_missing.json()
        assert body_missing["model_sync"]["status"] == "weights_missing"
        assert body_missing["model_sync"]["expected_model_path"] == "models/missing-weights-v9.pt"
        assert body_missing["model_sync"]["reloaded_sessions"] == 0
        assert "missing-weights-v9.pt" in body_missing["model_sync"]["message"]

        health_res = client.get("/health")
        assert health_res.status_code == 200
        health_body = health_res.json()
        assert health_body["telemetry_enabled"] is True
        assert health_body["active_model_version_id"] == v_missing_uuid
        assert health_body["active_model_sync_status"] == "weights_missing"


def test_sync_active_model_once_and_background_telemetry_resilience(tmp_path: Path):
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    weight_file = models_dir / "yolo-active.pt"
    weight_file.write_bytes(b"active-weights")

    active_mv_uuid = "0199b001-0000-7000-8000-000000000901"
    cam_uuid = "0199a001-0000-7000-8000-000000000902"

    mgr = LiveEngineManager(max_sessions=4)
    sync_mgr = CameraSyncManager(
        manager=mgr,
        dispatch_enabled=False,
        sync_interval_s=15.0,
        telemetry_enabled=True,
        telemetry_push_interval_s=5.0,
        models_dir=models_dir,
        enable_reporting_sync=True,
    )
    sync_mgr.telemetry_push_interval_s = 0.05

    with patch(
        "app.live_sync.fetch_internal_active_model_version_sync",
        return_value={
            "id": active_mv_uuid,
            "version": "3.1.0",
            "modelPath": "models/yolo-active.pt",
        },
    ):
        synced = sync_mgr.sync_active_model_once()
        assert synced is not None
        assert synced["status"] == "synced"
        assert sync_mgr.active_model_version_id == active_mv_uuid
        assert Path(sync_mgr.active_model_path).resolve() == weight_file.resolve()

    sess = LiveStreamSession(
        camera_id="CAM-BG-01",
        camera_uuid=cam_uuid,
        source="0",
        model_version_id=active_mv_uuid,
        dispatch_enabled=False,
    )
    with mgr._lock:
        mgr._sessions["CAM-BG-01"] = sess

    sess._telemetry_frames_delta = 30
    sess._infer_latencies_ms = [20.0, 30.0, 40.0]

    delivered_payloads: list[dict] = []
    attempt_counter = {"count": 0}
    delivered_event = threading.Event()

    def _flaky_reporting_push(payload: dict) -> bool:
        attempt_counter["count"] += 1
        if attempt_counter["count"] <= 2:
            return False
        delivered_payloads.append(payload)
        delivered_event.set()
        return True

    with patch("app.live_sync.push_ai_performance_telemetry_sync", side_effect=_flaky_reporting_push):
        sync_mgr.start_telemetry_background()
        try:
            assert delivered_event.wait(timeout=3.0), "Background reporter should retry and deliver buffered telemetry"
        finally:
            sync_mgr.stop_telemetry_background()
            with mgr._lock:
                mgr._sessions.clear()

    assert attempt_counter["count"] >= 3
    assert len(delivered_payloads) == 1
    assert delivered_payloads[0]["cameraId"] == cam_uuid
    assert delivered_payloads[0]["modelVersionId"] == active_mv_uuid
    assert delivered_payloads[0]["frameCount"] == 30
