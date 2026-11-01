"""
Regression tests for P0 security fixes (SEC-01 through SEC-05).
Verifies:
1. SEC-01: Model path traversal & arbitrary loading blocked
2. SEC-02: MinIO bucket created with secure private policy (no public anonymous access)
3. SEC-03: Poison-pill head-of-line blocking resolved with retry counter in SQLite buffer
4. SEC-04: Dynamic safety rules honor enable_no_helmet / enable_no_vest flags
5. SEC-05: Backend client includes Bearer token for inter-service M2M authentication
"""

import asyncio
from pathlib import Path
from unittest.mock import patch, MagicMock
import pytest
from fastapi import HTTPException

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
    VIOLATION_NO_VEST,
    SEVERITY_HIGH,
    LABEL_PERSON,
    STATE_COMPLIANT,
    STATE_CANDIDATE,
)
import app.events.offline_buffer as offline_buffer_mod
from app.events import flush_offline_buffer
from app.rules.safety import evaluate
from app.association.associator import PersonPPE, MATCHED
from app.detection.adapter import Detection as CVDet
from app.storage.minio_client import ensure_bucket_exists
from app.clients.backend_client import (
    fetch_system_default_rules,
    fetch_effective_rules,
    _get_auth_headers,
)
from app import config


# ─────────────────────────────────────────────────────────────────────────────
# 1. SEC-01: Model Sandboxing & Path Traversal Prevention
# ─────────────────────────────────────────────────────────────────────────────
def test_sec_01_model_path_sandboxing():
    import importlib
    server = importlib.import_module("test-ui.server")
    start_fn = server.start

    # Invalid job_id path traversal
    with pytest.raises(HTTPException) as exc_info:
        start_fn(job_id="../etc/passwd")
    assert exc_info.value.status_code == 400

    # Arbitrary model path outside models/ directory
    with pytest.raises(HTTPException) as exc_info:
        start_fn(job_id="valid_job_123", model="/etc/shadow")
    assert exc_info.value.status_code == 400

    # Non-existent model inside models/
    with pytest.raises(HTTPException) as exc_info:
        start_fn(job_id="valid_job_123", model="non_existent_fake_model.pt")
    assert exc_info.value.status_code == 400


# ─────────────────────────────────────────────────────────────────────────────
# 2. SEC-02: Private Bucket Access (No Anonymous Public Policy)
# ─────────────────────────────────────────────────────────────────────────────
def test_sec_02_minio_private_bucket_policy():
    mock_client = MagicMock()
    mock_client.bucket_exists.return_value = False

    ensure_bucket_exists(mock_client, "ppe-evidences")

    mock_client.make_bucket.assert_called_once_with("ppe-evidences")
    # Must NOT set any public anonymous read policy
    mock_client.set_bucket_policy.assert_not_called()


# ─────────────────────────────────────────────────────────────────────────────
# 3. SEC-03: Poison-Pill Head-of-Line Starvation Fix in SQLite Buffer
# ─────────────────────────────────────────────────────────────────────────────
def _sample_event(cam="CAM-01") -> ViolationEvent:
    return ViolationEvent(
        cameraCode=cam,
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


def test_sec_03_poison_pill_offline_buffer_recovery(tmp_path):
    test_db = tmp_path / "test_buffer.db"
    ev1 = _sample_event("CAM-POISON")
    ev2 = _sample_event("CAM-VALID")

    with patch.object(offline_buffer_mod, "DB_PATH", test_db):
        offline_buffer_mod.save_event(ev1)
        offline_buffer_mod.save_event(ev2)
        assert offline_buffer_mod.pending_count() == 2

        # Simulate batch publish failure (e.g. Backend rejects poison event)
        with patch("app.events.publish_batch_sync", side_effect=RuntimeError("HTTP 422 Unprocessable Entity")):
            flushed = flush_offline_buffer(sync=True)
            assert flushed == 0

        # Verify retry counter was incremented
        with offline_buffer_mod._get_conn() as conn:
            rows = conn.execute("SELECT id, retried FROM offline_events").fetchall()
            for r in rows:
                assert r[1] == 1  # retried incremented to 1

        # Simulate 2 more failures to hit max_retries=3 for poison event
        with offline_buffer_mod._get_conn() as conn:
            conn.execute("UPDATE offline_events SET retried = 3 WHERE id = 1")

        # Now only ev2 (id=2, retried=1) should be pending
        assert offline_buffer_mod.pending_count(max_retries=3) == 1
        ids, events = offline_buffer_mod.load_batch(max_retries=3)
        assert len(events) == 1
        assert events[0].cameraCode == "CAM-VALID"


# ─────────────────────────────────────────────────────────────────────────────
# 4. SEC-04: Dynamic Safety Rules Gating (enable_no_helmet / enable_no_vest)
# ─────────────────────────────────────────────────────────────────────────────
def test_sec_04_dynamic_safety_rules_filtering():
    # Person missing both helmet and vest
    detections = [CVDet("person", 0.95, (100, 100, 200, 400), track_id=1)]
    associations = [PersonPPE(person_track_id=1, person_bbox=(100, 100, 200, 400), status=MATCHED)]

    # 1. Default: both required -> candidate with both violations
    states = evaluate(detections, associations, 1920, 1080)
    assert states[0].status == STATE_CANDIDATE
    assert set(states[0].violations) == {VIOLATION_NO_HELMET, VIOLATION_NO_VEST}

    # 2. Helmet rule disabled: only require vest
    states_vest_only = evaluate(detections, associations, 1920, 1080, required_ppe=("safety_vest",))
    assert states_vest_only[0].status == STATE_CANDIDATE
    assert states_vest_only[0].violations == [VIOLATION_NO_VEST]

    # 3. Both rules disabled: required_ppe=() -> compliant!
    states_none = evaluate(detections, associations, 1920, 1080, required_ppe=())
    assert states_none[0].status == STATE_COMPLIANT
    assert len(states_none[0].violations) == 0


# ─────────────────────────────────────────────────────────────────────────────
# 5. SEC-05: M2M Authentication Headers for Backend Client
# ─────────────────────────────────────────────────────────────────────────────
def test_sec_05_backend_client_auth_headers():
    # When token is set, _get_auth_headers must produce Bearer token
    with patch.object(config, "INTERNAL_SERVICE_TOKEN", "mock-secret-jwt-token"):
        headers = _get_auth_headers()
        assert headers == {"Authorization": "Bearer mock-secret-jwt-token"}

        mock_resp = MagicMock()
        mock_resp.json.return_value = {"success": True, "data": {"confidenceThreshold": 0.88}}
        mock_resp.raise_for_status.return_value = None

        with patch("httpx.AsyncClient.get", return_value=mock_resp) as mock_get:
            asyncio.run(fetch_system_default_rules())
            # Verify Authorization header was passed to HTTP client
            _, kwargs = mock_get.call_args
            assert kwargs.get("headers") == {"Authorization": "Bearer mock-secret-jwt-token"}
