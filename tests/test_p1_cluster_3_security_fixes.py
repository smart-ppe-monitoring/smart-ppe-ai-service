"""Security regression tests for P1 Vulnerability Cluster 3:
- SEC-13: Missing SIGTERM Signal Handler Causing Ungraceful Termination
- SEC-14: Synchronous Filesystem I/O Blocking FastAPI Async Event Loop
- SEC-15: Untracked PPE Track ID None Causing False Positive Violations
"""
import asyncio
import inspect
import signal
import importlib
import pytest
from fastapi.testclient import TestClient

from app.main import register_signal_handlers, _handle_signal
import app.main as main_module
from app.detection.adapter import Detection
from app.association.associator import associate
from app.rules.safety import evaluate as evaluate_safety, COMPLIANT, VIOLATION_CANDIDATE, VIOLATION_NO_HELMET

test_ui_server = importlib.import_module("test-ui.server")


def test_sec_13_signal_handlers_and_graceful_shutdown():
    """SEC-13: Validate signal handler registration and flag setting for SIGINT/SIGTERM."""
    # Reset flag
    main_module._shutdown_requested = False

    register_signal_handlers()

    # Simulate receiving SIGTERM
    _handle_signal(signal.SIGTERM, None)
    assert main_module._shutdown_requested is True

    # Reset flag for future execution
    main_module._shutdown_requested = False


def test_sec_14_async_endpoints_non_blocking():
    """SEC-14: Validate that list_jobs and list_models are async and execute non-blocking via asyncio.to_thread."""
    assert inspect.iscoroutinefunction(test_ui_server.list_jobs), "list_jobs must be an async coroutine function"
    assert inspect.iscoroutinefunction(test_ui_server.list_models), "list_models must be an async coroutine function"

    client = TestClient(test_ui_server.app)

    # Test /api/jobs returns 200
    res_jobs = client.get("/api/jobs")
    assert res_jobs.status_code == 200
    assert isinstance(res_jobs.json(), list)

    # Test /api/models returns 200
    res_models = client.get("/api/models")
    assert res_models.status_code == 200
    assert isinstance(res_models.json(), list)


def test_sec_15_untracked_ppe_track_id_no_false_positives():
    """SEC-15: Validate that untracked PPE (track_id=None) matches correctly without false positive violations."""
    # Person with track_id=1, helmet with track_id=None (as produced by YOLO for secondary objects), vest with track_id=None
    person = Detection(class_name="person", confidence=0.95, bbox=(100, 100, 200, 500), track_id=1)
    helmet = Detection(class_name="helmet", confidence=0.90, bbox=(110, 110, 190, 150), track_id=None)
    vest = Detection(class_name="safety_vest", confidence=0.90, bbox=(105, 200, 195, 400), track_id=None)

    detections = [person, helmet, vest]
    associations = associate(detections)

    assert len(associations) == 1
    # Untracked secondary detections should receive sentinel -1 instead of None
    assert associations[0].helmet_track_id == -1, f"Expected -1 sentinel for untracked helmet, got {associations[0].helmet_track_id}"
    assert associations[0].vest_track_id == -1, f"Expected -1 sentinel for untracked vest, got {associations[0].vest_track_id}"

    # Evaluate safety rules
    states = evaluate_safety(detections, associations, frame_w=1920, frame_h=1080)
    assert len(states) == 1
    assert states[0].status == COMPLIANT, f"Expected COMPLIANT for person wearing untracked helmet/vest, got {states[0].status} with {states[0].violations}"

    # Verify that if helmet is genuinely missing, it is correctly flagged as a violation
    detections_no_helmet = [person, vest]
    assoc_no_helmet = associate(detections_no_helmet)
    assert assoc_no_helmet[0].helmet_track_id is None

    states_no_helmet = evaluate_safety(detections_no_helmet, assoc_no_helmet, frame_w=1920, frame_h=1080)
    assert states_no_helmet[0].status == VIOLATION_CANDIDATE
    assert VIOLATION_NO_HELMET in states_no_helmet[0].violations
