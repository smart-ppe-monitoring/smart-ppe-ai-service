"""
Tests for contract schema validation.
Ensures ViolationEvent payload conforms to Contract 2.1.0-PPE-STABLE.
"""

import pytest
from app.schemas.violation_event import (
    ViolationEvent, Evidence, Detection, ViolationItem,
    BoundingBox, FrameResolution, EventSummary, EventMetadata
)
from app.constants import (
    VIOLATION_NO_HELMET, VIOLATION_NO_VEST,
    SEVERITY_HIGH, SEVERITY_MEDIUM,
    LABEL_PERSON, CONTRACT_VERSION,
)


def _make_event(**overrides) -> dict:
    base = {
        "cameraCode": "CAM-GATE-01",
        "timestamp": "2026-10-13T08:30:00.000Z",
        "evidence": {
            "snapshotUrl": "http://localhost:9000/ppe-evidences/evidences/SITE-01/CAM-GATE-01/1234567890_track12.jpg",
            "frameResolution": {"width": 1920, "height": 1080},
        },
        "detections": [
            {
                "class": "PERSON",
                "confidence": 0.96,
                "box": {"x": 420.0, "y": 180.0, "width": 240.0, "height": 620.0},
                "trackId": 12,
            }
        ],
        "violations": [
            {
                "violationType": VIOLATION_NO_HELMET,
                "severity": SEVERITY_HIGH,
                "confidence": 0.94,
                "description": "Không đội mũ bảo hộ",
                "box": {"x": 470.0, "y": 180.0, "width": 110.0, "height": 95.0},
            }
        ],
    }
    base.update(overrides)
    return base


class TestViolationEventSchema:
    def test_valid_single_violation(self):
        event = ViolationEvent.model_validate(_make_event())
        assert event.cameraCode == "CAM-GATE-01"
        assert len(event.violations) == 1
        assert event.violations[0].violationType == VIOLATION_NO_HELMET

    def test_multiple_violations(self):
        data = _make_event()
        data["violations"].append({
            "violationType": VIOLATION_NO_VEST,
            "severity": SEVERITY_MEDIUM,
            "confidence": 0.89,
            "description": "Không mặc áo phản quang",
        })
        event = ViolationEvent.model_validate(data)
        assert len(event.violations) == 2

    def test_camera_code_required(self):
        from pydantic import ValidationError
        data = _make_event()
        del data["cameraCode"]
        with pytest.raises(ValidationError):
            ViolationEvent.model_validate(data)

    def test_timestamp_required(self):
        from pydantic import ValidationError
        data = _make_event()
        del data["timestamp"]
        with pytest.raises(ValidationError):
            ViolationEvent.model_validate(data)

    def test_violations_required(self):
        from pydantic import ValidationError
        data = _make_event()
        del data["violations"]
        with pytest.raises(ValidationError):
            ViolationEvent.model_validate(data)

    def test_default_frame_resolution(self):
        data = _make_event()
        data["evidence"] = {
            "snapshotUrl": "http://localhost:9000/ppe-evidences/test.jpg"
        }
        event = ViolationEvent.model_validate(data)
        assert event.evidence.frameResolution.width == 1920
        assert event.evidence.frameResolution.height == 1080

    def test_clip_url_optional(self):
        data = _make_event()
        event = ViolationEvent.model_validate(data)
        assert event.evidence.clipUrl is None

    def test_metadata_contract_version(self):
        data = _make_event()
        data["metadata"] = {
            "confirmedDurationMs": 2300,
            "cooldownDurationMs": 30000,
            "model": "yolov8x-ppe-v2.pt",
        }
        event = ViolationEvent.model_validate(data)
        assert event.metadata.contractVersion == CONTRACT_VERSION

    def test_detection_track_id_present(self):
        event = ViolationEvent.model_validate(_make_event())
        assert event.detections[0].trackId == 12

    def test_serializes_to_json(self):
        event = ViolationEvent.model_validate(_make_event())
        json_str = event.model_dump_json(by_alias=True)
        assert "cameraCode" in json_str
        assert "violations" in json_str
