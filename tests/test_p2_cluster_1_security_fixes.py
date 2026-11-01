"""
Regression tests for P2 Security Fixes Cluster (SEC-20, SEC-21, SEC-22).

- SEC-20: Video Reader HTTP/HTTPS Stream Protocol Recognition
- SEC-21: Video Reader Iterative Frame Reading Immunity to RecursionError
- SEC-22: Untracked Secondary PPE Equipment Detections Preserved in Violation Events
"""

import numpy as np
import pytest
from unittest.mock import MagicMock, patch

from app.constants import (
    LABEL_PERSON,
    LABEL_HARD_HAT,
    LABEL_SAFETY_VEST,
    VIOLATION_NO_HELMET,
)
from app.detection.adapter import Detection as BoundingBoxDetection
from app.main import _build_violation_event
from app.rules.deduplication import ViolationEvent as DedupEvent
from app.schemas.violation_event import ViolationEvent, Detection as SchemaDetection
from app.video.video_reader import VideoReader, VideoSourceError


# ==============================================================================
# SEC-20: Video Reader HTTP/HTTPS Stream Protocol Recognition
# ==============================================================================

def test_sec_20_stream_protocol_recognition():
    """Verify VideoReader recognizes HTTP, HTTPS, RTSP, RTMP and HLS streams."""
    stream_sources = [
        0,
        1,
        "rtsp://admin:pass@192.168.1.100:554/live",
        "rtsps://camera.lan/feed",
        "http://192.168.1.50:8080/video.mjpeg",
        "https://stream.site.com/live/channel1",
        "rtmp://live.site.com/app/stream",
        "rtmps://live.site.com/app/stream",
        "http://cdn.site.com/live/manifest.m3u8",
        "/var/media/streams/index.m3u8",
    ]
    for src in stream_sources:
        reader = VideoReader(src)
        assert reader._is_stream is True, f"Failed to recognize {src} as stream"

    file_sources = [
        "sample.mp4",
        "/data/cctv_recording.avi",
        "C:\\videos\\gate_01.mkv",
        "test_feed.mov",
    ]
    for src in file_sources:
        reader = VideoReader(src)
        assert reader._is_stream is False, f"Erroneously identified {src} as stream"


def test_sec_20_http_stream_tolerance_not_terminated_on_single_empty_frame():
    """Verify HTTP stream applies empty_frame_tolerance instead of halting on first empty frame."""
    reader = VideoReader("http://192.168.1.50/mjpeg", empty_frame_tolerance=5)
    reader._opened = True

    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    mock_cap = MagicMock()
    # Sequence: 1 empty frame, then 1 valid frame
    mock_cap.read.side_effect = [(False, None), (True, dummy_frame)]
    reader._cap = mock_cap

    # In old code: would immediately return None because _is_stream was False.
    # In patched code: retries within tolerance and retrieves the valid frame.
    result = reader.read()
    assert result is not None
    frame_num, frame = result
    assert frame_num == 1
    assert frame.shape == (480, 640, 3)


# ==============================================================================
# SEC-21: Video Reader Iterative Frame Reading Immunity to RecursionError
# ==============================================================================

def test_sec_21_iterative_read_no_recursion_error():
    """
    Verify read() uses iterative loop, successfully tolerating > 1000 consecutive
    empty frames without exceeding Python's call stack limit (RecursionError).
    """
    # Exceed standard Python recursion limit of 1000
    tolerance = 1500
    reader = VideoReader("rtsp://mock-cam/live", empty_frame_tolerance=tolerance)
    reader._opened = True

    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    mock_cap = MagicMock()

    # Return 1200 empty frames followed by 1 valid frame
    empty_count = 1200
    responses = [(False, None)] * empty_count + [(True, dummy_frame)]
    mock_cap.read.side_effect = responses
    reader._cap = mock_cap

    # Must complete iteratively without RecursionError
    result = reader.read()
    assert result is not None
    frame_num, frame = result
    assert frame_num == 1
    assert reader._empty_streak == 0
    assert mock_cap.read.call_count == empty_count + 1


def test_sec_21_iterative_read_gracefully_halts_when_tolerance_exceeded():
    """Verify that when empty frames exceed tolerance, read() returns None cleanly."""
    tolerance = 10
    reader = VideoReader("http://mock-cam/live", empty_frame_tolerance=tolerance)
    reader._opened = True

    mock_cap = MagicMock()
    mock_cap.read.return_value = (False, None)
    reader._cap = mock_cap

    result = reader.read()
    assert result is None
    assert reader._empty_streak == tolerance + 1


# ==============================================================================
# SEC-22: Untracked Secondary PPE Equipment Detections Preserved
# ==============================================================================

def test_sec_22_untracked_ppe_detections_preserved_in_violation_event():
    """
    Verify _build_violation_event preserves secondary PPE detections (helmet, vest)
    even when track_id is None, populating schema detections properly.
    """
    ev = DedupEvent(
        event_id="EVT-001",
        camera_id="CAM-01",
        person_track_id=10,
        canonical_person_id=10,
        violation_type="NO_HELMET",
        start_t=100.0,
        confirmed_t=102.5,
        duration_s=2.5,
    )

    all_detections = [
        # Tracked person
        BoundingBoxDetection(
            class_name="person",
            confidence=0.95,
            bbox=(100.0, 150.0, 300.0, 650.0),
            track_id=10,
        ),
        # Untracked helmet (e.g., worn by or near worker, or another object)
        BoundingBoxDetection(
            class_name="helmet",
            confidence=0.88,
            bbox=(140.0, 160.0, 260.0, 240.0),
            track_id=None,
        ),
        # Untracked safety vest
        BoundingBoxDetection(
            class_name="safety_vest",
            confidence=0.82,
            bbox=(120.0, 250.0, 280.0, 480.0),
            track_id=None,
        ),
    ]

    event = _build_violation_event(
        ev=ev,
        frame_w=1920,
        frame_h=1080,
        snapshot_url="http://minio:9000/ppe-evidences/test.jpg",
        all_detections=all_detections,
        model_name="yolov8x-ppe.pt",
    )

    # 1. Total detections in schema must be 3 (not reduced to 1)
    assert len(event.detections) == 3

    labels = [d.classLabel for d in event.detections]
    assert LABEL_PERSON in labels
    assert LABEL_HARD_HAT in labels
    assert LABEL_SAFETY_VEST in labels

    # 2. Total persons summary should only count PERSON class
    assert event.summary.totalPersons == 1

    # 3. Tracked person retains trackId
    person_det = next(d for d in event.detections if d.classLabel == LABEL_PERSON)
    assert person_det.trackId == 10

    # 4. Untracked items have trackId as None in Pydantic model
    helmet_det = next(d for d in event.detections if d.classLabel == LABEL_HARD_HAT)
    vest_det = next(d for d in event.detections if d.classLabel == LABEL_SAFETY_VEST)
    assert helmet_det.trackId is None
    assert vest_det.trackId is None

    # 5. Serialization to dict (by_alias=True) should exclude trackId for untracked items
    # so downstream Zod (z.number().int().optional()) does not receive `trackId: null`
    payload = event.model_dump(by_alias=True)
    serialized_detections = payload["detections"]
    assert len(serialized_detections) == 3

    s_person = next(d for d in serialized_detections if d["class"] == LABEL_PERSON)
    s_helmet = next(d for d in serialized_detections if d["class"] == LABEL_HARD_HAT)
    s_vest = next(d for d in serialized_detections if d["class"] == LABEL_SAFETY_VEST)

    assert s_person.get("trackId") == 10
    assert "trackId" not in s_helmet
    assert "trackId" not in s_vest
