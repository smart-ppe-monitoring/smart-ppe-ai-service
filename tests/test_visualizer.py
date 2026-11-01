"""Kiểm thử visualizer: vẽ bounding box thường và vi phạm (HUD đỏ cảnh báo)."""
import numpy as np
from app.detection.adapter import Detection
from app.rules.safety import PersonState, VIOLATION_CANDIDATE, VIOLATION_NO_HELMET, VIOLATION_NO_VEST
from app.video.visualizer import draw_detections


def test_draw_detections_normal():
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    detections = [
        Detection(class_name="person", confidence=0.95, bbox=(50, 50, 200, 400), track_id=1),
        Detection(class_name="helmet", confidence=0.90, bbox=(70, 50, 150, 120), track_id=2),
    ]
    out = draw_detections(frame, detections)
    assert out is not None
    assert out.shape == (480, 640, 3)


def test_draw_detections_with_violations():
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    detections = [
        Detection(class_name="person", confidence=0.95, bbox=(50, 50, 200, 400), track_id=1),
    ]
    person_states = [
        PersonState(
            person_track_id=1,
            status=VIOLATION_CANDIDATE,
            violations=[VIOLATION_NO_HELMET, VIOLATION_NO_VEST],
        )
    ]
    out = draw_detections(frame, detections, person_states=person_states)
    assert out is not None
    # Kiểm tra xem có màu đỏ (vi phạm) xuất hiện trên frame không
    # Kênh BGR: Red là out[:, :, 2] > 0
    has_red = np.any(out[:, :, 2] > 0)
    assert has_red, "Frame phải chứa màu đỏ cảnh báo vi phạm"

