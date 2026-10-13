"""Vẽ bounding box + nhãn lên frame — phục vụ chạy demo và kiểm chứng bằng mắt."""
import cv2

from app.detection.adapter import HELMET, PERSON, SAFETY_VEST

# Màu BGR (OpenCV), khớp bảng màu đã dùng khi train (Person xanh dương, Helmet xanh lá, Vest cam)
COLORS = {
    PERSON: (255, 100, 60),
    HELMET: (80, 200, 0),
    SAFETY_VEST: (0, 170, 255),
}


def draw_detections(frame, detections):
    for d in detections:
        x1, y1, x2, y2 = d.bbox
        color = COLORS.get(d.class_name, (255, 255, 255))
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        if d.track_id is not None:
            label = f"{d.class_name} #{d.track_id} {d.confidence:.2f}"
        else:
            label = f"{d.class_name} {d.confidence:.2f}"
        cv2.putText(frame, label, (x1, max(14, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return frame
