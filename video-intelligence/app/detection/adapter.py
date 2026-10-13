"""VI-02: Chuẩn hóa output của detector về định dạng chung của project.

Mọi model YOLO đều được đưa về Detection(class_name, confidence, bbox x1y1x2y2)
thông qua ánh xạ TÊN lớp — pipeline không phụ thuộc ID số của từng model,
nên đổi model chỉ cần sửa config.
"""
from dataclasses import dataclass

PERSON = "person"
HELMET = "helmet"
SAFETY_VEST = "safety_vest"
CANONICAL_CLASSES = (PERSON, HELMET, SAFETY_VEST)

# Tên lớp của model (đã chuẩn hóa: thường, bỏ dấu gạch) -> tên chuẩn project
LABEL_ALIASES = {
    "person": PERSON,
    "helmet": HELMET,
    "hardhat": HELMET,
    "safetyvest": SAFETY_VEST,
    "vest": SAFETY_VEST,
}


@dataclass
class Detection:
    class_name: str
    confidence: float
    bbox: tuple  # (x1, y1, x2, y2) pixel, gốc trên-trái
    track_id: int = None  # None khi không chạy tracking hoặc đối tượng chưa được gán ID


def _norm(name: str) -> str:
    return str(name).strip().lower().replace("-", "").replace("_", "").replace(" ", "")


def map_class_name(model_name: str):
    """Tên lớp từ model -> tên chuẩn project; None nếu ngoài 3 lớp cần dùng."""
    return LABEL_ALIASES.get(_norm(model_name))


def normalize_results(results, frame_w: int, frame_h: int):
    """Chuyển ultralytics Results (1 frame) -> list[Detection].

    Bbox được đổi từ (x1,y1,x2,y2) float sang số nguyên và cắt về trong ảnh.
    """
    detections = []
    names = getattr(results, "names", None) or {}
    boxes = results.boxes
    if boxes is None or len(boxes) == 0:
        return detections

    track_ids = getattr(boxes, "id", None)
    for i, (xyxy, conf, cls) in enumerate(zip(boxes.xyxy, boxes.conf, boxes.cls)):
        name = map_class_name(names.get(int(cls), ""))
        if name is None:
            continue
        x1 = max(0.0, min(float(xyxy[0]), frame_w))
        y1 = max(0.0, min(float(xyxy[1]), frame_h))
        x2 = max(0.0, min(float(xyxy[2]), frame_w))
        y2 = max(0.0, min(float(xyxy[3]), frame_h))
        bbox = (int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2)))

        track_id = None
        if track_ids is not None and i < len(track_ids):
            tid = track_ids[i]
            if tid is not None and int(tid) >= 0:
                track_id = int(tid)

        detections.append(Detection(class_name=name, confidence=float(conf),
                                    bbox=bbox, track_id=track_id))
    return detections
