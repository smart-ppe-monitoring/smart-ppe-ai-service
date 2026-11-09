"""Vẽ bounding box + nhãn lên frame — phục vụ chạy demo, bằng chứng vi phạm và kiểm chứng bằng mắt."""
from typing import List, Optional
import cv2
import numpy as np

from app.detection.adapter import HELMET, PERSON, SAFETY_VEST

# Màu BGR (OpenCV), khớp bảng màu đã dùng khi train (Person xanh dương, Helmet xanh lá, Vest cam)
COLORS = {
    PERSON: (255, 100, 60),
    HELMET: (80, 200, 0),
    SAFETY_VEST: (0, 170, 255),
}

# Màu vi phạm (Đỏ tươi cảnh báo)
COLOR_VIOLATION = (0, 0, 255)
# Màu vùng cấm (Cam cảnh báo nguy hiểm trong BGR)
COLOR_ROI = (0, 140, 255)


def draw_roi_polygons(frame, roi_polygons: Optional[list] = None):
    """Vẽ các đa giác vùng cấm/nguy hiểm (Restricted Area ROI) bán trong suốt lên frame."""
    if not roi_polygons:
        return frame
    polygons = roi_polygons
    if isinstance(roi_polygons, dict):
        polygons = [roi_polygons.get("points", [])]
    elif isinstance(roi_polygons, (list, tuple)):
        if len(roi_polygons) > 0:
            first = roi_polygons[0]
            if isinstance(first, (list, tuple)) and len(first) == 2 and isinstance(first[0], (int, float)):
                polygons = [roi_polygons]
            elif isinstance(first, dict):
                polygons = [p.get("points", []) for p in roi_polygons if "points" in p]

    overlay = frame.copy()
    h, w = frame.shape[:2]
    has_drawn = False

    for poly in polygons:
        if not poly or len(poly) < 3:
            continue
        try:
            pts = np.array(poly, dtype=np.int32).reshape((-1, 1, 2))
            cv2.fillPoly(overlay, [pts], COLOR_ROI)
            cv2.polylines(frame, [pts], isClosed=True, color=COLOR_ROI, thickness=2, lineType=cv2.LINE_AA)
            has_drawn = True

            # Vẽ nhãn vùng cấm tại đỉnh đầu tiên
            start_x, start_y = pts[0][0]
            start_x = max(10, min(w - 180, int(start_x)))
            start_y = max(25, min(h - 10, int(start_y)))
            label = "RESTRICTED AREA"
            font = cv2.FONT_HERSHEY_SIMPLEX
            (tw, th), bl = cv2.getTextSize(label, font, 0.5, 1)
            cv2.rectangle(frame, (start_x, start_y - th - 4), (start_x + tw + 6, start_y + bl), (0, 0, 0), -1)
            cv2.putText(frame, label, (start_x + 3, start_y - 2), font, 0.5, COLOR_ROI, 1, cv2.LINE_AA)
        except Exception:
            continue

    if has_drawn:
        cv2.addWeighted(overlay, 0.20, frame, 0.80, 0, frame)
    return frame


def _draw_hud_corners(img, bbox, color, length=15, thickness=2):
    """Vẽ 4 góc HUD nổi bật quanh bounding box."""
    x1, y1, x2, y2 = (int(round(v)) for v in bbox)
    length = max(4, min(length, (x2 - x1) // 3, (y2 - y1) // 3))
    # Góc trên-trái
    cv2.line(img, (x1, y1), (x1 + length, y1), color, thickness)
    cv2.line(img, (x1, y1), (x1, y1 + length), color, thickness)
    # Góc trên-phải
    cv2.line(img, (x2, y1), (x2 - length, y1), color, thickness)
    cv2.line(img, (x2, y1), (x2, y1 + length), color, thickness)
    # Góc dưới-trái
    cv2.line(img, (x1, y2), (x1 + length, y2), color, thickness)
    cv2.line(img, (x1, y2), (x1, y2 - length), color, thickness)
    # Góc dưới-phải
    cv2.line(img, (x2, y2), (x2 - length, y2), color, thickness)
    cv2.line(img, (x2, y2), (x2, y2 - length), color, thickness)


def draw_detections(
    frame,
    detections,
    person_states=None,
    roi_polygons=None,
    enable_no_helmet: bool = True,
    enable_no_vest: bool = True,
):
    """Vẽ detections và trạng thái vi phạm lên frame.

    Args:
        frame: BGR numpy frame ảnh
        detections: Danh sách đối tượng Detection từ YOLO
        person_states: Danh sách PersonState từ evaluate_safety (tùy chọn)
        roi_polygons: Danh sách Polygon vùng hạn chế ROI (tùy chọn)
        enable_no_helmet: Ẩn bbox mũ nếu tắt kiểm tra vi phạm mũ
        enable_no_vest: Ẩn bbox áo nếu tắt kiểm tra vi phạm áo
    """
    if roi_polygons:
        draw_roi_polygons(frame, roi_polygons)

    # Lập bản đồ tra cứu vi phạm theo track_id của person
    violation_map = {}
    if person_states:
        for ps in person_states:
            if getattr(ps, "violations", None):
                violation_map[ps.person_track_id] = ps.violations

    h, w = frame.shape[:2]

    for d in detections:
        # Ẩn Bounding Box của loại bảo hộ đã tắt kiểm tra vi phạm
        if d.class_name == HELMET and not enable_no_helmet:
            continue
        if d.class_name == SAFETY_VEST and not enable_no_vest:
            continue

        raw_x1, raw_y1, raw_x2, raw_y2 = d.bbox
        x1 = max(0, min(w - 1, int(round(raw_x1))))
        y1 = max(0, min(h - 1, int(round(raw_y1))))
        x2 = max(0, min(w, int(round(raw_x2))))
        y2 = max(0, min(h, int(round(raw_y2))))
        if x2 <= x1 or y2 <= y1:
            continue

        is_violation = False
        violation_texts = []

        if d.class_name == PERSON and d.track_id is not None and d.track_id in violation_map:
            is_violation = True
            v_list = violation_map[d.track_id]
            for v in v_list:
                v_str = str(v).lower()
                if "helmet" in v_str:
                    violation_texts.append("NO HELMET")
                elif "vest" in v_str:
                    violation_texts.append("NO VEST")
                elif "restricted" in v_str or "area" in v_str:
                    violation_texts.append("RESTRICTED AREA")
                else:
                    violation_texts.append(str(v).upper())

        # Chọn màu và độ dày
        if is_violation:
            color = COLOR_VIOLATION
            thickness = 2
            # Vẽ lớp phủ bán trong suốt màu đỏ nhẹ lên bbox người vi phạm
            try:
                sub_img = frame[y1:y2, x1:x2]
                if sub_img.size > 0:
                    red_rect = np.full(sub_img.shape, (0, 0, 180), dtype=np.uint8)
                    cv2.addWeighted(red_rect, 0.25, sub_img, 0.75, 1.0, sub_img)
            except Exception:
                pass
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
            _draw_hud_corners(frame, (x1, y1, x2, y2), color, length=18, thickness=3)

            # Nếu vi phạm vùng cấm, đánh dấu điểm chân (foot point) trực quan
            if any("RESTRICTED" in vt for vt in violation_texts):
                foot_pt = (int((x1 + x2) / 2), int(y2))
                cv2.circle(frame, foot_pt, 5, COLOR_VIOLATION, -1)
                cv2.circle(frame, foot_pt, 8, (255, 255, 255), 1)

            label = f"VIOLATION #{d.track_id}: {', '.join(violation_texts)}"
        else:
            color = COLORS.get(d.class_name, (255, 255, 255))
            thickness = 2
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
            if d.track_id is not None:
                label = f"{d.class_name} #{d.track_id} {d.confidence:.2f}"
            else:
                label = f"{d.class_name} {d.confidence:.2f}"

        # Vẽ nhãn chữ kèm nền tối giúp dễ đọc
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.55 if is_violation else 0.5
        font_thickness = 2 if is_violation else 1
        (text_w, text_h), baseline = cv2.getTextSize(label, font, font_scale, font_thickness)

        text_y = max(text_h + 4, y1 - 6)
        cv2.rectangle(
            frame,
            (x1, text_y - text_h - 4),
            (x1 + text_w + 4, text_y + baseline - 2),
            (0, 0, 0),
            -1,
        )
        cv2.putText(
            frame,
            label,
            (x1 + 2, text_y - 2),
            font,
            font_scale,
            color if is_violation else (255, 255, 255),
            font_thickness,
            cv2.LINE_AA,
        )

    return frame
