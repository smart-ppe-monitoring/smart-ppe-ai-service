"""VI-08: Evidence Snapshot — lưu frame bằng chứng khi có CONFIRMED_VIOLATION.

- Ảnh = full frame đã vẽ bbox + nhãn, tô thêm lớp đỏ mờ (translucent) lên người
  vi phạm kèm nhãn loại vi phạm — nhìn là nhận ra ai không đủ yêu cầu.
  Chỉ vẽ trên bản copy, không làm đổi frame gốc (video output giữ nguyên).
- Lưu local thành evidences/<eventId>.jpg; đường dẫn chỉ để lưu trữ — không dùng
  làm URL công khai (Storage Backend sẽ trả imageUrl chính thức, VI-09).
- Hoặc lưu được ảnh, hoặc trạng thái lỗi rõ — không bao giờ crash pipeline,
  không giả định ảnh đã lưu thành công.
"""
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from app.detection.adapter import PERSON
from app.constants import (
    VIOLATION_NO_HELMET,
    VIOLATION_NO_VEST,
    VIOLATION_RESTRICTED_AREA,
)

OVERLAY_ALPHA = 0.35  # độ mờ của lớp đỏ phủ lên người vi phạm

# cv2.putText chỉ vẽ được ký tự ASCII -> nhãn không dấu
VIOLATION_LABELS = {
    "no_helmet": "VI PHAM: NO_HELMET",
    "no_safety_vest": "VI PHAM: NO_SAFETY_VEST",
    "restricted_area": "VI PHAM: RESTRICTED_AREA",
    VIOLATION_NO_HELMET: "VI PHAM: NO_HELMET",
    VIOLATION_NO_VEST: "VI PHAM: NO_SAFETY_VEST",
    VIOLATION_RESTRICTED_AREA: "VI PHAM: RESTRICTED_AREA",
}


@dataclass
class EvidenceResult:
    """Kết quả lưu snapshot của một event vi phạm."""
    event_id: str
    status: str          # "saved" | "error"
    path: str            # đường dẫn local ảnh vi phạm (rỗng nếu lỗi)
    reference_path: str = ""  # đường dẫn local ảnh đối chứng (rỗng nếu không có/lỗi)
    error: str = ""      # message lỗi (rỗng nếu saved)


def _draw_violation_overlay(img, bbox, label):
    """Tô lớp đỏ mờ lên vùng bbox + viền đỏ đậm + nhãn loại vi phạm (sửa trực tiếp img)."""
    x1, y1, x2, y2 = (int(v) for v in bbox)
    h, w = img.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 <= x1 or y2 <= y1:
        return
    roi = img[y1:y2, x1:x2]
    red = np.full(roi.shape, (0, 0, 255), dtype=np.uint8)
    img[y1:y2, x1:x2] = cv2.addWeighted(red, OVERLAY_ALPHA, roi, 1.0 - OVERLAY_ALPHA, 0)
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 3)

    scale = max(0.6, (x2 - x1) / 300.0)  # nhãn to theo chiều rộng người
    (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, scale, 2)
    ty = max(y1 - th - 10, 10)
    cv2.rectangle(img, (x1, ty - th - 6), (x1 + tw + 8, ty + baseline + 2), (0, 0, 255), -1)
    cv2.putText(img, label, (x1 + 4, ty), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 2)


def _draw_reference_overlay(img, label):
    """Vẽ thanh thông tin ảnh đối chứng màu xanh lá trên đầu ảnh."""
    h, w = img.shape[:2]
    banner_h = max(32, int(h * 0.06))
    cv2.rectangle(img, (0, 0), (w, banner_h), (34, 139, 34), -1)  # Forest Green
    scale = max(0.55, w / 1100.0)
    cv2.putText(img, label, (12, int(banner_h * 0.7)), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 2)


def _draw_reference_box(img, bbox, track_id, ref_type="COMPLIANT"):
    """Vẽ bounding box bao quanh người được đối chứng (để phân biệt rõ trong cảnh đông người)."""
    x1, y1, x2, y2 = (int(v) for v in bbox)
    h, w = img.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 <= x1 or y2 <= y1:
        return

    # Màu sắc: Xanh lá (0, 204, 0) nếu tuân thủ, Xanh cyan/cam (255, 180, 0) nếu t=0s
    color = (0, 204, 0) if ref_type == "COMPLIANT" else (255, 180, 0)
    label = f"DOI CHUNG: NGUOI #{track_id}" + (" (HOP LE)" if ref_type == "COMPLIANT" else " (T=0s)")

    # Vẽ viền dày 3px nổi bật
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)

    # Vẽ nhãn phía trên bbox
    scale = max(0.55, (x2 - x1) / 320.0)
    (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, scale, 2)
    ty = max(y1 - th - 8, 10)
    cv2.rectangle(img, (x1, ty - th - 4), (x1 + tw + 8, ty + baseline + 2), color, -1)
    text_color = (0, 0, 0) if ref_type != "COMPLIANT" else (255, 255, 255)
    cv2.putText(img, label, (x1 + 4, ty), cv2.FONT_HERSHEY_SIMPLEX, scale, text_color, 2)


def save_evidence(frame, detections, event, out_dir) -> EvidenceResult:
    """Lưu frame vi phạm `<out_dir>/<event.event_id>.jpg` và ảnh đối chứng `<event.event_id>_ref.jpg`.

    Tìm bbox người vi phạm trong detections để tô lớp đỏ mờ; không tìm thấy
    (track mất đúng frame phát event) vẫn lưu ảnh bình thường, chỉ bỏ overlay.
    Không bao giờ ném exception — mọi lỗi trả về status="error" kèm lý do.
    """
    event_id = event.event_id
    try:
        if frame is None:
            raise ValueError("frame rỗng — không có ảnh để lưu")
        snap = frame.copy()
        person = next((d for d in detections
                       if d.class_name == PERSON and d.track_id == event.person_track_id), None)
        if person is not None:
            label = VIOLATION_LABELS.get(event.violation_type, "VI PHAM")
            _draw_violation_overlay(snap, person.bbox, label)
        out_path = Path(out_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        ok, buf = cv2.imencode(".jpg", snap)
        if not ok:
            raise ValueError("không encode được JPEG")
        dest = out_path / f"{event_id}.jpg"
        dest.write_bytes(buf.tobytes())

        # Lưu ảnh đối chứng nếu có trong event
        ref_dest_str = ""
        ref_frame = getattr(event, "reference_frame", None)
        ref_bbox = getattr(event, "reference_bbox", None)
        if ref_frame is not None:
            try:
                ref_snap = ref_frame.copy()
                ref_type = getattr(event, "reference_type", "COMPLIANT")
                # 1. Vẽ bounding box lên người được đối chứng nếu có tọa độ bbox
                if ref_bbox is not None:
                    _draw_reference_box(ref_snap, ref_bbox, event.person_track_id, ref_type)

                # 2. Vẽ banner thông tin trên đầu ảnh
                ref_label = "ANH DOI CHUNG: TRANG THAI HOP LE (BEFORE)" if ref_type == "COMPLIANT" else "ANH DOI CHUNG: BAT DAU THEO DOI (T=0s)"
                _draw_reference_overlay(ref_snap, ref_label)
                ok_ref, buf_ref = cv2.imencode(".jpg", ref_snap)
                if ok_ref:
                    dest_ref = out_path / f"{event_id}_ref.jpg"
                    dest_ref.write_bytes(buf_ref.tobytes())
                    ref_dest_str = str(dest_ref)
            except Exception:
                pass  # Không để lỗi lưu ảnh đối chứng ảnh hưởng tới ảnh vi phạm

        return EvidenceResult(event_id=event_id, status="saved",
                              path=str(dest), reference_path=ref_dest_str, error="")
    except Exception as e:
        return EvidenceResult(event_id=event_id, status="error",
                              path="", reference_path="", error=f"{type(e).__name__}: {e}")
