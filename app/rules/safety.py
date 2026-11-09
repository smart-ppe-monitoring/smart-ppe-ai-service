"""VI-05: Size filter + safety rules — trạng thái an toàn của từng người, từng frame.

Theo tài liệu Phase 2:
  - Person quá nhỏ (< 60px), confidence thấp hoặc bị che khuất -> UNKNOWN,
    không tự kết luận vi phạm (mục tiêu: giảm false alarm).
  - "Che khuất" không đo được trực tiếp từ bbox đơn, nên dùng proxy: bbox chạm
    mép khung hình (người bị cắt) -> không xét được luật ở phần bị cắt.
  - Luật dựa trên kết quả ghép VI-04: thiếu PPE bắt buộc -> VIOLATION_CANDIDATE.
    Xác nhận vi phạm (CONFIRMED_VIOLATION) thuộc VI-06 (tồn tại liên tục 3 giây).
"""
from dataclasses import dataclass, field
import cv2
import numpy as np

from app.association.associator import UNKNOWN as ASSOC_UNKNOWN
from app.config import (
    MIN_PERSON_HEIGHT_PX,
    PERSON_CONFIDENCE_THRESHOLD,
    PERSON_EDGE_MARGIN_PX,
    PERSON_MIN_VISIBLE_WIDTH_PX,
    PERSON_PROFILE_ASPECT_RATIO_MIN,
    REQUIRED_PPE,
)
from app.detection.adapter import HELMET, PERSON, SAFETY_VEST
from app.constants import (
    VIOLATION_NO_HELMET,
    VIOLATION_NO_VEST,
    VIOLATION_RESTRICTED_AREA,
    STATE_COMPLIANT,
    STATE_CANDIDATE,
    STATE_CONFIRMED,
    STATE_UNKNOWN,
)

# Re-export state constants from app.constants for backward compatibility
COMPLIANT = STATE_COMPLIANT
VIOLATION_CANDIDATE = STATE_CANDIDATE
CONFIRMED_VIOLATION = STATE_CONFIRMED  # trạng thái cuối do VI-06 gán
UNKNOWN = STATE_UNKNOWN
# VIOLATION_NO_HELMET, VIOLATION_NO_VEST, VIOLATION_RESTRICTED_AREA imported from app.constants above

# Lớp PPE -> mã vi phạm tương ứng
_VIOLATION_BY_CLASS = {HELMET: VIOLATION_NO_HELMET, SAFETY_VEST: VIOLATION_NO_VEST}


def is_inside_roi(person_bbox, roi_polygons) -> bool:
    """Kiểm tra điểm chân (bottom-center) của người có nằm trong vùng nguy hiểm Polygon ROI không.

    Theo đề cương D08.pdf Mục 2.6 & Ví dụ 5:
    Điểm chân: P_foot = ((x1 + x2) / 2, y2).
    """
    if not roi_polygons:
        return False
    x1, y1, x2, y2 = person_bbox
    foot_x = float((x1 + x2) / 2.0)
    foot_y = float(y2)
    foot_pt = (foot_x, foot_y)

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

    for poly in polygons:
        if not poly or len(poly) < 3:
            continue
        try:
            poly_np = np.array(poly, dtype=np.float32)
            dist = cv2.pointPolygonTest(poly_np, foot_pt, False)
            if dist >= 0:
                return True
        except Exception:
            continue
    return False


@dataclass
class PersonState:
    """Trạng thái an toàn của một người trong một frame."""
    person_track_id: int
    status: str = UNKNOWN  # COMPLIANT | VIOLATION_CANDIDATE | UNKNOWN
    violations: list = field(default_factory=list)  # ["no_helmet", ...]
    reason: str = ""  # lý do UNKNOWN (quá nhỏ / confidence thấp / bị cắt khung)


def _judgeable(assoc, frame_w, frame_h=None, active_classes=None):
    """Mỗi luật PPE có xét được không:
    - Người bị cắt ở mép khung thì không kết luận phần đó.
    - Mũ nằm ở ĐỈNH đầu: chạm mép trên là có thể đã che mất mũ -> không xét.
    - Áo nằm GIỮA thân: cắt vài px ở mép bên không che được áo; chỉ coi là che khi
      phần người thấy được hẹp (người gần như đã ra khỏi khung) -> không xét.
    - Cắt mép dưới (chân) không che mũ/áo nên vẫn xét bình thường.
    - Người xoay ngang (profile view W/H < PERSON_PROFILE_ASPECT_RATIO_MIN) và không thấy áo:
      chưa đủ góc quan sát tin cậy để khẳng định vi phạm áo -> không xét được.
    """
    x1, y1, x2, y2 = assoc.person_bbox
    w = x2 - x1
    h = max(1, y2 - y1)
    aspect_ratio = w / h
    m = PERSON_EDGE_MARGIN_PX
    classes = active_classes if active_classes is not None else REQUIRED_PPE
    out = {}
    for cls in classes:
        if cls == HELMET:
            out[cls] = y1 > m
        elif cls == SAFETY_VEST:
            cut_side = (x1 <= m or x2 >= frame_w - m) and (w < PERSON_MIN_VISIBLE_WIDTH_PX)
            # Profile view filter: người xoay ngang (dáng hẹp) và không thấy áo -> không xét được áo
            is_profile = (aspect_ratio < PERSON_PROFILE_ASPECT_RATIO_MIN) and (assoc.vest_track_id is None)
            out[cls] = (not cut_side) and (not is_profile)
        else:
            out[cls] = True
    return out


def evaluate(
    detections,
    associations,
    frame_w,
    frame_h,
    required_ppe=None,
    enable_no_helmet: bool = True,
    enable_no_vest: bool = True,
    roi_polygons=None,
    enable_restricted_area: bool = True,
):
    """Tính trạng thái VI-05 cho từng người (kết quả ghép VI-04) trong một frame.

    Hỗ trợ tham số linh hoạt:
      - required_ppe: tuple/list các lớp bắt buộc (vd: ("helmet", "safety_vest"))
      - hoặc enable_no_helmet (bool) & enable_no_vest (bool) bật/tắt riêng từng luật
      - roi_polygons: danh sách các polygon vùng nguy hiểm
      - enable_restricted_area (bool): bật/tắt kiểm tra xâm nhập vùng cấm
    """
    if required_ppe is not None:
        active_required = list(required_ppe)
    else:
        active_required = []
        if enable_no_helmet:
            active_required.append(HELMET)
        if enable_no_vest:
            active_required.append(SAFETY_VEST)

    persons = {d.track_id: d for d in detections
               if d.class_name == PERSON and d.track_id is not None}
    states = []
    for a in associations:
        p = persons.get(a.person_track_id)
        if p is None:
            continue
        bbox_h = a.person_bbox[3] - a.person_bbox[1]
        if bbox_h < MIN_PERSON_HEIGHT_PX:
            states.append(PersonState(a.person_track_id, UNKNOWN,
                                      reason=f"người quá nhỏ (< {MIN_PERSON_HEIGHT_PX}px)"))
            continue
        if p.confidence < PERSON_CONFIDENCE_THRESHOLD:
            states.append(PersonState(a.person_track_id, UNKNOWN,
                                      reason=f"confidence thấp ({p.confidence:.2f})"))
            continue
        if a.status == ASSOC_UNKNOWN:
            states.append(PersonState(a.person_track_id, UNKNOWN,
                                      reason="VI-04 không ghép được PPE"))
            continue

        judgeable = _judgeable(a, frame_w, frame_h, active_required)
        violations = []
        unjudgeable = []
        for cls in active_required:
            if not judgeable.get(cls, True):
                unjudgeable.append(cls)
                continue
            missing = (a.helmet_track_id is None) if cls == HELMET else (a.vest_track_id is None)
            if missing:
                violations.append(_VIOLATION_BY_CLASS[cls])

        # Check restricted area intrusion if enabled and roi_polygons provided
        if enable_restricted_area and roi_polygons:
            if is_inside_roi(a.person_bbox, roi_polygons):
                violations.append(VIOLATION_RESTRICTED_AREA)

        if violations:
            states.append(PersonState(a.person_track_id, VIOLATION_CANDIDATE, violations))
        elif not unjudgeable:
            states.append(PersonState(a.person_track_id, COMPLIANT))
        else:
            states.append(PersonState(a.person_track_id, UNKNOWN,
                                      reason="bị cắt khung hoặc xoay ngang — không xét được: " + ", ".join(unjudgeable)))
    return states
