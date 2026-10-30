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

from app.association.associator import UNKNOWN as ASSOC_UNKNOWN
from app.config import (
    MIN_PERSON_HEIGHT_PX,
    PERSON_CONFIDENCE_THRESHOLD,
    PERSON_EDGE_MARGIN_PX,
    PERSON_MIN_VISIBLE_WIDTH_PX,
    REQUIRED_PPE,
)
from app.detection.adapter import HELMET, PERSON, SAFETY_VEST
from app.constants import (
    VIOLATION_NO_HELMET,
    VIOLATION_NO_VEST,
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
# VIOLATION_NO_HELMET and VIOLATION_NO_VEST imported from app.constants above

# Lớp PPE -> mã vi phạm tương ứng
_VIOLATION_BY_CLASS = {HELMET: VIOLATION_NO_HELMET, SAFETY_VEST: VIOLATION_NO_VEST}


@dataclass
class PersonState:
    """Trạng thái an toàn của một người trong một frame."""
    person_track_id: int
    status: str = UNKNOWN  # COMPLIANT | VIOLATION_CANDIDATE | UNKNOWN
    violations: list = field(default_factory=list)  # ["no_helmet", ...]
    reason: str = ""  # lý do UNKNOWN (quá nhỏ / confidence thấp / bị cắt khung)


def _judgeable(assoc, frame_w):
    """Mỗi luật PPE có xét được không — người bị cắt ở mép khung thì không kết luận phần đó.

    - Mũ nằm ở ĐỈNH đầu: chạm mép trên là có thể đã che mất mũ -> không xét.
    - Áo nằm GIỮA thân: cắt vài px ở mép bên không che được áo; chỉ coi là che khi
      phần người thấy được hẹp (người gần như đã ra khỏi khung) -> không xét.
    - Cắt mép dưới (chân) không che mũ/áo nên vẫn xét bình thường.
    """
    x1, y1, x2, _ = assoc.person_bbox
    m = PERSON_EDGE_MARGIN_PX
    out = {}
    for cls in REQUIRED_PPE:
        if cls == HELMET:
            out[cls] = y1 > m
        elif cls == SAFETY_VEST:
            cut_side = x1 <= m or x2 >= frame_w - m
            out[cls] = not (cut_side and (x2 - x1) < PERSON_MIN_VISIBLE_WIDTH_PX)
        else:
            out[cls] = True
    return out


def evaluate(detections, associations, frame_w, frame_h, required_ppe=None):
    """Tính trạng thái VI-05 cho từng người (kết quả ghép VI-04) trong một frame."""
    active_required = required_ppe if required_ppe is not None else REQUIRED_PPE
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

        judgeable = _judgeable(a, frame_w)
        violations = []
        unjudgeable = []
        for cls in active_required:
            if not judgeable[cls]:
                unjudgeable.append(cls)
                continue
            missing = (a.helmet_track_id is None) if cls == HELMET else (a.vest_track_id is None)
            if missing:
                violations.append(_VIOLATION_BY_CLASS[cls])

        if violations:
            states.append(PersonState(a.person_track_id, VIOLATION_CANDIDATE, violations))
        elif not unjudgeable:
            states.append(PersonState(a.person_track_id, COMPLIANT))
        else:
            states.append(PersonState(a.person_track_id, UNKNOWN,
                                      reason="bị cắt khung — không xét được: " + ", ".join(unjudgeable)))
    return states
