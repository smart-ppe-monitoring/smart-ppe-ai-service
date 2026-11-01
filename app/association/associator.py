"""VI-04: Ghép Helmet/Safety Vest với đúng Person bằng hình học bbox.

Theo mục 8 (Heuristic ghép Helmet/Vest) của tài liệu Phase 2:
  - Helmet: tâm mũ phải nằm trong vùng ngang trung tâm của người (lề 15% mỗi bên)
    và ở phần trên người (trên 40% chiều cao tính từ đỉnh).
  - Safety Vest: dùng vùng thân kết hợp overlap/khoảng cách (không áp dụng quy tắc
    mũ). 1 PPE chỉ ghép cho tối đa 1 người; không dựa vào tổng số PPE.
  - Người có bbox cao dưới MIN_PERSON_HEIGHT_PX -> UNKNOWN (không tự kết luận).

Đây là heuristic: có thể sai với camera top-down hoặc người cúi; cần đánh giá
FP/FN trên video thật (VI-12).
"""
from dataclasses import dataclass

from app.config import (
    MIN_PERSON_HEIGHT_PX,
    PPE_HEAD_X_MARGIN,
    PPE_HEAD_Y_MAX,
    PPE_TORSO_Y_MIN,
    PPE_TORSO_Y_MAX,
    PPE_TORSO_IOU_MIN,
)
from app.detection.adapter import HELMET, PERSON, SAFETY_VEST

UNKNOWN = "unknown"   # người quá nhỏ / thiếu bằng chứng — không kết luận
MATCHED = "matched"


@dataclass
class PersonPPE:
    """Kết quả ghép PPE của một người trong một frame.

    helmet_track_id/vest_track_id = None nghĩa là frame này không ghép được PPE đó
    (có thể thật sự không có, hoặc model miss — quyết định ở VI-05/VI-06).
    """
    person_track_id: int
    person_bbox: tuple  # (x1, y1, x2, y2)
    helmet_track_id: int = None
    vest_track_id: int = None
    status: str = MATCHED  # MATCHED | UNKNOWN


def _center(bbox):
    return ((bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0)


def _iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter <= 0:
        return 0.0
    area = lambda r: (r[2] - r[0]) * (r[3] - r[1])
    return inter / (area(a) + area(b) - inter)


def _helmet_candidates(person_bbox, helmets):
    """Các mũ thỏa điều kiện vùng đầu của người; trả (điểm gần mỏ neo đầu, mũ)."""
    x1, y1, x2, y2 = person_bbox
    w, h = x2 - x1, y2 - y1
    anchor_x, anchor_y = (x1 + x2) / 2.0, y1 + 0.10 * h  # vị trí đầu dự kiến
    out = []
    for hm in helmets:
        hx, hy = _center(hm.bbox)
        if x1 + PPE_HEAD_X_MARGIN * w <= hx <= x2 - PPE_HEAD_X_MARGIN * w \
                and hy < y1 + PPE_HEAD_Y_MAX * h:
            dist = ((hx - anchor_x) ** 2 + (hy - anchor_y) ** 2) ** 0.5
            out.append((dist, hm))
    return out


def _vest_candidates(person_bbox, vests):
    """Các áo thỏa điều kiện vùng thân (tâm trong dải thân HOẶC overlap đủ lớn)."""
    x1, y1, x2, y2 = person_bbox
    h = y2 - y1
    torso = (x1, y1 + PPE_TORSO_Y_MIN * h, x2, y1 + PPE_TORSO_Y_MAX * h)
    cx, cy = (x1 + x2) / 2.0, y1 + 0.55 * h  # tâm vùng thân dự kiến
    out = []
    for vt in vests:
        vx, vy = _center(vt.bbox)
        in_torso = torso[0] <= vx <= torso[2] and torso[1] <= vy <= torso[3]
        iou = _iou(vt.bbox, torso)
        if in_torso or iou >= PPE_TORSO_IOU_MIN:
            dist = ((vx - cx) ** 2 + (vy - cy) ** 2) ** 0.5
            out.append(((-iou, dist), vt))  # ưu tiên overlap lớn trước, rồi gần nhất
    return out


def _assign(person_bboxes, candidates_fn, ppe_list):
    """Ghép 1-1 tham lam theo độ phù hợp: mỗi PPE tối đa 1 người."""
    pairs = []
    for person_idx, person_bbox in enumerate(person_bboxes):
        for score, ppe in candidates_fn(person_bbox, ppe_list):
            pairs.append((score, person_idx, ppe))
    pairs.sort(key=lambda t: t[0])

    assigned = {}
    used_ppe = set()
    for _, person_idx, ppe in pairs:
        ppe_id = id(ppe)
        if person_idx in assigned or ppe_id in used_ppe:
            continue
        # SEC-15: If ppe.track_id is None (untracked secondary detection in YOLO),
        # assign sentinel -1 instead of None to prevent false positive missing PPE violations
        assigned[person_idx] = ppe.track_id if ppe.track_id is not None else -1
        used_ppe.add(ppe_id)
    return assigned


def associate(detections, min_height=MIN_PERSON_HEIGHT_PX):
    """Ghép PPE cho từng người trong 1 frame -> list[PersonPPE].

    Chỉ ghép cho Person có track_id (theo dõi được); mũ/áo không khớp ai thì bỏ qua.
    """
    persons = [d for d in detections if d.class_name == PERSON and d.track_id is not None]
    helmets = [d for d in detections if d.class_name == HELMET]
    vests = [d for d in detections if d.class_name == SAFETY_VEST]
    if not persons:
        return []

    ok_idx = [i for i, p in enumerate(persons)
              if (p.bbox[3] - p.bbox[1]) >= min_height]
    helmet_map = _assign([persons[i].bbox for i in ok_idx], _helmet_candidates, helmets)
    vest_map = _assign([persons[i].bbox for i in ok_idx], _vest_candidates, vests)

    results = []
    for i, p in enumerate(persons):
        if i not in ok_idx:
            results.append(PersonPPE(p.track_id, p.bbox, status=UNKNOWN))
        else:
            results.append(PersonPPE(
                person_track_id=p.track_id,
                person_bbox=p.bbox,
                helmet_track_id=helmet_map.get(i),
                vest_track_id=vest_map.get(i),
            ))
    return results
