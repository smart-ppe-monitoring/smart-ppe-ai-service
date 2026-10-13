"""VI-07: Chống trùng — cooldown 30 giây, khóa cameraId + trackId + violationType.

- VI-06 đã đảm bảo mỗi CHUỖI vi phạm chỉ phát 1 event (một vi phạm kéo dài không
  tạo event mỗi frame). VI-07 chặn thêm các trường hợp lặp:
    * Tái phạm trong vòng cooldown (COMPLIANT rồi vi phạm lại) -> chặn.
    * Track đổi ID (ByteTrack cấp ID mới cho cùng một người) -> nối về cùng
      "canonical id" rồi áp cooldown -> không phát trùng.
- Khóa chống trùng: (camera_id, canonical_person_id, violation_type), trong đó
  canonical_person_id là trackId đã được ổn định qua các lần đổi ID — thỏa khóa
  tối thiểu cameraId + trackId + violationType theo tài liệu.

Xử lý track đổi ID: khi một track biến mất, nhớ bbox cuối + thời điểm mất; nếu một
track MỚI xuất hiện trong ID_SWITCH_GRACE_S (2s) và tâm bbox gần track cũ
(< ID_SWITCH_MAX_DIST_RATIO * chiều cao), coi là cùng một người (nối canonical id).
Heuristic — cần đánh giá trên video thật với người đi ra/vào khung nhiều.
"""
import uuid
from dataclasses import dataclass

from app.config import (
    COOLDOWN_DURATION_S,
    DEFAULT_CAMERA_ID,
    ID_SWITCH_GRACE_S,
    ID_SWITCH_MAX_DIST_RATIO,
)
from app.detection.adapter import PERSON


@dataclass
class ViolationEvent:
    """Sự kiện vi phạm cuối cùng được phát đi (đầu vào cho VI-08 snapshot / VI-09 contract)."""
    event_id: str
    camera_id: str
    person_track_id: int          # track_id hiện tại tại thời điểm phát
    canonical_person_id: int      # id ổn định qua các lần đổi track ID
    violation_type: str
    start_t: float
    confirmed_t: float
    duration_s: float


def _center(bbox):
    return ((bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0)


def _center_dist(a, b):
    ca, cb = _center(a), _center(b)
    return ((ca[0] - cb[0]) ** 2 + (ca[1] - cb[1]) ** 2) ** 0.5


class ViolationDeduplicator:
    """Lọc event trùng — 1 instance riêng cho mỗi camera (trạng thái theo camera)."""

    def __init__(self, camera_id=DEFAULT_CAMERA_ID, cooldown_s=COOLDOWN_DURATION_S,
                 id_switch_grace_s=ID_SWITCH_GRACE_S,
                 id_switch_max_dist_ratio=ID_SWITCH_MAX_DIST_RATIO):
        self.camera_id = camera_id
        self.cooldown_s = float(cooldown_s)
        self.id_switch_grace_s = float(id_switch_grace_s)
        self.id_switch_max_dist_ratio = float(id_switch_max_dist_ratio)
        self._canonical = {}       # track_id -> canonical_person_id
        self._next_canonical = 0
        self._last_bbox = {}       # track_id -> bbox frame trước (để nhớ vị trí khi track mất)
        self._last_seen = {}       # track_id -> thời điểm cuối cùng còn thấy
        self._recently_lost = {}   # canonical_id -> (bbox cuối, thời điểm thấy lần cuối)
        self._last_emit = {}       # (canonical_id, violation_type) -> thời điểm phát gần nhất
        self._prev_ids = set()

    def _update_links(self, persons, t):
        current_ids = set(persons.keys())
        lost_ids = self._prev_ids - current_ids
        new_ids = current_ids - self._prev_ids

        for tid in lost_ids:
            if tid in self._canonical and tid in self._last_bbox:
                self._recently_lost[self._canonical[tid]] = (
                    self._last_bbox[tid], self._last_seen.get(tid, t))

        # hết hạn: track mất quá lâu -> không còn nghi đổi ID nữa
        for can_id in list(self._recently_lost.keys()):
            if t - self._recently_lost[can_id][1] > self.id_switch_grace_s:
                del self._recently_lost[can_id]

        for tid in new_ids:
            bbox = persons[tid]
            best = None
            best_dist = float("inf")
            best_bbox = None
            for can_id in list(self._recently_lost.keys()):
                old_bbox, lost_t = self._recently_lost[can_id]
                d = _center_dist(bbox, old_bbox)
                if d < best_dist:
                    best, best_dist, best_bbox = can_id, d, old_bbox
            if best is not None and best_bbox is not None:
                ref_h = max(bbox[3] - bbox[1], best_bbox[3] - best_bbox[1])
                if best_dist <= self.id_switch_max_dist_ratio * ref_h:
                    self._canonical[tid] = best  # nối: cùng một người
                    continue
            self._canonical[tid] = self._next_canonical
            self._next_canonical += 1

        for tid, bbox in persons.items():
            self._last_bbox[tid] = bbox
            self._last_seen[tid] = t
        self._prev_ids = current_ids

    def filter(self, confirmed_events, detections, t):
        """Lọc các event VI-06 đã xác nhận -> danh sách ViolationEvent được phát.

        Nhận detections của frame hiện tại để theo dõi track xuất hiện/mất
        (phục vụ nối ID khi track đổi).
        """
        t = float(t)
        persons = {d.track_id: d.bbox for d in detections
                   if d.class_name == PERSON and d.track_id is not None}
        self._update_links(persons, t)

        out = []
        for c in confirmed_events:
            can = self._canonical.get(c.person_track_id, c.person_track_id)
            key = (can, c.violation_type)
            last = self._last_emit.get(key)
            if last is not None and t - last < self.cooldown_s:
                continue  # trùng trong cooldown -> chặn
            self._last_emit[key] = t
            out.append(ViolationEvent(
                event_id=uuid.uuid4().hex[:12],
                camera_id=self.camera_id,
                person_track_id=c.person_track_id,
                canonical_person_id=can,
                violation_type=c.violation_type,
                start_t=c.start_t,
                confirmed_t=c.confirmed_t,
                duration_s=c.duration_s,
            ))
        return out
