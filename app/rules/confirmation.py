"""VI-06: Xác nhận vi phạm theo thời gian — dùng timestamp, không dựa cứng vào số frame.

Quy tắc:
  - Mỗi cặp (người, loại vi phạm) có một "chuỗi" vi phạm: bắt đầu từ frame đầu tiên
    ở trạng thái VIOLATION_CANDIDATE, cộng dồn thời gian (giây) của các frame
    CANDIDATE liên tiếp nhau.
  - Chuỗi đạt ngưỡng thời gian của loại vi phạm tương ứng (Mũ: 2.0s, Áo: 4.0s cấu hình được)
    -> phát event CONFIRMED_VIOLATION đúng một lần tại thời điểm vượt ngưỡng.

Xử lý track mất / che khuất:
  - Track biến mất khỏi frame -> reset chuỗi (đối tượng không còn được theo dõi).
  - COMPLIANT (thấy lại PPE) -> reset chuỗi (vi phạm đã chấm dứt).
  - UNKNOWN (che khuất/quá nhỏ/confidence thấp/xoay ngang) -> GIỮ NGUYÊN chuỗi nhưng KHÔNG
    cộng thời gian (thiếu bằng chứng chưa phải hết vi phạm; tránh che khuất ngắn
    làm mất chuỗi). Đánh đổi: nếu vi phạm thật sự kết thúc trong lúc UNKNOWN, có
    thể xác nhận sai — chấp nhận để giảm false negative; cần đánh giá trên video thật.
"""
from dataclasses import dataclass

from app.config import (
    CONFIRMATION_DURATION_S,
    CONFIRMATION_DURATION_HELMET_S,
    CONFIRMATION_DURATION_VEST_S,
    CONFIRMATION_DURATION_MAP,
    COMPLIANT_GRACE_PERIOD_S,
)
from app.constants import (
    VIOLATION_NO_HELMET,
    VIOLATION_NO_VEST,
)
from app.rules.safety import COMPLIANT, UNKNOWN, VIOLATION_CANDIDATE


@dataclass
class ConfirmedViolation:
    """Sự kiện vi phạm đã được xác nhận đủ thời lượng."""
    person_track_id: int
    violation_type: str  # "no_helmet" | "no_safety_vest"
    start_t: float       # thời điểm bắt đầu chuỗi vi phạm (giây)
    confirmed_t: float   # thời điểm xác nhận (giây)
    duration_s: float    # tổng thời gian CANDIDATE trong chuỗi
    reference_frame: object = None  # Ảnh đối chứng chụp trước đó (tuân thủ hoặc t=0s)
    reference_bbox: tuple = None    # Bounding box của người tại thời điểm chụp đối chứng (x1, y1, x2, y2)
    reference_type: str = "COMPLIANT"  # "COMPLIANT" | "START"


class ViolationConfirmation:
    """Bộ xác nhận theo thời gian — 1 instance riêng cho mỗi camera/nguồn video."""

    def __init__(
        self,
        duration_s=CONFIRMATION_DURATION_S,
        duration_map=None,
        grace_period_s=COMPLIANT_GRACE_PERIOD_S,
        max_gap_s=2.0,
        max_compliant_frames=50,
    ):
        self.duration_s = float(duration_s)
        self.grace_period_s = float(grace_period_s)
        # Nếu duration_map được truyền vào tường minh: sử dụng nó
        # Nếu không truyền:
        # - Nếu duration_s khác DEFAULT (ví dụ test truyền ViolationConfirmation(3.0)): dùng duration_s cho tất cả loại vi phạm
        # - Nếu duration_s là mặc định: sử dụng CONFIRMATION_DURATION_MAP ({NO_HELMET: 2.0s, NO_VEST: 4.0s})
        if duration_map is not None:
            self.duration_map = dict(duration_map)
        elif self.duration_s != float(CONFIRMATION_DURATION_S):
            self.duration_map = {
                VIOLATION_NO_HELMET: self.duration_s,
                VIOLATION_NO_VEST: self.duration_s,
            }
        else:
            self.duration_map = dict(CONFIRMATION_DURATION_MAP)

        # SEC-16: Cap maximum allowable frame interval delta to prevent instant violation on stream reconnect
        self.max_gap_s = float(max_gap_s)
        # SEC-26: Bound maximum cached compliant frames to prevent memory exhaustion (CWE-400)
        self.max_compliant_frames = int(max_compliant_frames)
        # (person, violation_type) -> [start_t, acc, prev_t, was_candidate]
        self._streaks = {}
        self._emitted = set()
        # Bộ đệm ảnh đối chứng: lưu (frame, bbox) hợp lệ gần nhất hoặc frame bắt đầu vi phạm
        self._compliant_frames = {}  # pid -> (frame, bbox)
        self._start_frames = {}      # (pid, vt) -> (frame, bbox)
        self._last_compliant_t = {}  # pid -> float (thời điểm gần nhất tuân thủ)

    def get_duration(self, violation_type: str) -> float:
        """Lấy ngưỡng thời gian xác nhận (giây) theo loại vi phạm."""
        val = self.duration_map.get(violation_type, self.duration_s)
        try:
            return max(0.5, float(val))
        except (TypeError, ValueError):
            return 2.0

    def update(self, t, person_states, frame=None, detections=None):
        """Nhận trạng thái VI-05 của một frame tại thời điểm t (giây).

        Trả danh sách ConfirmedViolation mới được xác nhận trong frame này.
        """
        t = float(t)
        present = {}      # person_track_id -> PersonState
        candidates = set()  # (person, violation_type) đang là CANDIDATE frame này

        # Trích xuất tọa độ bbox của từng person trong frame hiện tại
        person_bboxes = {}
        if detections is not None:
            for d in detections:
                if getattr(d, "class_name", "") == "person" and getattr(d, "track_id", None) is not None:
                    person_bboxes[d.track_id] = tuple(d.bbox) if hasattr(d, "bbox") else None

        for s in person_states:
            present[s.person_track_id] = s
            if s.status == COMPLIANT:
                self._last_compliant_t[s.person_track_id] = t
                if frame is not None:
                    # Cập nhật ảnh tuân thủ hợp lệ gần nhất kèm bbox của track này
                    self._compliant_frames[s.person_track_id] = (frame.copy(), person_bboxes.get(s.person_track_id))
            elif s.status == VIOLATION_CANDIDATE:
                for vt in s.violations:
                    candidates.add((s.person_track_id, vt))

        # SEC-26: Dọn dẹp chủ động toàn bộ compliant_frames của các track không còn trong present
        # và áp dụng giới hạn dung lượng cache tối đa để ngăn chặn rò rỉ RAM (OOM DoS)
        active_pids = set(present.keys())
        for pid in list(self._compliant_frames.keys()):
            if pid not in active_pids:
                self._compliant_frames.pop(pid, None)

        for pid in list(self._last_compliant_t.keys()):
            if pid not in active_pids:
                self._last_compliant_t.pop(pid, None)

        if len(self._compliant_frames) > self.max_compliant_frames:
            overflow = len(self._compliant_frames) - self.max_compliant_frames
            for k in list(self._compliant_frames.keys())[:overflow]:
                self._compliant_frames.pop(k, None)

        confirmed = []

        for key in list(self._streaks.keys()):
            pid, vt = key
            start_t, acc, prev_t, was_cand = self._streaks[key]

            if pid not in present:
                # track mất -> reset streak và giải phóng bộ đệm để tránh rò rỉ bộ nhớ
                del self._streaks[key]
                self._emitted.discard(key)
                self._compliant_frames.pop(pid, None)
                self._start_frames.pop(key, None)
                self._last_compliant_t.pop(pid, None)
                continue

            s = present[pid]
            if key in candidates:
                if was_cand:
                    # SEC-16: Validate time delta, ignore backward timestamps and reset on network gap
                    dt = t - prev_t
                    if dt <= 0:
                        dt = 0.0
                    elif dt > self.max_gap_s:
                        # Reset streak on camera reconnect / network drop to prevent instant false confirmation
                        acc = 0.0
                        start_t = t
                        dt = 0.0
                    acc += dt
                self._streaks[key] = (start_t, acc, t, True)
                req_duration = self.get_duration(vt)
                if key not in self._emitted and acc >= req_duration:
                    self._emitted.add(key)
                    # Chọn ảnh đối chứng: ưu tiên ảnh hợp lệ gần nhất, nếu không có thì dùng ảnh lúc bắt đầu t=0s
                    ref_data = self._compliant_frames.get(pid)
                    ref_type = "COMPLIANT"
                    if ref_data is None:
                        ref_data = self._start_frames.get(key)
                        ref_type = "START"

                    ref_frame = None
                    ref_bbox = None
                    if isinstance(ref_data, tuple):
                        ref_frame, ref_bbox = ref_data
                    else:
                        ref_frame = ref_data

                    # SEC-26: Giải phóng frame t=0s trong _start_frames sau khi đã đính kèm vào event phát ra
                    self._start_frames.pop(key, None)

                    confirmed.append(ConfirmedViolation(
                        person_track_id=pid, violation_type=vt,
                        start_t=round(start_t, 3), confirmed_t=round(t, 3),
                        duration_s=round(acc, 3),
                        reference_frame=ref_frame,
                        reference_bbox=ref_bbox,
                        reference_type=ref_type))
            elif s.status == COMPLIANT or s.status == VIOLATION_CANDIDATE:
                # thấy lại PPE / vi phạm này đã hết (còn CANDIDATE loại khác) -> reset
                del self._streaks[key]
                self._emitted.discard(key)
                self._start_frames.pop(key, None)
            else:
                # UNKNOWN -> giữ chuỗi, không cộng thời gian
                self._streaks[key] = (start_t, acc, t, False)

        for key in candidates:
            if key not in self._streaks:
                last_comp = self._last_compliant_t.get(key[0])
                if self.grace_period_s > 0 and last_comp is not None and (t - last_comp) < self.grace_period_s:
                    continue
                self._streaks[key] = (t, 0.0, t, True)
                if frame is not None:
                    # Lưu frame và bbox tại giây 0s bắt đầu vi phạm
                    self._start_frames[key] = (frame.copy(), person_bboxes.get(key[0]))

        return confirmed

    def reset(self):
        """Reset toàn bộ chuỗi theo dõi và bộ đệm (dùng khi luồng ngắt quãng hoặc chuyển cảnh)."""
        self._streaks.clear()
        self._emitted.clear()
        self._compliant_frames.clear()
        self._start_frames.clear()
        self._last_compliant_t.clear()
