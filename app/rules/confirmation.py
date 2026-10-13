"""VI-06: Xác nhận vi phạm theo thời gian — dùng timestamp, không dựa cứng vào số frame.

Quy tắc:
  - Mỗi cặp (người, loại vi phạm) có một "chuỗi" vi phạm: bắt đầu từ frame đầu tiên
    ở trạng thái VIOLATION_CANDIDATE, cộng dồn thời gian (giây) của các frame
    CANDIDATE liên tiếp nhau.
  - Chuỗi đạt CONFIRMATION_DURATION_S (3s đề xuất, cấu hình được) -> phát event
    CONFIRMED_VIOLATION đúng một lần tại thời điểm vượt ngưỡng.

Xử lý track mất / che khuất:
  - Track biến mất khỏi frame -> reset chuỗi (đối tượng không còn được theo dõi).
  - COMPLIANT (thấy lại PPE) -> reset chuỗi (vi phạm đã chấm dứt).
  - UNKNOWN (che khuất/quá nhỏ/confidence thấp) -> GIỮ NGUYÊN chuỗi nhưng KHÔNG
    cộng thời gian (thiếu bằng chứng chưa phải hết vi phạm; tránh che khuất ngắn
    làm mất chuỗi). Đánh đổi: nếu vi phạm thật sự kết thúc trong lúc UNKNOWN, có
    thể xác nhận sai — chấp nhận để giảm false negative; cần đánh giá trên video thật.
"""
from dataclasses import dataclass

from app.config import CONFIRMATION_DURATION_S
from app.rules.safety import COMPLIANT, UNKNOWN, VIOLATION_CANDIDATE


@dataclass
class ConfirmedViolation:
    """Sự kiện vi phạm đã được xác nhận đủ thời lượng."""
    person_track_id: int
    violation_type: str  # "no_helmet" | "no_safety_vest"
    start_t: float       # thời điểm bắt đầu chuỗi vi phạm (giây)
    confirmed_t: float   # thời điểm xác nhận (giây)
    duration_s: float    # tổng thời gian CANDIDATE trong chuỗi


class ViolationConfirmation:
    """Bộ xác nhận theo thời gian — 1 instance riêng cho mỗi camera/nguồn video."""

    def __init__(self, duration_s=CONFIRMATION_DURATION_S):
        self.duration_s = float(duration_s)
        # (person, violation_type) -> [start_t, acc, prev_t, was_candidate]
        self._streaks = {}
        self._emitted = set()

    def update(self, t, person_states):
        """Nhận trạng thái VI-05 của một frame tại thời điểm t (giây).

        Trả danh sách ConfirmedViolation mới được xác nhận trong frame này.
        """
        t = float(t)
        present = {}      # person_track_id -> PersonState
        candidates = set()  # (person, violation_type) đang là CANDIDATE frame này
        for s in person_states:
            present[s.person_track_id] = s
            if s.status == VIOLATION_CANDIDATE:
                for vt in s.violations:
                    candidates.add((s.person_track_id, vt))

        confirmed = []

        for key in list(self._streaks.keys()):
            pid, vt = key
            start_t, acc, prev_t, was_cand = self._streaks[key]

            if pid not in present:
                # track mất -> reset
                del self._streaks[key]
                self._emitted.discard(key)
                continue

            s = present[pid]
            if key in candidates:
                if was_cand:
                    acc += t - prev_t
                self._streaks[key] = (start_t, acc, t, True)
                if key not in self._emitted and acc >= self.duration_s:
                    self._emitted.add(key)
                    confirmed.append(ConfirmedViolation(
                        person_track_id=pid, violation_type=vt,
                        start_t=round(start_t, 3), confirmed_t=round(t, 3),
                        duration_s=round(acc, 3)))
            elif s.status == COMPLIANT or s.status == VIOLATION_CANDIDATE:
                # thấy lại PPE / vi phạm này đã hết (còn CANDIDATE loại khác) -> reset
                del self._streaks[key]
                self._emitted.discard(key)
            else:
                # UNKNOWN -> giữ chuỗi, không cộng thời gian
                self._streaks[key] = (start_t, acc, t, False)

        for key in candidates:
            if key not in self._streaks:
                self._streaks[key] = (t, 0.0, t, True)

        return confirmed
