"""VI-06: Kiểm thử xác nhận vi phạm theo thời gian (timestamp).

Chạy:  python tests/test_confirmation.py  hoặc  pytest tests/test_confirmation.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.rules.confirmation import ViolationConfirmation
from app.rules.safety import (
    COMPLIANT,
    UNKNOWN,
    VIOLATION_CANDIDATE,
    VIOLATION_NO_HELMET,
    VIOLATION_NO_VEST,
    PersonState,
)


def feed(conf, t, states):
    return conf.update(float(t), states)


def cand(pid, *vts):
    return [PersonState(pid, VIOLATION_CANDIDATE, list(vts))]


def run_tests():
    failed = []

    def check(name, cond):
        print(f"[{'OK  ' if cond else 'FAIL'}] {name}")
        if not cond:
            failed.append(name)

    # 1. Vi phạm liên tục đủ 3 giây -> xác nhận đúng 1 event tại thời điểm vượt ngưỡng
    c = ViolationConfirmation(3.0)
    evs = []
    for t in range(0, 5):
        evs += feed(c, t, cand(1, VIOLATION_NO_HELMET))
    check("3 giây liên tục -> xác nhận", len(evs) == 1 and evs[0].confirmed_t == 3.0
          and evs[0].person_track_id == 1 and evs[0].duration_s == 3.0)

    # 2. Chưa đủ 3 giây -> không xác nhận
    c = ViolationConfirmation(3.0)
    evs = []
    for t in range(0, 3):  # t=0,1,2 -> chỉ 2 giây
        evs += feed(c, t, cand(1, VIOLATION_NO_HELMET))
    check("dưới 3 giây -> không xác nhận", evs == [])

    # 3. COMPLIANT làm reset chuỗi -> phải đủ 3 giây mới sau đó
    c = ViolationConfirmation(3.0)
    evs = []
    for t in range(0, 2):            # 2 giây candidate
        evs += feed(c, t, cand(1, VIOLATION_NO_HELMET))
    evs += feed(c, 2, [PersonState(1, COMPLIANT)])  # có mũ lại -> reset
    for t in range(3, 7):            # candidate lại: t=3..6 -> xác nhận tại t=6
        evs += feed(c, t, cand(1, VIOLATION_NO_HELMET))
    check("COMPLIANT reset chuỗi", len(evs) == 1 and evs[0].confirmed_t == 6.0)

    # 4. UNKNOWN không làm reset chuỗi, nhưng không tính thời gian
    c = ViolationConfirmation(3.0)
    evs = []
    evs += feed(c, 0, cand(1, VIOLATION_NO_HELMET))
    evs += feed(c, 1, cand(1, VIOLATION_NO_HELMET))
    evs += feed(c, 2, [PersonState(1, UNKNOWN, reason="occluded")])  # pause
    evs += feed(c, 3, [PersonState(1, UNKNOWN, reason="small")])     # pause
    evs += feed(c, 4, cand(1, VIOLATION_NO_HELMET))  # candidate tiếp
    # Tại t=4: candidate chạy được t=0..1 (1s) + t=4 (0s) = 1s -> chưa đủ 3s
    check("UNKNOWN giữ chuỗi, không cộng thời gian", evs == [])
    evs += feed(c, 5, cand(1, VIOLATION_NO_HELMET))
    evs += feed(c, 6, cand(1, VIOLATION_NO_HELMET))  # t=4..6 là thêm 2s -> tổng 3s
    assert len(evs) == 1 and evs[0].confirmed_t == 6.0

    # 5. Track biến mất (không có trong state) -> reset
    c = ViolationConfirmation(3.0)
    evs = []
    evs += feed(c, 0, cand(1, VIOLATION_NO_HELMET))
    evs += feed(c, 1, cand(1, VIOLATION_NO_HELMET))
    evs += feed(c, 2, [])                            # track 1 mất
    evs += feed(c, 3, cand(1, VIOLATION_NO_HELMET))  # xuất hiện lại -> reset
    evs += feed(c, 4, cand(1, VIOLATION_NO_HELMET))
    check("track mất -> reset, chưa xác nhận lại ngay", evs == [])
    evs += feed(c, 6, cand(1, VIOLATION_NO_HELMET))  # t=3..6 đủ 3s
    check("sau reset cần đủ 3 giây mới", len(evs) == 1 and evs[0].confirmed_t == 6.0)

    # 6. Tính theo TIMESTAMP chứ không theo số frame (FPS biến thiên)
    c = ViolationConfirmation(3.0)
    # 7 frame, nhưng phân bố trong đúng 3.0s (0.0 -> 3.0)
    timestamps = [0.0, 0.4, 0.9, 1.5, 2.1, 2.7, 3.0]
    evs = []
    for ts in timestamps:
        evs += feed(c, ts, cand(1, VIOLATION_NO_HELMET))
    check("tính theo timestamp (7 frame, 3.0s)", len(evs) == 1 and evs[0].confirmed_t == 3.0)

    # 7. Nhiều frame nhưng timestamp chưa đủ 3s -> KHÔNG xác nhận
    c = ViolationConfirmation(3.0)
    timestamps = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5]  # 6 frame nhưng chỉ 2.5s
    evs = []
    for ts in timestamps:
        evs += feed(c, ts, cand(1, VIOLATION_NO_HELMET))
    check("6 frame nhưng 2.5s -> chưa xác nhận", evs == [])

    # 8. Hai loại vi phạm độc lập: người 1 thiếu mũ từ t=0, thiếu áo từ t=2
    c = ViolationConfirmation(3.0)
    evs = []
    for t in (0, 1):
        evs += feed(c, t, cand(1, VIOLATION_NO_HELMET))
    for t in (2, 3, 4, 5):
        evs += feed(c, t, cand(1, VIOLATION_NO_HELMET, VIOLATION_NO_VEST))
    # helmet đủ 3s tại t=3; vest đủ 3s tại t=5
    check("2 loại vi phạm -> 2 event",
          len(evs) == 2
          and {e.violation_type for e in evs} == {VIOLATION_NO_HELMET, VIOLATION_NO_VEST}
          and {e.confirmed_t for e in evs} == {3.0, 5.0})

    # 8b. Đội mũ nhưng vẫn thiếu áo -> chỉ reset helmet, vest chạy tiếp
    c = ViolationConfirmation(3.0)
    evs = []
    for t in range(0, 3):  # thiếu cả hai t=0,1,2
        evs += feed(c, t, cand(1, VIOLATION_NO_HELMET, VIOLATION_NO_VEST))
    # tại t=3: đội mũ, vẫn thiếu áo
    evs += feed(c, 3, [PersonState(1, VIOLATION_CANDIDATE, [VIOLATION_NO_VEST])])
    check("chỉ reset loại vi phạm đã hết", len(evs) == 1 and evs[0].violation_type == VIOLATION_NO_VEST)

    # 9. Đã xác nhận một lần, sau đó COMPLIANT, rồi vi phạm lại -> xác nhận LẦN HAI
    c = ViolationConfirmation(3.0)
    evs = []
    for t in range(0, 4):
        evs += feed(c, t, cand(1, VIOLATION_NO_HELMET))
    evs += feed(c, 4, [PersonState(1, COMPLIANT)])
    for t in range(5, 9):
        evs += feed(c, t, cand(1, VIOLATION_NO_HELMET))
    check("tái phạm -> xác nhận lại", len(evs) == 2 and evs[1].confirmed_t == 8.0)

    # 10. Thời lượng cấu hình được
    c = ViolationConfirmation(1.0)
    evs = []
    for t in range(0, 3):
        evs += feed(c, t, cand(1, VIOLATION_NO_HELMET))
    check("duration cấu hình được (1s)", len(evs) == 1 and evs[0].confirmed_t == 1.0)

    return failed


def test_confirmation():
    failed = run_tests()
    assert len(failed) == 0, f"Failed tests: {failed}"


if __name__ == "__main__":
    failed = run_tests()
    print()
    if failed:
        print(f"KẾT QUẢ: {len(failed)} test FAIL — {failed}")
        sys.exit(1)
    print("KẾT QUẢ: tất cả test OK")
