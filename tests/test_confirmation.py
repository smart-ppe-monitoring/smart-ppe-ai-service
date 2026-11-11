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

    # 11. SEC-26: Dọn dẹp bộ đệm compliant_frames khi track biến mất và giới hạn dung lượng cache
    import numpy as np
    c = ViolationConfirmation(2.0, max_compliant_frames=3)
    dummy_frame = np.zeros((10, 10, 3), dtype=np.uint8)
    # Track 10 tuân thủ ở t=0
    c.update(0.0, [PersonState(10, COMPLIANT)], frame=dummy_frame)
    check("SEC-26: lưu frame tuân thủ của track 10", 10 in c._compliant_frames)

    # Ở t=1, track 10 rời khỏi camera
    c.update(1.0, [], frame=dummy_frame)
    check("SEC-26: tự động thu hồi frame khi track 10 rời khỏi camera", 10 not in c._compliant_frames)

    # Bounded cache limit (max 3 frames)
    for tid in range(1, 6):
        c.update(2.0, [PersonState(i, COMPLIANT) for i in range(1, tid + 1)], frame=dummy_frame)
    check("SEC-26: số lượng compliant_frames không vượt quá max_compliant_frames", len(c._compliant_frames) <= 3)

    # 12. Differentiated confirmation duration (Helmet 2.0s vs Vest 4.0s)
    c_diff = ViolationConfirmation()
    evs_helmet = []
    evs_vest = []
    # Feed helmet candidate at t=0, 1, 2 -> should confirm at t=2.0
    for t in [0.0, 1.0, 2.0]:
        evs_helmet += feed(c_diff, t, cand(1, VIOLATION_NO_HELMET))
    check("mũ bảo hộ xác nhận tại 2.0s", len(evs_helmet) == 1 and evs_helmet[0].confirmed_t == 2.0)

    # Feed vest candidate at t=0, 1, 2 -> should NOT confirm yet at t=2.0
    c_vest = ViolationConfirmation()
    for t in [0.0, 1.0, 2.0]:
        evs_vest += feed(c_vest, t, cand(2, VIOLATION_NO_VEST))
    check("áo phản quang chưa xác nhận tại 2.0s (ngưỡng 4.0s)", len(evs_vest) == 0)

    # Feed vest up to t=3, 4 -> confirms at t=4.0
    for t in [3.0, 4.0]:
        evs_vest += feed(c_vest, t, cand(2, VIOLATION_NO_VEST))
    check("áo phản quang xác nhận tại 4.0s", len(evs_vest) == 1 and evs_vest[0].confirmed_t == 4.0)

    # 13. UNKNOWN (xoay ngang) tạm dừng streak áo mà không làm tăng streak
    c_pause = ViolationConfirmation()
    # t=0, 1 -> candidate áo (acc = 1.0s)
    feed(c_pause, 0.0, cand(3, VIOLATION_NO_VEST))
    feed(c_pause, 1.0, cand(3, VIOLATION_NO_VEST))
    # t=2, 3 -> UNKNOWN do xoay ngang
    feed(c_pause, 2.0, [PersonState(3, UNKNOWN, reason="xoay ngang")])
    feed(c_pause, 3.0, [PersonState(3, UNKNOWN, reason="xoay ngang")])
    # t=4 -> candidate áo lại (acc = 2.0s, chưa đủ 4.0s)
    evs_pause = feed(c_pause, 4.0, cand(3, VIOLATION_NO_VEST))
    check("UNKNOWN tạm dừng tích lũy streak áo (chưa kích hoạt vi phạm ở t=4)", len(evs_pause) == 0)

    # 14. COMPLIANT grace period trì hoãn khởi động streak vi phạm
    c_grace = ViolationConfirmation(duration_s=2.0, grace_period_s=1.0)
    # t=0.0: COMPLIANT
    feed(c_grace, 0.0, [PersonState(4, COMPLIANT)])
    # t=0.5: CANDIDATE nhưng chưa qua grace period 1.0s (0.5 - 0.0 = 0.5 < 1.0) -> streak chưa tạo
    feed(c_grace, 0.5, cand(4, VIOLATION_NO_HELMET))
    check("trong grace period (dt=0.5 < 1.0s) -> chưa tạo streak", (4, VIOLATION_NO_HELMET) not in c_grace._streaks)
    # t=1.0: Đã đủ grace period (1.0 - 0.0 = 1.0 >= 1.0) -> bắt đầu streak tại t=1.0
    feed(c_grace, 1.0, cand(4, VIOLATION_NO_HELMET))
    check("sau grace period -> bắt đầu streak", (4, VIOLATION_NO_HELMET) in c_grace._streaks)

    # 15. Kiểm tra thời gian xác nhận riêng biệt giữa Mũ (2s) và Áo (4s)
    c_diff = ViolationConfirmation()
    # Kiểm tra mũ: 2s
    for t in [0.0, 1.0, 1.9]:
        evs_h = feed(c_diff, t, cand(10, VIOLATION_NO_HELMET))
    check("Mũ chưa đủ 2.0s -> chưa xác nhận", len(evs_h) == 0)
    evs_h_conf = feed(c_diff, 2.0, cand(10, VIOLATION_NO_HELMET))
    check("Mũ đủ 2.0s -> xác nhận", len(evs_h_conf) == 1 and evs_h_conf[0].confirmed_t == 2.0)

    # Kiểm tra áo: 2s chưa được xác nhận, phải đủ 4s
    for t in [0.0, 1.0, 2.0, 3.0, 3.9]:
        evs_v = feed(c_diff, t, cand(20, VIOLATION_NO_VEST))
    check("Áo ở 2.0s hoặc 3.9s -> chưa xác nhận (cần 4.0s)", len(evs_v) == 0)
    evs_v_conf = feed(c_diff, 4.0, cand(20, VIOLATION_NO_VEST))
    check("Áo đủ 4.0s -> xác nhận", len(evs_v_conf) == 1 and evs_v_conf[0].confirmed_t == 4.0)

    # Kiểm tra alias chữ thường "no_safety_vest" cũng nhận đúng 4.0s
    for t in [0.0, 1.0, 2.0, 3.0, 3.9]:
        evs_alias = feed(c_diff, t, cand(30, "no_safety_vest"))
    check("Alias no_safety_vest ở 3.9s -> chưa xác nhận", len(evs_alias) == 0)
    evs_alias_conf = feed(c_diff, 4.0, cand(30, "no_safety_vest"))
    check("Alias no_safety_vest đủ 4.0s -> xác nhận", len(evs_alias_conf) == 1 and evs_alias_conf[0].confirmed_t == 4.0)

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
