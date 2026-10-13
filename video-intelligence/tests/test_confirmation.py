"""VI-06: Kiểm thử xác nhận vi phạm theo thời gian (timestamp).

Chạy:  python tests/test_confirmation.py
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

FAILED = []


def check(name, cond):
    print(f"[{'OK  ' if cond else 'FAIL'}] {name}")
    if not cond:
        FAILED.append(name)


def feed(conf, t, states):
    return conf.update(float(t), states)


def cand(pid, *vts):
    return [PersonState(pid, VIOLATION_CANDIDATE, list(vts))]


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

# 4. UNKNOWN (che khuất) giữ chuỗi nhưng không cộng thời gian
c = ViolationConfirmation(3.0)
evs = []
for t in range(0, 3):            # t=0,1,2 -> 2 giây
    evs += feed(c, t, cand(1, VIOLATION_NO_VEST))
for t in range(3, 5):            # che khuất 2 giây -> không cộng, không reset
    evs += feed(c, t, [PersonState(1, UNKNOWN, reason="bị cắt khung")])
for t in range(5, 7):            # candidate lại: t=5 (+0), t=6 (+1) -> đủ 3
    evs += feed(c, t, cand(1, VIOLATION_NO_VEST))
check("UNKNOWN giữ chuỗi, không cộng thời gian",
      len(evs) == 1 and evs[0].confirmed_t == 6.0 and evs[0].duration_s == 3.0
      and evs[0].start_t == 0.0)

# 5. Track biến mất -> reset chuỗi
c = ViolationConfirmation(3.0)
evs = []
for t in range(0, 3):            # t=0,1,2 -> 2 giây
    evs += feed(c, t, cand(1, VIOLATION_NO_HELMET))
evs += feed(c, 3, [])            # người biến mất khỏi frame -> reset
for t in range(4, 7):            # xuất hiện lại: t=4,5,6 -> mới 2 giây, chưa xác nhận
    evs += feed(c, t, cand(1, VIOLATION_NO_HELMET))
check("track mất -> reset, chưa xác nhận lại ngay", evs == [])
evs += feed(c, 7, cand(1, VIOLATION_NO_HELMET))  # t=7 -> đủ 3
check("sau reset cần đủ 3 giây mới", len(evs) == 1 and evs[0].confirmed_t == 7.0)

# 6. Timestamp không đều (fps thay đổi) -> tính bằng giây, không phải số frame
c = ViolationConfirmation(3.0)
evs = []
for t in (0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0):  # 7 frame nhưng đúng 3 giây
    evs += feed(c, t, cand(1, VIOLATION_NO_HELMET))
check("tính theo timestamp (7 frame, 3.0s)", len(evs) == 1 and evs[0].confirmed_t == 3.0)
c2 = ViolationConfirmation(3.0)
evs2 = []
for t in (0.0, 0.5, 1.0, 1.5, 2.0, 2.5):  # 6 frame, 2.5 giây -> chưa đủ
    evs2 += feed(c2, t, cand(1, VIOLATION_NO_HELMET))
check("6 frame nhưng 2.5s -> chưa xác nhận", evs2 == [])

# 7. Hai loại vi phạm cùng người -> 2 event riêng
c = ViolationConfirmation(3.0)
evs = []
for t in range(0, 5):
    evs += feed(c, t, cand(1, VIOLATION_NO_HELMET, VIOLATION_NO_VEST))
types = sorted(e.violation_type for e in evs)
check("2 loại vi phạm -> 2 event", len(evs) == 2 and types == [VIOLATION_NO_HELMET, VIOLATION_NO_VEST])

# 8. Một loại hết (có PPE lại) chỉ reset loại đó, loại kia vẫn chạy
c = ViolationConfirmation(3.0)
evs = []
for t in range(0, 3):            # cả hai loại: 2 giây
    evs += feed(c, t, cand(1, VIOLATION_NO_HELMET, VIOLATION_NO_VEST))
evs += feed(c, 3, cand(1, VIOLATION_NO_HELMET))  # có áo lại -> reset áo, mũ tiếp tục
check("chỉ reset loại vi phạm đã hết",
      len(evs) == 1 and evs[0].violation_type == VIOLATION_NO_HELMET and evs[0].confirmed_t == 3.0)

# 9. Sau khi xác nhận, vi phạm chấm dứt rồi tái phạm -> xác nhận lại lần nữa
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

print()
if FAILED:
    print(f"KẾT QUẢ: {len(FAILED)} test FAIL — {FAILED}")
    sys.exit(1)
print("KẾT QUẢ: tất cả test OK")
