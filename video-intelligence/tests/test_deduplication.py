"""VI-07: Kiểm thử chống trùng (cooldown + track đổi ID).

Chạy:  python tests/test_deduplication.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.detection.adapter import Detection
from app.rules.confirmation import ConfirmedViolation
from app.rules.deduplication import ViolationDeduplicator

FAILED = []


def check(name, cond):
    print(f"[{'OK  ' if cond else 'FAIL'}] {name}")
    if not cond:
        FAILED.append(name)


def D(cls, x1, y1, x2, y2, tid=1):
    return Detection(class_name=cls, confidence=0.9, bbox=(x1, y1, x2, y2), track_id=tid)


def conf(pid, vt="no_helmet", start=0.0, at=3.0, dur=3.0):
    return ConfirmedViolation(person_track_id=pid, violation_type=vt,
                              start_t=start, confirmed_t=at, duration_s=dur)


def frame(d, tid, x1, y1, x2, y2, t):
    return d.filter([], [D("person", x1, y1, x2, y2, tid=tid)], t)


P = lambda tid, x, y: D("person", x, y, x + 100, y + 300, tid=tid)

# 1. Event đầu tiên -> phát bình thường, có event_id, đúng khóa camera
d = ViolationDeduplicator(camera_id="cam-test")
evs = d.filter([conf(1)], [P(1, 10, 10)], 3.0)
check("event đầu tiên được phát", len(evs) == 1 and evs[0].event_id
      and evs[0].camera_id == "cam-test" and evs[0].person_track_id == 1)

# 2. Vi phạm kéo dài: các event xác nhận lặp lại của cùng chuỗi -> chỉ phát 1 lần
d = ViolationDeduplicator(camera_id="cam-test")
out = []
for t in (3.0, 3.1, 3.2, 4.0):
    out += d.filter([conf(1, at=t)], [P(1, 10, 10)], t)
check("vi phạm kéo dài chỉ 1 event", len(out) == 1)

# 3. Tái phạm trong cooldown 30s -> chặn
d = ViolationDeduplicator(camera_id="cam-test")
out = []
out += d.filter([conf(1, at=3.0)], [P(1, 10, 10)], 3.0)
out += d.filter([conf(1, at=20.0)], [P(1, 10, 10)], 20.0)   # tái phạm sau 17s < 30s
check("tái phạm trong 30s bị chặn", len(out) == 1)

# 4. Tái phạm sau cooldown -> phát lại
d = ViolationDeduplicator(camera_id="cam-test")
out = []
out += d.filter([conf(1, at=3.0)], [P(1, 10, 10)], 3.0)
out += d.filter([conf(1, at=40.0)], [P(1, 10, 10)], 40.0)
check("tái phạm sau 30s được phát lại", len(out) == 2)

# 5. Track đổi ID cùng vị trí -> chặn (coi là cùng người)
d = ViolationDeduplicator(camera_id="cam-test")
out = []
out += d.filter([conf(1, at=3.0)], [P(1, 10, 10)], 3.0)
out += d.filter([], [P(1, 10, 10)], 3.5)                     # track 1 vẫn còn
out += d.filter([], [P(9, 12, 11)], 4.0)                     # track 1 mất, track 9 mới gần đó
out += d.filter([conf(9, at=7.5)], [P(9, 12, 11)], 7.5)      # track 9 xác nhận vi phạm
check("track đổi ID gần đó -> chặn trùng", len(out) == 1)

# 6. Track mới ở xa (người khác) -> không bị chặn
d = ViolationDeduplicator(camera_id="cam-test")
out = []
out += d.filter([conf(1, at=3.0)], [P(1, 10, 10)], 3.0)
out += d.filter([], [P(1, 10, 10)], 3.5)
out += d.filter([], [P(9, 800, 10)], 4.0)                    # người mới ở xa
out += d.filter([conf(9, at=7.5)], [P(9, 800, 10)], 7.5)
check("track mới ở xa -> phát bình thường", len(out) == 2)

# 7. Khác violation_type cùng người -> không bị chặn (khóa có violationType)
d = ViolationDeduplicator(camera_id="cam-test")
out = []
out += d.filter([conf(1, vt="no_helmet", at=3.0)], [P(1, 10, 10)], 3.0)
out += d.filter([conf(1, vt="no_safety_vest", at=3.5)], [P(1, 10, 10)], 3.5)
check("khác loại vi phạm -> 2 event", len(out) == 2)

# 8. Track mất quá lâu mới xuất hiện lại -> coi là người mới (không nối ID)
d = ViolationDeduplicator(camera_id="cam-test")
out = []
out += d.filter([conf(1, at=3.0)], [P(1, 10, 10)], 3.0)
out += d.filter([], [P(1, 10, 10)], 3.5)                     # mất lúc 3.5
out += d.filter([], [P(9, 12, 11)], 7.0)                     # xuất hiện sau 3.5s > grace 2s
out += d.filter([conf(9, at=10.5)], [P(9, 12, 11)], 10.5)
check("xuất hiện lại sau grace -> người mới, phát event", len(out) == 2)

# 9. event_id là duy nhất
d = ViolationDeduplicator(camera_id="cam-test")
ids = set()
for t in (3.0, 40.0, 80.0):
    for e in d.filter([conf(1, at=t)], [P(1, 10, 10)], t):
        ids.add(e.event_id)
check("event_id duy nhất", len(ids) == 3)

print()
if FAILED:
    print(f"KẾT QUẢ: {len(FAILED)} test FAIL — {FAILED}")
    sys.exit(1)
print("KẾT QUẢ: tất cả test OK")
