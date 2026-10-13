"""VI-07: Kiểm thử chống trùng (cooldown + track đổi ID).

Chạy:  python tests/test_deduplication.py  hoặc  pytest tests/test_deduplication.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.detection.adapter import Detection
from app.rules.confirmation import ConfirmedViolation
from app.rules.deduplication import ViolationDeduplicator


def D(cls, x1, y1, x2, y2, tid=1):
    return Detection(class_name=cls, confidence=0.9, bbox=(x1, y1, x2, y2), track_id=tid)


def conf(pid, vt="no_helmet", start=0.0, at=3.0, dur=3.0):
    return ConfirmedViolation(person_track_id=pid, violation_type=vt,
                              start_t=start, confirmed_t=at, duration_s=dur)


def frame(d, tid, x1, y1, x2, y2, t):
    return d.filter([], [D("person", x1, y1, x2, y2, tid=tid)], t)


P = lambda tid, x, y: D("person", x, y, x + 100, y + 300, tid=tid)


def run_tests():
    failed = []

    def check(name, cond):
        print(f"[{'OK  ' if cond else 'FAIL'}] {name}")
        if not cond:
            failed.append(name)

    # 1. Event đầu tiên -> phát bình thường, có event_id, đúng khóa camera
    d = ViolationDeduplicator(camera_id="cam-test")
    evs = d.filter([conf(1)], [P(1, 10, 10)], 3.0)
    check("event đầu tiên được phát", len(evs) == 1 and evs[0].event_id
          and evs[0].camera_id == "cam-test" and evs[0].person_track_id == 1)

    # 2. Vi phạm kéo dài, xác nhận liên tục -> không phát thêm (cooldown chặn)
    d = ViolationDeduplicator(camera_id="cam-test")
    out = []
    for t in (3.0, 3.5, 4.0, 4.5):
        out += d.filter([conf(1, at=t)], [P(1, 10, 10)], t)
    check("vi phạm kéo dài chỉ 1 event", len(out) == 1)

    # 3. Hết vi phạm, tái phạm trong vòng 30s -> bị chặn trùng
    d = ViolationDeduplicator(camera_id="cam-test", cooldown_s=30.0)
    out = []
    out += d.filter([conf(1, at=3.0)], [P(1, 10, 10)], 3.0)  # emit tại t=3
    out += d.filter([], [P(1, 10, 10)], 5.0)                  # hết vi phạm
    out += d.filter([conf(1, at=20.0)], [P(1, 10, 10)], 20.0) # tái phạm lúc t=20 (cách 17s < 30s)
    check("tái phạm trong 30s bị chặn", len(out) == 1)

    # 4. Tái phạm sau 30s -> phát lại bình thường
    d = ViolationDeduplicator(camera_id="cam-test", cooldown_s=30.0)
    out = []
    out += d.filter([conf(1, at=3.0)], [P(1, 10, 10)], 3.0)
    out += d.filter([], [P(1, 10, 10)], 5.0)
    out += d.filter([conf(1, at=35.0)], [P(1, 10, 10)], 35.0) # t=35 cách 32s > 30s
    check("tái phạm sau 30s được phát lại", len(out) == 2 and out[1].confirmed_t == 35.0)

    # 5. Track đổi ID (id-switch) ở vị trí gần -> nhận diện cùng 1 người, chặn trùng
    d = ViolationDeduplicator(camera_id="cam-test")
    out = []
    out += d.filter([conf(1, at=3.0)], [P(1, 10, 10)], 3.0)  # track 1 vi phạm tại t=3
    # track 1 biến mất lúc t=3.5, track 9 xuất hiện lúc t=4.0 ở vị trí gần (15, 12 so với 10, 10)
    out += d.filter([], [P(1, 10, 10)], 3.5)
    out += d.filter([], [P(9, 15, 12)], 4.0)                 # track 9 xuất hiện
    out += d.filter([conf(9, at=7.0)], [P(9, 15, 12)], 7.0)  # track 9 bị phát hiện vi phạm lúc t=7
    # Khoảng cách giữa (10,10) và (15,12) rất nhỏ so với chiều cao 300px -> cùng canonical
    # t=7 cách t=3 chỉ 4s < cooldown 30s -> phải bị chặn!
    check("track đổi ID gần đó -> chặn trùng", len(out) == 1)

    # 6. Track mới xuất hiện ở vị trí XA -> người khác, phát bình thường
    d = ViolationDeduplicator(camera_id="cam-test")
    out = []
    out += d.filter([conf(1, at=3.0)], [P(1, 10, 10)], 3.0)
    # track 1 mất lúc 3.5, track 9 xuất hiện lúc 4.0 nhưng ở tít góc kia (500, 500)
    out += d.filter([], [P(1, 10, 10)], 3.5)
    out += d.filter([], [P(9, 500, 500)], 4.0)
    out += d.filter([conf(9, at=7.0)], [P(9, 500, 500)], 7.0)
    check("track mới ở xa -> phát bình thường", len(out) == 2 and out[1].person_track_id == 9)

    # 7. Khác loại vi phạm -> phát độc lập dù cùng người
    d = ViolationDeduplicator(camera_id="cam-test")
    out = []
    out += d.filter([conf(1, vt="no_helmet", at=3.0)], [P(1, 10, 10)], 3.0)
    out += d.filter([conf(1, vt="no_safety_vest", at=5.0)], [P(1, 10, 10)], 5.0)
    check("khác loại vi phạm -> 2 event", len(out) == 2
          and {e.violation_type for e in out} == {"no_helmet", "no_safety_vest"})

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

    return failed


def test_deduplication():
    failed = run_tests()
    assert len(failed) == 0, f"Failed tests: {failed}"


if __name__ == "__main__":
    failed = run_tests()
    print()
    if failed:
        print(f"KẾT QUẢ: {len(failed)} test FAIL — {failed}")
        sys.exit(1)
    print("KẾT QUẢ: tất cả test OK")
