"""VI-08: Kiểm thử lưu snapshot bằng chứng.

Chạy:  python tests/test_evidence.py  hoặc  pytest tests/test_evidence.py
"""
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np

from app.detection.adapter import Detection
from app.evidence.snapshot import save_evidence
from app.rules.deduplication import ViolationEvent


def ev(event_id="a1b2c3d4e5f6"):
    return ViolationEvent(event_id=event_id, camera_id="cam-test", person_track_id=1,
                          canonical_person_id=1, violation_type="no_helmet",
                          start_t=0.0, confirmed_t=3.0, duration_s=3.0)


def fake_frame(w=64, h=48):
    return np.zeros((h, w, 3), dtype=np.uint8)


def D(cls, x1, y1, x2, y2, tid=1):
    return Detection(class_name=cls, confidence=0.9, bbox=(x1, y1, x2, y2), track_id=tid)


def run_tests():
    failed = []

    def check(name, cond):
        print(f"[{'OK  ' if cond else 'FAIL'}] {name}")
        if not cond:
            failed.append(name)

    with tempfile.TemporaryDirectory() as tmp:
        # 1. Lưu thành công: đúng tên file, ảnh đọc lại được, đúng kích thước
        r = save_evidence(fake_frame(64, 48), [], ev("111111111111"), Path(tmp) / "sub" / "ev")
        p = Path(r.path)
        img = cv2.imread(str(p))
        check("lưu thành công status=saved", r.status == "saved" and r.error == "")
        check("tên file = <event_id>.jpg", p.name == "111111111111.jpg")
        check("ảnh đọc lại được, đúng kích thước",
              img is not None and img.shape == (48, 64, 3))

        # 2. out_dir chưa tồn tại -> tự tạo (đã kiểm ở trên: sub/ev được tạo)
        check("out_dir tự tạo", p.parent.is_dir())

        # 3. event_id khác nhau -> file khác nhau
        r2 = save_evidence(fake_frame(), [], ev("222222222222"), Path(tmp) / "ev")
        check("event khác -> file khác", r2.status == "saved" and Path(r2.path).name == "222222222222.jpg")

        # 4. Có detection người vi phạm -> ảnh có lớp đỏ mờ trên vùng bbox
        r3 = save_evidence(fake_frame(64, 48), [D("person", 10, 10, 50, 40, tid=1)],
                           ev("333333333333"), Path(tmp) / "ev")
        img3 = cv2.imread(Path(r3.path))
        cx, cy = 30, 25  # điểm giữa bbox (10,10)-(50,40)
        b, g, rr = img3[cy, cx]
        check("người vi phạm được tô đỏ (R > B tại giữa bbox)", rr > b + 40)

        # 5. Track người không có trong detections -> vẫn lưu ảnh, không crash, không overlay
        r4 = save_evidence(fake_frame(64, 48), [D("helmet", 5, 5, 20, 15, tid=7)],
                           ev("444444444444"), Path(tmp) / "ev")
        img4 = cv2.imread(Path(r4.path))
        check("không tìm thấy người -> vẫn lưu, không overlay",
              r4.status == "saved" and img4 is not None and img4[25, 30][2] == 0)

        # 6. out_dir bị chiếm bởi một FILE -> status=error, không ném exception
        blocked = Path(tmp) / "blocked"
        blocked.write_text("đây là file, không phải thư mục", encoding="utf-8")
        try:
            r5 = save_evidence(fake_frame(), [D("person", 10, 10, 50, 40)],
                               ev("555555555555"), blocked)
            check("thư mục lỗi -> status=error có message", r5.status == "error" and r5.error)
        except Exception as e:
            check("thư mục lỗi -> status=error có message (không ném)", False)
            print("   exception:", e)

        # 7. frame None -> status=error
        r6 = save_evidence(None, [], ev("666666666666"), Path(tmp) / "ev")
        check("frame None -> status=error", r6.status == "error" and r6.error)

        # 8. no_safety_vest -> nhãn tương ứng vẫn lưu OK
        ev_vest = ev("777777777777")
        ev_vest.violation_type = "no_safety_vest"
        r7 = save_evidence(fake_frame(), [D("person", 10, 10, 50, 40, tid=1)],
                           ev_vest, Path(tmp) / "ev")
        check("no_safety_vest lưu OK", r7.status == "saved")

    return failed


def test_evidence():
    failed = run_tests()
    assert len(failed) == 0, f"Failed tests: {failed}"


if __name__ == "__main__":
    failed = run_tests()
    print()
    if failed:
        print(f"KẾT QUẢ: {len(failed)} test FAIL — {failed}")
        sys.exit(1)
    print("KẾT QUẢ: tất cả test OK")
