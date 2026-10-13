"""VI-04: Kiểm thử ghép PPE–Person bằng bbox nhân tạo (không cần model).

Chạy:  python tests/test_association.py  hoặc  pytest tests/test_association.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.association.associator import MATCHED, UNKNOWN, associate
from app.detection.adapter import Detection


def D(cls, x1, y1, x2, y2, tid=1):
    return Detection(class_name=cls, confidence=0.9, bbox=(x1, y1, x2, y2), track_id=tid)


def run_tests():
    failed = []

    def check(name, cond, detail=""):
        tag = "OK  " if cond else "FAIL"
        print(f"[{tag}] {name}" + (f" — {detail}" if detail and not cond else ""))
        if not cond:
            failed.append(name)

    person = D("person", 0, 0, 100, 300)

    # 1. Mũ nằm đúng vùng đầu -> khớp
    r = associate([person, D("helmet", 35, 10, 65, 40, tid=7)])
    check("mũ trong vùng đầu được ghép", len(r) == 1 and r[0].helmet_track_id == 7 and r[0].status == MATCHED)

    # 2. Mũ quá thấp (tâm y >= 120) -> không khớp
    r = associate([person, D("helmet", 35, 130, 65, 160, tid=7)])
    check("mũ ở vùng thân không được tính là đội", len(r) == 1 and r[0].helmet_track_id is None)

    # 3. Mũ lệch hẳn ra ngoài mép ngang -> không khớp
    r = associate([person, D("helmet", 90, 10, 120, 40, tid=7)])
    check("mũ ngoài lề ngang không khớp", r[0].helmet_track_id is None)

    # 4. Áo nằm vùng thân -> khớp
    r = associate([person, D("safety_vest", 10, 90, 90, 240, tid=9)])
    check("áo trong vùng thân được ghép", r[0].vest_track_id == 9)

    # 5. Áo chỉ overlap vùng thân (tâm ngoài dải) nhưng IoU đủ -> vẫn khớp
    r = associate([person, D("safety_vest", 10, 200, 90, 300, tid=9)])
    check("áo overlap vùng thân được ghép", r[0].vest_track_id == 9)

    # 6. Một mũ, hai người gần nhau -> ghép cho người gần mỏ neo đầu nhất
    p1 = D("person", 0, 0, 100, 300, tid=1)
    p2 = D("person", 110, 0, 210, 300, tid=2)
    r = associate([p1, p2, D("helmet", 40, 10, 70, 40, tid=7)])
    who = next(x for x in r if x.helmet_track_id == 7)
    check("1 mũ chỉ ghép 1 người (người gần nhất)", who.person_track_id == 1 and
          all(x.helmet_track_id is None for x in r if x is not who))

    # 7. Người nhỏ hơn 60px -> UNKNOWN, không ghép PPE
    r = associate([D("person", 0, 0, 40, 50, tid=3), D("helmet", 10, 2, 30, 12, tid=7)])
    check("người < 60px trả UNKNOWN", r[0].status == UNKNOWN and r[0].helmet_track_id is None)

    # 8. PPE không thuộc ai (không có người) -> bị bỏ qua, không lỗi
    r = associate([D("helmet", 500, 10, 530, 40, tid=7), D("safety_vest", 500, 100, 530, 250, tid=9)])
    check("PPE không có người thì bỏ qua", r == [])

    # 9. Người thiếu cả mũ lẫn áo -> vẫn có kết quả với cả hai None
    r = associate([person])
    check("người không có PPE -> cả hai None", r[0].helmet_track_id is None and r[0].vest_track_id is None)

    # 10. Person không có track_id -> không ghép
    r = associate([D("person", 0, 0, 100, 300, tid=None)])
    check("person không track_id bị bỏ qua", r == [])

    return failed


def test_association():
    failed = run_tests()
    assert len(failed) == 0, f"Failed tests: {failed}"


if __name__ == "__main__":
    failed = run_tests()
    print()
    if failed:
        print(f"KẾT QUẢ: {len(failed)} test FAIL — {failed}")
        sys.exit(1)
    print("KẾT QUẢ: tất cả test OK")
