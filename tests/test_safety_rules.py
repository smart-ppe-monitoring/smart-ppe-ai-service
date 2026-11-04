"""VI-05: Kiểm thử size filter + safety rules bằng dữ liệu nhân tạo.

Chạy:  python tests/test_safety_rules.py  hoặc  pytest tests/test_safety_rules.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.association.associator import MATCHED, UNKNOWN as ASSOC_UNKNOWN, PersonPPE
from app.detection.adapter import Detection
from app.rules.safety import (
    COMPLIANT,
    UNKNOWN,
    VIOLATION_CANDIDATE,
    VIOLATION_NO_HELMET,
    VIOLATION_NO_VEST,
    evaluate,
)


def D(cls, x1, y1, x2, y2, tid=1, conf=0.9):
    return Detection(class_name=cls, confidence=conf, bbox=(x1, y1, x2, y2), track_id=tid)


# Khung 1000x1000; người chuẩn 100x400 tại (100,100) — không chạm mép
def state(person, helmet=None, vest=None, w=1000, h=1000, conf=0.9):
    dets = [person]
    if helmet: dets.append(helmet)
    if vest: dets.append(vest)
    asso = [PersonPPE(person.track_id, person.bbox,
                      helmet_track_id=helmet.track_id if helmet else None,
                      vest_track_id=vest.track_id if vest else None)]
    return evaluate(dets, asso, w, h)[0]


def run_tests():
    failed = []

    def check(name, cond):
        print(f"[{'OK  ' if cond else 'FAIL'}] {name}")
        if not cond:
            failed.append(name)

    P = D("person", 100, 100, 200, 500)
    HM = D("helmet", 110, 110, 190, 140, tid=2)
    VT = D("safety_vest", 105, 250, 195, 450, tid=3)

    # 1. Đủ PPE, trong khung, đủ lớn, confidence ổn -> COMPLIANT
    check("đủ mũ + áo -> COMPLIANT", state(P, HM, VT).status == COMPLIANT)

    # 2. Thiếu mũ -> CANDIDATE no_helmet
    s = state(P, None, VT)
    check("thiếu mũ -> CANDIDATE no_helmet",
          s.status == VIOLATION_CANDIDATE and s.violations == [VIOLATION_NO_HELMET])

    # 3. Thiếu áo -> CANDIDATE no_safety_vest
    s = state(P, HM, None)
    check("thiếu áo -> CANDIDATE no_safety_vest",
          s.status == VIOLATION_CANDIDATE and s.violations == [VIOLATION_NO_VEST])

    # 4. Thiếu cả mũ lẫn áo -> CANDIDATE cả hai
    s = state(P, None, None)
    check("thiếu cả mũ + áo -> CANDIDATE cả hai",
          s.status == VIOLATION_CANDIDATE and set(s.violations) == {VIOLATION_NO_HELMET, VIOLATION_NO_VEST})

    # 5. Người quá nhỏ (< 60px) -> UNKNOWN (do filter trước đó chặn, kiểm tra lại)
    s = state(D("person", 100, 100, 150, 150), HM, VT)
    check("người < 60px -> UNKNOWN", s.status == UNKNOWN and "nhỏ" in s.reason)

    # 6. Confidence người quá thấp -> UNKNOWN
    s = state(D("person", 100, 100, 200, 500, conf=0.3), HM, VT)
    check("confidence thấp -> UNKNOWN", s.status == UNKNOWN and "confidence" in s.reason)

    # 7. Đầu bị cắt mép trên (y1=0) + thiếu mũ -> UNKNOWN
    s = state(D("person", 100, 0, 200, 400), None, VT)
    check("đầu bị cắt + thiếu mũ -> UNKNOWN (không kết luận vi phạm)",
          s.status == UNKNOWN and "helmet" in s.reason)

    # 8. Thân bị cắt mép trái (x1=0) + phần thấy được hẹp (100px < 150px) + thiếu áo -> UNKNOWN
    s = state(D("person", 0, 100, 100, 500), HM, None)
    check("thân bị cắt sâu + thiếu áo -> UNKNOWN",
          s.status == UNKNOWN and "safety_vest" in s.reason)

    # 8b. Thân chạm mép trái nhưng phần thấy được rộng (250px >= 150px) + thiếu áo -> CANDIDATE
    #     (cắt vài px không che được áo — áo nằm giữa thân)
    s = state(D("person", 0, 100, 250, 500), HM, None)
    check("chạm mép vài px, thân còn rộng + thiếu áo -> CANDIDATE",
          s.status == VIOLATION_CANDIDATE and s.violations == [VIOLATION_NO_VEST])

    # 8c. Tương tự 8b nhưng có áo -> COMPLIANT
    s = state(D("person", 0, 100, 250, 500), HM, VT)
    check("chạm mép vài px, thân còn rộng + có áo -> COMPLIANT", s.status == COMPLIANT)

    # 9. Cắt mép dưới (chân, y2 = khung) không che mũ/áo -> vẫn xét bình thường
    s = state(D("person", 100, 100, 400, 1000), HM, VT)
    check("cắt mép dưới vẫn xét -> COMPLIANT", s.status == COMPLIANT)

    # 10. Mép dưới cắt + thiếu áo -> CANDIDATE (không bị bỏ sót)
    s = state(D("person", 100, 100, 400, 1000), HM, None)
    check("cắt mép dưới + thiếu áo -> CANDIDATE", s.status == VIOLATION_CANDIDATE)

    # 11. ASSOC UNKNOWN (người nhỏ theo VI-04) -> UNKNOWN
    det = D("person", 100, 100, 150, 150)
    r = evaluate([det], [PersonPPE(det.track_id, det.bbox, status=ASSOC_UNKNOWN)], 1000, 1000)
    check("assoc UNKNOWN -> UNKNOWN", r[0].status == UNKNOWN)

    # 12. Dynamic rules: enable_no_helmet=False -> bỏ qua lỗi mũ
    asso_no_helmet = [PersonPPE(P.track_id, P.bbox, helmet_track_id=None, vest_track_id=VT.track_id)]
    r_no_helmet = evaluate([P, VT], asso_no_helmet, 1000, 1000, enable_no_helmet=False, enable_no_vest=True)
    check("enable_no_helmet=False -> thiếu mũ vẫn COMPLIANT", r_no_helmet[0].status == COMPLIANT)

    # 13. Dynamic rules: enable_no_vest=False -> bỏ qua lỗi áo
    asso_no_vest = [PersonPPE(P.track_id, P.bbox, helmet_track_id=HM.track_id, vest_track_id=None)]
    r_no_vest = evaluate([P, HM], asso_no_vest, 1000, 1000, enable_no_helmet=True, enable_no_vest=False)
    check("enable_no_vest=False -> thiếu áo vẫn COMPLIANT", r_no_vest[0].status == COMPLIANT)

    # 14. Dynamic rules: tắt cả mũ và áo -> thiếu cả 2 vẫn COMPLIANT
    asso_no_ppe = [PersonPPE(P.track_id, P.bbox, helmet_track_id=None, vest_track_id=None)]
    r_no_ppe = evaluate([P], asso_no_ppe, 1000, 1000, enable_no_helmet=False, enable_no_vest=False)
    check("tắt cả mũ và áo -> COMPLIANT", r_no_ppe[0].status == COMPLIANT)

    # 15. Profile view (xoay ngang W/H = 90/500 = 0.18 < 0.24) + thiếu áo -> UNKNOWN
    P_profile = D("person", 100, 100, 190, 600)
    HM_profile = D("helmet", 110, 110, 180, 150, tid=2)
    s_profile_novest = state(P_profile, HM_profile, None)
    check("profile view (W/H=0.18) + thiếu áo -> UNKNOWN (không phạt oan)",
          s_profile_novest.status == UNKNOWN and "safety_vest" in s_profile_novest.reason)

    # 16. Profile view (xoay ngang W/H = 0.18) + CÓ ÁO -> COMPLIANT
    VT_profile = D("safety_vest", 110, 200, 180, 450, tid=3)
    s_profile_vest = state(P_profile, HM_profile, VT_profile)
    check("profile view (W/H=0.18) + có áo -> COMPLIANT", s_profile_vest.status == COMPLIANT)

    # 17. Normal view (W/H = 175/500 = 0.35 >= 0.24) + thiếu áo -> CANDIDATE
    P_normal = D("person", 100, 100, 275, 600)
    HM_normal = D("helmet", 110, 110, 190, 150, tid=2)
    s_normal_novest = state(P_normal, HM_normal, None)
    check("normal view (W/H=0.35) + thiếu áo -> CANDIDATE",
          s_normal_novest.status == VIOLATION_CANDIDATE and s_normal_novest.violations == [VIOLATION_NO_VEST])

    # 18. Profile view sát mép đáy (y2=1000 frame_h=1000) + thiếu áo -> vẫn UNKNOWN (nhất quán biên)
    P_bottom_profile = D("person", 100, 500, 190, 1000)  # W=90, H=500 -> W/H = 0.18 < 0.24
    HM_bottom_profile = D("helmet", 110, 510, 180, 550, tid=2)
    s_bottom_profile = state(P_bottom_profile, HM_bottom_profile, None)
    check("profile view sát mép đáy (cut_bottom) + thiếu áo -> vẫn UNKNOWN",
          s_bottom_profile.status == UNKNOWN and "safety_vest" in s_bottom_profile.reason)

    return failed


def test_safety_rules():
    failed = run_tests()
    assert len(failed) == 0, f"Failed tests: {failed}"


if __name__ == "__main__":
    failed = run_tests()
    print()
    if failed:
        print(f"KẾT QUẢ: {len(failed)} test FAIL — {failed}")
        sys.exit(1)
    print("KẾT QUẢ: tất cả test OK")
