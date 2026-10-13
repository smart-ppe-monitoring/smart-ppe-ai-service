"""Cấu hình trung tâm của video-intelligence.

Có thể ghi đè qua biến môi trường (env), ưu tiên env > giá trị mặc định ở đây.
Các ngưỡng 3s/30s/60px/0.5 là giá trị khởi đầu theo tài liệu Phase 2, cần kiểm thử.
"""
import os


# --- Nguồn video (VI-01) ---
# Dạng: đường dẫn file video | "rtsp://..." | số nguyên (index webcam)
DEFAULT_SOURCE = os.environ.get("VIDEO_SOURCE", "")
FALLBACK_FPS = 25.0                      # dùng khi video không khai báo FPS
EMPTY_FRAME_TOLERANCE = 30               # số frame rỗng liên tiếp chịu được (webcam/RTSP)


# --- Model / detector (VI-02) ---
MODEL_PATH = os.environ.get("MODEL_PATH", "models/best.pt")
MODEL_IMGSZ = int(os.environ.get("MODEL_IMGSZ", "640"))
DEVICE = os.environ.get("DEVICE", "")          # rỗng = tự chọn (mps nếu có, ngược lại cpu)


# --- Ghép PPE–Person (VI-04, theo mục 8 tài liệu Phase 2) ---
PPE_HEAD_X_MARGIN = 0.15    # tâm mũ phải cách mép ngang bbox người tối thiểu 15% chiều rộng
PPE_HEAD_Y_MAX = 0.40       # tâm mũ phải nằm trên 40% chiều cao người (tính từ đỉnh)
PPE_TORSO_Y_MIN = 0.25      # vùng thân (cho safety vest): từ 25% chiều cao...
PPE_TORSO_Y_MAX = 0.85      # ...đến 85% chiều cao
PPE_TORSO_IOU_MIN = 0.10    # overlap tối thiểu áo–vùng thân khi tâm áo ngoài dải thân


# --- Safety rules (VI-05) ---
REQUIRED_PPE = ("helmet", "safety_vest")  # PPE bắt buộc theo quy định công trường (MVP)
PERSON_CONFIDENCE_THRESHOLD = 0.5         # confidence người tối thiểu để kết luận (detector đã lọc sẵn 0.5)
PERSON_EDGE_MARGIN_PX = 2                 # bbox cách mép khung <= số px này -> xem như bị cắt/che
PERSON_MIN_VISIBLE_WIDTH_PX = 150         # áo: cắt mép bên chỉ tính là "che" khi phần người thấy được hẹp hơn mức này


# --- Chống trùng (VI-07) ---
DEFAULT_CAMERA_ID = os.environ.get("CAMERA_ID", "cam-01")  # khóa trùng phải gồm cameraId
ID_SWITCH_GRACE_S = 2.0          # track mới xuất hiện trong vòng này sau khi track cũ mất -> nghi đổi ID
ID_SWITCH_MAX_DIST_RATIO = 0.5   # ...và tâm bbox gần track cũ (< 50% chiều cao người) -> nối là cùng một người


# --- Evidence snapshot (VI-08) ---
EVIDENCE_DIR = os.environ.get("EVIDENCE_DIR", "evidences")  # thư mục lưu frame bằng chứng (local)


# --- Ngưỡng safety rules (dùng từ VI-05 trở đi) ---
MIN_PERSON_HEIGHT_PX = 60                # bbox người thấp hơn ngưỡng -> UNKNOWN
CONFIRMATION_DURATION_S = 3.0            # vi phạm phải tồn tại liên tục 3 giây
COOLDOWN_DURATION_S = 30.0               # 30 giây không gửi lại event trùng
CONFIDENCE_THRESHOLD = 0.5               # confidence tối thiểu


def resolve_source(source: str):
    """Chuẩn hóa nguồn video: chuỗi số nguyên -> webcam index, còn lại giữ nguyên."""
    if source is None:
        return None
    s = str(source).strip()
    if s.isdigit():
        return int(s)
    return s
