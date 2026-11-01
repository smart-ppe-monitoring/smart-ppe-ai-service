"""
Environment configuration — reads from .env / environment variables.
Never log or expose secret values.
"""

import os
from dotenv import load_dotenv
from app.constants import (
    DEFAULT_CONFIDENCE_THRESHOLD,
    DEFAULT_CONFIRMATION_DURATION_S,
    DEFAULT_COOLDOWN_DURATION_S,
    MINIO_BUCKET,
)

load_dotenv()

# ─── Backend API ──────────────────────────────────────────────────────────────
BACKEND_BASE_URL: str = os.getenv("BACKEND_BASE_URL", "http://localhost:8893")
INTERNAL_SERVICE_TOKEN: str = os.getenv("INTERNAL_SERVICE_TOKEN", os.getenv("AI_SERVICE_TOKEN", ""))

# ─── Kafka ────────────────────────────────────────────────────────────────────
KAFKA_BROKER: str = os.getenv("KAFKA_BROKER", "localhost:9092")

# ─── MinIO ────────────────────────────────────────────────────────────────────
MINIO_ENDPOINT: str = os.getenv("MINIO_ENDPOINT", "localhost:9000")
MINIO_ACCESS_KEY: str = os.getenv("MINIO_ACCESS_KEY", "minio_admin")
MINIO_SECRET_KEY: str = os.getenv("MINIO_SECRET_KEY", "")   # NEVER log this
MINIO_SECURE: bool = os.getenv("MINIO_SECURE", "false").lower() == "true"
MINIO_BUCKET_NAME: str = os.getenv("MINIO_BUCKET", MINIO_BUCKET)

# ─── Safety Rule Fallbacks ────────────────────────────────────────────────────
CONFIDENCE_THRESHOLD: float = float(
    os.getenv("CONFIDENCE_THRESHOLD", str(DEFAULT_CONFIDENCE_THRESHOLD))
)
CONFIRMATION_DURATION_S: int = int(
    os.getenv("CONFIRMATION_DURATION_S", str(DEFAULT_CONFIRMATION_DURATION_S))
)
COOLDOWN_DURATION_S: int = int(
    os.getenv("COOLDOWN_DURATION_S", str(DEFAULT_COOLDOWN_DURATION_S))
)

# ─── Model ────────────────────────────────────────────────────────────────────
YOLO_MODEL_PATH: str = os.getenv("YOLO_MODEL_PATH", "models/yolo11m-best.pt")
MODEL_PATH: str = os.getenv("MODEL_PATH", YOLO_MODEL_PATH)
MODEL_IMGSZ: int = int(os.getenv("MODEL_IMGSZ", "640"))
DEVICE: str = os.getenv("DEVICE", "")           # empty = auto (mps if available, else cpu)

# ─── Video source ─────────────────────────────────────────────────────────────
DEFAULT_SOURCE: str = os.getenv("VIDEO_SOURCE", "")
FALLBACK_FPS: float = 25.0
EMPTY_FRAME_TOLERANCE: int = 30

# ─── Camera & Site identity ───────────────────────────────────────────────────
DEFAULT_CAMERA_ID: str = os.getenv("CAMERA_ID", "CAM-01")
SITE_CODE: str = os.getenv("SITE_CODE", "SITE-01")
SAVE_OUTPUT_VIDEO: bool = os.getenv("SAVE_OUTPUT_VIDEO", "true").lower() == "true"

# ─── PPE–Person Association (VI-04) ──────────────────────────────────────────
PPE_HEAD_X_MARGIN: float = 0.15
PPE_HEAD_Y_MAX: float = 0.40
PPE_TORSO_Y_MIN: float = 0.25
PPE_TORSO_Y_MAX: float = 0.85
PPE_TORSO_IOU_MIN: float = 0.10

# ─── Safety Rules filters (VI-05) ────────────────────────────────────────────
REQUIRED_PPE: tuple = ("helmet", "safety_vest")
PERSON_CONFIDENCE_THRESHOLD: float = 0.5
PERSON_EDGE_MARGIN_PX: int = 2
PERSON_MIN_VISIBLE_WIDTH_PX: int = 150
MIN_PERSON_HEIGHT_PX: int = 60

# ─── Track ID-switch handling (VI-07) ────────────────────────────────────────
ID_SWITCH_GRACE_S: float = 2.0
ID_SWITCH_MAX_DIST_RATIO: float = 0.5

# ─── Local evidence directory (offline fallback) ──────────────────────────────
EVIDENCE_DIR: str = os.getenv("EVIDENCE_DIR", "evidences")


def resolve_source(source: str):
    """Normalize video source: numeric string → webcam index, else keep as-is."""
    if source is None:
        return None
    s = str(source).strip()
    if s.isdigit():
        return int(s)
    return s
