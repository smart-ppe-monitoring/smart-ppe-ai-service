"""
Environment configuration — reads from .env / environment variables.
Never log or expose secret values.
"""

import os
from dotenv import load_dotenv
from app.constants import (
    DEFAULT_CONFIDENCE_THRESHOLD,
    DEFAULT_CONFIRMATION_DURATION_S,
    DEFAULT_CONFIRMATION_DURATION_HELMET_S,
    DEFAULT_CONFIRMATION_DURATION_VEST_S,
    DEFAULT_PROFILE_ASPECT_RATIO_MIN,
    DEFAULT_COMPLIANT_GRACE_PERIOD_S,
    DEFAULT_COOLDOWN_DURATION_S,
    MINIO_BUCKET,
    VIOLATION_NO_HELMET,
    VIOLATION_NO_VEST,
)

load_dotenv()

# ─── Backend API ──────────────────────────────────────────────────────────────
BACKEND_BASE_URL: str = os.getenv("BACKEND_BASE_URL", os.getenv("INCIDENT_SERVICE_URL", "http://localhost:8893"))
CAMERA_SERVICE_URL: str = os.getenv("CAMERA_SERVICE_URL", "http://localhost:8892")
SITE_SERVICE_URL: str = os.getenv("SITE_SERVICE_URL", "http://localhost:8891")
IDENTITY_SERVICE_URL: str = os.getenv("IDENTITY_SERVICE_URL", "http://localhost:9999")
GATEWAY_INTERNAL_SECRET: str = os.getenv("GATEWAY_INTERNAL_SECRET", "")
INTERNAL_SERVICE_TOKEN: str = os.getenv("INTERNAL_SERVICE_TOKEN", os.getenv("AI_SERVICE_TOKEN", ""))

# ─── Kafka ────────────────────────────────────────────────────────────────────
KAFKA_BROKER: str = os.getenv("KAFKA_BROKER", "localhost:9092")
KAFKA_ENABLED: bool = os.getenv("KAFKA_ENABLED", "false").strip().lower() in ("1", "true", "yes")


def _normalize_minio_endpoint(raw: str) -> tuple[str, bool]:
    val = (raw or "localhost:9000").strip().rstrip("/")
    if val.startswith("https://"):
        return val[len("https://"):], True
    if val.startswith("http://"):
        return val[len("http://"):], False
    return val, False


_MINIO_ENDPOINT_HOST, _MINIO_URL_SECURE = _normalize_minio_endpoint(
    os.getenv("MINIO_ENDPOINT", os.getenv("PPE_MINIO_ENDPOINT", "localhost:9000"))
)

# ─── MinIO ────────────────────────────────────────────────────────────────────
MINIO_ENDPOINT: str = _MINIO_ENDPOINT_HOST
MINIO_ACCESS_KEY: str = os.getenv("MINIO_ACCESS_KEY", os.getenv("PPE_MINIO_ROOT_USER", "minio_admin"))
MINIO_SECRET_KEY: str = os.getenv("MINIO_SECRET_KEY", os.getenv("PPE_MINIO_ROOT_PASSWORD", ""))   # NEVER log this
MINIO_SECURE: bool = os.getenv("MINIO_SECURE", "true" if _MINIO_URL_SECURE else "false").lower() == "true"
MINIO_BUCKET_NAME: str = os.getenv("MINIO_BUCKET", os.getenv("PPE_MINIO_BUCKET_EVIDENCES", MINIO_BUCKET))

# ─── Safety Rule Fallbacks ────────────────────────────────────────────────────
CONFIDENCE_THRESHOLD: float = float(
    os.getenv("CONFIDENCE_THRESHOLD", str(DEFAULT_CONFIDENCE_THRESHOLD))
)
CONFIRMATION_DURATION_S: int = int(
    os.getenv("CONFIRMATION_DURATION_S", str(DEFAULT_CONFIRMATION_DURATION_S))
)
CONFIRMATION_DURATION_HELMET_S: float = max(
    0.5, float(os.getenv("CONFIRMATION_DURATION_HELMET_S", str(DEFAULT_CONFIRMATION_DURATION_HELMET_S)))
)
CONFIRMATION_DURATION_VEST_S: float = max(
    0.5, float(os.getenv("CONFIRMATION_DURATION_VEST_S", str(DEFAULT_CONFIRMATION_DURATION_VEST_S)))
)
CONFIRMATION_DURATION_MAP: dict = {
    VIOLATION_NO_HELMET: CONFIRMATION_DURATION_HELMET_S,
    VIOLATION_NO_VEST: CONFIRMATION_DURATION_VEST_S,
}
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
PERSON_PROFILE_ASPECT_RATIO_MIN: float = float(
    os.getenv("PERSON_PROFILE_ASPECT_RATIO_MIN", str(DEFAULT_PROFILE_ASPECT_RATIO_MIN))
)
COMPLIANT_GRACE_PERIOD_S: float = float(
    os.getenv("COMPLIANT_GRACE_PERIOD_S", str(DEFAULT_COMPLIANT_GRACE_PERIOD_S))
)

# ─── Track ID-switch handling (VI-07) ────────────────────────────────────────
ID_SWITCH_GRACE_S: float = 2.0
ID_SWITCH_MAX_DIST_RATIO: float = 0.5

# ─── Local evidence directory (offline fallback) ──────────────────────────────
EVIDENCE_DIR: str = os.getenv("EVIDENCE_DIR", "evidences")

# ─── Auth / RS256 JWKS (decentralized verification) ──────────────────────────
JWT_PUBLIC_KEY: str = os.getenv("JWT_PUBLIC_KEY", "").replace("\\n", "\n")
JWT_PUBLIC_KEY_PATH: str = os.getenv("JWT_PUBLIC_KEY_PATH", "keys/public.key.pem")
AUTH_JWKS_URL: str = os.getenv(
    "AUTH_JWKS_URL",
    f"{IDENTITY_SERVICE_URL.rstrip('/')}/.well-known/jwks.json",
)
JWT_JWKS_CACHE_TTL_S: int = max(30, int(os.getenv("JWT_JWKS_CACHE_TTL_S", "300")))

# ─── 24/7 Live Camera Sync & Dynamic Limits ─────────────────────────────────
LIVE_MAX_SESSIONS_DEFAULT: int = max(1, int(os.getenv("LIVE_MAX_SESSIONS", "16")))
AI_AUTO_SYNC_ENABLED: bool = os.getenv("AI_AUTO_SYNC_ENABLED", "true").strip().lower() in ("1", "true", "yes")
AI_CAMERA_SYNC_INTERVAL_S: float = max(5.0, float(os.getenv("AI_CAMERA_SYNC_INTERVAL_S", "15")))
LIVE_DISPATCH_DEFAULT: bool = (os.getenv("LIVE_DISPATCH_ENABLED") or "true").strip().lower() in ("1", "true", "yes")
LIVE_ALLOW_HTTP_STREAMS: bool = os.getenv("LIVE_ALLOW_HTTP_STREAMS", "false").strip().lower() in ("1", "true", "yes")
LIVE_ALLOWED_RTSP_HOSTS: set[str] = {
    h.strip().lower() for h in os.getenv("LIVE_ALLOWED_RTSP_HOSTS", "").split(",") if h.strip()
}
LIVE_MAX_LOCAL_EVIDENCES: int = max(50, int(os.getenv("LIVE_MAX_LOCAL_EVIDENCES", "500")))
AI_ALLOWED_HOSTS: list[str] = [
    h.strip()
    for h in os.getenv("AI_ALLOWED_HOSTS", "127.0.0.1,localhost,ai-service").split(",")
    if h.strip()
]


def resolve_source(source: str):
    """Normalize video source: numeric string → webcam index, else keep as-is."""
    if source is None:
        return None
    s = str(source).strip()
    if s.isdigit():
        return int(s)
    return s
