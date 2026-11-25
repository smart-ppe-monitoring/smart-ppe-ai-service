"""
Backend HTTP client — lấy cấu hình Safety Rules động và kiểm tra health.
Không gọi Identity Service để kiểm tra quyền (Decentralized Auth).
"""

import logging
import math
from typing import Any, Optional
import httpx
from app import config
from app.constants import (
    HTTP_REPORTING_INTERNAL_ACTIVE_MODEL,
    HTTP_REPORTING_INTERNAL_TELEMETRY,
    HTTP_SAFETY_RULES_DEFAULT,
    HTTP_SAFETY_RULES_EFFECTIVE,
    HTTP_SAFETY_RULES_INTERNAL_DEFAULT,
    HTTP_SAFETY_RULES_INTERNAL_EFFECTIVE,
    DEFAULT_CONFIDENCE_THRESHOLD,
    DEFAULT_CONFIRMATION_DURATION_S,
    DEFAULT_CONFIRMATION_DURATION_HELMET_S,
    DEFAULT_CONFIRMATION_DURATION_VEST_S,
    DEFAULT_COOLDOWN_DURATION_S,
)

logger = logging.getLogger(__name__)


def _get_auth_headers() -> dict[str, str]:
    token = getattr(config, "INTERNAL_SERVICE_TOKEN", "")
    if token:
        return {"Authorization": f"Bearer {token}"}
    return {}


def _get_internal_m2m_headers() -> dict[str, str]:
    headers = _get_auth_headers()
    token = getattr(config, "INTERNAL_SERVICE_TOKEN", "")
    if token:
        headers["x-internal-service-token"] = token
    gw_secret = getattr(config, "GATEWAY_INTERNAL_SECRET", "")
    if gw_secret:
        headers["x-gateway-secret"] = gw_secret
    return headers


class SafetyRulesConfig:
    def __init__(
        self,
        confidence: float,
        confirmation_s: Optional[float] = None,
        cooldown_s: int = DEFAULT_COOLDOWN_DURATION_S,
        enable_no_helmet: bool = True,
        enable_no_vest: bool = True,
        enable_restricted_area: bool = False,
        model_path: Optional[str] = None,
        helmet_confirmation_s: Optional[float] = None,
        vest_confirmation_s: Optional[float] = None,
    ):
        resolved_helmet = (
            helmet_confirmation_s
            if helmet_confirmation_s is not None
            else (
                confirmation_s
                if confirmation_s is not None
                else DEFAULT_CONFIRMATION_DURATION_HELMET_S
            )
        )
        resolved_vest = (
            vest_confirmation_s
            if vest_confirmation_s is not None
            else DEFAULT_CONFIRMATION_DURATION_VEST_S
        )
        self.confidence = float(confidence)
        self.helmet_confirmation_s = float(resolved_helmet)
        self.confirmation_s = self.helmet_confirmation_s  # Backward-compatible alias
        self.vest_confirmation_s = float(resolved_vest)
        self.cooldown_s = int(cooldown_s)
        self.enable_no_helmet = enable_no_helmet
        self.enable_no_vest = enable_no_vest
        self.enable_restricted_area = enable_restricted_area
        self.model_path = model_path


_FALLBACK = SafetyRulesConfig(
    confidence=DEFAULT_CONFIDENCE_THRESHOLD,
    helmet_confirmation_s=float(DEFAULT_CONFIRMATION_DURATION_HELMET_S),
    vest_confirmation_s=float(DEFAULT_CONFIRMATION_DURATION_VEST_S),
    cooldown_s=DEFAULT_COOLDOWN_DURATION_S,
)


def _parse_safety_rules_payload(data: dict) -> SafetyRulesConfig:
    rules_dict = data.get("rules") if isinstance(data.get("rules"), dict) else {}
    model_version = data.get("modelVersion") if isinstance(data.get("modelVersion"), dict) else {}
    model_path = (
        model_version.get("modelPath")
        or data.get("modelVersionId")
        or rules_dict.get("modelVersionId")
    )
    raw_helmet = data.get("helmetConfirmationDuration")
    if raw_helmet is None:
        raw_helmet = data.get("confirmationDuration")
    if raw_helmet is None:
        raw_helmet = DEFAULT_CONFIRMATION_DURATION_HELMET_S

    raw_vest = data.get("vestConfirmationDuration")
    if raw_vest is None:
        raw_vest = DEFAULT_CONFIRMATION_DURATION_VEST_S

    return SafetyRulesConfig(
        confidence=float(data.get("confidenceThreshold", DEFAULT_CONFIDENCE_THRESHOLD)),
        helmet_confirmation_s=float(raw_helmet),
        vest_confirmation_s=float(raw_vest),
        cooldown_s=int(data.get("cooldownDuration", DEFAULT_COOLDOWN_DURATION_S)),
        enable_no_helmet=bool(data.get("enableNoHelmet", rules_dict.get("enableNoHelmet", True))),
        enable_no_vest=bool(data.get("enableNoVest", rules_dict.get("enableNoVest", True))),
        enable_restricted_area=bool(data.get("enableRestrictedArea", rules_dict.get("enableRestrictedArea", False))),
        model_path=model_path,
    )


async def fetch_system_default_rules() -> SafetyRulesConfig:
    """
    GET {SITE_SERVICE_URL}/api/v1/safety-rules/internal/system-default
    Falls back to DEFAULT_* constants if Site Service is unreachable.
    """
    base_url = getattr(config, "SITE_SERVICE_URL", "http://localhost:8891").rstrip("/")
    url = f"{base_url}{HTTP_SAFETY_RULES_INTERNAL_DEFAULT}"
    headers = _get_auth_headers()
    gw_secret = getattr(config, "GATEWAY_INTERNAL_SECRET", "")
    if gw_secret:
        headers["x-gateway-secret"] = gw_secret
    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            resp = await client.get(url, headers=headers)
            resp.raise_for_status()
            data = resp.json().get("data", {})
            if isinstance(data, dict):
                return _parse_safety_rules_payload(data)
            return _FALLBACK
    except Exception as exc:
        logger.warning("Failed to fetch safety rules from Site Service (%s). Using fallback defaults.", exc)
        return _FALLBACK


async def fetch_effective_rules(camera_id: str, site_id: str) -> SafetyRulesConfig:
    """
    GET {SITE_SERVICE_URL}/api/v1/safety-rules/internal/effective?cameraId={cameraId}&siteId={siteId}
    Falls back to system-default if camera-specific config is unavailable.
    """
    base_url = getattr(config, "SITE_SERVICE_URL", "http://localhost:8891").rstrip("/")
    url = f"{base_url}{HTTP_SAFETY_RULES_INTERNAL_EFFECTIVE}"
    params = {}
    if camera_id:
        params["cameraId"] = camera_id
    if site_id:
        params["siteId"] = site_id
    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            resp = await client.get(url, params=params, headers=_get_internal_m2m_headers())
            resp.raise_for_status()
            data = resp.json().get("data", {})
            if isinstance(data, dict):
                return _parse_safety_rules_payload(data)
            return _FALLBACK
    except Exception as exc:
        logger.warning("Effective rules fetch failed (%s). Falling back to system default.", exc)
        return await fetch_system_default_rules()


def fetch_effective_rules_sync(
    camera_id: Optional[str] = None,
    site_id: Optional[str] = None,
) -> Optional[SafetyRulesConfig]:
    """
    GET {SITE_SERVICE_URL}/api/v1/safety-rules/internal/effective
    Lấy cấu hình Safety Rules hiệu lực (Camera Override -> Site -> System Default) qua M2M.
    Trả về None nếu Site Service chưa sẵn sàng để caller giữ nguyên cấu hình đang chạy thay vì đè mặc định.
    """
    if not camera_id and not site_id:
        return None
    base_url = getattr(config, "SITE_SERVICE_URL", "http://localhost:8891").rstrip("/")
    url = f"{base_url}{HTTP_SAFETY_RULES_INTERNAL_EFFECTIVE}"
    params: dict[str, str] = {}
    if camera_id:
        params["cameraId"] = str(camera_id)
    if site_id:
        params["siteId"] = str(site_id)
    try:
        with httpx.Client(timeout=2.5) as client:
            resp = client.get(url, params=params, headers=_get_internal_m2m_headers())
            resp.raise_for_status()
            payload = resp.json()
            data = payload.get("data") if isinstance(payload, dict) else None
            if not isinstance(data, dict):
                return None
            return _parse_safety_rules_payload(data)
    except Exception as exc:
        logger.debug("Sync effective safety rules fetch skipped (%s): %s", camera_id or site_id, exc)
        return None


def check_internal_camera_exists_sync(
    camera_id: Optional[str] = None,
    code: Optional[str] = None,
) -> Optional[bool]:
    """
    Kiểm tra camera có còn tồn tại trong Camera Service không qua M2M internal lookup.
    Trả về True (200), False (404 - đã bị xóa khỏi DB), hoặc None (lỗi mạng/5xx để tránh dừng nhầm).
    """
    if not camera_id and not code:
        return None
    base_url = getattr(config, "CAMERA_SERVICE_URL", "http://localhost:8892").rstrip("/")
    url = f"{base_url}/api/v1/cameras/internal/lookup"
    params = {"id": str(camera_id)} if camera_id else {"code": str(code).upper()}
    try:
        with httpx.Client(timeout=2.5) as client:
            resp = client.get(url, params=params, headers=_get_internal_m2m_headers())
            if resp.status_code == 200:
                return True
            if resp.status_code == 404:
                return False
            return None
    except Exception as exc:
        logger.debug("Camera existence check skipped (%s): %s", code or camera_id, exc)
        return None


def fetch_internal_active_streams_sync(status: str = "ONLINE") -> Optional[list[dict]]:
    """
    GET {CAMERA_SERVICE_URL}/api/v1/cameras/internal/active-streams
    Truy xuất danh sách camera ONLINE kèm URL stream gốc & cấu hình ROI từ Camera Service.
    Trả về None nếu Camera Service chưa sẵn sàng hoặc xác thực thất bại.
    """
    base_url = getattr(config, "CAMERA_SERVICE_URL", "http://localhost:8892").rstrip("/")
    url = f"{base_url}/api/v1/cameras/internal/active-streams"
    try:
        with httpx.Client(timeout=5.0) as client:
            resp = client.get(url, params={"status": status}, headers=_get_internal_m2m_headers())
            resp.raise_for_status()
            payload = resp.json()
            data = payload.get("data") if isinstance(payload, dict) else None
            return data if isinstance(data, list) else []
    except Exception as exc:
        logger.warning("Failed to fetch internal active streams from Camera Service: %s", exc)
        return None


def lookup_internal_camera_sync(camera_id: Optional[str] = None, code: Optional[str] = None) -> Optional[dict]:
    """
    GET {CAMERA_SERVICE_URL}/api/v1/cameras/internal/lookup
    Tra cứu thông tin camera nội bộ theo UUID hoặc code (không lộ RTSP password ra FE).
    """
    if not camera_id and not code:
        return None
    base_url = getattr(config, "CAMERA_SERVICE_URL", "http://localhost:8892").rstrip("/")
    url = f"{base_url}/api/v1/cameras/internal/lookup"
    params = {"id": camera_id} if camera_id else {"code": str(code).upper()}
    try:
        with httpx.Client(timeout=4.0) as client:
            resp = client.get(url, params=params, headers=_get_internal_m2m_headers())
            if resp.status_code == 404:
                return None
            resp.raise_for_status()
            payload = resp.json()
            data = payload.get("data") if isinstance(payload, dict) else None
            return data if isinstance(data, dict) else None
    except Exception as exc:
        logger.warning("Failed internal camera lookup (%s): %s", code or camera_id, exc)
        return None


def fetch_dynamic_live_max_sessions_sync() -> Optional[int]:
    """
    GET {IDENTITY_SERVICE_URL}/api/v1/system/public
    Đọc cấu hình động `ai.live_max_sessions` từ Identity SystemConfig.
    """
    base_url = getattr(config, "IDENTITY_SERVICE_URL", "http://localhost:9999").rstrip("/")
    url = f"{base_url}/api/v1/system/public"
    headers = {}
    gw_secret = getattr(config, "GATEWAY_INTERNAL_SECRET", "")
    if gw_secret:
        headers["x-gateway-secret"] = gw_secret
    try:
        with httpx.Client(timeout=4.0) as client:
            resp = client.get(url, headers=headers)
            resp.raise_for_status()
            payload = resp.json()
            data = payload.get("data") if isinstance(payload, dict) else None
            if not isinstance(data, dict):
                return None
            raw_val = data.get("ai.live_max_sessions")
            if raw_val is None:
                return None
            parsed = int(raw_val)
            return max(1, min(256, parsed))
    except Exception as exc:
        logger.debug("Dynamic ai.live_max_sessions fetch skipped (%s)", exc)
        return None


def fetch_internal_active_model_version_sync(
    model_type: str = "OBJECT_DETECTION",
) -> Optional[dict]:
    """
    GET {REPORTING_SERVICE_URL}/api/v1/ai-models/internal/active-version
    Lấy phiên bản mô hình AI đang được triển khai (DEPLOYED) qua M2M internal token.
    """
    base_url = getattr(config, "REPORTING_SERVICE_URL", "http://localhost:8894").rstrip("/")
    url = f"{base_url}{HTTP_REPORTING_INTERNAL_ACTIVE_MODEL}"
    params = {"modelType": model_type} if model_type else {}
    try:
        with httpx.Client(timeout=3.0) as client:
            resp = client.get(url, params=params, headers=_get_internal_m2m_headers())
            if resp.status_code == 404:
                return None
            resp.raise_for_status()
            payload = resp.json()
            data = payload.get("data") if isinstance(payload, dict) else None
            return data if isinstance(data, dict) else None
    except Exception as exc:
        logger.debug("Active model version fetch skipped (%s)", exc)
        return None


def push_ai_performance_telemetry_sync(payload: dict[str, Any]) -> bool:
    """
    POST {REPORTING_SERVICE_URL}/api/v1/statistics/internal/record-ai-performance
    Đẩy telemetry hiệu năng AI (fps, latencyMs, p95LatencyMs, e2eAlertLatencyMs, frameCount) sang Reporting Service.
    """
    if not isinstance(payload, dict):
        return False

    camera_id = str(payload.get("cameraId") or "").strip()
    if not camera_id:
        return False

    try:
        fps = float(payload.get("fps", 0.0))
        latency_ms = float(payload.get("latencyMs", 0.0))
        frame_count = int(payload.get("frameCount", 0))
    except (TypeError, ValueError):
        return False

    if not math.isfinite(fps) or fps <= 0 or not math.isfinite(latency_ms) or latency_ms <= 0 or frame_count < 1:
        return False

    clean_payload: dict[str, Any] = {
        "cameraId": camera_id,
        "fps": round(fps, 2),
        "latencyMs": round(latency_ms, 2),
        "frameCount": frame_count,
    }

    for optional_key in ("modelVersionId", "statDate"):
        raw_val = payload.get(optional_key)
        if isinstance(raw_val, str) and raw_val.strip():
            clean_payload[optional_key] = raw_val.strip()

    for numeric_key in ("p95LatencyMs", "e2eAlertLatencyMs"):
        raw_num = payload.get(numeric_key)
        if raw_num is not None:
            try:
                num_val = float(raw_num)
                if math.isfinite(num_val) and num_val >= 0:
                    clean_payload[numeric_key] = round(num_val, 2)
            except (TypeError, ValueError):
                pass

    base_url = getattr(config, "REPORTING_SERVICE_URL", "http://localhost:8894").rstrip("/")
    url = f"{base_url}{HTTP_REPORTING_INTERNAL_TELEMETRY}"
    try:
        with httpx.Client(timeout=3.0) as client:
            resp = client.post(url, json=clean_payload, headers=_get_internal_m2m_headers())
            if 200 <= resp.status_code < 300:
                return True
            logger.warning(
                "Reporting telemetry push rejected (HTTP %d) for camera %s",
                resp.status_code,
                camera_id,
            )
            return False
    except Exception as exc:
        logger.debug("Reporting telemetry push skipped for camera %s (%s)", camera_id, exc)
        return False
