"""
Backend HTTP client — lấy cấu hình Safety Rules động và kiểm tra health.
Không gọi Identity Service để kiểm tra quyền (Decentralized Auth).
"""

import logging
from typing import Optional
import httpx
from app import config
from app.constants import (
    HTTP_SAFETY_RULES_DEFAULT,
    HTTP_SAFETY_RULES_EFFECTIVE,
    DEFAULT_CONFIDENCE_THRESHOLD,
    DEFAULT_CONFIRMATION_DURATION_S,
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
    def __init__(self, confidence: float, confirmation_s: float, cooldown_s: int,
                 enable_no_helmet: bool = True, enable_no_vest: bool = True,
                 enable_restricted_area: bool = False, model_path: Optional[str] = None):
        self.confidence = float(confidence)
        self.confirmation_s = float(confirmation_s)
        self.cooldown_s = int(cooldown_s)
        self.enable_no_helmet = enable_no_helmet
        self.enable_no_vest = enable_no_vest
        self.enable_restricted_area = enable_restricted_area
        self.model_path = model_path


_FALLBACK = SafetyRulesConfig(
    confidence=DEFAULT_CONFIDENCE_THRESHOLD,
    confirmation_s=float(DEFAULT_CONFIRMATION_DURATION_S),
    cooldown_s=DEFAULT_COOLDOWN_DURATION_S,
)


async def fetch_system_default_rules() -> SafetyRulesConfig:
    """
    GET /api/v1/safety-rules/system-default
    Falls back to DEFAULT_* constants if Backend is unreachable.
    """
    url = f"{config.BACKEND_BASE_URL}{HTTP_SAFETY_RULES_DEFAULT}"
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(url, headers=_get_auth_headers())
            resp.raise_for_status()
            data = resp.json().get("data", {})
            rules_dict = data.get("rules") if isinstance(data.get("rules"), dict) else {}
            return SafetyRulesConfig(
                confidence=float(data.get("confidenceThreshold", DEFAULT_CONFIDENCE_THRESHOLD)),
                confirmation_s=float(data.get("confirmationDuration", DEFAULT_CONFIRMATION_DURATION_S)),
                cooldown_s=int(data.get("cooldownDuration", DEFAULT_COOLDOWN_DURATION_S)),
                enable_no_helmet=data.get("enableNoHelmet", rules_dict.get("enableNoHelmet", True)),
                enable_no_vest=data.get("enableNoVest", rules_dict.get("enableNoVest", True)),
                enable_restricted_area=data.get("enableRestrictedArea", rules_dict.get("enableRestrictedArea", False)),
                model_path=data.get("modelVersionId", rules_dict.get("modelVersionId")),
            )
    except Exception as exc:
        logger.warning("Failed to fetch safety rules from Backend (%s). Using fallback defaults.", exc)
        return _FALLBACK


async def fetch_effective_rules(camera_id: str, site_id: str) -> SafetyRulesConfig:
    """
    GET /api/v1/safety-rules/effective?cameraId={cameraId}&siteId={siteId}
    Falls back to system-default if camera-specific config is unavailable.
    """
    url = f"{config.BACKEND_BASE_URL}{HTTP_SAFETY_RULES_EFFECTIVE}"
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(url, params={"cameraId": camera_id, "siteId": site_id}, headers=_get_auth_headers())
            resp.raise_for_status()
            data = resp.json().get("data", {})
            rules_dict = data.get("rules") if isinstance(data.get("rules"), dict) else {}
            return SafetyRulesConfig(
                confidence=float(data.get("confidenceThreshold", DEFAULT_CONFIDENCE_THRESHOLD)),
                confirmation_s=float(data.get("confirmationDuration", DEFAULT_CONFIRMATION_DURATION_S)),
                cooldown_s=int(data.get("cooldownDuration", DEFAULT_COOLDOWN_DURATION_S)),
                enable_no_helmet=data.get("enableNoHelmet", rules_dict.get("enableNoHelmet", True)),
                enable_no_vest=data.get("enableNoVest", rules_dict.get("enableNoVest", True)),
                enable_restricted_area=data.get("enableRestrictedArea", rules_dict.get("enableRestrictedArea", False)),
                model_path=data.get("modelVersionId", rules_dict.get("modelVersionId")),
            )
    except Exception as exc:
        logger.warning("Effective rules fetch failed (%s). Falling back to system default.", exc)
        return await fetch_system_default_rules()


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
