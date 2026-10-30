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


class SafetyRulesConfig:
    def __init__(self, confidence: float, confirmation_s: int, cooldown_s: int,
                 enable_no_helmet: bool = True, enable_no_vest: bool = True,
                 enable_restricted_area: bool = False):
        self.confidence = confidence
        self.confirmation_s = confirmation_s
        self.cooldown_s = cooldown_s
        self.enable_no_helmet = enable_no_helmet
        self.enable_no_vest = enable_no_vest
        self.enable_restricted_area = enable_restricted_area


_FALLBACK = SafetyRulesConfig(
    confidence=DEFAULT_CONFIDENCE_THRESHOLD,
    confirmation_s=DEFAULT_CONFIRMATION_DURATION_S,
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
                confirmation_s=int(data.get("confirmationDuration", DEFAULT_CONFIRMATION_DURATION_S)),
                cooldown_s=int(data.get("cooldownDuration", DEFAULT_COOLDOWN_DURATION_S)),
                enable_no_helmet=data.get("enableNoHelmet", rules_dict.get("enableNoHelmet", True)),
                enable_no_vest=data.get("enableNoVest", rules_dict.get("enableNoVest", True)),
                enable_restricted_area=data.get("enableRestrictedArea", rules_dict.get("enableRestrictedArea", False)),
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
                confirmation_s=int(data.get("confirmationDuration", DEFAULT_CONFIRMATION_DURATION_S)),
                cooldown_s=int(data.get("cooldownDuration", DEFAULT_COOLDOWN_DURATION_S)),
                enable_no_helmet=data.get("enableNoHelmet", rules_dict.get("enableNoHelmet", True)),
                enable_no_vest=data.get("enableNoVest", rules_dict.get("enableNoVest", True)),
                enable_restricted_area=data.get("enableRestrictedArea", rules_dict.get("enableRestrictedArea", False)),
            )
    except Exception as exc:
        logger.warning("Effective rules fetch failed (%s). Falling back to system default.", exc)
        return await fetch_system_default_rules()
