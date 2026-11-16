"""
RS256 JWKS + Cookie/Bearer Authentication & RBAC/Site-Scoping cho smart-ppe-ai-service.

Tuân thủ kiến trúc Decentralized Auth của hệ thống:
- Trích xuất JWT từ HttpOnly cookie `accessToken` hoặc header `Authorization: Bearer <token>`.
- Chặn tuyệt đối truyền token qua query string (?token=... / ?ticket=..., CWE-598).
- Chỉ chấp nhận thuật toán RS256 (chống alg=none / key confusion).
- Xác thực chữ ký offline qua JWT_PUBLIC_KEY / file PEM / JWKS cache từ Identity Service.
- Hỗ trợ M2M xác thực qua `x-internal-service-token` (constant-time SHA-256 compare).
- Tách biệt purpose claim (chặn 2FA_VERIFICATION / TEMP) và mustChangePassword.
- Kiểm tra RBAC (`PERMISSIONS`) và phạm vi công trường (`siteId` / `siteIds`).
"""

from __future__ import annotations

import base64
from dataclasses import dataclass, field
import hashlib
import hmac
import json
import logging
from pathlib import Path
import threading
import time
from typing import Any, Callable, Iterable, Optional, Set
import urllib.request

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import padding, rsa
from cryptography.hazmat.primitives.serialization import load_pem_public_key
from fastapi import Depends, HTTPException, Request

from app import config

logger = logging.getLogger("ai_auth")

ROLE_ADMIN = "ADMIN"
ROLE_SAFETY_MANAGER = "SAFETY_MANAGER"
ROLE_SAFETY_OFFICER = "SAFETY_OFFICER"

PERM_PPE_CAMERA_READ = "PPE_CAMERA_READ"
PERM_PPE_CAMERA_MANAGE = "PPE_CAMERA_MANAGE"
PERM_PPE_INCIDENT_READ = "PPE_INCIDENT_READ"
PERM_PPE_MODEL_READ = "PPE_MODEL_READ"
PERM_SYSTEM_CONFIG_READ = "SYSTEM_CONFIG_READ"
PERM_SYSTEM_CONFIG_MANAGE = "SYSTEM_CONFIG_MANAGE"

ALL_KNOWN_PERMISSIONS: Set[str] = {
    PERM_PPE_CAMERA_READ,
    PERM_PPE_CAMERA_MANAGE,
    PERM_PPE_INCIDENT_READ,
    PERM_PPE_MODEL_READ,
    PERM_SYSTEM_CONFIG_READ,
    PERM_SYSTEM_CONFIG_MANAGE,
}

_SAFETY_MANAGER_FALLBACK_PERMS: Set[str] = {
    "PPE_SITE_READ",
    "PPE_SITE_MANAGE",
    "PPE_ZONE_READ",
    "PPE_ZONE_MANAGE",
    "PPE_RESTRICTED_AREA_READ",
    "PPE_RESTRICTED_AREA_MANAGE",
    PERM_PPE_CAMERA_READ,
    PERM_PPE_CAMERA_MANAGE,
    "PPE_SAFETY_CONFIG_READ",
    "PPE_SAFETY_CONFIG_MANAGE",
    PERM_PPE_INCIDENT_READ,
    "PPE_INCIDENT_REVIEW",
    "PPE_INCIDENT_UPDATE",
    "PPE_INCIDENT_MANAGE",
    "PPE_REPORTING_READ",
    "PPE_REPORTING_EXPORT",
    "PPE_AI_METRICS_READ",
    PERM_PPE_MODEL_READ,
    "NOTIFICATION_READ",
    "NOTIFICATION_CREATE",
    "FILE_READ",
    "FILE_UPLOAD",
    "FILE_MANAGE",
}

_SAFETY_OFFICER_FALLBACK_PERMS: Set[str] = {
    "PPE_SITE_READ",
    "PPE_ZONE_READ",
    "PPE_RESTRICTED_AREA_READ",
    PERM_PPE_CAMERA_READ,
    "PPE_SAFETY_CONFIG_READ",
    PERM_PPE_INCIDENT_READ,
    "PPE_INCIDENT_REVIEW",
    "PPE_INCIDENT_UPDATE",
    "PPE_REPORTING_READ",
    "NOTIFICATION_READ",
    "FILE_READ",
    "FILE_UPLOAD",
}

_ROOT_DIR = Path(__file__).resolve().parent.parent

_jwks_lock = threading.Lock()
_jwks_cached_keys: list[tuple[Optional[str], rsa.RSAPublicKey]] = []
_jwks_cached_at: float = 0.0
_jwks_last_forced_refresh_at: float = 0.0


def clear_auth_caches() -> None:
    """Xóa cache JWKS trong bộ nhớ (hữu ích cho test hoặc key rotation tức thì)."""
    global _jwks_cached_keys, _jwks_cached_at, _jwks_last_forced_refresh_at
    with _jwks_lock:
        _jwks_cached_keys = []
        _jwks_cached_at = 0.0
        _jwks_last_forced_refresh_at = 0.0


@dataclass(frozen=True)
class AuthContext:
    user_id: str
    role: Optional[str] = None
    permissions: Set[str] = field(default_factory=set)
    site_ids: Optional[Set[str]] = None  # None = không giới hạn (ADMIN / SYSTEM_INTERNAL)
    is_internal: bool = False
    is_admin: bool = False

    def has_any_permission(self, required: Iterable[str]) -> bool:
        if self.is_internal or self.is_admin:
            return True
        req = set(required)
        return bool(self.permissions.intersection(req))

    def can_access_site(self, site_id: Optional[str]) -> bool:
        if self.is_internal or self.is_admin or self.site_ids is None:
            return True
        if not site_id:
            return False
        return str(site_id) in self.site_ids


def _b64url_decode(segment: str) -> bytes:
    if not segment or not isinstance(segment, str):
        raise ValueError("Empty base64url segment")
    padded = segment + "=" * (-len(segment) % 4)
    return base64.urlsafe_b64decode(padded.encode("ascii"))


def _constant_time_token_equal(provided: str, expected: str) -> bool:
    if not provided or not expected:
        return False
    p_hash = hashlib.sha256(provided.encode("utf-8")).digest()
    e_hash = hashlib.sha256(expected.encode("utf-8")).digest()
    return hmac.compare_digest(p_hash, e_hash)


def _extract_internal_token(request: Request) -> Optional[str]:
    expected = (config.INTERNAL_SERVICE_TOKEN or "").strip()
    if not expected:
        return None
    provided = (request.headers.get("x-internal-service-token") or "").strip()
    if provided and _constant_time_token_equal(provided, expected):
        return provided
    return None


def _extract_jwt_token(request: Request) -> Optional[str]:
    cookie_token = request.cookies.get("accessToken")
    if isinstance(cookie_token, str) and cookie_token.strip():
        return cookie_token.strip()

    auth_header = request.headers.get("authorization")
    if isinstance(auth_header, str):
        parts = auth_header.strip().split(None, 1)
        if len(parts) == 2 and parts[0].lower() == "bearer" and parts[1].strip():
            return parts[1].strip()

    return None


def _load_static_pem_key() -> Optional[rsa.RSAPublicKey]:
    pem_text = (getattr(config, "JWT_PUBLIC_KEY", "") or "").strip()
    if pem_text:
        try:
            key = load_pem_public_key(pem_text.encode("utf-8"))
            if isinstance(key, rsa.RSAPublicKey):
                return key
        except Exception as e:
            logger.warning("Failed loading JWT_PUBLIC_KEY PEM: %s", type(e).__name__)

    pem_path_raw = (getattr(config, "JWT_PUBLIC_KEY_PATH", "") or "").strip()
    if pem_path_raw:
        p = Path(pem_path_raw)
        candidates = [p] if p.is_absolute() else [p, _ROOT_DIR / p]
        for candidate in candidates:
            try:
                if candidate.is_file():
                    key = load_pem_public_key(candidate.read_bytes())
                    if isinstance(key, rsa.RSAPublicKey):
                        return key
            except Exception as e:
                logger.warning("Failed reading JWT public key file %s: %s", candidate, type(e).__name__)
    return None


def _parse_jwks_payload(data: Any) -> list[tuple[Optional[str], rsa.RSAPublicKey]]:
    if not isinstance(data, dict):
        return []
    keys_raw = data.get("keys")
    if not isinstance(keys_raw, list):
        return []
    parsed: list[tuple[Optional[str], rsa.RSAPublicKey]] = []
    for item in keys_raw:
        if not isinstance(item, dict) or item.get("kty") != "RSA":
            continue
        n_b64 = item.get("n")
        e_b64 = item.get("e")
        if not isinstance(n_b64, str) or not isinstance(e_b64, str):
            continue
        try:
            n_int = int.from_bytes(_b64url_decode(n_b64), "big")
            e_int = int.from_bytes(_b64url_decode(e_b64), "big")
            pub_key = rsa.RSAPublicNumbers(e_int, n_int).public_key()
            kid = item.get("kid")
            parsed.append((str(kid) if kid is not None else None, pub_key))
        except Exception:
            continue
    return parsed


def _fetch_jwks_from_identity() -> list[tuple[Optional[str], rsa.RSAPublicKey]]:
    jwks_url = (getattr(config, "AUTH_JWKS_URL", "") or "").strip()
    if not jwks_url or not (jwks_url.startswith("http://") or jwks_url.startswith("https://")):
        return []
    try:
        req = urllib.request.Request(jwks_url, headers={"Accept": "application/json"})
        with urllib.request.urlopen(req, timeout=3.0) as resp:
            if resp.status != 200:
                return []
            payload = json.loads(resp.read().decode("utf-8"))
        return _parse_jwks_payload(payload)
    except Exception as e:
        logger.debug("JWKS fetch from %s failed: %s", jwks_url, type(e).__name__)
        return []


def _get_candidate_rsa_keys(kid: Optional[str], force_jwks: bool = False) -> list[rsa.RSAPublicKey]:
    global _jwks_cached_keys, _jwks_cached_at, _jwks_last_forced_refresh_at

    candidates: list[rsa.RSAPublicKey] = []
    static_key = _load_static_pem_key()
    if static_key is not None and not force_jwks:
        candidates.append(static_key)
        return candidates

    now = time.monotonic()
    ttl = float(getattr(config, "JWT_JWKS_CACHE_TTL_S", 300))

    with _jwks_lock:
        cache_expired = (not _jwks_cached_keys) or ((now - _jwks_cached_at) > ttl)
        kid_missing = bool(
            kid
            and _jwks_cached_keys
            and not any(cached_kid == kid for cached_kid, _ in _jwks_cached_keys)
            and (now - _jwks_last_forced_refresh_at) >= 5.0
        )
        cooldown_ok = (now - _jwks_last_forced_refresh_at) >= 5.0
        should_fetch_jwks = cache_expired or kid_missing or (force_jwks and cooldown_ok)

    if should_fetch_jwks:
        fetched = _fetch_jwks_from_identity()
        with _jwks_lock:
            if fetched:
                _jwks_cached_keys = fetched
                _jwks_cached_at = time.monotonic()
            if kid_missing or force_jwks:
                _jwks_last_forced_refresh_at = time.monotonic()
            jwks_snapshot = list(_jwks_cached_keys)
    else:
        with _jwks_lock:
            jwks_snapshot = list(_jwks_cached_keys)

    if kid:
        matched = [k for k_id, k in jwks_snapshot if k_id == kid]
        candidates.extend(matched)
        if not matched:
            candidates.extend(k for _, k in jwks_snapshot)
    else:
        candidates.extend(k for _, k in jwks_snapshot)

    if static_key is not None:
        candidates.append(static_key)

    return candidates


def verify_jwt_token(token: str) -> dict[str, Any]:
    parts = token.split(".")
    if len(parts) != 3 or not all(parts):
        raise HTTPException(401, "Token xác thực không hợp lệ")

    header_b64, payload_b64, sig_b64 = parts
    try:
        header = json.loads(_b64url_decode(header_b64).decode("utf-8"))
        payload = json.loads(_b64url_decode(payload_b64).decode("utf-8"))
        sig_bytes = _b64url_decode(sig_b64)
    except Exception:
        raise HTTPException(401, "Token xác thực không hợp lệ")

    if not isinstance(header, dict) or not isinstance(payload, dict):
        raise HTTPException(401, "Token xác thực không hợp lệ")

    alg = header.get("alg")
    if alg != "RS256":
        raise HTTPException(
            401,
            f"Thuật toán '{alg}' không được chấp nhận. Hệ thống chỉ hỗ trợ RS256.",
        )

    kid_raw = header.get("kid")
    kid = str(kid_raw) if kid_raw is not None else None
    signing_input = f"{header_b64}.{payload_b64}".encode("ascii")

    keys = _get_candidate_rsa_keys(kid, force_jwks=False)
    verified = False
    for pub_key in keys:
        try:
            pub_key.verify(sig_bytes, signing_input, padding.PKCS1v15(), hashes.SHA256())
            verified = True
            break
        except InvalidSignature:
            continue
        except Exception:
            continue

    # Nếu chưa verify được (vd. xoay khóa JWKS trên Identity Service), thử tải JWKS một lần
    if not verified:
        jwks_keys = _get_candidate_rsa_keys(kid, force_jwks=True)
        for pub_key in jwks_keys:
            try:
                pub_key.verify(sig_bytes, signing_input, padding.PKCS1v15(), hashes.SHA256())
                verified = True
                break
            except InvalidSignature:
                continue
            except Exception:
                continue
        if not keys and not jwks_keys:
            raise HTTPException(500, "Cấu hình khóa công khai RS256 chưa sẵn sàng")

    if not verified:
        raise HTTPException(401, "Token xác thực không hợp lệ hoặc đã hết hạn")

    now = time.time()
    exp = payload.get("exp")
    if not isinstance(exp, (int, float)) or isinstance(exp, bool) or now > float(exp):
        raise HTTPException(401, "Token xác thực không hợp lệ hoặc đã hết hạn")

    nbf = payload.get("nbf")
    if isinstance(nbf, (int, float)) and not isinstance(nbf, bool) and now < float(nbf):
        raise HTTPException(401, "Token xác thực chưa có hiệu lực")

    purpose = payload.get("purpose")
    if purpose is not None and purpose != "ACCESS":
        raise HTTPException(401, "Token không có quyền truy cập API nghiệp vụ")

    must_change = payload.get("mustChangePassword")
    if must_change is True or (isinstance(must_change, str) and must_change.lower() == "true"):
        raise HTTPException(
            403,
            "Bạn bắt buộc phải đổi mật khẩu trước khi thực hiện các thao tác khác trên hệ thống",
        )

    return payload


def _resolve_permissions(payload: dict[str, Any]) -> tuple[Set[str], bool]:
    role = payload.get("role")
    role_str = str(role) if role is not None else None
    is_admin = role_str == ROLE_ADMIN

    raw_perms = payload.get("permissions")
    if isinstance(raw_perms, list) and len(raw_perms) > 0:
        return {str(p) for p in raw_perms if p}, is_admin

    if is_admin:
        return set(ALL_KNOWN_PERMISSIONS), True
    if role_str == ROLE_SAFETY_MANAGER:
        return set(_SAFETY_MANAGER_FALLBACK_PERMS), False
    if role_str == ROLE_SAFETY_OFFICER:
        return set(_SAFETY_OFFICER_FALLBACK_PERMS), False
    return set(), False


def _resolve_site_ids(payload: dict[str, Any], is_admin: bool) -> Optional[Set[str]]:
    if is_admin:
        return None
    sites: Set[str] = set()
    site_id = payload.get("siteId")
    if isinstance(site_id, str) and site_id.strip():
        sites.add(site_id.strip())
    site_ids = payload.get("siteIds")
    if isinstance(site_ids, list):
        for s in site_ids:
            if isinstance(s, str) and s.strip():
                sites.add(s.strip())
    return sites


def authenticate_request(request: Request, allow_internal: bool = True) -> AuthContext:
    # SEC-14: Chặn truyền token qua query string (?token=... hoặc ?ticket=...)
    if request.query_params.get("token") or request.query_params.get("ticket"):
        raise HTTPException(
            400,
            "Không được phép truyền token xác thực qua query string để tránh rò rỉ qua log và browser history",
        )

    if allow_internal and _extract_internal_token(request) is not None:
        return AuthContext(
            user_id="SYSTEM_INTERNAL",
            role="SYSTEM_INTERNAL",
            permissions=set(ALL_KNOWN_PERMISSIONS),
            site_ids=None,
            is_internal=True,
            is_admin=True,
        )

    token = _extract_jwt_token(request)
    if not token:
        raise HTTPException(401, "Unauthorized")

    payload = verify_jwt_token(token)
    user_id = str(payload.get("id") or payload.get("sub") or "")
    if not user_id:
        raise HTTPException(401, "Token xác thực không hợp lệ")

    perms, is_admin = _resolve_permissions(payload)
    site_ids = _resolve_site_ids(payload, is_admin=is_admin)

    return AuthContext(
        user_id=user_id,
        role=str(payload.get("role")) if payload.get("role") is not None else None,
        permissions=perms,
        site_ids=site_ids,
        is_internal=False,
        is_admin=is_admin,
    )


def require_any_permission(
    *permissions: str,
    allow_internal: bool = True,
) -> Callable[[Request], AuthContext]:
    """Dependency kiểm tra JWT/M2M + ít nhất một permission trong danh sách."""

    def _dependency(request: Request) -> AuthContext:
        auth = authenticate_request(request, allow_internal=allow_internal)
        if permissions and not auth.has_any_permission(permissions):
            raise HTTPException(403, "Forbidden: không đủ quyền truy cập")
        return auth

    return _dependency


def require_admin_or_permission(
    *permissions: str,
    allow_internal: bool = True,
) -> Callable[[Request], AuthContext]:
    """Dependency yêu cầu ADMIN / M2M hoặc permission quản trị được chỉ định."""

    def _dependency(request: Request) -> AuthContext:
        auth = authenticate_request(request, allow_internal=allow_internal)
        if auth.is_internal or auth.is_admin:
            return auth
        if permissions and auth.has_any_permission(permissions):
            return auth
        raise HTTPException(403, "Forbidden: yêu cầu quyền quản trị")

    return _dependency


def get_auth_context_dep(
    allow_internal: bool = True,
) -> Callable[[Request], AuthContext]:
    def _dep(request: Request) -> AuthContext:
        return authenticate_request(request, allow_internal=allow_internal)

    return Depends(_dep)
