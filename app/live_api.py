"""
Shared Live Stream & Model API Helpers/Router cho smart-ppe-ai-service.
Dùng chung bởi production server (app/server.py) và Test-UI sandbox (test-ui/server.py).
"""

from __future__ import annotations

import asyncio
import ipaddress
from pathlib import Path
import re
import socket
from typing import Annotated, Any, Callable, List, Optional
from urllib.parse import urlparse

from fastapi import APIRouter, Depends, HTTPException, Response
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, ConfigDict, Field, StringConstraints

from app import config
from app.auth import (
    AuthContext,
    PERM_PPE_CAMERA_MANAGE,
    PERM_PPE_CAMERA_READ,
    PERM_PPE_INCIDENT_READ,
    PERM_PPE_MODEL_READ,
    require_any_permission,
)
from app.live_engine import (
    MAX_ROI_POINTS,
    MAX_ROI_POLYGONS,
    get_evidence_owner,
    live_engine_manager,
)
from app.video.video_reader import is_youtube_url

VIDEO_EXT = {".mp4", ".mov", ".avi", ".mkv", ".m4v"}
JOB_ID_RE = re.compile(r"^[A-Za-z0-9_-]{3,64}$")
EVIDENCE_FILE_RE = re.compile(r"^[0-9a-f]{12}(?:_ref)?\.jpg$")

LIVE_SOURCE_MAX_LEN = 2048
LIVE_MAX_WEBCAM_INDEX = 9
LIVE_ALLOWED_STREAM_SCHEMES = {"rtsp", "rtsps"}

SafeId = Annotated[str, StringConstraints(pattern=r"^[A-Za-z0-9_-]{1,64}$")]
RoiCoord = Annotated[float, Field(ge=0, le=10000, allow_inf_nan=False)]
RoiPoint = Annotated[List[RoiCoord], Field(min_length=2, max_length=2)]
RoiPolygon = Annotated[List[RoiPoint], Field(min_length=3, max_length=MAX_ROI_POINTS)]
RoiPolygons = Annotated[List[RoiPolygon], Field(max_length=MAX_ROI_POLYGONS)]
Fps = Annotated[float, Field(ge=1.0, le=60.0, allow_inf_nan=False)]
ConfirmSeconds = Annotated[float, Field(ge=0.5, le=30.0, allow_inf_nan=False)]


class LiveStartRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    camera_id: SafeId = "CAM-01"
    source: Annotated[str, StringConstraints(min_length=1, max_length=LIVE_SOURCE_MAX_LEN)]
    model: Optional[Annotated[str, StringConstraints(max_length=255)]] = None
    target_fps: Fps = 25.0
    ai_fps: Fps = 15.0
    roi_polygons: Optional[RoiPolygons] = None
    confirmation_duration_s: ConfirmSeconds = 2.0
    confirmation_duration_vest_s: Optional[ConfirmSeconds] = None
    enable_no_helmet: bool = True
    enable_no_vest: bool = True
    enable_restricted_area: bool = True
    enable_scene_cut: bool = True


class LiveConfigUpdateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    target_fps: Optional[Fps] = None
    ai_fps: Optional[Fps] = None
    roi_polygons: Optional[RoiPolygons] = None
    enable_no_helmet: Optional[bool] = None
    enable_no_vest: Optional[bool] = None
    enable_restricted_area: Optional[bool] = None
    enable_scene_cut: Optional[bool] = None
    confirmation_duration_s: Optional[ConfirmSeconds] = None
    confirmation_duration_vest_s: Optional[ConfirmSeconds] = None


_BLOCKED_STREAM_HOSTS = {
    "localhost",
    "ip6-localhost",
    "ip6-loopback",
    "metadata.google.internal",
    "metadata",
}
_CLOUD_METADATA_IPV6 = ipaddress.ip_address("fd00:ec2::254")
_ZERO_NET_IPV4 = ipaddress.ip_network("0.0.0.0/8")


def _is_dangerous_ip(ip_obj: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    mapped = getattr(ip_obj, "ipv4_mapped", None)
    if mapped is not None:
        return _is_dangerous_ip(mapped)
    if isinstance(ip_obj, ipaddress.IPv4Address) and ip_obj in _ZERO_NET_IPV4:
        return True
    if ip_obj == _CLOUD_METADATA_IPV6:
        return True
    return bool(
        ip_obj.is_loopback
        or ip_obj.is_link_local
        or ip_obj.is_unspecified
        or ip_obj.is_multicast
        or ip_obj.is_reserved
    )


def _is_blocked_stream_host(
    hostname: str,
    allowed_rtsp_hosts: Optional[set[str]] = None,
) -> bool:
    host = hostname.strip().lower().rstrip(".")
    if not host:
        return True
    if (
        host in _BLOCKED_STREAM_HOSTS
        or host.endswith(".localhost")
        or host.endswith(".internal")
        or host.endswith(".local")
    ):
        return True

    try:
        ip_obj = ipaddress.ip_address(host)
    except ValueError:
        ip_obj = None

    if ip_obj is not None:
        return _is_dangerous_ip(ip_obj)

    explicitly_allowlisted = bool(allowed_rtsp_hosts and host in allowed_rtsp_hosts)
    if "." not in host and not explicitly_allowlisted:
        return True

    try:
        addr_infos = socket.getaddrinfo(host, None, type=socket.SOCK_STREAM)
    except OSError:
        return False

    for info in addr_infos:
        sockaddr = info[4]
        if not sockaddr:
            continue
        raw_ip = sockaddr[0].split("%", 1)[0]
        try:
            resolved_ip = ipaddress.ip_address(raw_ip)
        except ValueError:
            continue
        if _is_dangerous_ip(resolved_ip):
            return True

    return False


def resolve_live_source(
    raw: str,
    upload_dir: Path,
    allowed_rtsp_hosts: Optional[set[str]] = None,
    allow_http_streams: bool = False,
):
    """Allowlist kiểm soát nguồn video hợp lệ, chặn đọc file tùy ý / SSRF qua FFmpeg."""
    src = raw.strip()
    if any(ord(c) < 0x20 for c in src):
        raise HTTPException(400, "Nguồn video chứa ký tự không hợp lệ")

    # 1. Webcam index
    if src.isdigit():
        idx = int(src)
        if idx > LIVE_MAX_WEBCAM_INDEX:
            raise HTTPException(400, "Chỉ số webcam không hợp lệ")
        return idx

    # 2. Video đã upload (theo job_id hoặc tên file)
    stem, dot, ext = src.partition(".")
    if upload_dir.exists() and JOB_ID_RE.match(stem) and (not dot or f".{ext.lower()}" in VIDEO_EXT):
        candidates = [upload_dir / src] if dot else [upload_dir / f"{stem}{e}" for e in sorted(VIDEO_EXT)]
        for cand in candidates:
            resolved = cand.resolve()
            if resolved.is_relative_to(upload_dir.resolve()) and resolved.is_file():
                return str(resolved)

    # 3. YouTube
    if is_youtube_url(src):
        return src

    # 4. RTSP / optional HTTP stream
    try:
        parsed = urlparse(src)
    except ValueError:
        parsed = None

    allowed_schemes = set(LIVE_ALLOWED_STREAM_SCHEMES)
    if allow_http_streams:
        allowed_schemes.update({"http", "https"})

    if parsed and parsed.scheme.lower() in allowed_schemes and parsed.hostname:
        host_norm = parsed.hostname.strip().lower().rstrip(".")
        if parsed.scheme.lower() in LIVE_ALLOWED_STREAM_SCHEMES and allowed_rtsp_hosts:
            if host_norm not in allowed_rtsp_hosts:
                raise HTTPException(400, "Host RTSP không nằm trong danh sách cho phép")
        if _is_blocked_stream_host(host_norm, allowed_rtsp_hosts=allowed_rtsp_hosts):
            raise HTTPException(400, "Host nguồn video bị chặn theo chính sách chống SSRF")
        return src

    raise HTTPException(
        400,
        "Nguồn video không được hỗ trợ. Chỉ chấp nhận: chỉ số webcam, job_id video đã upload, "
        "URL YouTube hoặc URL rtsp(s)://",
    )


def resolve_model_path(model_name: Optional[str], models_dir: Path) -> Optional[str]:
    if not model_name:
        return None
    model_cand = (models_dir / Path(model_name).name).resolve()
    if (
        not model_cand.exists()
        or not model_cand.is_relative_to(models_dir.resolve())
        or model_cand.suffix.lower() not in (".pt", ".onnx")
    ):
        raise HTTPException(400, f"Mô hình không hợp lệ hoặc không tồn tại: {model_name}")
    return str(model_cand)


def scan_models_from_disk(models_dir: Path) -> list[dict]:
    available = []
    if models_dir.exists():
        for p in models_dir.glob("*.pt"):
            if p.stat().st_size > 1000:
                available.append({
                    "name": p.stem,
                    "path": f"models/{p.name}",
                    "size_mb": round(p.stat().st_size / (1024 * 1024), 1),
                    "is_default": "yolo11m" in p.name,
                })
    available.sort(key=lambda x: (not x["is_default"], x["name"]))
    return available


def scan_uploaded_videos(upload_dir: Path) -> list[dict]:
    videos = []
    if upload_dir.exists():
        for p in sorted(upload_dir.glob("*")):
            if p.is_file() and p.suffix.lower() in VIDEO_EXT and p.stat().st_size > 10000:
                videos.append({
                    "filename": p.name,
                    "stem": p.stem,
                    "size_mb": round(p.stat().st_size / (1024 * 1024), 2),
                })
    return videos


def can_access_live_session(session: Any, auth: Optional[AuthContext]) -> bool:
    """Kiểm tra quyền truy cập một phiên Live theo site_id hoặc người tạo phiên test."""
    if auth is None or auth.is_internal or auth.is_admin or auth.site_ids is None:
        return True
    if getattr(session, "managed", False):
        return auth.can_access_site(getattr(session, "site_id", None))
    owner_uid = getattr(session, "owner_user_id", None)
    owner_sid = getattr(session, "owner_site_id", None)
    if owner_uid is not None:
        if owner_uid == auth.user_id:
            return True
        if owner_sid and auth.can_access_site(owner_sid):
            return True
        return False
    return auth.can_access_site(getattr(session, "site_id", None))


def _assert_session_access(
    session: Any,
    auth: Optional[AuthContext],
    enforce_auth: bool,
) -> None:
    if not enforce_auth:
        return
    if not can_access_live_session(session, auth):
        raise HTTPException(
            403,
            "Bạn không có quyền truy cập hoặc điều khiển camera thuộc công trường này",
        )


def _noop_auth() -> Optional[AuthContext]:
    return None


def create_common_live_router(
    models_dir: Path,
    upload_dir: Path,
    after_config_fn: Optional[Callable[[], Any]] = None,
    enforce_auth: bool = False,
) -> APIRouter:
    """Router chứa các endpoint vận hành Live Engine & Models dùng chung."""
    router = APIRouter()

    model_read_dep = (
        require_any_permission(PERM_PPE_CAMERA_READ, PERM_PPE_MODEL_READ)
        if enforce_auth
        else _noop_auth
    )
    camera_read_dep = (
        require_any_permission(PERM_PPE_CAMERA_READ)
        if enforce_auth
        else _noop_auth
    )
    camera_manage_dep = (
        require_any_permission(PERM_PPE_CAMERA_MANAGE)
        if enforce_auth
        else _noop_auth
    )
    evidence_read_dep = (
        require_any_permission(PERM_PPE_CAMERA_READ, PERM_PPE_INCIDENT_READ)
        if enforce_auth
        else _noop_auth
    )

    @router.get("/api/models", tags=["Model Management"], summary="Liệt kê các mô hình YOLO (.pt) khả dụng")
    async def list_models(_auth: Optional[AuthContext] = Depends(model_read_dep)):
        return await asyncio.to_thread(scan_models_from_disk, models_dir)

    @router.get("/api/live/videos", tags=["Live Stream AI Engine"], summary="Danh sách video trong uploads/ sẵn sàng phát trực tiếp")
    async def list_live_videos(_auth: Optional[AuthContext] = Depends(camera_read_dep)):
        return await asyncio.to_thread(scan_uploaded_videos, upload_dir)

    @router.get("/api/live/{camera_id}/stream", tags=["Live Stream AI Engine"], summary="Phát luồng MJPEG trực tiếp")
    def get_live_stream(camera_id: str, auth: Optional[AuthContext] = Depends(camera_read_dep)):
        session = live_engine_manager.get_session(camera_id)
        if not session:
            raise HTTPException(404, f"Không tìm thấy phiên phát cho camera {camera_id}")
        _assert_session_access(session, auth, enforce_auth)
        return StreamingResponse(
            session.generate_mjpeg(),
            media_type="multipart/x-mixed-replace; boundary=frame",
        )

    @router.get("/api/live/{camera_id}/stats", tags=["Live Stream AI Engine"], summary="Lấy số liệu thời gian thực")
    def get_live_stats(camera_id: str, auth: Optional[AuthContext] = Depends(camera_read_dep)):
        session = live_engine_manager.get_session(camera_id)
        if not session:
            raise HTTPException(404, f"Không tìm thấy phiên phát cho camera {camera_id}")
        _assert_session_access(session, auth, enforce_auth)
        return {
            "camera_id": session.camera_id,
            "camera_uuid": session.camera_uuid,
            "site_id": session.site_id,
            "status": session.status,
            "managed": session.managed,
            "reconnect_count": session.reconnect_count,
            "error": session.error_message,
            "metrics": session.metrics,
            "recent_violations": session.recent_violations,
        }

    @router.patch("/api/live/{camera_id}/config", tags=["Live Stream AI Engine"], summary="Cập nhật cấu hình nóng")
    def update_live_config(
        camera_id: str,
        req: LiveConfigUpdateRequest,
        auth: Optional[AuthContext] = Depends(camera_manage_dep),
    ):
        session = live_engine_manager.get_session(camera_id)
        if not session:
            raise HTTPException(404, f"Không tìm thấy phiên phát cho camera {camera_id}")
        _assert_session_access(session, auth, enforce_auth)
        session.update_config(
            target_fps=req.target_fps,
            ai_fps=req.ai_fps,
            roi_polygons=req.roi_polygons,
            enable_no_helmet=req.enable_no_helmet,
            enable_no_vest=req.enable_no_vest,
            enable_restricted_area=req.enable_restricted_area,
            enable_scene_cut=req.enable_scene_cut,
            confirmation_duration_s=req.confirmation_duration_s,
            confirmation_duration_vest_s=req.confirmation_duration_vest_s,
        )
        return {
            "status": "updated",
            "camera_id": camera_id,
            "target_fps": session.target_fps,
            "ai_fps": session.ai_fps,
        }

    @router.post("/api/live/{camera_id}/stop", tags=["Live Stream AI Engine"], summary="Dừng phiên phát trực tiếp")
    def stop_live_camera(
        camera_id: str,
        auth: Optional[AuthContext] = Depends(camera_manage_dep),
    ):
        session = live_engine_manager.get_session(camera_id)
        if not session:
            raise HTTPException(404, f"Không tìm thấy phiên phát cho camera {camera_id}")
        _assert_session_access(session, auth, enforce_auth)
        stopped = live_engine_manager.stop_session(camera_id)
        if not stopped:
            raise HTTPException(404, f"Không tìm thấy phiên phát cho camera {camera_id}")
        return {"status": "stopped", "camera_id": camera_id}

    @router.get("/api/live/sessions", tags=["Live Stream AI Engine"], summary="Liệt kê danh sách các camera đang phát trực tiếp")
    def list_live_sessions(
        response: Response,
        auth: Optional[AuthContext] = Depends(camera_read_dep),
    ):
        response.headers["X-Live-Max-Sessions"] = str(live_engine_manager.get_max_sessions())
        sessions = live_engine_manager.list_sessions()
        if (
            enforce_auth
            and auth is not None
            and not (auth.is_internal or auth.is_admin or auth.site_ids is None)
        ):
            filtered = []
            for item in sessions:
                sess = live_engine_manager.get_session(item.get("camera_id", ""))
                if sess is not None and can_access_live_session(sess, auth):
                    filtered.append(item)
                elif sess is None and item.get("managed") and auth.can_access_site(item.get("site_id")):
                    filtered.append(item)
            return filtered
        return sessions

    @router.get("/api/live/evidence/{filename}", tags=["Live Stream AI Engine"], summary="Lấy ảnh bằng chứng vi phạm trực tiếp")
    def get_live_evidence(
        filename: str,
        auth: Optional[AuthContext] = Depends(evidence_read_dep),
    ):
        if not EVIDENCE_FILE_RE.match(filename):
            raise HTTPException(400, "Tên file bằng chứng không hợp lệ")
        ev_root = Path(config.EVIDENCE_DIR).resolve()
        target = (ev_root / filename).resolve()
        if not target.is_relative_to(ev_root):
            raise HTTPException(404, "Không tìm thấy ảnh bằng chứng trực tiếp")

        if (
            enforce_auth
            and auth is not None
            and not (auth.is_internal or auth.is_admin or auth.site_ids is None)
        ):
            owner_info = get_evidence_owner(filename)
            if owner_info is not None:
                ev_site_id, ev_owner_uid, ev_managed = owner_info
                if ev_managed:
                    if not auth.can_access_site(ev_site_id):
                        raise HTTPException(403, "Không có quyền truy cập ảnh bằng chứng thuộc công trường này")
                else:
                    allowed = (ev_owner_uid == auth.user_id) or (
                        bool(ev_site_id) and auth.can_access_site(ev_site_id)
                    )
                    if not allowed:
                        raise HTTPException(403, "Không có quyền truy cập ảnh bằng chứng này")
            else:
                raise HTTPException(
                    403 if target.is_file() else 404,
                    "Không có quyền truy cập ảnh bằng chứng trực tiếp",
                )

        if not target.is_file():
            raise HTTPException(404, "Không tìm thấy ảnh bằng chứng trực tiếp")
        return FileResponse(target, media_type="image/jpeg")

    return router
