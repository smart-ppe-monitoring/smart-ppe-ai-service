"""
Production Live AI Server (:8000) — smart-ppe-ai-service.

Khởi chạy độc lập chế độ giám sát 24/7:
- Tự động đồng bộ camera ONLINE từ Camera Service (qua M2M token, không lộ RTSP password).
- Đọc cấu hình `ai.live_max_sessions` động từ Identity Service (hoặc PATCH /api/live/config).
- Bật sẵn dispatch sự kiện vi phạm thật tới MinIO + Incident Service.
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
import logging
import os
from pathlib import Path
import threading
from typing import Annotated, Optional

from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from pydantic import BaseModel, ConfigDict, Field, StringConstraints

from app import config
from app.auth import (
    AuthContext,
    PERM_PPE_CAMERA_MANAGE,
    PERM_PPE_CAMERA_READ,
    PERM_SYSTEM_CONFIG_MANAGE,
    PERM_SYSTEM_CONFIG_READ,
    require_admin_or_permission,
    require_any_permission,
)
from app.clients.backend_client import lookup_internal_camera_sync
from app.events import flush_producer
from app.live_api import (
    ConfirmSeconds,
    Fps,
    LIVE_SOURCE_MAX_LEN,
    RoiPolygons,
    SafeId,
    can_access_live_session,
    create_common_live_router,
    resolve_live_source,
    resolve_model_path,
)
from app.live_engine import (
    LiveSessionLimitError,
    live_engine_manager,
    preload_detector,
)
from app.live_sync import CameraSyncManager, _extract_roi_polygons_from_config

logger = logging.getLogger("ai_server")

ROOT_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = (ROOT_DIR / "models").resolve()
UPLOAD_DIR = (ROOT_DIR / "test-ui" / "uploads").resolve()

LIVE_DISPATCH_ENABLED = config.LIVE_DISPATCH_DEFAULT
LIVE_ALLOWED_RTSP_HOSTS = set(config.LIVE_ALLOWED_RTSP_HOSTS)

camera_sync_manager = CameraSyncManager(
    manager=live_engine_manager,
    dispatch_enabled=LIVE_DISPATCH_ENABLED,
    sync_interval_s=config.AI_CAMERA_SYNC_INTERVAL_S,
)


class ProductionLiveStartRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    camera_id: SafeId = "CAM-01"
    camera_uuid: Optional[SafeId] = None
    source: Optional[Annotated[str, StringConstraints(min_length=1, max_length=LIVE_SOURCE_MAX_LEN)]] = None
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
    auto_reconnect: bool = True
    loop_on_eof: bool = True


class LiveRuntimeConfigPatchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    max_sessions: Optional[Annotated[int, Field(ge=1, le=256)]] = None
    trigger_sync: bool = False


@asynccontextmanager
async def lifespan(_app: FastAPI):
    if not os.getenv("PYTEST_CURRENT_TEST"):
        rec_model = MODELS_DIR / "yolo11s_joint_phase3.pt"
        target = str(rec_model) if rec_model.exists() else config.YOLO_MODEL_PATH
        threading.Thread(target=preload_detector, args=(target,), daemon=True, name="model-prewarm").start()
        if config.AI_AUTO_SYNC_ENABLED:
            camera_sync_manager.start_background()
    yield
    camera_sync_manager.stop_background()
    live_engine_manager.stop_all()
    try:
        flush_producer(timeout=2.0)
    except Exception:
        pass


app = FastAPI(
    title="Smart PPE Monitoring — Production AI Live Server",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=config.AI_ALLOWED_HOSTS or ["127.0.0.1", "localhost"],
)


@app.get("/health", tags=["Health"], summary="Healthcheck của AI Production Server")
def health_check():
    sessions = live_engine_manager.list_sessions()
    active = sum(1 for s in sessions if s.get("status") not in ("stopped", "error"))
    sync_status = camera_sync_manager.get_status()
    return {
        "status": "ok",
        "service": "smart-ppe-ai-service",
        "mode": "production",
        "dispatch_enabled": LIVE_DISPATCH_ENABLED,
        "auto_sync_enabled": config.AI_AUTO_SYNC_ENABLED,
        "active_sessions": active,
        "max_sessions": live_engine_manager.get_max_sessions(),
        "sync_ok": sync_status.get("last_sync_ok"),
    }


@app.get("/api/live/config", tags=["Live Stream AI Engine"], summary="Xem cấu hình giới hạn phiên động hiện tại")
def get_live_runtime_config(
    _auth: AuthContext = Depends(
        require_any_permission(PERM_PPE_CAMERA_READ, PERM_SYSTEM_CONFIG_READ)
    ),
):
    sessions = live_engine_manager.list_sessions()
    active = sum(1 for s in sessions if s.get("status") not in ("stopped", "error"))
    return {
        "max_sessions": live_engine_manager.get_max_sessions(),
        "active_sessions": active,
        "dispatch_enabled": LIVE_DISPATCH_ENABLED,
        "auto_sync_enabled": config.AI_AUTO_SYNC_ENABLED,
        "sync_interval_s": config.AI_CAMERA_SYNC_INTERVAL_S,
        "sync": camera_sync_manager.get_status(),
    }


@app.patch("/api/live/config", tags=["Live Stream AI Engine"], summary="Cập nhật nóng cấu hình động AI Live Engine")
async def patch_live_runtime_config(
    req: LiveRuntimeConfigPatchRequest,
    _auth: AuthContext = Depends(require_admin_or_permission(PERM_SYSTEM_CONFIG_MANAGE)),
):
    if req.max_sessions is not None:
        live_engine_manager.set_max_sessions(req.max_sessions)
    sync_status = None
    if req.trigger_sync:
        sync_status = await asyncio.to_thread(camera_sync_manager.sync_once)
    return {
        "status": "updated",
        "max_sessions": live_engine_manager.get_max_sessions(),
        "sync": sync_status,
    }


@app.post("/api/live/sync", tags=["Live Stream AI Engine"], summary="Kích hoạt đồng bộ ngay toàn bộ camera ONLINE 24/7")
async def trigger_live_sync(
    auth: AuthContext = Depends(require_admin_or_permission(PERM_PPE_CAMERA_MANAGE)),
):
    status = await asyncio.to_thread(camera_sync_manager.sync_once)
    sessions = live_engine_manager.list_sessions()
    if not (auth.is_internal or auth.is_admin or auth.site_ids is None):
        filtered = []
        for item in sessions:
            sess = live_engine_manager.get_session(item.get("camera_id", ""))
            if sess is not None and can_access_live_session(sess, auth):
                filtered.append(item)
        sessions = filtered
    return {
        "status": "synced",
        **status,
        "sessions": sessions,
    }


def _needs_internal_camera_lookup(source: Optional[str]) -> bool:
    if not source:
        return True
    s = source.strip()
    return "***" in s or "%2A%2A%2A" in s.upper()


@app.post("/api/live/start", tags=["Live Stream AI Engine"], summary="Khởi chạy phiên AI trực tiếp (hỗ trợ tra cứu M2M cho camera đã đăng ký)")
async def start_production_live(
    req: ProductionLiveStartRequest,
    auth: AuthContext = Depends(require_any_permission(PERM_PPE_CAMERA_MANAGE)),
):
    resolved_site_id: Optional[str] = None
    resolved_cam_uuid: Optional[str] = None
    resolved_cam_code: str = req.camera_id
    resolved_roi = req.roi_polygons
    source_val: object = None
    managed_flag = False
    dispatch_flag = False

    if _needs_internal_camera_lookup(req.source):
        cam = await asyncio.to_thread(
            lookup_internal_camera_sync,
            req.camera_uuid,
            req.camera_id,
        )
        if not cam or not cam.get("streamUrl"):
            raise HTTPException(
                404,
                f"Không thể tra cứu URL luồng gốc cho camera {req.camera_id} từ Camera Service",
            )
        resolved_site_id = str(cam.get("siteId") or "") or None
        if not auth.can_access_site(resolved_site_id):
            raise HTTPException(
                403,
                "Bạn không có quyền khởi chạy camera thuộc công trường này",
            )
        source_val = cam["streamUrl"]
        resolved_cam_code = str(cam.get("code") or req.camera_id)
        resolved_cam_uuid = str(cam.get("id") or req.camera_uuid or "") or None
        if resolved_roi is None:
            resolved_roi = _extract_roi_polygons_from_config(cam.get("config"))
        managed_flag = bool(resolved_site_id)
        dispatch_flag = LIVE_DISPATCH_ENABLED
    else:
        existing_sess = live_engine_manager.get_session(req.camera_id)
        if existing_sess is not None and not can_access_live_session(existing_sess, auth):
            raise HTTPException(
                403,
                "Bạn không có quyền ghi đè phiên camera thuộc công trường khác",
            )
        source_val = resolve_live_source(
            req.source or "",
            upload_dir=UPLOAD_DIR,
            allowed_rtsp_hosts=LIVE_ALLOWED_RTSP_HOSTS,
            allow_http_streams=config.LIVE_ALLOW_HTTP_STREAMS,
        )
        if auth.site_ids:
            resolved_site_id = next(iter(sorted(auth.site_ids)))

    model_path = resolve_model_path(req.model, MODELS_DIR)

    try:
        session = live_engine_manager.start_session(
            camera_id=resolved_cam_code,
            camera_uuid=resolved_cam_uuid,
            source=source_val,
            site_id=resolved_site_id,
            model_path=model_path,
            target_fps=req.target_fps,
            ai_fps=req.ai_fps,
            roi_polygons=resolved_roi,
            confirmation_duration_s=req.confirmation_duration_s,
            confirmation_duration_vest_s=req.confirmation_duration_vest_s,
            enable_no_helmet=req.enable_no_helmet,
            enable_no_vest=req.enable_no_vest,
            enable_restricted_area=req.enable_restricted_area,
            enable_scene_cut=req.enable_scene_cut,
            dispatch_enabled=dispatch_flag,
            auto_reconnect=req.auto_reconnect,
            loop_on_eof=req.loop_on_eof,
            managed=managed_flag,
        )
        session.owner_user_id = auth.user_id
        session.owner_site_id = resolved_site_id
    except LiveSessionLimitError as e:
        raise HTTPException(429, str(e))
    except ValueError as e:
        raise HTTPException(400, str(e))

    return {
        "status": "started",
        "camera_id": session.camera_id,
        "camera_uuid": session.camera_uuid,
        "site_id": session.site_id,
        "target_fps": session.target_fps,
        "ai_fps": session.ai_fps,
        "dispatch_enabled": session.dispatch_enabled,
        "max_sessions": live_engine_manager.get_max_sessions(),
    }


app.include_router(
    create_common_live_router(
        models_dir=MODELS_DIR,
        upload_dir=UPLOAD_DIR,
        enforce_auth=True,
    )
)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8000")))
