"""
Bộ đồng bộ Camera ONLINE 24/7 & Cấu hình động cho AI Live Engine.

Tự động:
1. Đọc cấu hình động `ai.live_max_sessions` từ Identity Service SystemConfig.
2. Truy vấn danh sách camera ONLINE (kèm RTSP URL gốc & ROI polygon) từ Camera Service qua M2M internal token.
3. Khởi động / cập nhật / dừng các phiên LiveStreamSession tương ứng 24/7.
4. Flush SQLite offline buffer khi Incident Service sẵn sàng.
"""

from __future__ import annotations

from datetime import datetime, timezone
import logging
import os
from pathlib import Path
import threading
import time
from typing import Any, Dict, List, Optional

from app import config
from app.constants import EVENT_AI_MODEL_DEPLOYED
from app.clients.backend_client import (
    check_internal_camera_exists_sync,
    fetch_dynamic_live_max_sessions_sync,
    fetch_effective_rules_sync,
    fetch_internal_active_model_version_sync,
    fetch_internal_active_streams_sync,
    push_ai_performance_telemetry_sync,
)
from app.events import flush_offline_buffer
from app.events.offline_buffer import pending_count
from app.live_engine import (
    LiveEngineManager,
    LiveSessionLimitError,
    UUID_RE,
    _normalize_model_path,
    live_engine_manager,
    preload_detector,
)
from app.video.video_reader import mask_source

logger = logging.getLogger("live_sync")


def _extract_roi_polygons_from_config(cfg: Optional[dict]) -> Optional[List[Any]]:
    if not isinstance(cfg, dict):
        return None
    raw_roi = cfg.get("roiPolygon")
    if not raw_roi or not isinstance(raw_roi, list):
        return None
    if len(raw_roi) >= 3 and isinstance(raw_roi[0], dict) and "x" in raw_roi[0]:
        return [raw_roi]
    return raw_roi


def _resolve_safe_local_model_path(
    raw_model_path: Optional[str],
    models_dir: Path,
) -> Optional[str]:
    """Xác thực an toàn đường dẫn weights cục bộ trong models_dir (chặn traversal/URL bên ngoài)."""
    if not raw_model_path or not isinstance(raw_model_path, str):
        return None
    normalized = raw_model_path.strip().replace("\\", "/")
    if not normalized or ".." in normalized or "://" in normalized:
        return None
    if normalized.startswith("models/"):
        normalized = normalized[len("models/"):]
    if not normalized or normalized.startswith("/"):
        return None
    try:
        base_dir = models_dir.resolve()
        candidate = (base_dir / normalized).resolve()
    except OSError:
        return None
    if (
        not candidate.is_relative_to(base_dir)
        or candidate.suffix.lower() not in (".pt", ".onnx")
        or not candidate.is_file()
    ):
        return None
    return str(candidate)


class CameraSyncManager:
    """Quản lý vòng đời tự động bật/tắt/đồng bộ các camera công trường ONLINE 24/7 và đẩy telemetry AI."""

    def __init__(
        self,
        manager: LiveEngineManager = live_engine_manager,
        dispatch_enabled: bool = config.LIVE_DISPATCH_DEFAULT,
        sync_interval_s: float = config.AI_CAMERA_SYNC_INTERVAL_S,
        telemetry_enabled: bool = config.AI_TELEMETRY_ENABLED,
        telemetry_push_interval_s: float = config.AI_TELEMETRY_PUSH_INTERVAL_S,
        models_dir: Optional[Path] = None,
        enable_reporting_sync: Optional[bool] = None,
    ):
        self.manager = manager
        self.dispatch_enabled = bool(dispatch_enabled)
        self.sync_interval_s = max(5.0, float(sync_interval_s))
        self.telemetry_enabled = bool(telemetry_enabled)
        self.telemetry_push_interval_s = max(5.0, float(telemetry_push_interval_s))
        self.models_dir = (
            models_dir
            if models_dir is not None
            else (Path(__file__).resolve().parent.parent / "models")
        )
        self.enable_reporting_sync = (
            bool(enable_reporting_sync)
            if enable_reporting_sync is not None
            else (not bool(os.getenv("PYTEST_CURRENT_TEST")))
        )
        self._sync_lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

        self.last_sync_at: Optional[str] = None
        self.last_sync_ok: Optional[bool] = None
        self.last_sync_error: Optional[str] = None
        self.last_synced_count: int = 0
        self.skipped_over_limit: List[str] = []
        self.dynamic_max_sessions: Optional[int] = None
        self._known_db_camera_codes: set[str] = set()

        self.active_model_version_id: Optional[str] = None
        self.active_model_version: Optional[str] = None
        self.active_model_path: Optional[str] = None
        self.last_model_sync_at: Optional[str] = None
        self.last_telemetry_push_at: Optional[str] = None
        self.last_telemetry_pushed_count: int = 0
        self.last_telemetry_failed_count: int = 0
        self._last_telemetry_monotonic: float = 0.0

    def get_status(self) -> Dict[str, Any]:
        return {
            "last_sync_at": self.last_sync_at,
            "last_sync_ok": self.last_sync_ok,
            "last_sync_error": self.last_sync_error,
            "last_synced_count": self.last_synced_count,
            "skipped_over_limit": list(self.skipped_over_limit),
            "dynamic_max_sessions": self.dynamic_max_sessions,
            "max_sessions": self.manager.get_max_sessions(),
            "dispatch_enabled": self.dispatch_enabled,
            "active_model_version_id": self.active_model_version_id,
            "active_model_version": self.active_model_version,
            "active_model_path": self.active_model_path,
            "last_model_sync_at": self.last_model_sync_at,
            "telemetry_enabled": self.telemetry_enabled,
            "telemetry_push_interval_s": self.telemetry_push_interval_s,
            "last_telemetry_push_at": self.last_telemetry_push_at,
            "last_telemetry_pushed_count": self.last_telemetry_pushed_count,
            "last_telemetry_failed_count": self.last_telemetry_failed_count,
        }

    def apply_deployed_model_sync(
        self,
        *,
        version_id: Optional[str] = None,
        version: Optional[str] = None,
        model_path: Optional[str] = None,
        models_dir: Optional[Path] = None,
        trigger: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Áp dụng thông tin phiên bản mô hình vừa deploy và hot-reload các phiên managed nếu weights thay đổi."""
        now_iso = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        target_models_dir = models_dir if models_dir is not None else self.models_dir

        if version_id and UUID_RE.match(str(version_id)):
            self.active_model_version_id = str(version_id)
        if isinstance(version, str) and version.strip():
            self.active_model_version = version.strip()
        self.last_model_sync_at = now_iso

        resolved_local_path = _resolve_safe_local_model_path(model_path, target_models_dir)
        reloaded_sessions = 0

        if resolved_local_path is not None:
            self.active_model_path = resolved_local_path
            if not os.getenv("PYTEST_CURRENT_TEST"):
                threading.Thread(
                    target=preload_detector,
                    args=(resolved_local_path,),
                    daemon=True,
                    name="model-deploy-prewarm",
                ).start()

            for sess in self.manager.iter_sessions():
                if self.active_model_version_id:
                    sess.model_version_id = self.active_model_version_id
                if (
                    sess.managed
                    and sess.status in ("initializing", "running", "reconnecting")
                    and _normalize_model_path(sess.model_path) != _normalize_model_path(resolved_local_path)
                ):
                    try:
                        self.manager.start_session(
                            camera_id=sess.camera_id,
                            camera_uuid=sess.camera_uuid,
                            source=sess.source,
                            site_id=sess.site_id,
                            model_path=resolved_local_path,
                            model_version_id=self.active_model_version_id,
                            target_fps=sess.target_fps,
                            ai_fps=sess.ai_fps,
                            roi_polygons=sess.roi_polygons,
                            confirmation_duration_s=sess.confirmation_duration_s,
                            confirmation_duration_vest_s=sess.confirmation_duration_vest_s,
                            cooldown_s=sess.cooldown_s,
                            confidence_threshold=sess.confidence_threshold,
                            enable_no_helmet=sess.enable_no_helmet,
                            enable_no_vest=sess.enable_no_vest,
                            enable_restricted_area=sess.enable_restricted_area,
                            enable_scene_cut=sess.enable_scene_cut,
                            dispatch_enabled=sess.dispatch_enabled,
                            auto_reconnect=sess.auto_reconnect,
                            loop_on_eof=sess.loop_on_eof,
                            managed=sess.managed,
                        )
                        reloaded_sessions += 1
                    except Exception as reload_err:
                        logger.warning(
                            "[%s] Failed hot-reloading deployed model '%s': %s",
                            sess.camera_id,
                            resolved_local_path,
                            reload_err,
                        )

            logger.info(
                "Applied deployed AI model sync (%s v%s, path=%s, reloaded=%d)",
                self.active_model_version_id or "unknown",
                self.active_model_version or "unknown",
                Path(resolved_local_path).name,
                reloaded_sessions,
            )
            return {
                "applied": True,
                "trigger": trigger or EVENT_AI_MODEL_DEPLOYED,
                "version_id": self.active_model_version_id,
                "version": self.active_model_version,
                "model_path": resolved_local_path,
                "reloaded_sessions": reloaded_sessions,
                "reason": "model_reloaded",
            }

        for sess in self.manager.iter_sessions():
            if self.active_model_version_id:
                sess.model_version_id = self.active_model_version_id

        return {
            "applied": bool(self.active_model_version_id),
            "trigger": trigger or EVENT_AI_MODEL_DEPLOYED,
            "version_id": self.active_model_version_id,
            "version": self.active_model_version,
            "model_path": self.active_model_path,
            "reloaded_sessions": 0,
            "reason": "version_tracked_local_weights_unavailable",
        }

    def sync_active_model_once(self) -> Optional[Dict[str, Any]]:
        """Truy vấn active model version từ Reporting Service và đồng bộ nội bộ."""
        active = fetch_internal_active_model_version_sync()
        if not active:
            return None
        return self.apply_deployed_model_sync(
            version_id=active.get("id"),
            version=active.get("version"),
            model_path=active.get("modelPath"),
            trigger="reporting_poll",
        )

    def flush_telemetry_once(self, force: bool = False) -> Dict[str, Any]:
        """Thu thập telemetry thực tế theo từng camera và đẩy về Reporting Service định kỳ."""
        if not self.telemetry_enabled:
            return {"skipped": True, "reason": "telemetry_disabled", "pushed": 0, "failed": 0}

        now_mono = time.monotonic()
        if (
            not force
            and self._last_telemetry_monotonic > 0
            and (now_mono - self._last_telemetry_monotonic) < self.telemetry_push_interval_s
        ):
            return {"skipped": True, "reason": "interval_not_elapsed", "pushed": 0, "failed": 0}

        pushed = 0
        failed = 0
        for sess in self.manager.iter_sessions():
            drained = sess.drain_telemetry_batch(
                default_model_version_id=self.active_model_version_id,
            )
            if not drained:
                continue
            payload, rollback = drained
            if push_ai_performance_telemetry_sync(payload):
                pushed += 1
            else:
                sess.restore_telemetry_batch(rollback)
                failed += 1

        self._last_telemetry_monotonic = time.monotonic()
        self.last_telemetry_push_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        self.last_telemetry_pushed_count = pushed
        self.last_telemetry_failed_count = failed
        return {"skipped": False, "pushed": pushed, "failed": failed}

    def sync_once(self) -> Dict[str, Any]:
        """Thực hiện 1 lượt đồng bộ cấu hình động + danh sách camera ONLINE 24/7."""
        with self._sync_lock:
            now_iso = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

            # 1. Cập nhật giới hạn phiên động từ Identity SystemConfig nếu khả dụng
            dyn_limit = fetch_dynamic_live_max_sessions_sync()
            if dyn_limit is not None:
                self.dynamic_max_sessions = dyn_limit
                if self.manager.get_max_sessions() != dyn_limit:
                    self.manager.set_max_sessions(dyn_limit)
                    logger.info(
                        "Dynamic ai.live_max_sessions updated -> %d",
                        dyn_limit,
                    )

            # 1.5. Lấy thông tin model đang DEPLOYED từ Reporting Service nếu chưa đồng bộ
            if self.enable_reporting_sync and self.active_model_version_id is None:
                self.sync_active_model_once()

            # 2. Nếu có sự kiện tồn đọng ở offline SQLite buffer, thử flush lại
            try:
                if pending_count() > 0:
                    flushed = flush_offline_buffer(sync=True)
                    if flushed > 0:
                        logger.info("Flushed %d buffered violation events to Backend", flushed)
            except Exception as flush_err:
                logger.debug("Offline buffer flush skipped: %s", flush_err)

            # 3. Lấy danh sách camera ONLINE từ Camera Service (giữ nguyên URL gốc qua M2M)
            active_cams = fetch_internal_active_streams_sync(status="ONLINE")
            if active_cams is None:
                self.last_sync_at = now_iso
                self.last_sync_ok = False
                self.last_sync_error = "Camera Service chưa sẵn sàng hoặc từ chối xác thực nội bộ"
                return self.get_status()

            desired_map: Dict[str, dict] = {}
            for cam in active_cams:
                if not isinstance(cam, dict):
                    continue
                code = str(cam.get("code") or cam.get("id") or "").strip()
                stream_url = str(cam.get("streamUrl") or "").strip()
                if not code or not stream_url:
                    continue
                desired_map[code] = cam
            self._known_db_camera_codes.update(desired_map.keys())

            # 4. Dừng phiên được quản lý (managed=True) khi không còn ONLINE
            #    và dừng phiên mồ côi (managed=False) nếu camera DB đã bị xóa
            current_sessions = self.manager.list_sessions()
            for sess in current_sessions:
                cid = sess["camera_id"]
                if sess.get("managed") and cid not in desired_map:
                    logger.info("[%s] Stopping managed session (camera no longer active ONLINE)", cid)
                    self.manager.stop_session(cid)
                    continue

                if (
                    not sess.get("managed")
                    and sess.get("status") not in ("stopped", "error")
                    and cid not in desired_map
                ):
                    sess_obj = self.manager.get_session(cid)
                    linked_uuid = (
                        sess.get("camera_uuid")
                        or (getattr(sess_obj, "_sync_camera_uuid", None) if sess_obj else None)
                        or (cid if UUID_RE.match(cid) else None)
                    )
                    linked_code = (
                        None
                        if linked_uuid
                        else (cid if cid in self._known_db_camera_codes else None)
                    )
                    if linked_uuid or linked_code:
                        exists = check_internal_camera_exists_sync(
                            camera_id=linked_uuid,
                            code=linked_code,
                        )
                        if exists is False:
                            logger.info(
                                "[%s] Stopping orphaned unmanaged session (camera deleted from DB)",
                                cid,
                            )
                            self.manager.stop_session(cid)
                            self._known_db_camera_codes.discard(cid)

            # 4b. Nếu max_sessions động vừa bị giảm thấp hơn số phiên đang chạy, cắt bớt phiên managed vượt ngưỡng
            after_cleanup = [
                s for s in self.manager.list_sessions() if s.get("status") not in ("stopped", "error")
            ]
            current_limit = self.manager.get_max_sessions()
            if len(after_cleanup) > current_limit:
                unmanaged_active = sum(1 for s in after_cleanup if not s.get("managed"))
                allowed_managed = max(0, current_limit - unmanaged_active)
                active_managed = [s for s in after_cleanup if s.get("managed")]
                if len(active_managed) > allowed_managed:
                    desired_order = list(desired_map.keys())
                    rank_map = {code: idx for idx, code in enumerate(desired_order)}
                    # Giữ ưu tiên theo thứ tự desired_map, dừng các phiên ở cuối
                    active_managed.sort(key=lambda s: rank_map.get(s["camera_id"], 10**6), reverse=True)
                    excess_count = len(active_managed) - allowed_managed
                    for excess_sess in active_managed[:excess_count]:
                        ecid = excess_sess["camera_id"]
                        logger.info(
                            "[%s] Stopping managed session due to reduced max_sessions=%d",
                            ecid,
                            current_limit,
                        )
                        self.manager.stop_session(ecid)

            # 5. Bật hoặc cập nhật cấu hình các camera ONLINE
            started_or_kept = 0
            skipped: List[str] = []
            for code, cam in desired_map.items():
                cfg = cam.get("config") or {}
                if cfg.get("enabled") is False or cfg.get("processingEnabled") is False:
                    existing = self.manager.get_session(code)
                    if existing and existing.managed:
                        self.manager.stop_session(code)
                    continue

                capture_fps = float(cfg.get("captureFps") or 10.0)
                ai_fps = max(1.0, min(25.0, capture_fps))
                target_fps = max(25.0, ai_fps)
                roi_polygons = _extract_roi_polygons_from_config(cfg)

                existing_session = self.manager.get_session(code)
                rules = fetch_effective_rules_sync(
                    camera_id=cam.get("id"),
                    site_id=cam.get("siteId"),
                )
                if rules is not None:
                    eff_no_helmet = bool(rules.enable_no_helmet)
                    eff_no_vest = bool(rules.enable_no_vest)
                    eff_restricted = bool(rules.enable_restricted_area)
                    eff_confidence = float(rules.confidence)
                    eff_confirm_helmet = float(rules.helmet_confirmation_s)
                    eff_confirm_vest = float(rules.vest_confirmation_s)
                    eff_cooldown = float(rules.cooldown_s)
                elif existing_session is not None:
                    # Site Service tạm gián đoạn -> giữ nguyên cấu hình đang chạy, tuyệt đối không reset về mặc định
                    eff_no_helmet = existing_session.enable_no_helmet
                    eff_no_vest = existing_session.enable_no_vest
                    eff_restricted = existing_session.enable_restricted_area
                    eff_confidence = existing_session.confidence_threshold
                    eff_confirm_helmet = existing_session.confirmation_duration_s
                    eff_confirm_vest = existing_session.confirmation_duration_vest_s
                    eff_cooldown = existing_session.cooldown_s
                else:
                    eff_no_helmet = True
                    eff_no_vest = True
                    eff_restricted = True
                    eff_confidence = config.CONFIDENCE_THRESHOLD
                    eff_confirm_helmet = float(config.CONFIRMATION_DURATION_HELMET_S)
                    eff_confirm_vest = float(config.CONFIRMATION_DURATION_VEST_S)
                    eff_cooldown = float(config.COOLDOWN_DURATION_S)

                try:
                    self.manager.start_session(
                        camera_id=code,
                        camera_uuid=cam.get("id"),
                        source=cam["streamUrl"],
                        site_id=cam.get("siteId"),
                        model_path=self.active_model_path,
                        model_version_id=self.active_model_version_id,
                        target_fps=target_fps,
                        ai_fps=ai_fps,
                        roi_polygons=roi_polygons,
                        confirmation_duration_s=eff_confirm_helmet,
                        confirmation_duration_vest_s=eff_confirm_vest,
                        cooldown_s=eff_cooldown,
                        confidence_threshold=eff_confidence,
                        enable_no_helmet=eff_no_helmet,
                        enable_no_vest=eff_no_vest,
                        enable_restricted_area=eff_restricted,
                        enable_scene_cut=True,
                        dispatch_enabled=self.dispatch_enabled,
                        auto_reconnect=True,
                        loop_on_eof=True,
                        managed=True,
                    )
                    started_or_kept += 1
                except LiveSessionLimitError:
                    skipped.append(code)
                    logger.warning(
                        "[%s] Skipped camera sync (%s): reached max concurrent sessions (%d)",
                        code,
                        mask_source(cam.get("streamUrl")),
                        self.manager.get_max_sessions(),
                    )
                except Exception as exc:
                    logger.error("[%s] Failed starting synced camera: %s", code, exc)

            self.last_sync_at = now_iso
            self.last_sync_ok = True
            self.last_sync_error = None
            self.last_synced_count = started_or_kept
            self.skipped_over_limit = skipped
            return self.get_status()

    def _run_worker(self) -> None:
        logger.info(
            "24/7 Camera auto-sync started (interval=%.1fs, dispatch=%s)",
            self.sync_interval_s,
            self.dispatch_enabled,
        )
        while not self._stop_event.is_set():
            try:
                self.sync_once()
            except Exception as exc:
                logger.error("Unexpected error in CameraSyncManager loop: %s", exc)
            if self.enable_reporting_sync and self.telemetry_enabled:
                try:
                    self.flush_telemetry_once(force=False)
                except Exception as telem_exc:
                    logger.debug("Telemetry flush loop error: %s", telem_exc)
            if self._stop_event.wait(self.sync_interval_s):
                break
        logger.info("24/7 Camera auto-sync worker stopped")

    def start_background(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run_worker,
            name="camera-auto-sync",
            daemon=True,
        )
        self._thread.start()

    def stop_background(self) -> None:
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=3.0)
        self._thread = None


camera_sync_manager = CameraSyncManager()
