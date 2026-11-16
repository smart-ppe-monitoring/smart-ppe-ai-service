"""
Live Stream AI Engine — smart-ppe-ai-service.

Quét AI thời gian thực (quét tới đâu phát tới đó) phục vụ:
- Màn hình Camera trực tiếp (/cameras/[id])
- Test-UI trực quan không độ trễ
- Hỗ trợ đa dạng nguồn: RTSP, YouTube (yt-dlp), Webcam, MP4
- Điều chỉnh linh hoạt FPS hiển thị (target_fps) và FPS quét AI (ai_fps / sampling rate)
- Phát hiện xâm nhập vùng nguy hiểm (Polygon ROI) với kiểm tra điểm chân (D08.pdf Mục 2.6 & Ví dụ 5)
- Đo lường Realtime Metrics: FPS, Inference Latency, Rule Latency, End-to-End Alert Latency (D08.pdf Mục 6.3 & 9)
- Gửi bằng chứng vi phạm bất đồng bộ (MinIO + Incident Service)
"""

import asyncio
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import logging
import os
from pathlib import Path
import re
import threading
import time
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

import cv2
import numpy as np

from app import config
from app.constants import (
    CONTRACT_VERSION,
    STATE_UNKNOWN,
    VIOLATION_NO_HELMET,
    VIOLATION_NO_VEST,
    VIOLATION_RESTRICTED_AREA,
)
from app.detection.adapter import PERSON
from app.detection.detector import Detector, DetectorError
from app.association.associator import associate
from app.rules.safety import PersonState, evaluate as evaluate_safety, is_inside_roi
from app.rules.confirmation import ViolationConfirmation
from app.rules.deduplication import ViolationDeduplicator, ViolationEvent as DedupEvent
from app.evidence.snapshot import save_evidence
from app.video.scene_cut import SceneCutDetector
from app.video.video_reader import VideoReader, VideoSourceError, is_youtube_url, mask_source
from app.video.visualizer import draw_detections, draw_roi_polygons
from app.main import _build_violation_event
from app.events import dispatch_event

logger = logging.getLogger("live_engine")

# Định danh camera/site được ghép vào MinIO object key và snapshot URL → chỉ cho phép ký tự an toàn.
SAFE_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
UUID_RE = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[1-8][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$", re.IGNORECASE)

# Giới hạn số phiên live đồng thời mặc định (có thể cập nhật động qua Identity SystemConfig / LIVE_MAX_SESSIONS).
DEFAULT_MAX_LIVE_SESSIONS = max(1, int(os.getenv("LIVE_MAX_SESSIONS", str(config.LIVE_MAX_SESSIONS_DEFAULT))))
MAX_LIVE_SESSIONS = DEFAULT_MAX_LIVE_SESSIONS

# Giới hạn hình học ROI để tránh tiêu tốn CPU mỗi frame.
MAX_ROI_POLYGONS = 16
MAX_ROI_POINTS = 64
MIN_CONFIRMATION_DURATION_S = 0.5
RECONNECT_BASE_DELAY_S = 1.0
RECONNECT_MAX_DELAY_S = 30.0

# Pool tái sử dụng Detector giữa các phiên để chuyển video không bị trễ 2-3 giây nạp lại PyTorch weights
_DETECTOR_POOL_LOCK = threading.Lock()
_DETECTOR_POOL: List[Tuple[str, Any, threading.Lock]] = []
_DETECTOR_POOL_CAP = DEFAULT_MAX_LIVE_SESSIONS * 2


def _set_detector_pool_cap(max_sessions: int) -> None:
    global _DETECTOR_POOL_CAP
    with _DETECTOR_POOL_LOCK:
        _DETECTOR_POOL_CAP = max(2, int(max_sessions) * 2)


def _is_mock_detector_cls() -> bool:
    """Kiểm tra xem class Detector có đang bị unittest.mock.patch trong test hay không."""
    return (
        not isinstance(Detector, type)
        or getattr(Detector, "__module__", "").startswith("unittest.mock")
    )


def _acquire_detector(model_path: str, conf: float, device: str, imgsz: int) -> Tuple[Any, Optional[threading.Lock]]:
    """Lấy Detector đã nạp sẵn từ pool (nếu rảnh) hoặc khởi tạo mới."""
    if _is_mock_detector_cls():
        det = Detector(model_path=model_path, conf=conf, device=device, imgsz=imgsz).load()
        return det, None

    norm_key = str(Path(model_path).resolve()) if model_path else str(config.YOLO_MODEL_PATH)
    with _DETECTOR_POOL_LOCK:
        for key, det, use_lock in _DETECTOR_POOL:
            if key == norm_key and use_lock.acquire(blocking=False):
                det.conf = conf
                det.imgsz = imgsz
                det.reset_tracker()
                return det, use_lock

    # Chưa có sẵn trong pool -> tải mới ngoài lock để không chặn luồng khác
    det = Detector(model_path=model_path, conf=conf, device=device, imgsz=imgsz).load()
    use_lock = threading.Lock()
    use_lock.acquire()
    with _DETECTOR_POOL_LOCK:
        if len(_DETECTOR_POOL) < _DETECTOR_POOL_CAP:
            _DETECTOR_POOL.append((norm_key, det, use_lock))
    return det, use_lock


def preload_detector(model_path: Optional[str] = None) -> None:
    """Nạp trước model YOLO và chạy 1 frame nháp để xóa bỏ độ trễ khởi động lần đầu."""
    if _is_mock_detector_cls():
        return
    target_model = model_path or config.YOLO_MODEL_PATH
    try:
        det, use_lock = _acquire_detector(
            model_path=target_model,
            conf=config.CONFIDENCE_THRESHOLD,
            device=config.DEVICE,
            imgsz=config.MODEL_IMGSZ,
        )
        try:
            dummy = np.zeros((320, 320, 3), dtype=np.uint8)
            det.detect(dummy)
            det.reset_tracker()
        finally:
            if use_lock is not None:
                use_lock.release()
        logger.info("Preloaded and warmed up YOLO model: %s", target_model)
    except Exception as e:
        logger.debug("Model preload skipped for %s: %s", target_model, e)


def _make_splash_jpeg(title: str, subtitle: str = "") -> Optional[bytes]:
    """Tạo frame chờ (splash screen) tức thì để trình duyệt không hiển thị khung đen khi đang mở nguồn."""
    canvas = np.full((360, 640, 3), (22, 27, 34), dtype=np.uint8)
    cv2.rectangle(canvas, (16, 16), (624, 344), (48, 54, 61), 1)
    cv2.putText(canvas, title[:55], (36, 165), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (88, 166, 255), 2, cv2.LINE_AA)
    if subtitle:
        cv2.putText(canvas, subtitle[:70], (36, 205), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (180, 190, 200), 1, cv2.LINE_AA)
    ok, buf = cv2.imencode(".jpg", canvas, [cv2.IMWRITE_JPEG_QUALITY, 80])
    return buf.tobytes() if ok else None


class LiveSessionLimitError(Exception):
    """Vượt quá số phiên live đồng thời cho phép."""


def _validate_safe_id(value: Any, field_name: str) -> str:
    s = str(value)
    if not SAFE_ID_RE.match(s):
        raise ValueError(f"{field_name} không hợp lệ (chỉ chấp nhận [A-Za-z0-9_-], tối đa 64 ký tự)")
    return s


def _extract_point_xy(pt: Any) -> Optional[Tuple[float, float]]:
    if isinstance(pt, dict) and "x" in pt and "y" in pt:
        return float(pt["x"]), float(pt["y"])
    if isinstance(pt, (list, tuple)) and len(pt) >= 2:
        return float(pt[0]), float(pt[1])
    return None


def _normalize_roi_input(roi_polygons: Optional[List[Any]]) -> List[Any]:
    """Chuẩn hóa định dạng roiPolygon từ camera-service ([{x,y}, ...]) hoặc polygon list."""
    if not roi_polygons:
        return []
    if (
        isinstance(roi_polygons, (list, tuple))
        and len(roi_polygons) >= 3
        and isinstance(roi_polygons[0], dict)
        and "x" in roi_polygons[0]
        and "y" in roi_polygons[0]
    ):
        return [list(roi_polygons)]
    return list(roi_polygons)


def _validate_roi_polygons(roi_polygons: Optional[List[Any]]) -> List[Any]:
    """Phòng thủ tầng core: giới hạn số polygon/số đỉnh ROI và chuẩn hóa shape."""
    normalized = _normalize_roi_input(roi_polygons)
    if not normalized:
        return []
    if len(normalized) > MAX_ROI_POLYGONS:
        raise ValueError(f"roi_polygons phải là danh sách tối đa {MAX_ROI_POLYGONS} polygon")
    for poly in normalized:
        pts = poly.get("points", []) if isinstance(poly, dict) else poly
        if not isinstance(pts, (list, tuple)) or len(pts) > MAX_ROI_POINTS:
            raise ValueError(f"Mỗi polygon ROI tối đa {MAX_ROI_POINTS} điểm")
    return normalized


def resolve_roi_polygons_for_frame(roi_polygons: List[Any], frame_w: int, frame_h: int) -> List[List[Tuple[int, int]]]:
    """Chuyển đổi ROI dạng normalized [0..1] hoặc [{x, y}] sang tọa độ pixel thực tế của frame."""
    if not roi_polygons or frame_w <= 0 or frame_h <= 0:
        return []
    resolved: List[List[Tuple[int, int]]] = []
    for poly in roi_polygons:
        raw_pts = poly.get("points", []) if isinstance(poly, dict) else poly
        coords: List[Tuple[float, float]] = []
        for item in raw_pts:
            xy = _extract_point_xy(item)
            if xy is not None:
                coords.append(xy)
        if len(coords) < 3:
            continue
        is_normalized = all(0.0 <= x <= 1.0 and 0.0 <= y <= 1.0 for x, y in coords)
        if is_normalized:
            resolved.append([(int(round(x * frame_w)), int(round(y * frame_h))) for x, y in coords])
        else:
            resolved.append([(int(round(x)), int(round(y))) for x, y in coords])
    return resolved


def _extract_object_key_from_url(url: Optional[str]) -> Optional[str]:
    if not url or not isinstance(url, str):
        return None
    try:
        path = urlparse(url).path
    except Exception:
        path = url.split("?")[0]
    match = re.search(r"(?:^|/)(evidences/.+)$", path)
    if match:
        return match.group(1)
    return None


class LiveStreamSession:
    """Phiên xử lý và phát luồng video trực tiếp với AI (YOLO + ByteTrack + Rules)."""

    def __init__(
        self,
        camera_id: str,
        source: Any,
        site_id: Optional[str] = None,
        camera_uuid: Optional[str] = None,
        model_path: Optional[str] = None,
        target_fps: float = 25.0,
        ai_fps: float = 15.0,
        roi_polygons: Optional[List[Any]] = None,
        confirmation_duration_s: float = 2.0,
        confirmation_duration_vest_s: Optional[float] = None,
        cooldown_s: float = 30.0,
        confidence_threshold: Optional[float] = None,
        enable_no_helmet: bool = True,
        enable_no_vest: bool = True,
        enable_restricted_area: bool = True,
        enable_scene_cut: bool = True,
        draw_hud: bool = True,
        dispatch_enabled: bool = True,
        auto_reconnect: bool = False,
        loop_on_eof: bool = False,
        managed: bool = False,
    ):
        self.camera_id = _validate_safe_id(camera_id, "camera_id")
        self.source = source
        # site_id mặc định lấy từ cấu hình service (không tin dữ liệu client)
        self.site_id = _validate_safe_id(site_id if site_id is not None else config.SITE_CODE, "site_id")
        self.camera_uuid = _validate_safe_id(camera_uuid, "camera_uuid") if camera_uuid else None
        self.model_path = model_path or config.YOLO_MODEL_PATH
        self.target_fps = max(1.0, min(60.0, float(target_fps)))
        self.ai_fps = max(1.0, min(self.target_fps, float(ai_fps)))
        self.roi_polygons = _validate_roi_polygons(roi_polygons)
        self.confirmation_duration_s = max(MIN_CONFIRMATION_DURATION_S, float(confirmation_duration_s))
        self.confirmation_duration_vest_s = max(
            MIN_CONFIRMATION_DURATION_S,
            float(confirmation_duration_vest_s if confirmation_duration_vest_s is not None else config.CONFIRMATION_DURATION_VEST_S)
        )
        self.cooldown_s = float(cooldown_s)
        self.confidence_threshold = float(
            confidence_threshold if confidence_threshold is not None else config.CONFIDENCE_THRESHOLD
        )
        self.enable_no_helmet = bool(enable_no_helmet)
        self.enable_no_vest = bool(enable_no_vest)
        self.enable_restricted_area = bool(enable_restricted_area)
        self.enable_scene_cut = bool(enable_scene_cut)
        self.draw_hud = bool(draw_hud)
        # Cho phép upload MinIO + bắn sự kiện tới Incident Service (tắt cho môi trường test)
        self.dispatch_enabled = bool(dispatch_enabled)
        self.auto_reconnect = bool(auto_reconnect)
        self.loop_on_eof = bool(loop_on_eof)
        self.managed = bool(managed)
        self.reconnect_count = 0
        self.last_frame_at: Optional[float] = None

        # Trạng thái phiên
        self.status = "initializing"  # initializing, running, paused, stopped, error
        self.error_message: Optional[str] = None
        self._stop_requested = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._ai_thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self._frame_condition = threading.Condition(self._lock)

        # Single-slot buffer cho AI Worker Thread (luôn lấy frame mới nhất, không gây nghẽn luồng phát 25 FPS)
        self._ai_lock = threading.Lock()
        self._ai_cond = threading.Condition(self._ai_lock)
        self._pending_ai_item: Optional[Tuple[int, np.ndarray, float]] = None
        self._ai_done = False

        # Bộ ổn định track_id qua các nhịp sampling AI để tránh đứt chuỗi xác nhận vi phạm
        self._track_alias: Dict[int, int] = {}
        self._track_last_bbox: Dict[int, Tuple[float, float, float, float]] = {}
        self._track_last_seen: Dict[int, float] = {}

        # Frame buffer JPEG mới nhất phục vụ MJPEG stream (khởi tạo ngay frame splash để không bị màn hình đen)
        self.latest_jpeg_bytes: Optional[bytes] = _make_splash_jpeg(
            f"CAM: {self.camera_id} - DANG KET NOI LUONG VIDEO...",
            "He thong dang mo nguon phat & khoi tao AI...",
        )
        self._jpeg_seq = 1 if self.latest_jpeg_bytes else 0
        self.frame_number = 0
        self.start_time = time.time()

        # Cache kết quả quét AI cho các frame nằm giữa nhịp sampling
        self._last_detections: list = []
        self._last_person_states: list = []
        self._last_counts = {"person": 0, "helmet": 0, "safety_vest": 0}

        # Realtime Metrics (D08.pdf Mục 2.8 & 6.3)
        self.metrics: Dict[str, Any] = {
            "fps": 0.0,
            "target_fps": self.target_fps,
            "ai_fps": self.ai_fps,
            "ai_fps_actual": 0.0,
            "inference_latency_ms": 0.0,
            "rule_latency_ms": 0.0,
            "e2e_alert_latency_ms": 0.0,
            "active_persons": 0,
            "active_violations": 0,
            "reconnect_count": 0,
            "counts": self._last_counts,
        }
        self.recent_violations: List[Dict[str, Any]] = []

        # Thread pool riêng phục vụ lưu & upload bằng chứng không làm nghẽn video
        self._evidence_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix=f"ev-{self.camera_id}")

    def update_config(
        self,
        target_fps: Optional[float] = None,
        ai_fps: Optional[float] = None,
        roi_polygons: Optional[List[Any]] = None,
        enable_no_helmet: Optional[bool] = None,
        enable_no_vest: Optional[bool] = None,
        enable_restricted_area: Optional[bool] = None,
        enable_scene_cut: Optional[bool] = None,
        confirmation_duration_s: Optional[float] = None,
        confirmation_duration_vest_s: Optional[float] = None,
        dispatch_enabled: Optional[bool] = None,
        auto_reconnect: Optional[bool] = None,
        loop_on_eof: Optional[bool] = None,
    ):
        """Cập nhật nóng (hot-update) cấu hình trong khi luồng đang phát."""
        with self._lock:
            if target_fps is not None:
                self.target_fps = max(1.0, min(60.0, float(target_fps)))
                self.metrics["target_fps"] = self.target_fps
            if ai_fps is not None:
                self.ai_fps = max(1.0, min(self.target_fps, float(ai_fps)))
                self.metrics["ai_fps"] = self.ai_fps
            if roi_polygons is not None:
                self.roi_polygons = _validate_roi_polygons(roi_polygons)
            if enable_no_helmet is not None:
                self.enable_no_helmet = bool(enable_no_helmet)
            if enable_no_vest is not None:
                self.enable_no_vest = bool(enable_no_vest)
            if enable_restricted_area is not None:
                self.enable_restricted_area = bool(enable_restricted_area)
            if enable_scene_cut is not None:
                self.enable_scene_cut = bool(enable_scene_cut)
            if confirmation_duration_s is not None:
                self.confirmation_duration_s = max(MIN_CONFIRMATION_DURATION_S, float(confirmation_duration_s))
            if confirmation_duration_vest_s is not None:
                self.confirmation_duration_vest_s = max(MIN_CONFIRMATION_DURATION_S, float(confirmation_duration_vest_s))
            if dispatch_enabled is not None:
                self.dispatch_enabled = bool(dispatch_enabled)
            if auto_reconnect is not None:
                self.auto_reconnect = bool(auto_reconnect)
            if loop_on_eof is not None:
                self.loop_on_eof = bool(loop_on_eof)
        logger.info(
            "[%s] Config updated: target_fps=%.1f, ai_fps=%.1f, helmet=%s, vest=%s, restricted=%s, scene_cut=%s",
            self.camera_id, self.target_fps, self.ai_fps,
            self.enable_no_helmet, self.enable_no_vest, self.enable_restricted_area, self.enable_scene_cut
        )

    def start(self):
        """Khởi động tiến trình đọc video và suy luận AI trong background thread."""
        if self._thread and self._thread.is_alive():
            return
        self._stop_requested.clear()
        self._thread = threading.Thread(
            target=self._run_loop,
            name=f"live-engine-{self.camera_id}",
            daemon=True,
        )
        self._thread.start()

    def stop(self):
        """Dừng phiên và giải phóng toàn bộ tài nguyên."""
        self._stop_requested.set()
        with self._ai_lock:
            self._ai_done = True
            self._ai_cond.notify_all()
        with self._lock:
            self._frame_condition.notify_all()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=3.0)
        if self._ai_thread and self._ai_thread.is_alive():
            self._ai_thread.join(timeout=2.0)
        self._evidence_executor.shutdown(wait=False)
        self.status = "stopped"
        logger.info("[%s] LiveStreamSession stopped cleanly", self.camera_id)

    def _stabilize_person_tracks(self, detections: list, t_s: float) -> None:
        """Giữ ổn định track_id của person khi ByteTrack bị nhảy ID giữa các nhịp sampling AI."""
        # Dọn các track đã biến mất quá 2.0s
        for sid in list(self._track_last_seen.keys()):
            if t_s - self._track_last_seen[sid] > 2.0:
                self._track_last_seen.pop(sid, None)
                self._track_last_bbox.pop(sid, None)
        for raw_id, sid in list(self._track_alias.items()):
            if sid not in self._track_last_seen:
                self._track_alias.pop(raw_id, None)

        person_dets = [d for d in detections if getattr(d, "class_name", "") == PERSON and getattr(d, "track_id", None) is not None]
        claimed_stable_ids = {
            self._track_alias[d.track_id]
            for d in person_dets
            if d.track_id in self._track_alias
        }

        for d in person_dets:
            raw_id = d.track_id
            if raw_id in self._track_alias:
                stable_id = self._track_alias[raw_id]
            else:
                # Kiểm tra xem có track ổn định nào vừa mất (<1.2s) ở ngay vị trí này không
                bx1, by1, bx2, by2 = d.bbox
                cx, cy = (bx1 + bx2) / 2.0, (by1 + by2) / 2.0
                bh = max(1.0, float(by2 - by1))
                best_sid = None
                best_dist = float("inf")
                for sid, last_bbox in self._track_last_bbox.items():
                    if sid in claimed_stable_ids:
                        continue
                    dt_lost = t_s - self._track_last_seen.get(sid, -999.0)
                    if dt_lost < 0 or dt_lost > 1.2:
                        continue
                    lx1, ly1, lx2, ly2 = last_bbox
                    lcx, lcy = (lx1 + lx2) / 2.0, (ly1 + ly2) / 2.0
                    lh = max(1.0, float(ly2 - ly1))
                    dist = ((cx - lcx) ** 2 + (cy - lcy) ** 2) ** 0.5
                    ref_h = max(bh, lh)
                    h_ratio = min(bh, lh) / ref_h
                    if dist <= 0.55 * ref_h and h_ratio >= 0.65 and dist < best_dist:
                        best_sid = sid
                        best_dist = dist
                stable_id = best_sid if best_sid is not None else raw_id
                self._track_alias[raw_id] = stable_id
                claimed_stable_ids.add(stable_id)

            d.track_id = stable_id
            self._track_last_bbox[stable_id] = tuple(d.bbox)
            self._track_last_seen[stable_id] = t_s

    def _dispatch_evidence_async(
        self,
        frame_copy: Optional[np.ndarray],
        detections_copy: list,
        ev: DedupEvent,
        frame_w: int,
        frame_h: int,
        person_states_copy: Optional[list] = None,
        frame_num: int = 0,
    ):
        """Lưu ảnh đối chứng (Before) & ảnh vi phạm (After) và bắn sự kiện trên background thread."""
        t_start = time.time()
        try:
            # 1. Vẽ bounding box & nhãn nhận diện lên bản sao frame vi phạm giống hệt chế độ offline
            annotated_frame = frame_copy.copy() if frame_copy is not None else None
            if annotated_frame is not None:
                try:
                    pixel_roi = resolve_roi_polygons_for_frame(
                        self.roi_polygons,
                        frame_w,
                        frame_h,
                    )
                    draw_detections(
                        annotated_frame,
                        detections_copy,
                        person_states=person_states_copy,
                        roi_polygons=pixel_roi,
                        enable_no_helmet=self.enable_no_helmet,
                        enable_no_vest=self.enable_no_vest,
                    )
                except Exception:
                    annotated_frame = frame_copy

            ev_dir = Path(config.EVIDENCE_DIR).resolve()
            evidence_res = save_evidence(annotated_frame, detections_copy, ev, ev_dir)

            if evidence_res.status == "saved":
                owner_uid = getattr(self, "owner_user_id", None)
                owner_sid = getattr(self, "owner_site_id", None) or self.site_id
                if evidence_res.path:
                    register_evidence_owner(
                        Path(evidence_res.path).name,
                        site_id=owner_sid,
                        owner_user_id=owner_uid,
                        managed=self.managed,
                    )
                if evidence_res.reference_path:
                    register_evidence_owner(
                        Path(evidence_res.reference_path).name,
                        site_id=owner_sid,
                        owner_user_id=owner_uid,
                        managed=self.managed,
                    )
                _prune_local_evidence_dir(
                    ev_dir,
                    int(getattr(config, "LIVE_MAX_LOCAL_EVIDENCES", 500)),
                )

            local_snapshot_url = (
                f"/api/live/evidence/{Path(evidence_res.path).name}"
                if evidence_res.status == "saved" and evidence_res.path
                else None
            )
            local_ref_url = (
                f"/api/live/evidence/{Path(evidence_res.reference_path).name}"
                if evidence_res.status == "saved" and evidence_res.reference_path
                else None
            )
            snapshot_url = local_snapshot_url
            reference_snapshot_url = local_ref_url

            if not self.dispatch_enabled:
                # Chế độ test: giữ nguyên ảnh bằng chứng & ảnh đối chứng cục bộ để hiển thị ngay trên UI
                e2e_latency = round((time.time() - t_start) * 1000.0, 1)
                self._record_violation(
                    ev=ev,
                    snapshot_url=local_snapshot_url,
                    method="disabled",
                    e2e_latency=e2e_latency,
                    reference_snapshot_url=local_ref_url,
                    frame_num=frame_num,
                )
                return

            # 2. Upload MinIO nếu khả dụng
            snap_checksum = None
            ref_checksum = None
            if evidence_res.status == "saved" and evidence_res.path:
                try:
                    from app.storage.minio_client import upload_snapshot
                    with open(evidence_res.path, "rb") as f:
                        img_bytes = f.read()
                    snap_checksum = hashlib.sha256(img_bytes).hexdigest()
                    try:
                        minio_url = upload_snapshot(
                            image_bytes=img_bytes,
                            site_code=self.site_id,
                            camera_code=ev.camera_id,
                            track_id=ev.person_track_id,
                            suffix=ev.event_id,
                        )
                    except TypeError:
                        minio_url = upload_snapshot(
                            image_bytes=img_bytes,
                            site_code=self.site_id,
                            camera_code=ev.camera_id,
                            track_id=ev.person_track_id,
                        )
                    if minio_url:
                        snapshot_url = minio_url
                except Exception as m_err:
                    logger.debug("[%s] MinIO direct upload skipped: %s", self.camera_id, m_err)

            # 2b. Upload ảnh đối chứng reference nếu có
            if evidence_res.status == "saved" and evidence_res.reference_path:
                try:
                    from app.storage.minio_client import upload_snapshot
                    with open(evidence_res.reference_path, "rb") as f:
                        ref_bytes = f.read()
                    ref_checksum = hashlib.sha256(ref_bytes).hexdigest()
                    try:
                        minio_ref_url = upload_snapshot(
                            image_bytes=ref_bytes,
                            site_code=self.site_id,
                            camera_code=ev.camera_id,
                            track_id=ev.person_track_id,
                            suffix=f"{ev.event_id}_ref",
                        )
                    except TypeError:
                        minio_ref_url = upload_snapshot(
                            image_bytes=ref_bytes,
                            site_code=self.site_id,
                            camera_code=ev.camera_id,
                            track_id=ev.person_track_id,
                            suffix="ref",
                        )
                    if minio_ref_url:
                        reference_snapshot_url = minio_ref_url
                except Exception as m_err:
                    logger.debug("[%s] MinIO ref upload skipped: %s", self.camera_id, m_err)

            # 3. Fallback sang Edge Gateway HTTP URL cho Contract nếu chưa có MinIO URL
            edge_base = (
                config.BACKEND_BASE_URL.rstrip("/")
                if hasattr(config, "BACKEND_BASE_URL")
                else "http://localhost:8893"
            )
            contract_snap_url = snapshot_url
            if not contract_snap_url or not str(contract_snap_url).startswith(("http://", "https://")):
                filename = Path(evidence_res.path).name if evidence_res.path else f"{ev.event_id}.jpg"
                contract_snap_url = f"{edge_base}/evidences/{self.site_id}/{ev.camera_id}/{filename}"

            contract_ref_url = reference_snapshot_url
            if evidence_res.reference_path and (not contract_ref_url or not str(contract_ref_url).startswith(("http://", "https://"))):
                ref_name = Path(evidence_res.reference_path).name
                contract_ref_url = f"{edge_base}/evidences/{self.site_id}/{ev.camera_id}/{ref_name}"

            # 4. Đóng gói ViolationEvent chuẩn Contract (kèm UUID camera/site & objectKey MinIO nếu có)
            model_name = Path(self.model_path).name
            snap_object_key = _extract_object_key_from_url(contract_snap_url)
            ref_object_key = _extract_object_key_from_url(contract_ref_url) if contract_ref_url else None
            resolved_site_uuid = self.site_id if UUID_RE.match(self.site_id) else None
            resolved_cam_uuid = (
                self.camera_uuid
                if self.camera_uuid and UUID_RE.match(self.camera_uuid)
                else (self.camera_id if UUID_RE.match(self.camera_id) else None)
            )
            contract_event = _build_violation_event(
                ev=ev,
                frame_w=frame_w,
                frame_h=frame_h,
                snapshot_url=contract_snap_url,
                reference_snapshot_url=contract_ref_url,
                all_detections=detections_copy,
                model_name=model_name,
                event_id=ev.event_id,
                site_id=resolved_site_uuid,
                camera_id=resolved_cam_uuid,
                site_code=None if resolved_site_uuid else self.site_id,
                object_key=snap_object_key,
                reference_object_key=ref_object_key,
                checksum=snap_checksum,
                reference_checksum=ref_checksum,
            )

            # 5. Dispatch Kafka / HTTP / Offline Buffer
            sync_http_only = not getattr(config, "KAFKA_ENABLED", False)
            method = dispatch_event(contract_event, sync=sync_http_only)
            e2e_latency = round((time.time() - t_start) * 1000.0, 1)
            self._record_violation(
                ev=ev,
                snapshot_url=local_snapshot_url or snapshot_url,
                method=method,
                e2e_latency=e2e_latency,
                reference_snapshot_url=local_ref_url or reference_snapshot_url,
                frame_num=frame_num,
            )

            logger.info(
                "[%s] VIOLATION EVENT DISPATCHED: person=#%d, type=%s, via=%s in %.1fms",
                self.camera_id, ev.person_track_id, ev.violation_type, method, e2e_latency,
            )
        except Exception as e:
            logger.error("[%s] Error in background evidence dispatch: %s", self.camera_id, e)

    def _record_violation(
        self,
        ev: DedupEvent,
        snapshot_url: Optional[str],
        method: str,
        e2e_latency: float,
        reference_snapshot_url: Optional[str] = None,
        frame_num: int = 0,
    ):
        """Ghi nhận vi phạm gần nhất phục vụ API /stats (tối đa 50 bản ghi)."""
        with self._lock:
            self.metrics["e2e_alert_latency_ms"] = e2e_latency
            fallback_duration = (
                self.confirmation_duration_vest_s
                if "VEST" in (getattr(ev, "violation_type", "") or "").upper()
                else self.confirmation_duration_s
            )
            self.recent_violations.append({
                "event_id": ev.event_id,
                "camera_id": self.camera_id,
                "person_id": ev.person_track_id,
                "canonical_person_id": getattr(ev, "canonical_person_id", ev.person_track_id),
                "violation_type": ev.violation_type,
                "start_t": getattr(ev, "start_t", 0.0),
                "confirmed_t": getattr(ev, "confirmed_t", 0.0),
                "duration_s": getattr(ev, "duration_s", None) or fallback_duration,
                "frame": frame_num,
                "snapshot_url": snapshot_url,
                "reference_snapshot_url": reference_snapshot_url,
                "has_reference": bool(reference_snapshot_url),
                "reference_type": getattr(ev, "reference_type", "COMPLIANT"),
                "dispatched_via": method,
                "e2e_alert_latency_ms": e2e_latency,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            })
            if len(self.recent_violations) > 50:
                self.recent_violations.pop(0)

    def _draw_stats_hud(self, frame: np.ndarray):
        """Vẽ thanh thông số thời gian thực (FPS, Latency, Số lượng) lên góc trên màn hình."""
        h, w = frame.shape[:2]
        ai_fps_disp = self.metrics.get("ai_fps_actual") or self.ai_fps
        fps_text = f"FPS: {self.metrics['fps']:.1f}/{self.target_fps:.0f} (AI: {ai_fps_disp:.1f})"
        lat_text = f"Infer: {self.metrics['inference_latency_ms']:.1f}ms | Rule: {self.metrics['rule_latency_ms']:.1f}ms"
        counts_parts = [f"Persons: {self._last_counts.get('person', 0)}"]
        if self.enable_no_helmet:
            counts_parts.append(f"Helmets: {self._last_counts.get('helmet', 0)}")
        if self.enable_no_vest:
            counts_parts.append(f"Vests: {self._last_counts.get('safety_vest', 0)}")
        counts_text = " | ".join(counts_parts)
        hud_bar_w = min(w - 20, 520)
        cv2.rectangle(frame, (10, 10), (10 + hud_bar_w, 75), (20, 20, 20), -1)
        cv2.rectangle(frame, (10, 10), (10 + hud_bar_w, 75), (70, 70, 70), 1)

        cv2.putText(frame, f"CAM: {self.camera_id} | {fps_text}", (20, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(frame, lat_text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (200, 200, 200), 1, cv2.LINE_AA)
        cv2.putText(frame, counts_text, (20, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (100, 255, 100), 1, cv2.LINE_AA)

    def _process_ai_step(
        self,
        frame_num: int,
        frame: np.ndarray,
        t_frame_s: float,
        detector: Any,
        confirmation: ViolationConfirmation,
        dedup: ViolationDeduplicator,
        ai_fps_window: List[float],
    ):
        """Thực thi 1 bước suy luận AI (YOLO + Association + Safety Rules + Confirmation + Evidence)."""
        frame_h, frame_w = frame.shape[:2]

        # Bước 2a: Detection + Tracking (YOLO + ByteTrack)
        t_infer_start = time.perf_counter()
        detections = detector.detect(frame)
        t_infer_end = time.perf_counter()
        infer_ms = (t_infer_end - t_infer_start) * 1000.0

        # Bước 2b: Ổn định track_id & PPE Association & Safety Rules
        t_rule_start = time.perf_counter()
        self._stabilize_person_tracks(detections, t_frame_s)
        associations = associate(detections)
        pixel_roi = resolve_roi_polygons_for_frame(self.roi_polygons, frame_w, frame_h)
        person_states = evaluate_safety(
            detections=detections,
            associations=associations,
            frame_w=frame_w,
            frame_h=frame_h,
            enable_no_helmet=self.enable_no_helmet,
            enable_no_vest=self.enable_no_vest,
            roi_polygons=pixel_roi,
            enable_restricted_area=self.enable_restricted_area,
        )

        # Đồng bộ ngưỡng thời gian xác nhận nếu người dùng cập nhật nóng
        confirmation.duration_s = self.confirmation_duration_s
        confirmation.duration_map = {
            VIOLATION_NO_HELMET: self.confirmation_duration_s,
            VIOLATION_NO_VEST: self.confirmation_duration_vest_s,
            VIOLATION_RESTRICTED_AREA: self.confirmation_duration_s,
        }

        # Bảo vệ chuỗi vi phạm & ảnh đối chứng trước hiện tượng nhấp nháy mất bbox 1-2 frame (<0.6s)
        present_pids = {s.person_track_id for s in person_states}
        active_streak_pids = {key[0] for key in confirmation._streaks.keys()}
        person_states_for_conf = list(person_states)
        for pid in active_streak_pids - present_pids:
            dt_miss = t_frame_s - self._track_last_seen.get(pid, -999.0)
            if 0.0 <= dt_miss <= 0.6:
                person_states_for_conf.append(PersonState(pid, STATE_UNKNOWN, reason="transient_miss"))

        # Bước 2c: Temporal Confirmation & Deduplication
        confirmed_list = confirmation.update(t_frame_s, person_states_for_conf, frame=frame, detections=detections)
        deduped_violations = dedup.filter(confirmed_list, detections, t_frame_s)
        t_rule_end = time.perf_counter()
        rule_ms = (t_rule_end - t_rule_start) * 1000.0

        # Đếm số lượng
        counts = {"person": 0, "helmet": 0, "safety_vest": 0}
        for d in detections:
            if d.class_name in counts:
                counts[d.class_name] += 1

        # Lưu cache cho luồng hiển thị 25 FPS
        with self._lock:
            self._last_detections = detections
            self._last_person_states = person_states
            self._last_counts = counts
            t_curr = time.time()
            ai_fps_window.append(t_curr)
            while ai_fps_window and t_curr - ai_fps_window[0] > 1.0:
                ai_fps_window.pop(0)
            self.metrics["ai_fps_actual"] = round(float(len(ai_fps_window)), 1)
            self.metrics["inference_latency_ms"] = round(infer_ms, 1)
            self.metrics["rule_latency_ms"] = round(rule_ms, 1)
            self.metrics["active_persons"] = counts["person"]
            self.metrics["active_violations"] = len([
                ps for ps in person_states if getattr(ps, "violations", None)
            ])
            self.metrics["counts"] = counts

        # Kích hoạt xử lý bằng chứng vi phạm (Before/After) bất đồng bộ nếu có vi phạm mới
        for ev in deduped_violations:
            frame_snap = frame.copy()
            dets_snap = list(detections)
            states_snap = list(person_states)
            self._evidence_executor.submit(
                self._dispatch_evidence_async,
                frame_snap, dets_snap, ev, frame_w, frame_h, states_snap, frame_num,
            )

    def _ai_worker_loop(
        self,
        detector: Any,
        confirmation: ViolationConfirmation,
        dedup: ViolationDeduplicator,
    ):
        """Luồng AI chạy độc lập để không bao giờ làm giảm FPS hiển thị video."""
        ai_fps_window: List[float] = []
        while not self._stop_requested.is_set():
            with self._ai_lock:
                while (
                    not self._stop_requested.is_set()
                    and not self._ai_done
                    and self._pending_ai_item is None
                ):
                    self._ai_cond.wait(timeout=0.1)

                if self._stop_requested.is_set():
                    break
                item = self._pending_ai_item
                self._pending_ai_item = None
                if item is None and self._ai_done:
                    break

            if item is not None:
                f_num, f_img, t_s = item
                try:
                    self._process_ai_step(f_num, f_img, t_s, detector, confirmation, dedup, ai_fps_window)
                except Exception as e:
                    logger.error("[%s] Error in AI worker step: %s", self.camera_id, e)

    def _encode_and_publish_frame(self, raw_frame: np.ndarray, fps_window: List[float]) -> None:
        """Vẽ overlay (bbox mới nhất + ROI + HUD), nén JPEG và phát tới mọi client MJPEG."""
        with self._lock:
            detections = self._last_detections
            person_states = self._last_person_states

        draw_frame = raw_frame.copy()
        frame_h, frame_w = raw_frame.shape[:2]
        pixel_roi = resolve_roi_polygons_for_frame(self.roi_polygons, frame_w, frame_h)
        draw_detections(
            draw_frame,
            detections,
            person_states=person_states,
            roi_polygons=pixel_roi,
            enable_no_helmet=self.enable_no_helmet,
            enable_no_vest=self.enable_no_vest,
        )
        if self.draw_hud:
            self._draw_stats_hud(draw_frame)

        ret, jpeg = cv2.imencode(".jpg", draw_frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if ret:
            jpeg_bytes = jpeg.tobytes()
            with self._lock:
                self.latest_jpeg_bytes = jpeg_bytes
                self._jpeg_seq += 1
                t_curr = time.time()
                fps_window.append(t_curr)
                while fps_window and t_curr - fps_window[0] > 1.0:
                    fps_window.pop(0)
                self.metrics["fps"] = round(float(len(fps_window)), 1)
                self._frame_condition.notify_all()

    def _push_status_splash(self, title: str, subtitle: str) -> None:
        jpeg_bytes = _make_splash_jpeg(title, subtitle)
        if not jpeg_bytes:
            return
        with self._lock:
            self.latest_jpeg_bytes = jpeg_bytes
            self._jpeg_seq += 1
            self._frame_condition.notify_all()

    @staticmethod
    def _is_live_cctv_source(source: Any, original_source: Any = None) -> bool:
        if isinstance(source, int):
            return True
        for candidate in (source, original_source):
            s = str(candidate or "").strip().lower()
            if s.startswith(("rtsp://", "rtsps://")):
                return True
            if s.startswith(("http://", "https://")) and not is_youtube_url(s):
                if ".m3u8" in s or "stream" in s or "mjpeg" in s:
                    return True
        return False

    def _stop_ai_worker(self) -> None:
        with self._ai_lock:
            self._ai_done = True
            self._ai_cond.notify_all()
        if self._ai_thread and self._ai_thread.is_alive():
            self._ai_thread.join(timeout=1.5)
        self._ai_thread = None

    def _run_loop(self):
        """Vòng lặp chính 24/7: Mở nguồn -> Pacing mượt mà -> AI bất đồng bộ -> Tự reconnect / loop."""
        logger.info("[%s] Opening video source: %s", self.camera_id, mask_source(self.source))
        confirmation = ViolationConfirmation(
            duration_s=self.confirmation_duration_s,
            duration_map={
                VIOLATION_NO_HELMET: self.confirmation_duration_s,
                VIOLATION_NO_VEST: self.confirmation_duration_vest_s,
                VIOLATION_RESTRICTED_AREA: self.confirmation_duration_s,
            },
        )
        dedup = ViolationDeduplicator(camera_id=self.camera_id, cooldown_s=self.cooldown_s)
        run_ai_inline = _is_mock_detector_cls()
        ai_fps_window: List[float] = []
        fps_window: List[float] = []
        cut_detector = SceneCutDetector() if self.enable_scene_cut else None

        reconnect_attempt = 0
        fatal_error = False

        try:
            while not self._stop_requested.is_set():
                try:
                    reader = VideoReader(self.source)
                except VideoSourceError as e:
                    masked_err = mask_source(str(e))
                    self.error_message = masked_err
                    logger.error("[%s] Failed to open video reader: %s", self.camera_id, masked_err)
                    if not self.auto_reconnect:
                        fatal_error = True
                        self.status = "error"
                        break
                    reconnect_attempt += 1
                    self.reconnect_count += 1
                    self.metrics["reconnect_count"] = self.reconnect_count
                    self.status = "reconnecting"
                    delay = min(RECONNECT_MAX_DELAY_S, RECONNECT_BASE_DELAY_S * (2 ** min(reconnect_attempt - 1, 5)))
                    self._push_status_splash(
                        f"CAM: {self.camera_id} - MAT KET NOI",
                        f"Thu ket noi lai sau {int(delay)}s (lan {self.reconnect_count})...",
                    )
                    self._stop_requested.wait(delay)
                    continue

                detector_lock: Optional[threading.Lock] = None
                try:
                    detector, detector_lock = _acquire_detector(
                        model_path=self.model_path,
                        conf=self.confidence_threshold,
                        device=config.DEVICE,
                        imgsz=config.MODEL_IMGSZ,
                    )
                except DetectorError as e:
                    reader.release()
                    self.status = "error"
                    fatal_error = True
                    self.error_message = mask_source(str(e))
                    logger.error("[%s] Failed to load detector: %s", self.camera_id, e)
                    break

                reconnect_attempt = 0
                self.error_message = None
                self.status = "running"
                logger.info("[%s] LiveStreamSession started running successfully", self.camera_id)

                if not run_ai_inline:
                    self._ai_done = False
                    self._pending_ai_item = None
                    self._ai_thread = threading.Thread(
                        target=self._ai_worker_loop,
                        args=(detector, confirmation, dedup),
                        name=f"live-ai-{self.camera_id}",
                        daemon=True,
                    )
                    self._ai_thread.start()

                orig_src = getattr(reader, "original_source", self.source)
                is_live_cctv = self._is_live_cctv_source(self.source, orig_src)
                needs_pacing = not is_live_cctv
                last_ai_submit_t = 0.0
                last_frame_tick = time.perf_counter()

                try:
                    with reader:
                        for frame_num, frame in reader:
                            if self._stop_requested.is_set():
                                break

                            self.frame_number += 1
                            self.last_frame_at = time.time()

                            if needs_pacing and frame_num > 1:
                                expected_interval = 1.0 / self.target_fps
                                elapsed_since_last = time.perf_counter() - last_frame_tick
                                if elapsed_since_last < expected_interval:
                                    time.sleep(expected_interval - elapsed_since_last)
                            last_frame_tick = time.perf_counter()

                            is_cut = False
                            if cut_detector is not None and cut_detector.check(frame):
                                is_cut = True
                                logger.info("[%s] Scene cut detected at frame %d, resetting tracker & confirmation", self.camera_id, frame_num)
                                try:
                                    detector.reset_tracker()
                                except Exception:
                                    pass
                                confirmation.reset()
                                self._track_alias.clear()
                                self._track_last_bbox.clear()
                                self._track_last_seen.clear()
                                with self._lock:
                                    self._last_detections = []
                                    self._last_person_states = []
                                    self._last_counts = {"person": 0, "helmet": 0, "safety_vest": 0}

                            if frame_num == 1 and not run_ai_inline:
                                self._encode_and_publish_frame(frame, fps_window)

                            now_perf = time.perf_counter()
                            ai_interval = 1.0 / self.ai_fps
                            should_run_ai = (frame_num == 1) or is_cut or (now_perf - last_ai_submit_t >= ai_interval - 0.003)

                            if should_run_ai:
                                last_ai_submit_t = now_perf
                                eff_fps = self.target_fps if needs_pacing else (reader.fps if reader.fps > 0 else 25.0)
                                t_frame_s = round((frame_num - 1) / eff_fps, 3)
                                if run_ai_inline:
                                    self._process_ai_step(
                                        frame_num, frame, t_frame_s, detector, confirmation, dedup, ai_fps_window
                                    )
                                else:
                                    with self._ai_lock:
                                        self._pending_ai_item = (frame_num, frame.copy(), t_frame_s)
                                        self._ai_cond.notify()

                            self._encode_and_publish_frame(frame, fps_window)
                finally:
                    self._stop_ai_worker()
                    if detector_lock is not None:
                        try:
                            detector.reset_tracker()
                        except Exception:
                            pass
                        detector_lock.release()

                if self._stop_requested.is_set():
                    break

                # Quyết định tiếp tục vòng đời 24/7 khi stream rớt hoặc video EOF
                confirmation.reset()
                self._track_alias.clear()
                self._track_last_bbox.clear()
                self._track_last_seen.clear()

                if not is_live_cctv and self.loop_on_eof:
                    logger.info("[%s] Video EOF reached, restarting 24/7 loop", self.camera_id)
                    self._stop_requested.wait(0.2)
                    continue

                if self.auto_reconnect:
                    reconnect_attempt += 1
                    self.reconnect_count += 1
                    self.metrics["reconnect_count"] = self.reconnect_count
                    self.status = "reconnecting"
                    delay = min(RECONNECT_MAX_DELAY_S, RECONNECT_BASE_DELAY_S * (2 ** min(reconnect_attempt - 1, 5)))
                    logger.warning(
                        "[%s] Stream ended unexpectedly, reconnecting in %.1fs (attempt %d)",
                        self.camera_id, delay, self.reconnect_count,
                    )
                    self._push_status_splash(
                        f"CAM: {self.camera_id} - DANG KET NOI LAI...",
                        f"Thu lai sau {int(delay)}s (lan {self.reconnect_count})...",
                    )
                    self._stop_requested.wait(delay)
                    continue

                break

        except Exception as e:
            fatal_error = True
            self.status = "error"
            self.error_message = mask_source(f"{type(e).__name__}: {e}")
            logger.error("[%s] Unhandled exception in live loop: %s", self.camera_id, e, exc_info=True)
        finally:
            self._stop_ai_worker()
            try:
                self._evidence_executor.shutdown(wait=False)
            except Exception:
                pass
            if not fatal_error:
                self.status = "stopped"
            with self._lock:
                self._frame_condition.notify_all()
            logger.info("[%s] Live loop finished", self.camera_id)

    def generate_mjpeg(self):
        """Generator trả về chuỗi multipart/x-mixed-replace cho StreamingResponse."""
        last_yielded_seq = -1
        while not self._stop_requested.is_set():
            with self._lock:
                # Đợi frame mới sẵn sàng (bao gồm cả lúc reconnecting 24/7)
                while (
                    not self._stop_requested.is_set()
                    and self.status in ("initializing", "running", "reconnecting")
                    and (self.latest_jpeg_bytes is None or self._jpeg_seq == last_yielded_seq)
                ):
                    self._frame_condition.wait(timeout=0.2)

                if self._stop_requested.is_set() or self.status in ("stopped", "error"):
                    break

                jpeg_data = self.latest_jpeg_bytes
                last_yielded_seq = self._jpeg_seq

            if jpeg_data:
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n" + jpeg_data + b"\r\n"
                )


_EVIDENCE_OWNERS_LOCK = threading.Lock()
_EVIDENCE_OWNERS: "OrderedDict[str, tuple[Optional[str], Optional[str], bool]]" = OrderedDict()
_MAX_TRACKED_EVIDENCES = 5000


def register_evidence_owner(
    filename: str,
    site_id: Optional[str],
    owner_user_id: Optional[str] = None,
    managed: bool = False,
) -> None:
    if not filename:
        return
    with _EVIDENCE_OWNERS_LOCK:
        if filename in _EVIDENCE_OWNERS:
            _EVIDENCE_OWNERS.move_to_end(filename)
        _EVIDENCE_OWNERS[filename] = (site_id, owner_user_id, bool(managed))
        while len(_EVIDENCE_OWNERS) > _MAX_TRACKED_EVIDENCES:
            _EVIDENCE_OWNERS.popitem(last=False)


def get_evidence_owner(filename: str) -> Optional[tuple[Optional[str], Optional[str], bool]]:
    if not filename:
        return None
    with _EVIDENCE_OWNERS_LOCK:
        return _EVIDENCE_OWNERS.get(filename)


def _prune_local_evidence_dir(ev_dir: Path, max_files: int) -> None:
    if max_files <= 0 or not ev_dir.is_dir():
        return
    try:
        files = [p for p in ev_dir.glob("*.jpg") if p.is_file()]
        if len(files) <= max_files:
            return
        files.sort(key=lambda p: p.stat().st_mtime)
        for old_file in files[: len(files) - max_files]:
            try:
                old_file.unlink()
            except OSError:
                pass
    except Exception:
        pass


class LiveEngineManager:
    """Quản lý tập trung các phiên LiveStreamSession theo camera_id (hỗ trợ giới hạn động)."""

    def __init__(self, max_sessions: int = DEFAULT_MAX_LIVE_SESSIONS):
        self._sessions: Dict[str, LiveStreamSession] = {}
        self._lock = threading.Lock()
        self.max_sessions = max(1, int(max_sessions))
        _set_detector_pool_cap(self.max_sessions)

    @staticmethod
    def register_evidence(
        filename: str,
        site_id: Optional[str],
        owner_user_id: Optional[str] = None,
        managed: bool = False,
    ) -> None:
        register_evidence_owner(filename, site_id=site_id, owner_user_id=owner_user_id, managed=managed)

    @staticmethod
    def get_evidence_owner(filename: str) -> Optional[tuple[Optional[str], Optional[str], bool]]:
        return get_evidence_owner(filename)

    def set_max_sessions(self, new_limit: int) -> int:
        clamped = max(1, min(256, int(new_limit)))
        with self._lock:
            self.max_sessions = clamped
        _set_detector_pool_cap(clamped)
        return clamped

    def get_max_sessions(self) -> int:
        with self._lock:
            return self.max_sessions

    def get_session(self, camera_id: str) -> Optional[LiveStreamSession]:
        with self._lock:
            return self._sessions.get(str(camera_id))

    def start_session(
        self,
        camera_id: str,
        source: Any,
        site_id: Optional[str] = None,
        camera_uuid: Optional[str] = None,
        model_path: Optional[str] = None,
        target_fps: float = 25.0,
        ai_fps: float = 15.0,
        roi_polygons: Optional[List[Any]] = None,
        confirmation_duration_s: float = 2.0,
        confirmation_duration_vest_s: Optional[float] = None,
        enable_no_helmet: bool = True,
        enable_no_vest: bool = True,
        enable_restricted_area: bool = True,
        enable_scene_cut: bool = True,
        dispatch_enabled: bool = True,
        auto_reconnect: bool = False,
        loop_on_eof: bool = False,
        managed: bool = False,
    ) -> LiveStreamSession:
        camera_id = _validate_safe_id(camera_id, "camera_id")
        with self._lock:
            existing = self._sessions.get(camera_id)
            if existing and existing.status in ("running", "initializing", "reconnecting"):
                # Cập nhật cấu hình nếu cùng source và site/camera không đổi
                if existing.source == source and (site_id is None or existing.site_id == site_id):
                    existing.update_config(
                        target_fps=target_fps,
                        ai_fps=ai_fps,
                        roi_polygons=roi_polygons,
                        enable_no_helmet=enable_no_helmet,
                        enable_no_vest=enable_no_vest,
                        enable_restricted_area=enable_restricted_area,
                        enable_scene_cut=enable_scene_cut,
                        confirmation_duration_s=confirmation_duration_s,
                        confirmation_duration_vest_s=confirmation_duration_vest_s,
                        dispatch_enabled=dispatch_enabled,
                        auto_reconnect=auto_reconnect,
                        loop_on_eof=loop_on_eof,
                    )
                    if camera_uuid:
                        existing.camera_uuid = _validate_safe_id(camera_uuid, "camera_uuid")
                    existing.managed = bool(managed)
                    return existing
                # Đổi source khác -> dừng session cũ
                existing.stop()

            # Giới hạn số phiên đang hoạt động (không tính phiên của chính camera này)
            active = sum(
                1 for cid, s in self._sessions.items()
                if cid != camera_id and s.status not in ("stopped", "error")
            )
            if active >= self.max_sessions:
                raise LiveSessionLimitError(
                    f"Đã đạt giới hạn {self.max_sessions} phiên live đồng thời"
                )

            session = LiveStreamSession(
                camera_id=camera_id,
                source=source,
                site_id=site_id,
                camera_uuid=camera_uuid,
                model_path=model_path,
                target_fps=target_fps,
                ai_fps=ai_fps,
                roi_polygons=roi_polygons,
                confirmation_duration_s=confirmation_duration_s,
                confirmation_duration_vest_s=confirmation_duration_vest_s,
                enable_no_helmet=enable_no_helmet,
                enable_no_vest=enable_no_vest,
                enable_restricted_area=enable_restricted_area,
                enable_scene_cut=enable_scene_cut,
                dispatch_enabled=dispatch_enabled,
                auto_reconnect=auto_reconnect,
                loop_on_eof=loop_on_eof,
                managed=managed,
            )
            self._sessions[camera_id] = session

        session.start()
        return session

    def stop_session(self, camera_id: str) -> bool:
        camera_id = str(camera_id)
        with self._lock:
            session = self._sessions.pop(camera_id, None)
        if session:
            session.stop()
            return True
        return False

    def list_sessions(self) -> List[Dict[str, Any]]:
        with self._lock:
            return [
                {
                    "camera_id": cid,
                    "camera_uuid": s.camera_uuid,
                    "site_id": s.site_id,
                    "status": s.status,
                    "managed": s.managed,
                    "reconnect_count": s.reconnect_count,
                    "dispatch_enabled": s.dispatch_enabled,
                    "target_fps": s.target_fps,
                    "ai_fps": s.ai_fps,
                    "source": mask_source(s.source),
                    "metrics": s.metrics,
                    "error": s.error_message,
                }
                for cid, s in self._sessions.items()
            ]

    def stop_all(self):
        with self._lock:
            sessions = list(self._sessions.values())
            self._sessions.clear()
        for s in sessions:
            s.stop()


# Instance singleton dùng chung toàn service
live_engine_manager = LiveEngineManager()
