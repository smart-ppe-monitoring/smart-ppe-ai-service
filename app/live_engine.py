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
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import logging
import os
from pathlib import Path
import re
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from app import config
from app.constants import (
    CONTRACT_VERSION,
    VIOLATION_NO_HELMET,
    VIOLATION_NO_VEST,
    VIOLATION_RESTRICTED_AREA,
)
from app.detection.detector import Detector, DetectorError
from app.association.associator import associate
from app.rules.safety import evaluate as evaluate_safety, is_inside_roi
from app.rules.confirmation import ViolationConfirmation
from app.rules.deduplication import ViolationDeduplicator, ViolationEvent as DedupEvent
from app.evidence.snapshot import save_evidence
from app.video.video_reader import VideoReader, VideoSourceError, is_youtube_url, mask_source
from app.video.visualizer import draw_detections, draw_roi_polygons
from app.main import _build_violation_event
from app.events import dispatch_event

logger = logging.getLogger("live_engine")

# Định danh camera/site được ghép vào MinIO object key và snapshot URL → chỉ cho phép ký tự an toàn.
SAFE_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")

# Giới hạn số phiên live đồng thời (mỗi phiên nạp 1 model YOLO + threads riêng).
MAX_LIVE_SESSIONS = max(1, int(os.getenv("LIVE_MAX_SESSIONS", "4")))

# Giới hạn hình học ROI để tránh tiêu tốn CPU mỗi frame.
MAX_ROI_POLYGONS = 16
MAX_ROI_POINTS = 64
MIN_CONFIRMATION_DURATION_S = 0.5


class LiveSessionLimitError(Exception):
    """Vượt quá số phiên live đồng thời cho phép."""


def _validate_safe_id(value: Any, field_name: str) -> str:
    s = str(value)
    if not SAFE_ID_RE.match(s):
        raise ValueError(f"{field_name} không hợp lệ (chỉ chấp nhận [A-Za-z0-9_-], tối đa 64 ký tự)")
    return s


def _validate_roi_polygons(roi_polygons: Optional[List[Any]]) -> List[Any]:
    """Phòng thủ tầng core: giới hạn số polygon/số đỉnh ROI."""
    if not roi_polygons:
        return []
    if not isinstance(roi_polygons, (list, tuple)) or len(roi_polygons) > MAX_ROI_POLYGONS:
        raise ValueError(f"roi_polygons phải là danh sách tối đa {MAX_ROI_POLYGONS} polygon")
    for poly in roi_polygons:
        pts = poly.get("points", []) if isinstance(poly, dict) else poly
        if not isinstance(pts, (list, tuple)) or len(pts) > MAX_ROI_POINTS:
            raise ValueError(f"Mỗi polygon ROI tối đa {MAX_ROI_POINTS} điểm")
    return list(roi_polygons)


class LiveStreamSession:
    """Phiên xử lý và phát luồng video trực tiếp với AI (YOLO + ByteTrack + Rules)."""

    def __init__(
        self,
        camera_id: str,
        source: Any,
        site_id: Optional[str] = None,
        model_path: Optional[str] = None,
        target_fps: float = 25.0,
        ai_fps: float = 15.0,
        roi_polygons: Optional[List[Any]] = None,
        confirmation_duration_s: float = 2.0,
        cooldown_s: float = 30.0,
        confidence_threshold: float = 0.8,
        enable_no_helmet: bool = True,
        enable_no_vest: bool = True,
        enable_restricted_area: bool = True,
        draw_hud: bool = True,
        dispatch_enabled: bool = True,
    ):
        self.camera_id = _validate_safe_id(camera_id, "camera_id")
        self.source = source
        # site_id mặc định lấy từ cấu hình service (không tin dữ liệu client)
        self.site_id = _validate_safe_id(site_id if site_id is not None else config.SITE_CODE, "site_id")
        self.model_path = model_path or config.YOLO_MODEL_PATH
        self.target_fps = max(1.0, min(60.0, float(target_fps)))
        self.ai_fps = max(1.0, min(self.target_fps, float(ai_fps)))
        self.roi_polygons = _validate_roi_polygons(roi_polygons)
        self.confirmation_duration_s = max(MIN_CONFIRMATION_DURATION_S, float(confirmation_duration_s))
        self.cooldown_s = float(cooldown_s)
        self.confidence_threshold = float(confidence_threshold)
        self.enable_no_helmet = bool(enable_no_helmet)
        self.enable_no_vest = bool(enable_no_vest)
        self.enable_restricted_area = bool(enable_restricted_area)
        self.draw_hud = bool(draw_hud)
        # Cho phép upload MinIO + bắn sự kiện tới Incident Service (tắt cho môi trường test)
        self.dispatch_enabled = bool(dispatch_enabled)

        # Trạng thái phiên
        self.status = "initializing"  # initializing, running, paused, stopped, error
        self.error_message: Optional[str] = None
        self._stop_requested = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self._frame_condition = threading.Condition(self._lock)

        # Frame buffer JPEG mới nhất phục vụ MJPEG stream
        self.latest_jpeg_bytes: Optional[bytes] = None
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
            "inference_latency_ms": 0.0,
            "rule_latency_ms": 0.0,
            "e2e_alert_latency_ms": 0.0,
            "active_persons": 0,
            "active_violations": 0,
            "counts": self._last_counts,
        }
        self.recent_violations: List[Dict[str, Any]] = []

        # Thread pool riêng phục vụ upload bằng chứng không làm nghẽn video
        self._evidence_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix=f"ev-{self.camera_id}")

    def update_config(
        self,
        target_fps: Optional[float] = None,
        ai_fps: Optional[float] = None,
        roi_polygons: Optional[List[Any]] = None,
        enable_no_helmet: Optional[bool] = None,
        enable_no_vest: Optional[bool] = None,
        enable_restricted_area: Optional[bool] = None,
        confirmation_duration_s: Optional[float] = None,
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
            if confirmation_duration_s is not None:
                self.confirmation_duration_s = max(MIN_CONFIRMATION_DURATION_S, float(confirmation_duration_s))
        logger.info(
            "[%s] Config updated: target_fps=%.1f, ai_fps=%.1f, helmet=%s, vest=%s, restricted=%s",
            self.camera_id, self.target_fps, self.ai_fps,
            self.enable_no_helmet, self.enable_no_vest, self.enable_restricted_area
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
        with self._lock:
            self._frame_condition.notify_all()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=3.0)
        self._evidence_executor.shutdown(wait=False)
        self.status = "stopped"
        logger.info("[%s] LiveStreamSession stopped cleanly", self.camera_id)

    def _dispatch_evidence_async(self, frame_copy, detections_copy, ev: DedupEvent, frame_w: int, frame_h: int):
        """Lưu ảnh và bắn sự kiện vi phạm trên background thread không làm dừng luồng video."""
        t_start = time.time()
        try:
            # 1. Lưu ảnh bằng chứng snapshot cục bộ
            evidence_res = save_evidence(frame_copy, detections_copy, ev, config.EVIDENCE_DIR)
            snapshot_url = None
            reference_snapshot_url = None

            if not self.dispatch_enabled:
                # Chế độ test: chỉ giữ bằng chứng cục bộ, KHÔNG upload MinIO / bắn sự kiện ra hệ thống thật
                self._record_violation(ev, None, "disabled", round((time.time() - t_start) * 1000.0, 1))
                return

            # 2. Upload MinIO nếu khả dụng
            if evidence_res.status == "saved" and evidence_res.path:
                try:
                    from app.storage.minio_client import upload_snapshot
                    with open(evidence_res.path, "rb") as f:
                        img_bytes = f.read()
                    snapshot_url = upload_snapshot(
                        image_bytes=img_bytes,
                        site_code=self.site_id,
                        camera_code=ev.camera_id,
                        track_id=ev.person_track_id,
                    )
                    try:
                        Path(evidence_res.path).unlink(missing_ok=True)
                    except Exception:
                        pass
                except Exception as m_err:
                    logger.debug("[%s] MinIO direct upload skipped: %s", self.camera_id, m_err)

            # 2b. Upload ảnh đối chứng reference nếu có
            if evidence_res.status == "saved" and evidence_res.reference_path:
                try:
                    from app.storage.minio_client import upload_snapshot
                    with open(evidence_res.reference_path, "rb") as f:
                        ref_bytes = f.read()
                    reference_snapshot_url = upload_snapshot(
                        image_bytes=ref_bytes,
                        site_code=self.site_id,
                        camera_code=ev.camera_id,
                        track_id=ev.person_track_id,
                        suffix="ref",
                    )
                    try:
                        Path(evidence_res.reference_path).unlink(missing_ok=True)
                    except Exception:
                        pass
                except Exception as m_err:
                    logger.debug("[%s] MinIO ref upload skipped: %s", self.camera_id, m_err)

            # 3. Fallback sang Edge Gateway HTTP URL nếu chưa có MinIO URL
            edge_base = (
                config.BACKEND_BASE_URL.rstrip("/")
                if hasattr(config, "BACKEND_BASE_URL")
                else "http://localhost:8893"
            )
            if not snapshot_url or not str(snapshot_url).startswith(("http://", "https://")):
                filename = Path(evidence_res.path).name if evidence_res.path else f"{ev.event_id}.jpg"
                snapshot_url = f"{edge_base}/evidences/{self.site_id}/{ev.camera_id}/{filename}"

            if evidence_res.reference_path and (not reference_snapshot_url or not str(reference_snapshot_url).startswith(("http://", "https://"))):
                ref_name = Path(evidence_res.reference_path).name
                reference_snapshot_url = f"{edge_base}/evidences/{self.site_id}/{ev.camera_id}/{ref_name}"

            # 4. Đóng gói ViolationEvent chuẩn Contract
            model_name = Path(self.model_path).name
            contract_event = _build_violation_event(
                ev=ev,
                frame_w=frame_w,
                frame_h=frame_h,
                snapshot_url=snapshot_url,
                reference_snapshot_url=reference_snapshot_url,
                all_detections=detections_copy,
                model_name=model_name,
            )

            # 5. Dispatch Kafka / HTTP / Offline Buffer
            method = dispatch_event(contract_event)
            e2e_latency = round((time.time() - t_start) * 1000.0, 1)
            self._record_violation(ev, snapshot_url, method, e2e_latency)

            logger.info(
                "[%s] VIOLATION EVENT DISPATCHED: person=#%d, type=%s, via=%s in %.1fms",
                self.camera_id, ev.person_track_id, ev.violation_type, method, e2e_latency,
            )
        except Exception as e:
            logger.error("[%s] Error in background evidence dispatch: %s", self.camera_id, e)

    def _record_violation(self, ev: DedupEvent, snapshot_url: Optional[str], method: str, e2e_latency: float):
        """Ghi nhận vi phạm gần nhất phục vụ API /stats (tối đa 50 bản ghi)."""
        with self._lock:
            self.metrics["e2e_alert_latency_ms"] = e2e_latency
            self.recent_violations.append({
                "event_id": ev.event_id,
                "person_id": ev.person_track_id,
                "violation_type": ev.violation_type,
                "snapshot_url": snapshot_url,
                "dispatched_via": method,
                "e2e_alert_latency_ms": e2e_latency,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            })
            if len(self.recent_violations) > 50:
                self.recent_violations.pop(0)

    def _draw_stats_hud(self, frame: np.ndarray):
        """Vẽ thanh thông số thời gian thực (FPS, Latency, Số lượng) lên góc trên màn hình."""
        h, w = frame.shape[:2]
        fps_text = f"FPS: {self.metrics['fps']:.1f}/{self.target_fps:.0f} (AI: {self.ai_fps:.0f})"
        lat_text = f"Infer: {self.metrics['inference_latency_ms']:.1f}ms | Rule: {self.metrics['rule_latency_ms']:.1f}ms"
        counts_text = (
            f"Persons: {self._last_counts.get('person', 0)} | "
            f"Helmets: {self._last_counts.get('helmet', 0)} | "
            f"Vests: {self._last_counts.get('safety_vest', 0)}"
        )
        hud_bar_w = min(w - 20, 520)
        cv2.rectangle(frame, (10, 10), (10 + hud_bar_w, 75), (20, 20, 20), -1)
        cv2.rectangle(frame, (10, 10), (10 + hud_bar_w, 75), (70, 70, 70), 1)

        cv2.putText(frame, f"CAM: {self.camera_id} | {fps_text}", (20, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(frame, lat_text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (200, 200, 200), 1, cv2.LINE_AA)
        cv2.putText(frame, counts_text, (20, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (100, 255, 100), 1, cv2.LINE_AA)

    def _run_loop(self):
        """Vòng lặp chính xử lý luồng: Đọc frame -> Pacing -> Inference/Cache -> Rules -> Draw -> Encode."""
        logger.info("[%s] Opening video source: %s", self.camera_id, mask_source(self.source))
        try:
            reader = VideoReader(self.source)
        except VideoSourceError as e:
            self.status = "error"
            self.error_message = mask_source(str(e))
            logger.error("[%s] Failed to open video reader: %s", self.camera_id, e)
            return

        try:
            detector = Detector(
                model_path=self.model_path,
                conf=self.confidence_threshold,
                device=config.DEVICE,
                imgsz=config.MODEL_IMGSZ,
            ).load()
        except DetectorError as e:
            self.status = "error"
            self.error_message = mask_source(str(e))
            logger.error("[%s] Failed to load detector: %s", self.camera_id, e)
            reader.release()
            return

        confirmation = ViolationConfirmation(duration_s=self.confirmation_duration_s)
        dedup = ViolationDeduplicator(camera_id=self.camera_id, cooldown_s=self.cooldown_s)

        self.status = "running"
        logger.info("[%s] LiveStreamSession started running successfully", self.camera_id)

        last_ai_time = 0.0
        fps_window: List[float] = []
        last_frame_tick = time.time()

        try:
            with reader:
                for frame_num, frame in reader:
                    if self._stop_requested.is_set():
                        break

                    now = time.time()
                    frame_h, frame_w = frame.shape[:2]
                    self.frame_number = frame_num

                    # 1. Pacing: Điều phối tốc độ phát cho file MP4 (không cho chạy tua quá nhanh)
                    if not reader.is_stream:
                        expected_interval = 1.0 / self.target_fps
                        elapsed_since_last = now - last_frame_tick
                        if elapsed_since_last < expected_interval:
                            time.sleep(expected_interval - elapsed_since_last)
                    last_frame_tick = time.time()

                    # 2. Quyết định có chạy AI trên frame này hay không (Sampling Rate)
                    ai_interval = 1.0 / self.ai_fps
                    should_run_ai = (now - last_ai_time >= ai_interval - 0.003)

                    if should_run_ai:
                        last_ai_time = now
                        # Bước 2a: Detection + Tracking (YOLO + ByteTrack)
                        t_infer_start = time.perf_counter()
                        detections = detector.detect(frame)
                        t_infer_end = time.perf_counter()
                        infer_ms = (t_infer_end - t_infer_start) * 1000.0

                        # Bước 2b: PPE Association & Safety Rules (Độc lập với Model AI)
                        t_rule_start = time.perf_counter()
                        associations = associate(detections)
                        person_states = evaluate_safety(
                            detections=detections,
                            associations=associations,
                            frame_w=frame_w,
                            frame_h=frame_h,
                            enable_no_helmet=self.enable_no_helmet,
                            enable_no_vest=self.enable_no_vest,
                            roi_polygons=self.roi_polygons,
                            enable_restricted_area=self.enable_restricted_area,
                        )

                        # Bước 2c: Temporal Confirmation & Deduplication
                        t_frame_s = round((frame_num - 1) / (reader.fps if reader.fps > 0 else 25.0), 3)
                        confirmed_list = confirmation.update(t_frame_s, person_states, frame=frame, detections=detections)
                        deduped_violations = dedup.filter(confirmed_list, detections, t_frame_s)
                        t_rule_end = time.perf_counter()
                        rule_ms = (t_rule_end - t_rule_start) * 1000.0

                        # Lưu cache cho các frame tiếp theo
                        self._last_detections = detections
                        self._last_person_states = person_states

                        # Đếm số lượng
                        counts = {"person": 0, "helmet": 0, "safety_vest": 0}
                        for d in detections:
                            if d.class_name in counts:
                                counts[d.class_name] += 1
                        self._last_counts = counts

                        # Kích hoạt xử lý bằng chứng vi phạm bất đồng bộ nếu có vi phạm mới
                        for ev in deduped_violations:
                            frame_snap = frame.copy()
                            dets_snap = list(detections)
                            self._evidence_executor.submit(
                                self._dispatch_evidence_async,
                                frame_snap, dets_snap, ev, frame_w, frame_h,
                            )

                        with self._lock:
                            self.metrics["inference_latency_ms"] = round(infer_ms, 1)
                            self.metrics["rule_latency_ms"] = round(rule_ms, 1)
                            self.metrics["active_persons"] = counts["person"]
                            self.metrics["active_violations"] = len([
                                ps for ps in person_states if getattr(ps, "violations", None)
                            ])
                            self.metrics["counts"] = counts
                    else:
                        # Tái sử dụng kết quả AI của frame liền trước (Cached Overlay)
                        detections = self._last_detections
                        person_states = self._last_person_states

                    # 3. Vẽ Bounding Box, Polygon ROI và HUD thống kê
                    draw_frame = frame.copy()
                    draw_detections(
                        draw_frame,
                        detections,
                        person_states=person_states,
                        roi_polygons=self.roi_polygons,
                    )
                    if self.draw_hud:
                        self._draw_stats_hud(draw_frame)

                    # 4. Mã hóa JPEG cho luồng MJPEG
                    ret, jpeg = cv2.imencode(".jpg", draw_frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                    if ret:
                        jpeg_bytes = jpeg.tobytes()
                        with self._lock:
                            self.latest_jpeg_bytes = jpeg_bytes
                            # Tính rolling FPS
                            t_curr = time.time()
                            fps_window.append(t_curr)
                            while fps_window and t_curr - fps_window[0] > 1.0:
                                fps_window.pop(0)
                            self.metrics["fps"] = round(float(len(fps_window)), 1)
                            self._frame_condition.notify_all()

        except Exception as e:
            self.status = "error"
            self.error_message = mask_source(f"{type(e).__name__}: {e}")
            logger.error("[%s] Unhandled exception in live loop: %s", self.camera_id, e, exc_info=True)
        finally:
            self.status = "stopped"
            with self._lock:
                self._frame_condition.notify_all()
            logger.info("[%s] Live loop finished", self.camera_id)

    def generate_mjpeg(self):
        """Generator trả về chuỗi multipart/x-mixed-replace cho StreamingResponse."""
        last_yielded_frame = -1
        while not self._stop_requested.is_set():
            with self._lock:
                # Đợi frame mới sẵn sàng
                while (
                    not self._stop_requested.is_set()
                    and self.status == "running"
                    and (self.latest_jpeg_bytes is None or self.frame_number == last_yielded_frame)
                ):
                    self._frame_condition.wait(timeout=0.2)

                if self._stop_requested.is_set() or self.status in ("stopped", "error"):
                    break

                jpeg_data = self.latest_jpeg_bytes
                last_yielded_frame = self.frame_number

            if jpeg_data:
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n" + jpeg_data + b"\r\n"
                )


class LiveEngineManager:
    """Quản lý tập trung các phiên LiveStreamSession theo camera_id."""

    def __init__(self, max_sessions: int = MAX_LIVE_SESSIONS):
        self._sessions: Dict[str, LiveStreamSession] = {}
        self._lock = threading.Lock()
        self.max_sessions = max(1, int(max_sessions))

    def get_session(self, camera_id: str) -> Optional[LiveStreamSession]:
        with self._lock:
            return self._sessions.get(str(camera_id))

    def start_session(
        self,
        camera_id: str,
        source: Any,
        site_id: Optional[str] = None,
        model_path: Optional[str] = None,
        target_fps: float = 25.0,
        ai_fps: float = 15.0,
        roi_polygons: Optional[List[Any]] = None,
        confirmation_duration_s: float = 2.0,
        enable_no_helmet: bool = True,
        enable_no_vest: bool = True,
        enable_restricted_area: bool = True,
        dispatch_enabled: bool = True,
    ) -> LiveStreamSession:
        camera_id = _validate_safe_id(camera_id, "camera_id")
        with self._lock:
            existing = self._sessions.get(camera_id)
            if existing and existing.status in ("running", "initializing"):
                # Cập nhật cấu hình nếu cùng source
                if existing.source == source:
                    existing.update_config(
                        target_fps=target_fps,
                        ai_fps=ai_fps,
                        roi_polygons=roi_polygons,
                        enable_no_helmet=enable_no_helmet,
                        enable_no_vest=enable_no_vest,
                        enable_restricted_area=enable_restricted_area,
                        confirmation_duration_s=confirmation_duration_s,
                    )
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
                model_path=model_path,
                target_fps=target_fps,
                ai_fps=ai_fps,
                roi_polygons=roi_polygons,
                confirmation_duration_s=confirmation_duration_s,
                enable_no_helmet=enable_no_helmet,
                enable_no_vest=enable_no_vest,
                enable_restricted_area=enable_restricted_area,
                dispatch_enabled=dispatch_enabled,
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
                    "status": s.status,
                    "target_fps": s.target_fps,
                    "ai_fps": s.ai_fps,
                    "source": mask_source(s.source),
                    "metrics": s.metrics,
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
