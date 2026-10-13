"""
Entry point — smart-ppe-ai-service.
Pipeline hoàn chỉnh:
Video/RTSP -> Detection (YOLO) -> Tracking (ByteTrack) -> Association (PPE-Person)
-> Safety Rules -> Temporal Confirmation -> Deduplication
-> Evidence Snapshot (MinIO / Local) -> Contract Schema Validation
-> Dispatch (Kafka -> HTTP Fallback -> Offline Buffer).
"""

import argparse
import asyncio
from datetime import datetime, timezone
import logging
from pathlib import Path
import time
from typing import Optional

import cv2

from app import config
from app.constants import (
    CONTRACT_VERSION,
    LABEL_PERSON,
    LABEL_HARD_HAT,
    LABEL_SAFETY_VEST,
    VIOLATION_NO_HELMET,
    VIOLATION_NO_VEST,
    SEVERITY_HIGH,
    SEVERITY_MEDIUM,
)
from app.clients.backend_client import fetch_system_default_rules, SafetyRulesConfig
from app.detection.detector import Detector, DetectorError
from app.association.associator import associate
from app.rules.safety import evaluate as evaluate_safety
from app.rules.confirmation import ViolationConfirmation
from app.rules.deduplication import ViolationDeduplicator, ViolationEvent as DedupEvent
from app.evidence.snapshot import save_evidence
from app.video.video_reader import VideoReader, VideoSourceError
from app.video.visualizer import draw_detections
from app.schemas.violation_event import (
    ViolationEvent,
    Evidence,
    Detection as SchemaDetection,
    ViolationItem,
    BoundingBox,
    FrameResolution,
    EventSummary,
    EventMetadata,
)
from app.events import dispatch_event, flush_offline_buffer, flush_producer

import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
    force=True,
)
logger = logging.getLogger("smart-ppe-ai-service")


def _map_violation_type(raw_type: str) -> str:
    """Map internal violation string to contract constant."""
    t = raw_type.lower()
    if "helmet" in t:
        return VIOLATION_NO_HELMET
    if "vest" in t:
        return VIOLATION_NO_VEST
    return raw_type.upper()


def _build_violation_event(
    ev: DedupEvent,
    frame_w: int,
    frame_h: int,
    snapshot_url: str,
    all_detections: list,
    model_name: str,
) -> ViolationEvent:
    """Build a Contract 2.1.0-PPE-STABLE compliant ViolationEvent."""
    v_type = _map_violation_type(ev.violation_type)
    severity = SEVERITY_HIGH if v_type == VIOLATION_NO_HELMET else SEVERITY_MEDIUM
    desc = "Không đội mũ bảo hộ lao động" if v_type == VIOLATION_NO_HELMET else "Không mặc áo phản quang"

    # Find the target person detection
    target_person = next(
        (d for d in all_detections if d.track_id == ev.person_track_id and d.class_name == "person"),
        None
    )
    v_box = None
    if target_person:
        v_box = BoundingBox(
            x=float(target_person.bbox[0]),
            y=float(target_person.bbox[1]),
            width=float(target_person.bbox[2] - target_person.bbox[0]),
            height=float(target_person.bbox[3] - target_person.bbox[1]),
        )

    # Convert person detections to schema format
    schema_detections = []
    for d in all_detections:
        if d.track_id is None:
            continue
        cls_label = LABEL_PERSON if d.class_name == "person" else (
            LABEL_HARD_HAT if d.class_name == "helmet" else LABEL_SAFETY_VEST
        )
        schema_detections.append(
            SchemaDetection(
                classLabel=cls_label,
                confidence=round(float(d.confidence), 2),
                box=BoundingBox(
                    x=float(d.bbox[0]),
                    y=float(d.bbox[1]),
                    width=float(d.bbox[2] - d.bbox[0]),
                    height=float(d.bbox[3] - d.bbox[1]),
                ),
                trackId=int(d.track_id),
            )
        )

    v_item = ViolationItem(
        violationType=v_type,
        severity=severity,
        description=desc,
        confidence=round(float(target_person.confidence if target_person else 0.9), 2),
        box=v_box,
    )

    now_iso = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"

    return ViolationEvent(
        cameraCode=ev.camera_id,
        timestamp=now_iso,
        evidence=Evidence(
            snapshotUrl=snapshot_url,
            clipUrl=None,
            frameResolution=FrameResolution(width=frame_w, height=frame_h),
        ),
        detections=schema_detections,
        violations=[v_item],
        summary=EventSummary(
            totalPersons=len([d for d in schema_detections if d.classLabel == LABEL_PERSON]),
            totalViolations=1,
            highestSeverity=severity,
        ),
        metadata=EventMetadata(
            confirmedDurationMs=int(ev.duration_s * 1000),
            cooldownDurationMs=int(config.COOLDOWN_DURATION_S * 1000),
            model=model_name,
            contractVersion=CONTRACT_VERSION,
        ),
    )


def run_pipeline(
    source: Optional[str] = None,
    camera_id: str = "CAM-01",
    out_dir: str = "outputs",
    show_every: int = 50,
    rules: Optional[SafetyRulesConfig] = None,
) -> int:
    """Run the video intelligence pipeline for a single source stream/file."""
    actual_source = source or config.VIDEO_SOURCE
    confirmation_duration = rules.confirmation_s if rules else config.CONFIRMATION_DURATION_S
    cooldown_duration = rules.cooldown_s if rules else config.COOLDOWN_DURATION_S
    conf_threshold = rules.confidence if rules else config.CONFIDENCE_THRESHOLD

    logger.info("Initializing VideoReader for source: %s", actual_source)
    try:
        reader = VideoReader(actual_source)
    except VideoSourceError as err:
        logger.error("Failed to open video source: %s", err)
        return 1

    try:
        detector = Detector(
            model_path=config.YOLO_MODEL_PATH,
            conf=conf_threshold,
            device=config.DEVICE,
            imgsz=config.MODEL_IMGSZ,
        ).load()
    except DetectorError as err:
        logger.error("Failed to load YOLO detector: %s", err)
        reader.release()
        return 1

    logger.info("Detector loaded: %s | device=%s | conf=%.2f", detector.model_path, detector.device, detector.conf)
    if reader.size:
        logger.info("Video resolution: %dx%d | FPS: %.2f", reader.size[0], reader.size[1], reader.fps)

    writer = None
    out_path = None
    n_frames = 0
    t_detect = 0.0

    confirmation = ViolationConfirmation(duration_s=confirmation_duration)
    dedup = ViolationDeduplicator(cooldown_s=cooldown_duration)

    last_flush_t = time.time()

    with reader:
        for num, frame in reader:
            n_frames += 1
            t0 = time.time()
            detections = detector.detect(frame)
            t_detect += time.time() - t0

            associations = associate(detections)
            frame_h, frame_w = frame.shape[:2]
            person_states = evaluate_safety(detections, associations, frame_w, frame_h)

            t_frame = (num - 1) / (reader.fps if reader.fps > 0 else 25.0)
            confirmed = confirmation.update(t_frame, person_states)

            # Draw visual bboxes on output frame
            draw_detections(frame, detections)

            # Filter duplicates and produce final violations
            for ev in dedup.filter(confirmed, detections, t_frame):
                logger.warning(
                    "VIOLATION CONFIRMED: camera=%s, person=#%d, type=%s, duration=%.1fs",
                    ev.camera_id, ev.person_track_id, ev.violation_type, ev.duration_s,
                )

                # 1. Save evidence snapshot locally
                evidence_res = save_evidence(frame, detections, ev, config.EVIDENCE_DIR)
                snapshot_url = evidence_res.path

                # 2. Upload to MinIO if client is ready, otherwise use local path
                if evidence_res.status == "saved" and evidence_res.path:
                    try:
                        from app.storage.minio_client import upload_snapshot
                        with open(evidence_res.path, "rb") as f:
                            img_bytes = f.read()
                        snapshot_url = upload_snapshot(
                            image_bytes=img_bytes,
                            site_code=config.SITE_CODE,
                            camera_code=ev.camera_id,
                            track_id=ev.person_track_id,
                        )
                        logger.info("Uploaded snapshot to MinIO: %s", snapshot_url)
                    except Exception as m_err:
                        logger.debug("MinIO direct upload skipped (%s), using local reference: %s", m_err, snapshot_url)

                # 3. Build contract-compliant event
                model_name = Path(config.YOLO_MODEL_PATH).name
                contract_event = _build_violation_event(
                    ev=ev,
                    frame_w=frame_w,
                    frame_h=frame_h,
                    snapshot_url=snapshot_url or f"file:///{evidence_res.path}",
                    all_detections=detections,
                    model_name=model_name,
                )

                # 4. Dispatch via Kafka -> HTTP fallback -> Offline Buffer
                method = dispatch_event(contract_event)
                logger.info("Event %s dispatched via [%s]", ev.event_id, method)

            # Periodically flush offline buffer (every 10 seconds)
            if time.time() - last_flush_t > 10.0:
                last_flush_t = time.time()
                flushed = flush_offline_buffer(sync=True)
                if flushed > 0:
                    logger.info("Flushed %d pending offline events", flushed)

            # Optional output video recording
            if config.SAVE_OUTPUT_VIDEO:
                if writer is None:
                    out_dir_path = Path(out_dir)
                    out_dir_path.mkdir(parents=True, exist_ok=True)
                    src_name = Path(str(actual_source)).stem if actual_source is not None else "stream"
                    out_path = out_dir_path / f"{src_name}_det.mp4"
                    writer = cv2.VideoWriter(
                        str(out_path),
                        cv2.VideoWriter_fourcc(*"mp4v"),
                        reader.fps if reader.fps > 0 else 25.0,
                        (frame_w, frame_h),
                    )
                writer.write(frame)

            if num % show_every == 0:
                logger.info(
                    "Processed frame %d — avg detect latency: %.1f ms/frame",
                    num, (t_detect / n_frames * 1000.0) if n_frames > 0 else 0.0,
                )

    if writer:
        writer.release()
        logger.info("Saved annotated video to %s (%d frames)", out_path, n_frames)

    # Ensure all buffered Kafka events and offline records are flushed
    flush_producer()
    flush_offline_buffer(sync=True)

    return 0


async def async_main():
    parser = argparse.ArgumentParser(description="smart-ppe-ai-service — Video Intelligence Pipeline")
    parser.add_argument("--source", default=None, help="Video source (file path, RTSP URL, or webcam index)")
    parser.add_argument("--camera-id", default="CAM-01", help="Camera identifier code")
    parser.add_argument("--out-dir", default="outputs", help="Directory for annotated output videos")
    parser.add_argument("--show-every", type=int, default=50, help="Log progress every N frames")
    args = parser.parse_args()

    # Step 1: Dynamic safety rules retrieval from Backend
    logger.info("Fetching dynamic safety rules from Backend...")
    rules = await fetch_system_default_rules()
    logger.info(
        "Effective rules: confidence=%.2f, confirmation=%ds, cooldown=%ds",
        rules.confidence, rules.confirmation_s, rules.cooldown_s,
    )

    # Step 2: Run pipeline
    run_pipeline(
        source=args.source,
        camera_id=args.camera_id,
        out_dir=args.out_dir,
        show_every=args.show_every,
        rules=rules,
    )


def main():
    try:
        asyncio.run(async_main())
    except KeyboardInterrupt:
        logger.info("Pipeline stopped by user. Resources released gracefully.")
        flush_producer(timeout=2.0)
        return 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
