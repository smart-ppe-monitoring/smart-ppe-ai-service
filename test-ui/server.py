import asyncio
import sys
from pathlib import Path

# Đảm bảo thư mục gốc dự án luôn nằm trong sys.path khi chạy trực tiếp
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
"""Test UI — FastAPI server phục vụ giao diện test 2 khung.

Giao diện này TÁCH BIỆT với core pipeline: chỉ dùng lại các API công khai
của app/ (VideoReader, Detector, draw_detections), không sửa core.

Chạy (từ thư mục gốc video-intelligence):
    python -m test-ui.server
    → mở http://127.0.0.1:8000
"""
import json
import logging
import os
import re
import shutil
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
import time
import uuid
from pathlib import Path
from typing import Annotated, Any, Dict, List, Optional
from urllib.parse import urlparse

import cv2
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field, StringConstraints

from app import config
from app.live_engine import (
    MAX_ROI_POINTS,
    MAX_ROI_POLYGONS,
    LiveSessionLimitError,
    LiveStreamSession,
    live_engine_manager,
    preload_detector,
)
from app.video.video_reader import VideoReader, VideoSourceError, is_youtube_url
from app.detection.detector import Detector, DetectorError
from app.association.associator import associate
from app.rules.safety import evaluate as evaluate_safety
from app.rules.confirmation import ViolationConfirmation
from app.rules.deduplication import ViolationDeduplicator
from app.evidence.snapshot import save_evidence
from app.config import CONFIRMATION_DURATION_S
from app.video.visualizer import draw_detections
from app.video.scene_cut import SceneCutDetector

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"
MODELS_DIR = (BASE_DIR.parent / "models").resolve()
RESULT_DIR = BASE_DIR / "results"
UPLOAD_DIR.mkdir(exist_ok=True)
RESULT_DIR.mkdir(exist_ok=True)

VIDEO_EXT = {".mp4", ".mov", ".avi", ".mkv", ".m4v"}

TAGS_METADATA = [
    {
        "name": "Live Stream AI Engine",
        "description": "Quét trực tiếp video thời gian thực từ Webcam, MP4, RTSP, YouTube. Hỗ trợ điều phối FPS mượt mà, suy luận AI bất đồng bộ, phát hiện vùng cấm Polygon ROI, Scene Cut, và trích xuất bằng chứng Before/After.",
    },
    {
        "name": "Offline Video Processing",
        "description": "Tải lên file video, chạy toàn bộ tiến trình nhận diện AI offline, lưu log JSONL và xuất video H.264.",
    },
    {
        "name": "Model Management",
        "description": "Quản lý và liệt kê các mô hình trọng số YOLO (.pt, .onnx) khả dụng trong hệ thống.",
    },
]

JOBS = {}
JOBS_LOCK = threading.Lock()

# Bounded thread pool for video processing (max 2 concurrent jobs to prevent DoS)
MAX_CONCURRENT_JOBS = 2
JOB_EXECUTOR = ThreadPoolExecutor(max_workers=MAX_CONCURRENT_JOBS, thread_name_prefix="video-worker")

# Chống DNS rebinding: chỉ chấp nhận Host header của loopback (cấu hình qua TEST_UI_ALLOWED_HOSTS).
TEST_UI_ALLOWED_HOSTS = [
    h.strip() for h in os.getenv("TEST_UI_ALLOWED_HOSTS", "127.0.0.1,localhost").split(",") if h.strip()
]


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Quản lý vòng đời server: làm nóng trước YOLO model khi khởi động và dọn dẹp khi tắt."""
    if not os.getenv("PYTEST_CURRENT_TEST"):
        rec_model = MODELS_DIR / "yolo11s_joint_phase3.pt"
        target = str(rec_model) if rec_model.exists() else config.YOLO_MODEL_PATH
        threading.Thread(target=preload_detector, args=(target,), daemon=True, name="model-prewarm").start()
    yield
    live_engine_manager.stop_all()
    JOB_EXECUTOR.shutdown(wait=False)


app = FastAPI(
    title="Smart PPE Monitoring — AI Vision & Live Engine API",
    description=(
        "Hệ thống thị giác máy tính giám sát an toàn lao động (PPE) và xâm nhập vùng nguy hiểm.\n\n"
        "### Các tính năng chính:\n"
        "- **📡 Quét trực tiếp (Live Stream AI)**: Hỗ trợ webcam, YouTube, file MP4, và RTSP với tốc độ hiển thị 25 FPS không độ trễ.\n"
        "- **🚨 Phát hiện vi phạm an toàn**: Kiểm tra Không đội mũ (`NO_HELMET`), Không mặc áo (`NO_VEST`), và Xâm nhập vùng nguy hiểm (`RESTRICTED_AREA`).\n"
        "- **📸 Bằng chứng đối chứng Before / After**: Tự động chụp và lưu cặp ảnh đối chứng hợp lệ và ảnh vi phạm kèm khoanh vùng HUD.\n"
        "- **⚙️ Điều chỉnh nóng (Hot-update)**: Cho phép bật/tắt luật, đổi target FPS, AI FPS, Scene Cut ngay khi video đang chạy.\n"
    ),
    version="2.1.0",
    openapi_tags=TAGS_METADATA,
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)
logger = logging.getLogger("test_ui_server")
app.add_middleware(TrustedHostMiddleware, allowed_hosts=TEST_UI_ALLOWED_HOSTS)

JOB_ID_RE = re.compile(r"^[a-zA-Z0-9_-]+$")


def _validate_job_id(job_id: str) -> str:
    """Validate job_id format and ensure it cannot traverse outside RESULT_DIR."""
    if not job_id or not JOB_ID_RE.match(job_id):
        raise HTTPException(400, "Mã job không hợp lệ")
    resolved_dir = (RESULT_DIR / job_id).resolve()
    if not resolved_dir.is_relative_to(RESULT_DIR):
        raise HTTPException(400, "Đường dẫn job không hợp lệ")
    return job_id


def _job_paths(job_id: str):
    _validate_job_id(job_id)
    job_dir = (RESULT_DIR / job_id).resolve()
    return {
        "dir": job_dir,
        "video": job_dir / "annotated.mp4",
        "log": job_dir / "log.jsonl",
        "meta": job_dir / "meta.json",
    }


def _convert_to_h264(video_path: Path):
    """Chuyển đổi video sang chuẩn H.264 (yuv420p) để phát trực tiếp trên trình duyệt."""
    if not video_path.exists():
        return
    ffmpeg_exe = None
    try:
        import imageio_ffmpeg
        ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        pass
    if not ffmpeg_exe:
        ffmpeg_exe = shutil.which("ffmpeg")
    if not ffmpeg_exe:
        return

    tmp_path = video_path.with_name(f"{video_path.stem}_h264.mp4")
    cmd = [
        ffmpeg_exe, "-y",
        "-i", str(video_path),
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        str(tmp_path),
    ]
    try:
        res = subprocess.run(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=120,
        )
        if res.returncode == 0 and tmp_path.exists():
            # SEC-23: Handle Windows file lock collision during atomic replace
            for attempt in range(5):
                try:
                    if os.name == "nt" and video_path.exists():
                        try:
                            video_path.unlink(missing_ok=True)
                        except PermissionError:
                            pass
                    tmp_path.replace(video_path)
                    break
                except PermissionError:
                    if attempt < 4:
                        time.sleep(0.5)
                    else:
                        logger.warning("Could not replace %s with %s due to Windows file lock", video_path, tmp_path)
    except subprocess.TimeoutExpired:
        logger.error("FFmpeg transcoding timed out after 120s for %s", video_path)
        if tmp_path.exists():
            try:
                tmp_path.unlink(missing_ok=True)
            except Exception:
                pass
    except Exception as e:
        logger.error("FFmpeg transcoding failed for %s: %s", video_path, e)


def _get_job(job_id: str):
    _validate_job_id(job_id)
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        if job is None:
            meta_path = _job_paths(job_id)["meta"]
            if meta_path.exists():
                try:
                    meta = json.loads(meta_path.read_text(encoding="utf-8"))
                    job = {"status": "done", "progress": 1.0, **meta}
                    JOBS[job_id] = job
                except Exception:
                    pass
    if job is None:
        raise HTTPException(404, "Không tìm thấy job")
    return job


def process_video(job_id, video_path, model_path=None):
    """Xử lý toàn bộ video: detect + tracking + vẽ box + ghi log JSONL. Chạy trong thread riêng."""
    job = _get_job(job_id)
    paths = _job_paths(job_id)
    if model_path:
        cand = (MODELS_DIR / Path(model_path).name).resolve()
        if cand.exists() and cand.is_relative_to(MODELS_DIR) and cand.suffix.lower() in (".pt", ".onnx"):
            model_path = str(cand)
        else:
            model_path = None
    try:
        with VideoReader(str(video_path)) as reader:
            detector = Detector(model_path=model_path).load()
            w, h = reader.size
            total = reader.frame_count
            fps = reader.fps
            with JOBS_LOCK:
                job.update(status="processing", progress=0.0,
                           total_frames=total, fps=fps,
                           size=[w, h], model=detector.model_path)

            ev_dir = paths["dir"] / "evidences"
            if ev_dir.exists():
                shutil.rmtree(ev_dir, ignore_errors=True)
            ev_dir.mkdir(parents=True, exist_ok=True)

            writer = cv2.VideoWriter(str(paths["video"]),
                                     cv2.VideoWriter_fourcc(*"mp4v"),
                                     reader.fps, (w, h))
            prev_ids = set()
            known_classes = {}  # track_id -> class_name (nhớ từ frame trước để ghi sự kiện "biến mất")
            confirmation = ViolationConfirmation()
            dedup = ViolationDeduplicator()
            cut_detector = SceneCutDetector()
            with open(paths["log"], "w", encoding="utf-8") as logf:
                for num, frame in reader:
                    if cut_detector.check(frame):
                        detector.reset_tracker()
                        confirmation.reset()
                    detections = detector.detect(frame)
                    associations = associate(detections)
                    person_states = evaluate_safety(detections, associations, w, h)
                    t_frame = round((num - 1) / reader.fps, 3)
                    confirmations = confirmation.update(t_frame, person_states, frame=frame, detections=detections)
                    violation_events = dedup.filter(confirmations, detections, t_frame)
                    draw_detections(frame, detections, person_states=person_states)
                    evidence_results = [save_evidence(frame, detections, ev, paths["dir"] / "evidences")
                                        for ev in violation_events]

                    ids = {d.track_id for d in detections if d.track_id is not None}
                    current_classes = {d.track_id: d.class_name for d in detections if d.track_id is not None}
                    events = [f"{current_classes.get(tid, 'đối tượng')} #{tid} xuất hiện"
                              for tid in sorted(ids - prev_ids)]
                    events += [f"{known_classes.get(tid, 'đối tượng')} #{tid} biến mất"
                               for tid in sorted(prev_ids - ids)]
                    known_classes.update(current_classes)
                    prev_ids = ids

                    counts = {"person": 0, "helmet": 0, "safety_vest": 0}
                    for d in detections:
                        counts[d.class_name] = counts.get(d.class_name, 0) + 1
                    entry = {
                        "frame": num,
                        "t": t_frame,
                        "detections": [
                            {
                                "class_name": d.class_name,
                                "track_id": d.track_id,
                                "confidence": round(d.confidence, 3),
                                "bbox": list(d.bbox),
                                "w": d.bbox[2] - d.bbox[0],
                                "h": d.bbox[3] - d.bbox[1],
                            }
                            for d in detections
                        ],
                        "summary": {**counts, "tracks": len(ids)},
                        "events": events,
                        "associations": [
                            {
                                "person": a.person_track_id,
                                "helmet": a.helmet_track_id,
                                "vest": a.vest_track_id,
                                "status": a.status,
                            }
                            for a in associations
                        ],
                        "persons": [
                            {
                                "track_id": s.person_track_id,
                                "status": s.status,
                                "violations": s.violations,
                                "reason": s.reason,
                            }
                            for s in person_states
                        ],
                        "confirmations": [
                            {
                                "person": c.person_track_id,
                                "violation_type": c.violation_type,
                                "start_t": c.start_t,
                                "confirmed_t": c.confirmed_t,
                                "duration_s": c.duration_s,
                            }
                            for c in confirmations
                        ],
                        "violations": [
                            {
                                "event_id": ev.event_id,
                                "camera_id": ev.camera_id,
                                "person": ev.person_track_id,
                                "canonical_person": ev.canonical_person_id,
                                "violation_type": ev.violation_type,
                                "start_t": ev.start_t,
                                "confirmed_t": ev.confirmed_t,
                                "duration_s": ev.duration_s,
                            }
                            for ev in violation_events
                        ],
                        "evidence": [
                            {
                                "event_id": r.event_id,
                                "status": r.status,
                                "path": r.path,
                                "reference_path": getattr(r, "reference_path", ""),
                                "has_reference": bool(getattr(r, "reference_path", "")),
                                "error": r.error,
                            }
                            for r in evidence_results
                        ],
                    }
                    logf.write(json.dumps(entry, ensure_ascii=False) + "\n")

                    writer.write(frame)

                    with JOBS_LOCK:
                        job["current_frame"] = num
                        job["progress"] = num / total if total else 0.0
                    if num % 100 == 0 or num == total:
                        print(f"  [{job.get('source', job_id)}] Frame {num}/{total} ({(num / total * 100 if total else 0):.1f}%)", flush=True)
            writer.release()
            _convert_to_h264(paths["video"])

        meta = {
            "job_id": job_id,
            "source": job.get("source") or Path(video_path).name,
            "fps": fps,
            "size": [w, h],
            "total_frames": total,
            "model": detector.model_path,
            "confirmation_duration_s": CONFIRMATION_DURATION_S,
            "status": "done",
            "processed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        paths["meta"].write_text(json.dumps(meta, ensure_ascii=False, indent=2))
        with JOBS_LOCK:
            job.update(status="done", progress=1.0)
    except Exception as e:
        with JOBS_LOCK:
            job.update(status="error", error=f"{type(e).__name__}: {e}")


MAX_UPLOAD_SIZE_BYTES = 500 * 1024 * 1024  # 500 MB limit to prevent Memory Exhaustion DoS
CHUNK_SIZE_BYTES = 1024 * 1024  # 1 MB chunks streaming


def _is_valid_video_magic(header: bytes) -> bool:
    """Validate initial header bytes against known video container signatures (MP4, MOV, AVI, MKV)."""
    if len(header) < 12:
        return False
    # MP4 / M4V / MOV (ISOBMFF / QuickTime boxes: ftyp, moov, mdat, wide, skip, free)
    if header[4:8] in (b"ftyp", b"moov", b"mdat", b"wide", b"skip", b"free"):
        return True
    # QuickTime box lengths with qt brand at offset 4
    if header[:4] in (b"\x00\x00\x00\x14", b"\x00\x00\x00\x20"):
        return True
    # AVI (RIFF ... AVI  / AVIX)
    if header[:4] == b"RIFF" and header[8:12] in (b"AVI ", b"AVIX"):
        return True
    # Matroska MKV / WebM (EBML header \x1a\x45\xdf\xa3)
    if header[:4] == b"\x1a\x45\xdf\xa3":
        return True
    return False


@app.post("/api/upload", tags=["Offline Video Processing"], summary="Tải video lên máy chủ")
async def upload(file: UploadFile = File(...)):
    suffix = Path(file.filename or "video.mp4").suffix.lower()
    if suffix not in VIDEO_EXT:
        raise HTTPException(400, f"File không phải video. Hỗ trợ: {', '.join(sorted(VIDEO_EXT))}")

    # SEC-24: Validate file magic header before saving to disk
    header = await file.read(64)
    await file.seek(0)
    if not _is_valid_video_magic(header):
        raise HTTPException(400, "Định dạng nội dung file không hợp lệ (không khớp signature video)")

    job_id = uuid.uuid4().hex[:12]
    dest = (UPLOAD_DIR / f"{job_id}{suffix}").resolve()
    if not dest.is_relative_to(UPLOAD_DIR):
        raise HTTPException(400, "Đường dẫn file không hợp lệ")

    total_size = 0
    try:
        with open(dest, "wb") as f_out:
            while chunk := await file.read(CHUNK_SIZE_BYTES):
                total_size += len(chunk)
                if total_size > MAX_UPLOAD_SIZE_BYTES:
                    raise HTTPException(
                        413,
                        f"Dung lượng file vượt quá giới hạn tối đa cho phép ({MAX_UPLOAD_SIZE_BYTES // (1024 * 1024)}MB)."
                    )
                f_out.write(chunk)
    except Exception:
        if dest.exists():
            dest.unlink(missing_ok=True)
        raise

    _job_paths(job_id)["dir"].mkdir(parents=True, exist_ok=True)
    with JOBS_LOCK:
        JOBS[job_id] = {"status": "uploaded", "progress": 0.0, "source": file.filename}
    return {"job_id": job_id}


@app.post("/api/jobs/{job_id}/start", tags=["Offline Video Processing"], summary="Khởi chạy tiến trình nhận diện AI offline")
def start(job_id: str, model: str = None):
    if not re.match(r"^[a-zA-Z0-9_-]+$", job_id):
        raise HTTPException(400, "Mã job không hợp lệ")
    if model:
        model_cand = (MODELS_DIR / Path(model).name).resolve()
        if not model_cand.exists() or not model_cand.is_relative_to(MODELS_DIR) or model_cand.suffix.lower() not in (".pt", ".onnx"):
            raise HTTPException(400, f"Mô hình không hợp lệ hoặc không tồn tại trong thư mục models: {model}")
        model = str(model_cand)
    job = _get_job(job_id)
    with JOBS_LOCK:
        if job.get("status") == "processing":
            return {"status": "processing"}
        job["status"] = "processing"
    src = next(UPLOAD_DIR.glob(f"{job_id}.*"), None)
    if src is None:
        with JOBS_LOCK:
            job["status"] = "error"
            job["error"] = "Không tìm thấy file upload"
        raise HTTPException(404, "Không tìm thấy file upload")
    JOB_EXECUTOR.submit(process_video, job_id, src, model)
    return {"status": "processing"}


@app.get("/api/jobs/{job_id}/status", tags=["Offline Video Processing"], summary="Xem trạng thái & tiến độ job offline")
def status(job_id: str):
    return _get_job(job_id)


@app.get("/api/jobs/{job_id}/log", tags=["Offline Video Processing"], summary="Tải nhật ký kết quả nhận diện JSONL")
def get_log(job_id: str):
    p = _job_paths(job_id)["log"]
    if not p.exists():
        raise HTTPException(404, "Chưa có log — job chưa chạy hoặc đang xử lý")
    return FileResponse(p, media_type="application/x-ndjson")


@app.get("/api/jobs/{job_id}/video", tags=["Offline Video Processing"], summary="Xem video kết quả đã vẽ nhãn (H.264 MP4)")
def get_video(job_id: str):
    p = _job_paths(job_id)["video"]
    if not p.exists():
        raise HTTPException(404, "Chưa có video — job chưa xong")
    return FileResponse(p, media_type="video/mp4")


EVIDENCE_FILE_RE = re.compile(r"^[0-9a-f]{12}(?:_ref)?\.jpg$")  # event_id hoặc event_id_ref


@app.get("/api/jobs/{job_id}/evidence/{filename}", tags=["Offline Video Processing"], summary="Lấy ảnh bằng chứng vi phạm offline")
def get_evidence(job_id: str, filename: str):
    if not EVIDENCE_FILE_RE.match(filename):
        raise HTTPException(400, "Tên file không hợp lệ")
    p = _job_paths(job_id)["dir"] / "evidences" / filename
    if not p.exists():
        raise HTTPException(404, "Không tìm thấy ảnh bằng chứng")
    return FileResponse(p, media_type="image/jpeg")


@app.get("/api/jobs/{job_id}/meta", tags=["Offline Video Processing"], summary="Lấy thông tin cấu hình và thống kê job")
def get_meta(job_id: str):
    p = _job_paths(job_id)["meta"]
    if not p.exists():
        raise HTTPException(404, "Chưa có meta — job chưa xong")
    return JSONResponse(json.loads(p.read_text(encoding="utf-8")))


def _scan_jobs_from_disk() -> list[dict]:
    jobs = []
    if RESULT_DIR.exists():
        for d in RESULT_DIR.iterdir():
            if d.is_dir():
                meta_p = d / "meta.json"
                if meta_p.exists():
                    try:
                        jobs.append(json.loads(meta_p.read_text(encoding="utf-8")))
                    except Exception:
                        pass
    jobs.sort(key=lambda x: x.get("processed_at", ""), reverse=True)
    return jobs


@app.get("/api/jobs", tags=["Offline Video Processing"], summary="Liệt kê danh sách video offline đã xử lý")
async def list_jobs():
    """SEC-14: Offload blocking filesystem disk traversal and JSON parsing to worker thread."""
    return await asyncio.to_thread(_scan_jobs_from_disk)


def _scan_models_from_disk() -> list[dict]:
    models_dir = BASE_DIR.parent / "models"
    available = []
    if models_dir.exists():
        for p in models_dir.glob("*.pt"):
            if p.stat().st_size > 1000:
                available.append({
                    "name": p.stem,
                    "path": f"models/{p.name}",
                    "size_mb": round(p.stat().st_size / (1024 * 1024), 1),
                    "is_default": "yolo11m" in p.name
                })
    available.sort(key=lambda x: (not x["is_default"], x["name"]))
    return available


@app.get("/api/models", tags=["Model Management"], summary="Liệt kê các mô hình YOLO (.pt) khả dụng")
async def list_models():
    """SEC-14: Offload model weight file scanning to worker thread."""
    return await asyncio.to_thread(_scan_models_from_disk)


def _scan_uploaded_videos() -> list[dict]:
    """Quét các file video MP4/MKV hợp lệ trong thư mục uploads/ sẵn sàng để quét trực tiếp."""
    videos = []
    if UPLOAD_DIR.exists():
        for p in sorted(UPLOAD_DIR.glob("*")):
            if p.is_file() and p.suffix.lower() in VIDEO_EXT and p.stat().st_size > 10000:
                videos.append({
                    "filename": p.name,
                    "stem": p.stem,
                    "size_mb": round(p.stat().st_size / (1024 * 1024), 2),
                })
    return videos


@app.get("/api/live/videos", tags=["Live Stream AI Engine"], summary="Danh sách video trong uploads/ sẵn sàng phát trực tiếp")
async def list_live_videos():
    """Liệt kê các video MP4/MKV trong uploads/ sẵn sàng để phát trực tiếp qua Live Engine."""
    return await asyncio.to_thread(_scan_uploaded_videos)


# ─── Live Stream AI Engine Endpoints (D08.pdf Mục 3.3, 2.6 & 6.3) ───────────────

# Ràng buộc biên HTTP cho dữ liệu live (chống DoS / dữ liệu rác)
LIVE_SOURCE_MAX_LEN = 2048
LIVE_MAX_WEBCAM_INDEX = 9
LIVE_ALLOWED_STREAM_SCHEMES = {"rtsp", "rtsps"}
# Allowlist host RTSP tùy chọn (phân tách bằng dấu phẩy). Rỗng = cho phép mọi host RTSP.
LIVE_ALLOWED_RTSP_HOSTS = {
    h.strip().lower() for h in os.getenv("LIVE_ALLOWED_RTSP_HOSTS", "").split(",") if h.strip()
}
# Mặc định Test-UI KHÔNG upload MinIO / bắn sự kiện tới Incident Service.
LIVE_DISPATCH_ENABLED = os.getenv("TEST_UI_LIVE_DISPATCH", "false").strip().lower() in ("1", "true", "yes")

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


def _resolve_live_source(raw: str):
    """Chỉ chấp nhận nguồn live thuộc allowlist, chặn đọc file tùy ý / SSRF qua FFmpeg.

    Hợp lệ:
      - Chỉ số webcam: "0".."9"
      - job_id của video đã upload (vd: "a1b2c3d4e5f6") hoặc tên file upload ("a1b2c3d4e5f6.mp4")
      - URL YouTube thuộc allowlist host (resolve qua yt-dlp)
      - URL rtsp:// / rtsps:// (tùy chọn giới hạn host qua LIVE_ALLOWED_RTSP_HOSTS)
    """
    src = raw.strip()
    if any(ord(c) < 0x20 for c in src):
        raise HTTPException(400, "Nguồn video chứa ký tự không hợp lệ")

    # 1. Webcam index
    if src.isdigit():
        idx = int(src)
        if idx > LIVE_MAX_WEBCAM_INDEX:
            raise HTTPException(400, "Chỉ số webcam không hợp lệ")
        return idx

    # 2. Video đã upload (theo job_id hoặc tên file) — không dùng glob với dữ liệu client
    stem, dot, ext = src.partition(".")
    if JOB_ID_RE.match(stem) and (not dot or f".{ext.lower()}" in VIDEO_EXT):
        candidates = [UPLOAD_DIR / src] if dot else [UPLOAD_DIR / f"{stem}{e}" for e in sorted(VIDEO_EXT)]
        for cand in candidates:
            resolved = cand.resolve()
            if resolved.is_relative_to(UPLOAD_DIR) and resolved.is_file():
                return str(resolved)

    # 3. YouTube (allowlist host chặt chẽ)
    if is_youtube_url(src):
        return src

    # 4. RTSP camera
    try:
        parsed = urlparse(src)
    except ValueError:
        parsed = None
    if parsed and parsed.scheme.lower() in LIVE_ALLOWED_STREAM_SCHEMES and parsed.hostname:
        if LIVE_ALLOWED_RTSP_HOSTS and parsed.hostname.lower() not in LIVE_ALLOWED_RTSP_HOSTS:
            raise HTTPException(400, "Host RTSP không nằm trong danh sách cho phép")
        return src

    raise HTTPException(
        400,
        "Nguồn video không được hỗ trợ. Chỉ chấp nhận: chỉ số webcam, job_id video đã upload, "
        "URL YouTube hoặc URL rtsp(s)://",
    )


@app.post("/api/live/start", tags=["Live Stream AI Engine"], summary="Khởi chạy phiên quét trực tiếp (Live Stream AI)")
def start_live(req: LiveStartRequest):
    """Khởi chạy phiên quét trực tiếp (Live Stream AI) cho camera."""
    source_val = _resolve_live_source(req.source)

    model_path = None
    if req.model:
        model_cand = (MODELS_DIR / Path(req.model).name).resolve()
        if not model_cand.exists() or not model_cand.is_relative_to(MODELS_DIR) or model_cand.suffix.lower() not in (".pt", ".onnx"):
            raise HTTPException(400, f"Mô hình không hợp lệ hoặc không tồn tại: {req.model}")
        model_path = str(model_cand)

    try:
        session = live_engine_manager.start_session(
            camera_id=req.camera_id,
            source=source_val,
            model_path=model_path,
            target_fps=req.target_fps,
            ai_fps=req.ai_fps,
            roi_polygons=req.roi_polygons,
            confirmation_duration_s=req.confirmation_duration_s,
            enable_no_helmet=req.enable_no_helmet,
            enable_no_vest=req.enable_no_vest,
            enable_restricted_area=req.enable_restricted_area,
            enable_scene_cut=req.enable_scene_cut,
            dispatch_enabled=LIVE_DISPATCH_ENABLED,
        )
    except LiveSessionLimitError as e:
        raise HTTPException(429, str(e))
    except ValueError as e:
        raise HTTPException(400, str(e))
    return {
        "status": "started",
        "camera_id": session.camera_id,
        "target_fps": session.target_fps,
        "ai_fps": session.ai_fps,
    }


@app.get("/api/live/{camera_id}/stream", tags=["Live Stream AI Engine"], summary="Phát luồng MJPEG trực tiếp (quét tới đâu phát tới đó)")
def get_live_stream(camera_id: str):
    """Phát luồng MJPEG trực tiếp (quét tới đâu phát tới đó) qua multipart/x-mixed-replace."""
    session = live_engine_manager.get_session(camera_id)
    if not session:
        raise HTTPException(404, f"Không tìm thấy phiên phát cho camera {camera_id}")
    return StreamingResponse(
        session.generate_mjpeg(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@app.get("/api/live/{camera_id}/stats", tags=["Live Stream AI Engine"], summary="Lấy số liệu thời gian thực (FPS, latency, vi phạm)")
def get_live_stats(camera_id: str):
    """Lấy số liệu thời gian thực: FPS, Inference Latency, Alert Latency, phát hiện hiện tại."""
    session = live_engine_manager.get_session(camera_id)
    if not session:
        raise HTTPException(404, f"Không tìm thấy phiên phát cho camera {camera_id}")
    return {
        "camera_id": session.camera_id,
        "status": session.status,
        "error": session.error_message,
        "metrics": session.metrics,
        "recent_violations": session.recent_violations,
    }


@app.patch("/api/live/{camera_id}/config", tags=["Live Stream AI Engine"], summary="Điều chỉnh cấu hình nóng (hot-update) khi đang chạy")
def update_live_config(camera_id: str, req: LiveConfigUpdateRequest):
    """Điều chỉnh nóng (hot-update) FPS, Sampling rate hoặc ROI ngay khi video đang chạy."""
    session = live_engine_manager.get_session(camera_id)
    if not session:
        raise HTTPException(404, f"Không tìm thấy phiên phát cho camera {camera_id}")
    session.update_config(
        target_fps=req.target_fps,
        ai_fps=req.ai_fps,
        roi_polygons=req.roi_polygons,
        enable_no_helmet=req.enable_no_helmet,
        enable_no_vest=req.enable_no_vest,
        enable_restricted_area=req.enable_restricted_area,
        enable_scene_cut=req.enable_scene_cut,
        confirmation_duration_s=req.confirmation_duration_s,
    )
    return {
        "status": "updated",
        "camera_id": camera_id,
        "target_fps": session.target_fps,
        "ai_fps": session.ai_fps,
    }


@app.post("/api/live/{camera_id}/stop", tags=["Live Stream AI Engine"], summary="Dừng phiên phát trực tiếp của camera")
def stop_live_camera(camera_id: str):
    """Dừng phiên live của camera."""
    stopped = live_engine_manager.stop_session(camera_id)
    if not stopped:
        raise HTTPException(404, f"Không tìm thấy phiên phát cho camera {camera_id}")
    return {"status": "stopped", "camera_id": camera_id}


@app.get("/api/live/sessions", tags=["Live Stream AI Engine"], summary="Liệt kê danh sách các camera đang phát trực tiếp")
def list_live_sessions():
    """Liệt kê toàn bộ các camera đang chạy phiên live stream."""
    return live_engine_manager.list_sessions()


@app.get("/api/live/evidence/{filename}", tags=["Live Stream AI Engine"], summary="Lấy ảnh bằng chứng vi phạm trực tiếp ({event_id}.jpg / _ref.jpg)")
def get_live_evidence(filename: str):
    """Phục vụ ảnh vi phạm ({event_id}.jpg) và ảnh đối chứng ({event_id}_ref.jpg) của luồng Live Stream."""
    if not EVIDENCE_FILE_RE.match(filename):
        raise HTTPException(400, "Tên file bằng chứng không hợp lệ")
    ev_root = Path(config.EVIDENCE_DIR).resolve()
    target = (ev_root / filename).resolve()
    if not target.is_relative_to(ev_root) or not target.is_file():
        raise HTTPException(404, "Không tìm thấy ảnh bằng chứng trực tiếp")
    return FileResponse(target, media_type="image/jpeg")


app.mount("/", StaticFiles(directory=BASE_DIR / "static", html=True), name="static")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
