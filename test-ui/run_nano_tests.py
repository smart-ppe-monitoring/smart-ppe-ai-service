"""Run evaluation for YOLO11n and YOLOv8n on representative test videos.
Complements the existing benchmarks to provide complete 5-model evaluation:
YOLO11n vs YOLO11s vs YOLO11m vs YOLOv8n vs YOLOv8s.
"""
import importlib
import json
import os
import sys
import time
import uuid
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

server = importlib.import_module("test-ui.server")
process_video = server.process_video
JOBS = server.JOBS
JOBS_LOCK = server.JOBS_LOCK
_job_paths = server._job_paths

UPLOADS_DIR = ROOT_DIR / "test-ui" / "uploads"
MODELS_DIR = ROOT_DIR / "models"

TESTS = [
    ("video2.mp4", "yolo11n-best.pt"),
    ("video2.mp4", "yolov8n-best.pt"),
    ("video7.mp4", "yolo11n-best.pt"),
    ("video7.mp4", "yolov8n-best.pt"),
    ("video8.mp4", "yolov8n-best.pt"),
]


def run_test(video_name, model_file):
    video_path = UPLOADS_DIR / video_name
    model_path = MODELS_DIR / model_file
    model_name = model_path.stem
    label = f"{video_name} [{model_name}]"
    job_id = uuid.uuid4().hex[:12]
    paths = _job_paths(job_id)
    paths["dir"].mkdir(parents=True, exist_ok=True)

    with JOBS_LOCK:
        JOBS[job_id] = {
            "status": "processing",
            "progress": 0.0,
            "source": f"{video_name} [{model_name}]",
        }

    t0 = time.time()
    try:
        process_video(job_id, str(video_path), model_path=str(model_path))
    except Exception as e:
        print(f"❌ Lỗi {label}: {e}", flush=True)
        return None
    elapsed = round(time.time() - t0, 2)

    log_file = paths["log"]
    meta_file = paths["meta"]
    meta = json.loads(meta_file.read_text(encoding="utf-8")) if meta_file.exists() else {}

    total_person = 0
    total_helmet = 0
    total_vest = 0
    unique_tracks = set()
    violations = []

    if log_file.exists():
        with open(log_file, "r", encoding="utf-8") as lf:
            for line in lf:
                if not line.strip():
                    continue
                entry = json.loads(line)
                summary = entry.get("summary", {})
                total_person += summary.get("person", 0)
                total_helmet += summary.get("helmet", 0)
                total_vest += summary.get("safety_vest", 0)
                for d in entry.get("detections", []):
                    if d.get("track_id") is not None:
                        unique_tracks.add(d["track_id"])
                for v in entry.get("violations", []):
                    violations.append(v)

    total_frames = meta.get("total_frames", 0)
    fps_proc = round(total_frames / elapsed, 1) if elapsed > 0 and total_frames > 0 else 0

    res = {
        "job_id": job_id,
        "video": video_name,
        "model": model_name,
        "total_frames": total_frames,
        "elapsed_s": elapsed,
        "fps_processing": fps_proc,
        "person_detections": total_person,
        "helmet_detections": total_helmet,
        "vest_detections": total_vest,
        "unique_tracks": len(unique_tracks),
        "violations_count": len(violations),
        "violations": violations,
    }
    print(f"✔ [{job_id}] {label}: {total_frames} frames trong {elapsed}s ({fps_proc} FPS) | "
          f"Tracks: {len(unique_tracks)} | Vi phạm: {len(violations)}", flush=True)
    return res


def main():
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        pass

    print("=" * 70, flush=True)
    print("CHẠY THỬ NGHIỆM ĐỐI CHIẾU NANO MODELS (YOLO11n & YOLOv8n)", flush=True)
    print("=" * 70, flush=True)

    results = []
    for vname, mfile in TESTS:
        r = run_test(vname, mfile)
        if r:
            results.append(r)

    out_file = ROOT_DIR / "test-ui" / "nano_test_results.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print()
    print(f"✅ Hoàn tất kiểm thử Nano models! Đã lưu tại: {out_file}", flush=True)


if __name__ == "__main__":
    main()
