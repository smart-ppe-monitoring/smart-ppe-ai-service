"""Batch test & comparison script for smart-ppe-ai-service test-ui.

Audits:
1. D:\\Python\\Projects\\DATN\\Train_results\\1_Model_Weights (verifies placeholder issue)
2. D:\\Python\\Projects\\DATN\\drive-download-20261018T154650Z-1-001 (valid trained weights)
3. Runs full AI pipeline on all videos in test-ui/uploads/
4. Outputs comprehensive comparison report.
"""
import glob
import importlib
import json
import os
import sys
import time
import uuid
from pathlib import Path

# Ensure root directory is on PYTHONPATH
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from ultralytics import YOLO

# Import test-ui server modules
server = importlib.import_module("test-ui.server")
process_video = server.process_video
JOBS = server.JOBS
JOBS_LOCK = server.JOBS_LOCK
_job_paths = server._job_paths

TRAIN_RESULTS_DIR = Path(r"D:\Python\Projects\DATN\Train_results\1_Model_Weights")
DRIVE_DOWNLOAD_DIR = Path(r"D:\Python\Projects\DATN\drive-download-20261018T154650Z-1-001")
UPLOADS_DIR = ROOT_DIR / "test-ui" / "uploads"
RESULTS_DIR = ROOT_DIR / "test-ui" / "results"

MODELS_MAP = {
    "yolo11s": ROOT_DIR / "models" / "yolo11s-best.pt",
    "yolo11m": ROOT_DIR / "models" / "yolo11m-best.pt",
    "yolov8s": ROOT_DIR / "models" / "yolov8s-best.pt",
    "yolo11n": ROOT_DIR / "models" / "yolo11n-best.pt",
}


def audit_train_results_models():
    print("=" * 70)
    print("PHẦN 1: KIỂM TRA MODEL TRƯỚC ĐÓ TẠI Train_results\\1_Model_Weights")
    print("=" * 70)
    results = {}
    if not TRAIN_RESULTS_DIR.exists():
        print(f"❌ Không tìm thấy thư mục: {TRAIN_RESULTS_DIR}")
        return results

    for f in sorted(TRAIN_RESULTS_DIR.iterdir()):
        if f.suffix in [".pt", ".onnx"]:
            size = f.stat().st_size
            sample_content = ""
            try:
                sample_content = f.read_text(encoding="utf-8", errors="ignore").strip()
            except Exception:
                pass

            if size < 1000:
                load_error = f"Kích thước chỉ {size} bytes, nội dung: '{sample_content}'. Đây là placeholder text, không chứa weights."
                status = "LỖI (File placeholder, không phải weights)"
            else:
                load_error = None
                try:
                    m = YOLO(str(f))
                    status = "Hợp lệ"
                except Exception as e:
                    load_error = f"{type(e).__name__}: {e}"
                    status = "LỖI (Không thể nạp)"

            results[f.name] = {
                "size_bytes": size,
                "status": status,
                "error": load_error,
                "sample_content": sample_content,
            }
            print(f"- File: {f.name} ({size} bytes)", flush=True)
            print(f"  Trạng thái: {status}", flush=True)
            print(f"  Nội dung file: '{sample_content}'", flush=True)
            if load_error:
                print(f"  Chi tiết lỗi khi nạp vào YOLO: {load_error}", flush=True)
            print(flush=True)
    return results


def audit_drive_download_models():
    print("=" * 70)
    print("PHẦN 2: KIỂM TRA MODEL HUẤN LUYỆN TẠI drive-download-20261018T154650Z-1-001")
    print("=" * 70)
    results = {}
    if not DRIVE_DOWNLOAD_DIR.exists():
        print(f"❌ Không tìm thấy thư mục: {DRIVE_DOWNLOAD_DIR}")
        return results

    for pt_file in sorted(DRIVE_DOWNLOAD_DIR.glob("**/*.pt")):
        size_mb = round(pt_file.stat().st_size / (1024 * 1024), 2)
        rel_path = pt_file.relative_to(DRIVE_DOWNLOAD_DIR)
        try:
            m = YOLO(str(pt_file))
            names = m.names
            num_params = sum(p.numel() for p in m.model.parameters()) if hasattr(m, "model") and m.model else 0
            results[str(rel_path)] = {
                "size_mb": size_mb,
                "status": "Hợp lệ (fine-tuned)",
                "names": names,
                "params": num_params,
            }
            print(f"✔ {rel_path} ({size_mb} MB) — {num_params:,} tham số — Lớp: {names}")
        except Exception as e:
            results[str(rel_path)] = {
                "size_mb": size_mb,
                "status": "Lỗi",
                "error": str(e),
            }
            print(f"❌ {rel_path} ({size_mb} MB) — Lỗi: {e}")
    print()
    return results


def run_pipeline_job(video_path: Path, model_path: Path, tag: str = None):
    video_name = video_path.name
    model_name = model_path.stem
    label = f"{video_name} ({model_name})"
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
        print(f"❌ Lỗi khi xử lý {label}: {e}")
        return None
    elapsed = round(time.time() - t0, 2)

    # Read log entries
    log_file = paths["log"]
    meta_file = paths["meta"]
    meta = json.loads(meta_file.read_text(encoding="utf-8")) if meta_file.exists() else {}

    total_person_detections = 0
    total_helmet_detections = 0
    total_vest_detections = 0
    unique_tracks = set()
    violations_confirmed = []

    if log_file.exists():
        with open(log_file, "r", encoding="utf-8") as lf:
            for line in lf:
                if not line.strip():
                    continue
                entry = json.loads(line)
                summary = entry.get("summary", {})
                total_person_detections += summary.get("person", 0)
                total_helmet_detections += summary.get("helmet", 0)
                total_vest_detections += summary.get("safety_vest", 0)
                for d in entry.get("detections", []):
                    if d.get("track_id") is not None:
                        unique_tracks.add(d["track_id"])
                for v in entry.get("violations", []):
                    violations_confirmed.append(v)

    total_frames = meta.get("total_frames", 0)
    fps_proc = round(total_frames / elapsed, 1) if elapsed > 0 and total_frames > 0 else 0

    result = {
        "job_id": job_id,
        "video": video_name,
        "model": model_name,
        "total_frames": total_frames,
        "elapsed_s": elapsed,
        "fps_processing": fps_proc,
        "person_detections": total_person_detections,
        "helmet_detections": total_helmet_detections,
        "vest_detections": total_vest_detections,
        "unique_tracks": len(unique_tracks),
        "violations_count": len(violations_confirmed),
        "violations": violations_confirmed,
    }
    print(f"✔ [{job_id}] {label}: {total_frames} frames trong {elapsed}s ({fps_proc} FPS) | "
          f"Track: {len(unique_tracks)} | Vi phạm: {len(violations_confirmed)}")
    return result


def main():
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        pass
    print("BẮT ĐẦU CHƯƠNG TRÌNH KIỂM THỬ TOÀN DIỆN", flush=True)
    print(f"Thời gian: {time.strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
    print(flush=True)

    # 1. Audit invalid models
    audit1 = audit_train_results_models()

    # 2. Audit valid drive-download models
    audit2 = audit_drive_download_models()

    # 3. Test all videos in test-ui/uploads with YOLO11s (Module 1 từ drive-download)
    print("=" * 70)
    print("PHẦN 3: XỬ LÝ TOÀN BỘ VIDEO TRONG test-ui/uploads/ VỚI YOLO11s (Module 1)")
    print("=" * 70)

    video_files = sorted(UPLOADS_DIR.glob("*.mp4"))
    if not video_files:
        print(f"❌ Không tìm thấy video trong {UPLOADS_DIR}")
        return

    model_11s = MODELS_MAP["yolo11s"]
    results_all_videos = []
    for vf in video_files:
        res = run_pipeline_job(vf, model_11s)
        if res:
            results_all_videos.append(res)

    # 4. So sánh Model YOLO11m vs YOLOv8s trên video đại diện (video2 & video7) đối chiếu với YOLO11s
    print()
    print("=" * 70)
    print("PHẦN 4: SO SÁNH ĐỐI CHỨNG CÁC KIẾN TRÚC (YOLO11m & YOLOv8s) TRÊN VIDEO MẪU")
    print("=" * 70)
    comparison_results = []
    # Thêm kết quả YOLO11s từ Phần 3 vào bảng so sánh
    for r in results_all_videos:
        if r["video"] in ["video2.mp4", "video7.mp4"]:
            comparison_results.append(r)

    for vname in ["video2.mp4", "video7.mp4"]:
        vpath = UPLOADS_DIR / vname
        if not vpath.exists():
            continue
        for mkey in ["yolo11m", "yolov8s"]:
            mpath = MODELS_MAP[mkey]
            if mpath.exists():
                res = run_pipeline_job(vpath, mpath)
                if res:
                    comparison_results.append(res)

    # 5. Export summary JSON
    summary_path = ROOT_DIR / "test-ui" / "batch_test_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump({
            "audit_invalid_models": audit1,
            "audit_valid_models": audit2,
            "results_all_videos_yolo11s": results_all_videos,
            "comparison_results": comparison_results,
            "completed_at": time.strftime("%Y-%m-%d %H:%M:%S")
        }, f, ensure_ascii=False, indent=2)

    print()
    print(f"✅ Hoàn tất kiểm thử toàn bộ video! Kết quả lưu tại: {summary_path}")


if __name__ == "__main__":
    main()
