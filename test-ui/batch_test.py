"""
Batch testing script to process all videos in test-ui/uploads using yolo11s_joint_phase3.pt
Results are saved to test-ui/results/<video_id> with:
- annotated.mp4 (H.264 video with HUD & bounding boxes)
- log.jsonl (frame-by-frame detections, violations, tracks)
- meta.json (job metadata)
- evidences/ (violation snapshots + reference snapshots)
"""

import json
import sys
import time
from pathlib import Path

# Add project root and test-ui to sys.path
TEST_UI_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = TEST_UI_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(TEST_UI_DIR))

import server

MODEL_NAME = "models/yolo11s_joint_phase3.pt"
MODEL_PATH = PROJECT_ROOT / MODEL_NAME


def run_batch():
    if not MODEL_PATH.exists():
        print(f"ERROR: Model weight not found at {MODEL_PATH}")
        sys.exit(1)

    print(f"=== BATCH VIDEO INFERENCE STARTING ===")
    print(f"Model: {MODEL_PATH} ({MODEL_PATH.stat().st_size / (1024*1024):.1f} MB)")
    print(f"Uploads Dir: {server.UPLOAD_DIR}")
    print(f"Results Dir: {server.RESULT_DIR}")
    print("-" * 60)

    # Find all test videos in uploads (all formats, size > 10KB)
    valid_exts = {".mp4", ".avi", ".mkv", ".mov"}
    video_files = sorted(
        [f for f in server.UPLOAD_DIR.iterdir() if f.is_file() and f.suffix.lower() in valid_exts and f.stat().st_size > 10000],
        key=lambda p: p.stem
    )

    print(f"Found {len(video_files)} video(s) to process: {[f.name for f in video_files]}")
    print("-" * 60)

    summary = []
    total_batch_start = time.time()

    for idx, video_path in enumerate(video_files, 1):
        job_id = video_path.stem
        print(f"\n[{idx}/{len(video_files)}] Processing: {video_path.name} (Job ID: {job_id})...")
        t0 = time.time()

        # Prepare directory and job state
        paths = server._job_paths(job_id)

        # Check if already processed
        meta_file = paths["meta"]
        if meta_file.exists() and paths["video"].exists():
            try:
                existing_meta = json.loads(meta_file.read_text(encoding="utf-8"))
                if existing_meta.get("status") == "done" and "yolo11s_joint_phase3" in existing_meta.get("model", ""):
                    ev_dir = paths["dir"] / "evidences"
                    ev_count = len(list(ev_dir.glob("*.jpg"))) if ev_dir.exists() else 0
                    mp4_size_mb = paths["video"].stat().st_size / (1024 * 1024)
                    print(f"  -> ALREADY PROCESSED ({existing_meta.get('total_frames')} frames, {ev_count} evidences). Skipping.")
                    summary.append({
                        "job_id": job_id,
                        "source": video_path.name,
                        "frames": existing_meta.get("total_frames", 0),
                        "fps": round(existing_meta.get("fps", 0), 1),
                        "size": existing_meta.get("size", [0, 0]),
                        "evidences": ev_count,
                        "video_mb": round(mp4_size_mb, 1),
                        "elapsed_s": 0.0,
                        "status": "OK (cached)"
                    })
                    continue
            except Exception:
                pass

        paths["dir"].mkdir(parents=True, exist_ok=True)
        with server.JOBS_LOCK:
            server.JOBS[job_id] = {
                "status": "uploaded",
                "progress": 0.0,
                "source": video_path.name,
            }

        try:
            server.process_video(job_id, video_path, model_path=MODEL_NAME)
            elapsed = time.time() - t0

            # Inspect results
            meta = server._get_job(job_id)
            ev_dir = paths["dir"] / "evidences"
            ev_count = len(list(ev_dir.glob("*.jpg"))) if ev_dir.exists() else 0
            annotated_mp4 = paths["video"]
            mp4_size_mb = annotated_mp4.stat().st_size / (1024 * 1024) if annotated_mp4.exists() else 0

            print(f"  -> SUCCESS in {elapsed:.1f}s | {meta.get('total_frames')} frames | {ev_count} evidence files | {mp4_size_mb:.1f} MB MP4")
            summary.append({
                "job_id": job_id,
                "source": video_path.name,
                "frames": meta.get("total_frames", 0),
                "fps": round(meta.get("fps", 0), 1),
                "size": meta.get("size", [0, 0]),
                "evidences": ev_count,
                "video_mb": round(mp4_size_mb, 1),
                "elapsed_s": round(elapsed, 1),
                "status": "OK"
            })
        except Exception as e:
            elapsed = time.time() - t0
            print(f"  -> FAILED after {elapsed:.1f}s: {e}")
            import traceback
            traceback.print_exc()
            summary.append({
                "job_id": job_id,
                "source": video_path.name,
                "status": f"ERROR: {e}",
                "elapsed_s": round(elapsed, 1)
            })

    total_time = time.time() - total_batch_start
    print("\n" + "=" * 70)
    print("=== BATCH INFERENCE SUMMARY TABLE ===")
    print("=" * 70)
    header = f"{'Video':<12} | {'Resolution':<12} | {'FPS':<6} | {'Frames':<8} | {'Evidences':<10} | {'Output MB':<10} | {'Time (s)':<8} | {'Status'}"
    print(header)
    print("-" * len(header))
    for s in summary:
        if s.get("status") == "OK":
            res = f"{s['size'][0]}x{s['size'][1]}"
            print(f"{s['source']:<12} | {res:<12} | {s['fps']:<6} | {s['frames']:<8} | {s['evidences']:<10} | {s['video_mb']:<10} | {s['elapsed_s']:<8} | {s['status']}")
        else:
            print(f"{s['source']:<12} | {'N/A':<12} | {'N/A':<6} | {'N/A':<8} | {'N/A':<10} | {'N/A':<10} | {s['elapsed_s']:<8} | {s['status']}")
    print("-" * len(header))
    print(f"Total time: {total_time:.1f}s ({total_time / 60:.2f} mins)")
    print("All results saved to: test-ui/results/")


if __name__ == "__main__":
    run_batch()
