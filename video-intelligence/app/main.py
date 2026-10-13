"""Entry point của video-intelligence.

Đã nối: VI-01 (đọc video) + VI-02 (detector chuẩn hóa + vẽ bbox)
+ VI-03 (tracking) + VI-04 (ghép PPE–Person).
Pipeline: đọc frame -> detect -> ghép PPE -> vẽ bbox -> ghi video output.

Chạy:
    python -m app.main --source videos/video3.mp4
"""
import argparse
import time
from pathlib import Path

import cv2

from app.video.video_reader import VideoSourceError, VideoReader
from app.detection.detector import Detector, DetectorError
from app.association.associator import associate
from app.rules.safety import evaluate as evaluate_safety
from app.rules.confirmation import ViolationConfirmation
from app.rules.deduplication import ViolationDeduplicator
from app.evidence.snapshot import save_evidence
from app.config import EVIDENCE_DIR
from app.video.visualizer import draw_detections


def _state_suffix(state):
    if state is None:
        return ""
    if state.status == "COMPLIANT":
        return " → OK"
    if state.status == "VIOLATION_CANDIDATE":
        return f" → VIOLATION_CANDIDATE ({', '.join(state.violations)})"
    return f" → UNKNOWN ({state.reason})"


def run(source, out_dir="outputs", show_every=50):
    with VideoReader(source) as reader:
        detector = Detector().load()
        print(f"Nguồn       : {reader.source}")
        print(f"Model       : {detector.model_path} | imgsz={detector.imgsz} | conf={detector.conf} | device={detector.device}")
        if reader.size:
            print(f"Kích thước  : {reader.size[0]}x{reader.size[1]} | FPS: {reader.fps:.2f}")

        writer = None
        out_path = None
        t_read = t_detect = 0.0
        n_frames = 0
        confirmation = ViolationConfirmation()
        dedup = ViolationDeduplicator()

        for num, frame in reader:
            n_frames += 1
            t0 = time.time()
            detections = detector.detect(frame)
            t_detect += time.time() - t0
            associations = associate(detections)
            person_states = evaluate_safety(detections, associations,
                                            frame.shape[1], frame.shape[0])
            t_frame = (num - 1) / reader.fps
            confirmed = confirmation.update(t_frame, person_states)
            draw_detections(frame, detections)
            for ev in dedup.filter(confirmed, detections, t_frame):
                print(f"  🚨 VI PHẠM: person #{ev.person_track_id} {ev.violation_type} — "
                      f"từ t={ev.start_t:.1f}s, xác nhận t={ev.confirmed_t:.1f}s "
                      f"({ev.duration_s:.1f}s) | {ev.camera_id} | id={ev.event_id}")
                evidence = save_evidence(frame, detections, ev, EVIDENCE_DIR)
                if evidence.status == "saved":
                    print(f"    📸 bằng chứng: {evidence.path}")
                else:
                    print(f"    ⚠ bằng chứng LỖI: {evidence.error}")

            if writer is None:
                out_path = Path(out_dir) / f"{Path(str(reader.source)).stem}_det.mp4"
                out_path.parent.mkdir(parents=True, exist_ok=True)
                writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"),
                                         reader.fps, (frame.shape[1], frame.shape[0]))
            writer.write(frame)

            if num % show_every == 0:
                from collections import Counter
                counts = Counter(d.class_name for d in detections)
                detail = " | ".join(f"{counts.get(c, 0)} {c}" for c in ("person", "helmet", "safety_vest"))
                n_tracks = len({d.track_id for d in detections if d.track_id is not None})
                states_by_id = {s.person_track_id: s for s in person_states}
                ppe = " | ".join(
                    f"person #{a.person_track_id}: mũ {'#' + str(a.helmet_track_id) if a.helmet_track_id else 'KHÔNG'} · "
                    f"áo {'#' + str(a.vest_track_id) if a.vest_track_id else 'KHÔNG'}"
                    + _state_suffix(states_by_id.get(a.person_track_id))
                    for a in associations)
                print(f"  frame {num:>6} — {detail} — {n_tracks} track đang theo dõi — "
                      f"detect {t_detect / n_frames * 1000:.0f} ms/frame (TB)")
                if ppe:
                    print(f"           ghép: {ppe}")

        if writer:
            writer.release()
            print(f"Đã ghi     : {out_path} ({n_frames} frame)")
        else:
            print("Không ghi được video output.")


def main():
    parser = argparse.ArgumentParser(description="Video Intelligence — pipeline phát hiện vi phạm PPE")
    parser.add_argument("--source", default=None,
                        help="Đường dẫn file video, 'rtsp://...' hoặc index webcam (vd: 0)")
    args = parser.parse_args()

    try:
        run(args.source)
    except VideoSourceError as e:
        print(f"[LỖI] {e}")
        return 1
    except DetectorError as e:
        print(f"[LỖI] {e}")
        return 1
    except KeyboardInterrupt:
        print("\nĐã dừng bởi người dùng — tài nguyên đã được giải phóng.")
        return 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
