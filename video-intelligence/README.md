# Video Intelligence — Phase 2 (SV2)

Pipeline: Video → Detection → Tracking → PPE–Person Association → Safety Rules → Snapshot → Violation Event.

**Giao diện test (tách biệt với core):** xem hướng dẫn tại [test-ui/README.md](test-ui/README.md) — chạy `python -m test-ui.server` rồi mở http://127.0.0.1:8000 để upload video và xem kết quả detect + tracking kèm log đồng bộ theo frame.

**Báo cáo đầy đủ VI-01 → VI-08** (đã làm gì, code ở đâu, cách chạy/test, còn thiếu gì): xem [docs/BAO-CAO-VI01-VI08.md](docs/BAO-CAO-VI01-VI08.md).

## Trạng thái

| Mã | Hạng mục | Trạng thái |
|---|---|---|
| VI-01 | Đọc video | ✅ Xong |
| VI-02 | Chuẩn hóa detector | ✅ Xong |
| VI-03 | Tracking | ✅ Xong |
| VI-04 | Ghép PPE–Person | ✅ Xong |
| VI-05 | Size filter + safety rules | ✅ Xong |
| VI-06 | Xác nhận thời gian | ✅ Xong |
| VI-07 | Chống trùng | ✅ Xong |
| VI-08 | Evidence Snapshot | ✅ Xong |
| VI-09 | Event JSON contract | ⬜ Chưa làm |
| VI-10 | Dual-Transport (HTTP/Kafka) | ⬜ Chưa làm |
| VI-11 | Lấy safety-config | ⬜ Chưa làm |
| VI-12 | Kiểm thử | ⬜ Chưa làm |
| VI-13 | Restricted Area (P2) | ⬜ Chưa làm |

## Cài đặt

```bash
cd video-intelligence
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Chạy (VI-01 + VI-02)

```bash
# 1. Chạy với video file — vẽ bbox + nhãn, ghi kết quả ra outputs/<tên>_det.mp4
python -m app.main --source videos/video3.mp4

# 2. Hoặc đặt biến môi trường rồi chạy không cần --source
export VIDEO_SOURCE=videos/video3.mp4
python -m app.main

# 3. Webcam (index 0) / RTSP
python -m app.main --source 0
python -m app.main --source rtsp://user:pass@ip:554/stream
```

Đổi model không cần sửa code:

```bash
export MODEL_PATH=models/best.pt     # hoặc đường dẫn model khác (yolo11n/s/v8n...)
export MODEL_IMGSZ=640               # kích thước ảnh đưa vào model
export CONFIDENCE_THRESHOLD=0.5      # ngưỡng tin cậy
export DEVICE=mps                    # hoặc cpu (mặc định: tự chọn mps nếu máy hỗ trợ)
```

Kết quả in ra: thông tin nguồn/model, tiến độ theo frame kèm số đối tượng và tốc độ detect TB (ms/frame). Video output ghi tại `outputs/`.

## Cấu hình nguồn

Nguồn video được chọn theo thứ tự: `--source` (CLI) > biến môi trường `VIDEO_SOURCE` > `DEFAULT_SOURCE` trong `app/config.py`. Nhận diện nguồn tự động:

- Số nguyên → webcam index
- Chuỗi bắt đầu `rtsp://` → camera RTSP
- Còn lại → đường dẫn file video

Các ngưỡng mặc định (3 giây xác nhận, 30 giây cooldown, 60 px, confidence 0.5) nằm trong `app/config.py`, dùng cho các bước VI-05 trở đi.
