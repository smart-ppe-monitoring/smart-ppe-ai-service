# Test UI — giao diện test 2 khung cho video-intelligence

Giao diện web để xem và đánh giá kết quả detect + tracking: upload video → xử lý (vẽ box + ghi log chi tiết) → xem lại với panel thông tin đồng bộ theo frame đang phát.

**Lưu ý:** đây là tool test, tách biệt hoàn toàn với core pipeline (`app/`). UI chỉ dùng lại các API công khai của core, không sửa core.

## Chạy

```bash
cd ~/Downloads/DATN/video-intelligence
source .venv/bin/activate
pip install -r test-ui/requirements.txt
python -m test-ui.server
# Mở trình duyệt: http://127.0.0.1:8000
```

## Cách dùng

1. Chọn file video → bấm **"Xử lý video"** → chờ thanh tiến độ (tốc độ ~40-200 ms/frame tùy video)
2. Khi xong, video có box + nhãn + track_id tự hiện bên trái
3. **Panel phải** tự đồng bộ theo frame đang phát:
   - **Vi phạm được xác nhận + chống trùng (VI-06 + VI-07)**: danh sách các sự kiện vi phạm CUỐI CÙNG được phát — đã tồn tại đủ 3 giây liên tục (VI-06, tính bằng timestamp) và đã lọc trùng (VI-07, cooldown 30 giây + nối track đổi ID). Mỗi mục ghi người, loại vi phạm, thời điểm xác nhận, thời lượng, camera, event_id
   - Nút **"Xem bằng chứng / frame lỗi"** ngay dưới danh sách vi phạm (VI-08): mở danh sách ảnh bằng chứng (full frame đã vẽ bbox, **người vi phạm tô lớp đỏ mờ + nhãn loại vi phạm**) của từng event — bấm ảnh để xem to; mục lưu lỗi hiện khung đỏ với **chi tiết lỗi bên cạnh** và link nhảy tới frame đó trong video
   - **Tóm tắt theo người**: mỗi người xuất hiện từ frame nào tới frame nào, mũ bảo hộ / áo phản quang có hay thiếu ở những đoạn nào, kèm đánh giá (ĐỦ BẢO HỘ / VI PHẠM DỰ KIẾN / VI PHẠM XÁC NHẬN / UNKNOWN). Dữ liệu ghép lấy từ VI-04/VI-05/VI-06/VI-07 chạy ở server (không còn heuristic JS)
   - Khung hình đang phát: danh sách detection (lớp, track_id, confidence, bbox, kích thước px), các sự kiện xuất hiện/biến mất kèm tên lớp (vd "helmet #5 xuất hiện")
   - **Ghép PPE–Person + Luật an toàn (VI-04 + VI-05)**: tại frame đang phát, mỗi người được ghép với mũ/áo nào (`person #5 ← mũ #8 · áo #7 → OK`) kèm trạng thái luật: `→ VI PHẠM? (no_safety_vest)` hoặc `→ UNKNOWN (lý do)`; PPE không ghép được với ai cũng được liệt kê
   - Người đang theo dõi tại frame hiện tại: trạng thái VI-05 của từng người (ĐỦ BẢO HỘ / VI PHẠM? / UNKNOWN)
4. Ô "Nhảy tới frame" để xem đúng một frame cụ thể

## API

| Endpoint | Chức năng |
|---|---|
| `POST /api/upload` | Upload video, trả `job_id` |
| `POST /api/jobs/{id}/start` | Chạy xử lý nền |
| `GET /api/jobs/{id}/status` | Tiến độ xử lý |
| `GET /api/jobs/{id}/log` | Toàn bộ log JSONL (1 dòng/frame) |
| `GET /api/jobs/{id}/video` | Video đã vẽ box (`annotated.mp4`) |
| `GET /api/jobs/{id}/evidence/{event_id}.jpg` | Ảnh bằng chứng của một event (VI-08) |
| `GET /api/jobs/{id}/meta` | Thông tin job (fps, kích thước, model...) |

## Định dạng log (log.jsonl — mỗi frame 1 dòng)

```json
{"frame": 200, "t": 6.71, "detections": [
  {"class_name": "person", "track_id": 2, "confidence": 0.83, "bbox": [685, 421, 900, 951], "w": 215, "h": 530}
],
 "summary": {"person": 2, "helmet": 2, "safety_vest": 1, "tracks": 5},
 "events": ["track 4 xuất hiện"],
 "associations": [
  {"person": 2, "helmet": 4, "vest": 3, "status": "matched"},
  {"person": 9, "helmet": null, "vest": null, "status": "unknown"}
 ],
 "persons": [
  {"track_id": 2, "status": "COMPLIANT", "violations": [], "reason": ""},
  {"track_id": 9, "status": "VIOLATION_CANDIDATE", "violations": ["no_helmet"], "reason": ""}
 ],
 "confirmations": [
  {"person": 9, "violation_type": "no_helmet", "start_t": 3.4, "confirmed_t": 6.4, "duration_s": 3.0}
 ],
 "violations": [
  {"event_id": "a1b2c3d4e5f6", "camera_id": "cam-01", "person": 9, "canonical_person": 9,
   "violation_type": "no_helmet", "start_t": 3.4, "confirmed_t": 6.4, "duration_s": 3.0}
 ],
 "evidence": [
  {"event_id": "a1b2c3d4e5f6", "status": "saved", "path": "test-ui/results/<job>/evidences/a1b2c3d4e5f6.jpg", "error": ""}
 ]}
```

- `associations`: kết quả ghép PPE–Person (VI-04): mỗi người một dòng, `helmet`/`vest` là track_id của PPE ghép được (null = không có), `status` = `matched` hoặc `unknown` (người < 60px).
- `persons`: trạng thái luật an toàn (VI-05): `COMPLIANT` (đủ PPE) / `VIOLATION_CANDIDATE` (thiếu PPE — chưa xác nhận, cần VI-06) / `UNKNOWN` (người nhỏ, confidence thấp, hoặc bị cắt khung — không kết luận).
- `confirmations`: sự kiện vi phạm được xác nhận (VI-06) — xuất hiện ở đúng frame vi phạm đạt đủ `confirmation_duration_s` (3s mặc định), tính bằng timestamp, kèm `start_t` / `confirmed_t` / `duration_s`. Chưa lọc trùng — mỗi chuỗi vi phạm chỉ phát 1 lần.
- `violations`: sự kiện vi phạm CUỐI CÙNG sau chống trùng (VI-07) — đây là thứ UI hiển thị. `event_id` là id duy nhất; khóa chống trùng là (camera_id, `canonical_person`, violation_type) với cooldown 30 giây; `canonical_person` ổn định khi track đổi ID (nối các track xuất hiện gần vị trí track cũ trong vòng 2 giây).
- `evidence`: kết quả lưu snapshot (VI-08) cho từng event ở chính frame phát event — `status` = `saved` (đã lưu ảnh full frame có vẽ bbox vào `results/<job_id>/evidences/<event_id>.jpg`; người vi phạm được tô lớp đỏ mờ + nhãn loại vi phạm) hoặc `error` (lưu lỗi, kèm message — pipeline vẫn chạy tiếp). `path` chỉ là đường dẫn lưu trữ local, không phải URL công khai; ảnh được serve qua endpoint `/api/jobs/{id}/evidence/...`.

## Nơi lưu dữ liệu

- `test-ui/uploads/` — video upload lên (tạm)
- `test-ui/results/<job_id>/` — `annotated.mp4`, `log.jsonl`, `meta.json`
