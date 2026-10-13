# Báo cáo tiến độ Video Intelligence — VI-01 → VI-08

> Người thực hiện: Trà (SV2 — Video Intelligence, nhóm D08).
> Ngày cập nhật: 13/10/2026. Phạm vi: VI-01 → VI-08 (Phase 2, tài liệu `Phase_2.docx`).

Mục đích tài liệu: trả lời nhanh 4 câu — **đã làm gì, code ở đâu, chạy/test thế nào, còn thiếu gì**.

---

## 0. Tổng quan

### Pipeline

```
Video → [VI-01 Đọc frame] → [VI-02 Detect] → [VI-03 Tracking]
      → [VI-04 Ghép PPE–Person] → [VI-05 Luật an toàn] → [VI-06 Xác nhận 3 giây]
      → [VI-07 Chống trùng 30 giây] → [VI-08 Snapshot bằng chứng]
      → (VI-09/10: đóng gói & gửi event — CHƯA LÀM)
```

### Nguyên tắc kiến trúc (quan trọng)

- **`app/`** = CORE pipeline chính thức. Mọi logic xử lý nằm ở đây.
- **`test-ui/`** = giao diện test **TÁCH HOÀN TOÀN khỏi core logic**. Nó chỉ để **xem kết quả** (upload video → xem video có box + log đồng bộ theo frame), **chỉ dùng lại API công khai của `app/`** (VideoReader, Detector, associate, evaluate, ViolationConfirmation, ViolationDeduplicator, save_evidence, draw_detections) — **không sửa một file nào trong core**. Test UI không thuộc pipeline chính thức, chỉ phục vụ đánh giá đúng/sai trong quá trình phát triển.
- Tất cả ngưỡng (3s, 30s, 60px, 0.5...) tập trung ở `app/config.py`, ghi đè được qua biến môi trường.

### Cấu trúc thư mục

```
video-intelligence/
├── app/                        # CORE — logic chính thức
│   ├── config.py               # Cấu hình trung tâm + toàn bộ ngưỡng
│   ├── main.py                 # Entry point CLI (pipeline đầu cuối)
│   ├── video/
│   │   ├── video_reader.py     # VI-01: VideoReader (file/webcam/RTSP)
│   │   └── visualizer.py       # Vẽ bbox + nhãn (demo/kiểm chứng)
│   ├── detection/
│   │   ├── adapter.py          # VI-02: chuẩn hóa Detection, LABEL_ALIASES
│   │   └── detector.py         # VI-02+03: Detector + ByteTrack (track=True)
│   ├── association/
│   │   └── associator.py       # VI-04: associate() ghép PPE–Person
│   ├── rules/
│   │   ├── safety.py           # VI-05: evaluate() luật an toàn, PersonState
│   │   ├── confirmation.py     # VI-06: ViolationConfirmation (3 giây)
│   │   └── deduplication.py    # VI-07: ViolationDeduplicator (cooldown 30s)
│   └── evidence/
│       └── snapshot.py         # VI-08: save_evidence (ảnh bằng chứng)
├── test-ui/                    # UI TEST — tách biệt, chỉ để xem
│   ├── server.py               # FastAPI: upload → xử lý nền → trả video + log
│   ├── static/index.html       # Giao diện 2 khung (video + log, vanilla JS)
│   ├── requirements.txt        # fastapi, uvicorn, python-multipart
│   ├── README.md               # Hướng dẫn UI + định dạng log
│   ├── uploads/                # Video upload (tạm)
│   └── results/<job_id>/       # annotated.mp4 + log.jsonl + meta.json + evidences/
├── tests/                      # Unit test (script tự chạy, không cần pytest)
├── models/best.pt              # Model YOLO11m 3 lớp (person/helmet/safety_vest)
├── videos/                     # Video mẫu test: video2..video8.mp4
├── outputs/                    # Video output khi chạy CLI core
├── evidences/                  # Ảnh bằng chứng khi chạy CLI core
├── docs/                       # Tài liệu
└── README.md                   # Trạng thái + cách chạy cơ bản
```

---

## 1. Trạng thái tổng thể

| Mã | Hạng mục | Trạng thái | Code chính |
|---|---|---|---|
| VI-01 | Đọc video | ✅ Xong | `app/video/video_reader.py` |
| VI-02 | Chuẩn hóa detector | ✅ Xong | `app/detection/adapter.py`, `app/detection/detector.py` |
| VI-03 | Tracking (ByteTrack) | ✅ Xong | `app/detection/detector.py` (`track=True`, `persist=True`) |
| VI-04 | Ghép PPE–Person | ✅ Xong | `app/association/associator.py` |
| VI-05 | Size filter + safety rules | ✅ Xong | `app/rules/safety.py` |
| VI-06 | Xác nhận thời gian (3s) | ✅ Xong | `app/rules/confirmation.py` |
| VI-07 | Chống trùng (30s) | ✅ Xong | `app/rules/deduplication.py` |
| VI-08 | Evidence Snapshot | ✅ Xong | `app/evidence/snapshot.py` |
| VI-09 | Event JSON contract | ⬜ Chưa làm | — (đã có hướng: ánh xạ ViolationEvent → schema docx) |
| VI-10 | Dual-Transport (HTTP/Kafka) | ⬜ Chưa làm | — (chờ Backend chốt endpoint) |
| VI-11 | Lấy safety-config từ Backend | ⬜ Chưa làm | — |
| VI-12 | Kiểm thử tổng thể | ⬜ Chưa làm | — (đã có 5 file unit test cơ bản) |
| VI-13 | Restricted Area | ⬜ Chưa làm (P2) | — (tính năng phát hiện vùng cấm, không phải đóng gói) |

---

## 2. Chi tiết từng bước đã làm

### VI-01 — Đọc video

- **Code:** `app/video/video_reader.py` — class `VideoReader` (dùng `with` để tự giải phóng tài nguyên), ném `VideoSourceError` khi mở lỗi; `app/config.py` — `resolve_source()` nhận diện nguồn tự động (số nguyên = webcam, `rtsp://` = camera mạng, còn lại = file).
- **Xử lý lỗi:** frame rỗng với webcam/RTSP → bỏ qua tối đa 30 frame liên tiếp (`EMPTY_FRAME_TOLERANCE`) rồi dừng; file hết → kết thúc tự nhiên.
- **Chạy:** `python -m app.main --source videos/video2.mp4` (hoặc đặt env `VIDEO_SOURCE`).
- **Đã test:** video bình thường, file không tồn tại (in `[LỖI]`, thoát mã 1), Ctrl+C dừng sạch.
- **Chưa test:** webcam, RTSP (chưa có thiết bị thật).

### VI-02 — Chuẩn hóa detector

- **Code:** `app/detection/adapter.py` — dataclass `Detection(class_name, confidence, bbox=(x1,y1,x2,y2) số nguyên pixel frame gốc, track_id)` + `normalize_results()`; `app/detection/detector.py` — class `Detector().load()` (tải model, tự chọn mps/cpu), `detect(frame)`.
- **Điểm quan trọng:** khả năng nhận diện 3 lớp `person/helmet/safety_vest` nằm **hoàn toàn trong file model** `models/best.pt` — code chỉ load và đọc tên lớp qua `LABEL_ALIASES` (chấp nhận cả "hardhat"/"vest" → tên chuẩn). **Đổi model chỉ cần đổi env `MODEL_PATH`, không sửa code.**
- **Model hiện tại:** `models/best.pt` — YOLO11m do SV1 train, 39MB, **là model tạm**; khi SV1 bàn giao model cuối (dự kiến YOLO11n/s) chỉ cần copy vào `models/` và đổi `MODEL_PATH`.
- **Chạy:** `python -m app.main --source videos/video3.mp4` → in tiến độ (số box từng lớp, ms/frame), ghi video output `outputs/<tên>_det.mp4` (codec mp4v, mở bằng QuickTime/VLC).
- **Tốc độ:** ~40–220 ms/frame trên Mac MPS (video dọc 900×1596), đủ xử lý offline.

### VI-03 — Tracking (ByteTrack)

- **Code:** `app/detection/detector.py` — `track=True` (mặc định), gọi `model.track(persist=True)`; `adapter.py` đọc `boxes.id` → `Detection.track_id`; nhãn video có dạng `person #2 0.87`.
- **Nguyên tắc:** mỗi lần chạy tạo một `Detector` mới → bộ nhớ tracker riêng theo nguồn (sẵn sàng cho nhiều camera, mỗi camera 1 instance).
- **Đã kiểm chứng (video2.mp4):** person#2 và person#4 giữ ID ổn định suốt hàng trăm frame, phân biệt được 2 người vào/ra khung hình.
- **Lưu ý:** ID đánh **chung cho mọi lớp** (helmet#1, person#2, vest#3 — một bộ đếm toàn cục); ID vẫn có thể switch sau che khuất lâu → đây là lý do VI-07 không chỉ dựa vào track_id. Lần đầu chạy track, ultralytics tự cài thêm thư viện `lap` (cần mạng).

### VI-04 — Ghép PPE–Person

- **Code:** `app/association/associator.py` — `associate(detections, min_height=MIN_PERSON_HEIGHT_PX) -> list[PersonPPE]`; `PersonPPE(person_track_id, person_bbox, helmet_track_id, vest_track_id, status=MATCHED/UNKNOWN)`.
- **Thuật toán:** mũ phải nằm trong **vùng đầu** (tâm mũ trong khoảng ngang 15%–85% bbox người và trên 40% chiều cao); áo nằm trong **vùng thân** (25%–85% chiều cao, hoặc IoU ≥ 0.10 với vùng thân); ghép tham lam 1-1 (mỗi PPE thuộc tối đa 1 người, mỗi người tối đa 1 mũ + 1 áo). Người < 60px → UNKNOWN.
- **Ngưỡng:** `PPE_HEAD_X_MARGIN=0.15`, `PPE_HEAD_Y_MAX=0.40`, `PPE_TORSO_Y_MIN=0.25`, `PPE_TORSO_Y_MAX=0.85`, `PPE_TORSO_IOU_MIN=0.10` (config.py).
- **Test:** `python tests/test_association.py` — 10 test, tất cả OK.

### VI-05 — Size filter + safety rules

- **Code:** `app/rules/safety.py` — `evaluate(detections, associations, frame_w, frame_h) -> list[PersonState]`; `PersonState(person_track_id, status, violations, reason)`.
- **Trạng thái:**
  - `COMPLIANT` — có đủ mũ + áo
  - `VIOLATION_CANDIDATE` — thiếu mũ (`no_helmet`) hoặc thiếu áo (`no_safety_vest`) — chưa kết luận, chờ VI-06
  - `UNKNOWN` — người quá nhỏ (< 60px), confidence thấp (< 0.5), hoặc bị cắt khung không xét được (ưu tiên UNKNOWN thay vì kết luận sai → giảm false alarm)
- **Ngưỡng cắt khung:** mũ không xét được khi đỉnh bbox chạm mép khung (`PERSON_EDGE_MARGIN_PX=2` — vì mũ nằm trên cùng, bị cắt trên là mất mũ); áo chỉ coi là bị che khi **vừa** cắt mép bên **và** phần người nhìn thấy hẹp hơn `PERSON_MIN_VISIBLE_WIDTH_PX=150px`. ⚠ **Ngưỡng 150px đã được hiệu chỉnh sau khi chạy video thật** (người đứng sát mép trái video2 bị đánh UNKNOWN oan) — theo docx "ngưỡng cần kiểm thử theo độ phân giải", cần xem lại khi đổi độ phân giải camera.
- **Test:** `python tests/test_safety_rules.py` — 13 test, tất cả OK.

### VI-06 — Xác nhận thời gian (3 giây)

- **Code:** `app/rules/confirmation.py` — `ViolationConfirmation(duration_s=CONFIRMATION_DURATION_S)`, `update(t, person_states) -> list[ConfirmedViolation]`.
- **Cơ chế:** tính bằng **timestamp** (không dựa cứng vào số frame); chuỗi vi phạm chỉ cộng dồn khi liên tục là CANDIDATE; `COMPLIANT` → reset chuỗi; `UNKNOWN` → tạm dừng (không cộng thời gian, không reset); track mất → reset. Khi đủ 3 giây liên tục → phát **đúng 1** `ConfirmedViolation(start_t, confirmed_t, duration_s)` tại thời điểm vượt ngưỡng.
- **Ngưỡng:** `CONFIRMATION_DURATION_S=3.0` (config, ghi đè được — VI-11 sau này sẽ kéo từ Backend).
- **Đã kiểm chứng (video2):** person #9 thiếu áo, 95 frame CANDIDATE → xác nhận đúng 1 lần tại t=6.53s; track nhiễu không bao giờ được xác nhận.
- **Test:** `python tests/test_confirmation.py` — 12 test, tất cả OK.

### VI-07 — Chống trùng (cooldown 30 giây)

- **Code:** `app/rules/deduplication.py` — `ViolationDeduplicator.filter(confirmed_events, detections, t) -> list[ViolationEvent]`; `ViolationEvent(event_id, camera_id, person_track_id, canonical_person_id, violation_type, start_t, confirmed_t, duration_s)`.
- **Cơ chế:**
  - Khóa chống trùng = `(camera_id, canonical_person_id, violation_type)`; tái phát trong 30 giây → chặn (thỏa khóa tối thiểu cameraId + trackId + violationType theo docx)
  - Một vi phạm kéo dài → chỉ 1 event (VI-06 đã lo việc này, VI-07 chặn thêm trường hợp lặp)
  - **Xử lý track đổi ID:** track mới xuất hiện trong vòng `ID_SWITCH_GRACE_S=2.0` giây và tâm bbox cách vị trí cuối của track vừa mất < 50% chiều cao người (`ID_SWITCH_MAX_DIST_RATIO=0.5`) → nối về cùng `canonical_person_id`. ⚠ Đây là heuristic — cần đánh giá thêm trên video thật có người ra/vào khung nhiều.
  - `event_id` = uuid4 đầy đủ (định dạng theo mẫu docx VI-09)
- **Test:** `python tests/test_deduplication.py` — 9 test, tất cả OK.

### VI-08 — Evidence Snapshot

- **Code:** `app/evidence/snapshot.py` — `save_evidence(frame, detections, event, out_dir) -> EvidenceResult(event_id, status="saved"|"error", path, error)`.
- **Cơ chế:** khi có event vi phạm → lưu **full frame đã vẽ bbox** thành `evidences/<eventId>.jpg`, **tô lớp đỏ mờ (alpha 0.35) lên người vi phạm + viền đỏ đậm + nhãn** `VI PHAM: NO_HELMET` / `VI PHAM: NO_SAFETY_VEST` (nhãn ASCII vì `cv2.putText` không vẽ được tiếng Việt). Chỉ vẽ trên bản copy — video output không bị ảnh hưởng.
- **Quy tắc lỗi:** mọi lỗi (thư mục không tạo được, không encode được, frame rỗng) → `status="error"` kèm message rõ, **pipeline không bao giờ crash**; không giả định ảnh đã lưu thành công. `path` chỉ là đường dẫn lưu trữ local, **không dùng làm URL công khai** (Storage Backend sẽ trả `imageUrl` chính thức sau này).
- **CHƯA LÀM (ghi rõ):** phần **gửi ảnh tới Media/Storage Backend** (multipart/form-data) — vì Backend của SV1 chưa tồn tại. Khi Backend chốt endpoint sẽ bổ sung client gửi + retry, đi cùng VI-09/VI-10.
- **Test:** `python tests/test_evidence.py` — 10 test, tất cả OK (gồm test pixel đỏ trên vùng bbox).

---

## 3. Test UI — tách hoàn toàn khỏi core logic

> **Nhắc lại nguyên tắc:** Test UI **chỉ để xem kết quả**, không phải pipeline chính thức. Nó không chứa logic xử lý nào — toàn bộ logic (detect, ghép, luật, xác nhận, chống trùng, snapshot) đều gọi từ `app/`. Sửa core không ảnh hưởng UI và ngược lại.

- **Backend UI:** `test-ui/server.py` (FastAPI). `process_video()` chạy trong thread nền, gọi lần lượt: `Detector().detect` → `associate` → `evaluate_safety` → `ViolationConfirmation.update` → `ViolationDeduplicator.filter` → `draw_detections` → `save_evidence`, và ghi **log JSONL mỗi frame 1 dòng**.

| Endpoint | Chức năng |
|---|---|
| `POST /api/upload` | Upload video → trả `job_id` |
| `POST /api/jobs/{id}/start` | Xử lý nền (poll tiến độ) |
| `GET /api/jobs/{id}/status` | Tiến độ xử lý |
| `GET /api/jobs/{id}/log` | Toàn bộ log JSONL |
| `GET /api/jobs/{id}/video` | Video đã vẽ box (`annotated.mp4`) |
| `GET /api/jobs/{id}/evidence/{event_id}.jpg` | Ảnh bằng chứng (VI-08) |
| `GET /api/jobs/{id}/meta` | Thông tin job |

- **Giao diện:** `test-ui/static/index.html` (vanilla JS, 1 file, không build step) — khung trái video, khung phải panel đồng bộ theo frame đang phát, gồm các thẻ:
  1. **Vi phạm được xác nhận + chống trùng (VI-06 + VI-07)** — thẻ đỏ, danh sách event cuối cùng (kèm camera, event_id, nối track đổi ID)
  2. **Nút "Xem bằng chứng / frame lỗi (N)"** — mở danh sách ảnh bằng chứng (người vi phạm tô đỏ); mục lưu lỗi hiện **chi tiết lỗi bên cạnh** + link nhảy tới frame đó trong video
  3. **Tóm tắt theo người** — mỗi người: xuất hiện frame nào→nào, mũ/áo có hay thiếu ở đoạn nào, đánh giá ĐỦ BẢO HỘ / VI PHẠM DỰ KIẾN / VI PHẠM XÁC NHẬN / UNKNOWN
  4. **Khung hình đang phát** — bảng detection (lớp, track, conf, bbox, kích thước px) + sự kiện track xuất hiện/biến mất
  5. **Ghép PPE–Person + Luật an toàn (VI-04 + VI-05)** — từng người ghép với mũ/áo nào, trạng thái luật
  6. **Người đang theo dõi** tại frame hiện tại
  - Ô "Nhảy tới frame" để xem đúng 1 frame.

- **Định dạng log (mỗi frame 1 dòng JSON):** `frame, t, detections[], summary, events[], associations[], persons[], confirmations[], violations[], evidence[]` — chi tiết từng field xem `test-ui/README.md`.

### Cách chạy

```bash
cd ~/Downloads/DATN/video-intelligence
source .venv/bin/activate
pip install -r test-ui/requirements.txt      # lần đầu
python -m test-ui.server
# Mở trình duyệt: http://127.0.0.1:8000
```

Lưu ý: chạy từ thư mục gốc `video-intelligence` để import được package `app`. Kết quả mỗi lần xử lý nằm ở `test-ui/results/<job_id>/` (`annotated.mp4`, `log.jsonl`, `meta.json`, `evidences/`).

---

## 4. Cách chạy toàn bộ test hiện có

### Unit test (5 file, script tự chạy — không cần pytest)

```bash
cd ~/Downloads/DATN/video-intelligence
source .venv/bin/activate
.venv/bin/python tests/test_association.py     # VI-04 — 10 test
.venv/bin/python tests/test_safety_rules.py    # VI-05 — 13 test
.venv/bin/python tests/test_confirmation.py    # VI-06 — 12 test
.venv/bin/python tests/test_deduplication.py   # VI-07 — 9 test
.venv/bin/python tests/test_evidence.py        # VI-08 — 10 test
```

Mỗi file in `[OK]`/`[FAIL]` từng test + dòng `KẾT QUẢ: tất cả test OK` ở cuối. (Tổng cộng 54 test, tất cả đang OK.)

### Chạy pipeline đầu cuối (core)

```bash
python -m app.main --source videos/video2.mp4
# → in tiến độ; khi có vi phạm đủ 3s + không trùng: dòng 🚨 VI PHẠM
# → ảnh bằng chứng: evidences/<eventId>.jpg (người vi phạm tô đỏ)
# → video output: outputs/video2_det.mp4
```

### Video mẫu đang có (`videos/`)

- `video2.mp4` — 900×1596, 429 frame, ~14s: 2 công nhân; **person #9 thiếu áo phản quang ~3s → phát 1 event** (dùng để test VI-06→08)
- `video3.mp4` — 145 frame, ~4.8s: ngắn hơn ngưỡng 3s → không có event (test âm tính)
- `video4/5/6/7/8.mp4` — các video khác (video5: 2586×1426, 1762 frame — test hiệu năng)

---

## 5. Cấu hình (app/config.py — ghi đè qua env)

| Biến env | Mặc định | Ý nghĩa |
|---|---|---|
| `VIDEO_SOURCE` | — | Nguồn video (file / rtsp / webcam index) |
| `MODEL_PATH` | `models/best.pt` | Model (đổi model = đổi biến này) |
| `MODEL_IMGSZ` | 640 | Kích thước ảnh đưa vào model |
| `DEVICE` | tự chọn | `mps` / `cpu` |
| `CAMERA_ID` | `cam-01` | cameraId cho khóa chống trùng + event |
| `EVIDENCE_DIR` | `evidences` | Thư mục lưu ảnh bằng chứng (core) |
| (hằng số) | `CONFIRMATION_DURATION_S=3.0`, `COOLDOWN_DURATION_S=30.0`, `MIN_PERSON_HEIGHT_PX=60`, `CONFIDENCE_THRESHOLD=0.5`, `PERSON_MIN_VISIBLE_WIDTH_PX=150` | Các ngưỡng — đều là **giá trị khởi đầu theo docx, cần kiểm thử theo độ phân giải**; VI-11 sẽ kéo từ Backend |

---

## 6. Chưa làm / còn thiếu / đang chờ — GHI RÕ

### Chưa làm (bước tiếp theo)

| Mã | Hạng mục | Ghi chú định hướng |
|---|---|---|
| VI-09 | Event JSON contract | Đã có hướng: ánh xạ `ViolationEvent` + `EvidenceResult` → schema docx (eventId UUID, eventType `NO_HELMET`/`NO_SAFETY_VEST`, siteId/zoneId null, occurredAt ISO 8601, confirmedDurationMs, confidence, status `CONFIRMED_VIOLATION`, personBoundingBox, evidence{imageUrl, imagePath, snapshotBase64, videoClipUrl}). Cần thêm `confidence` + `person_bbox` vào ViolationEvent; viết `docs/VI09-event-contract.md` để Backend duyệt |
| VI-10 | Dual-Transport HTTP/Kafka | Chưa có Backend → chưa gửi được; cơ chế `EVENT_TRANSPORT=http\|kafka`, không gửi trùng |
| VI-11 | Lấy safety-config | `GET /api/v1/sites/{siteId}/safety-config` theo đề xuất docx; fallback config.py |
| VI-12 | Kiểm thử tổng thể | Đã có 5 file unit test cơ bản (54 test); cần thêm test tích hợp pipeline đầu cuối, test nhiều người/che khuất, đánh giá false positive trên video thật |
| VI-13 | Restricted Area | P2 — tính năng phát hiện người vào vùng cấm (cần định nghĩa zone + rule mới) |

### Còn thiếu trong phần đã làm

1. **Gửi ảnh tới Storage Backend (VI-08):** chưa có client multipart — Backend SV1 chưa tồn tại; khi chốt endpoint sẽ bổ sung (kèm retry theo chính sách). Ảnh mới chỉ lưu local, `imageUrl` để null.
2. **Model cuối cùng:** đang dùng `models/best.pt` (YOLO11m tạm của SV1). Khi SV1 bàn giao model cuối → copy vào `models/`, đổi `MODEL_PATH`, không sửa code.
3. **siteId / zoneId:** hệ thống chưa có khái niệm zone — event sẽ gửi null, bổ sung khi Backend định nghĩa.
4. **occurredAt chuẩn:** video file không có giờ thật — dự kiến env `VIDEO_START_TIMESTAMP` + offset (hoặc giờ UTC lúc phát hiện), chốt cùng VI-09.
5. **Webcam/RTSP:** chưa test được (chưa có thiết bị).
6. **Các ngưỡng 60px / 150px / 3s / 30s:** là giá trị khởi đầu theo docx — cần kiểm thử thêm theo độ phân giải camera thật.
7. **Heuristic nối track đổi ID (VI-07):** cần đánh giá trên video thật có nhiều người ra/vào khung.
8. **Hiệu năng:** ~40–220 ms/frame trên MPS đủ cho offline; nếu cần real-time nhiều camera → cân nhắc model nhẹ hơn (YOLO11n/s) — phụ thuộc model SV1 bàn giao.

---

## 7. Phụ lục: nguồn tham chiếu

- Yêu cầu gốc: `Phase_2.docx` (mục 8 ghép PPE, mục 4 Violation Event JSON, mục 5 Evidence Snapshot Flow — các contract này là **đề xuất cần chốt với Backend**)
- Tổng quan đồ án: `TONG_QUAN_DU_AN.md`
- Hướng dẫn chi tiết UI: `test-ui/README.md`
