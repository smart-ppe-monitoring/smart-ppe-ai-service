# smart-ppe-ai-service

> **AI Edge Engine** — Video Intelligence & PPE Violation Detection  
> Contract Version: `2.1.0-PPE-STABLE` | Repo: `smart-ppe-monitoring/smart-ppe-ai-service`

---

## 1. Tổng quan

Service Python thực hiện toàn bộ pipeline AI:
`Video ➔ YOLO Detection ➔ ByteTrack ➔ PPE Association ➔ Safety Rules ➔ MinIO Upload ➔ Kafka / HTTP Event`

---

## 2. Cấu trúc thư mục

```
smart-ppe-ai-service/
├── app/
│   ├── constants.py          # Tất cả hằng số — KHÔNG hard-code ở nơi khác
│   ├── config.py             # Đọc .env — KHÔNG log secret
│   ├── main.py               # Entry point pipeline
│   ├── video/                # VI-01: Video reader (file/RTSP)
│   ├── detection/            # VI-02: YOLO adapter (normalize labels)
│   ├── tracking/             # VI-03: ByteTrack per-camera
│   ├── association/          # VI-04: PPE-Person IoU association
│   ├── rules/                # VI-05/06/07: Safety rules + temporal + dedup
│   ├── storage/              # VI-08: MinIO evidence upload
│   ├── clients/              # TASK-A: Backend HTTP client (dynamic config)
│   ├── events/               # VI-09/10: Kafka + HTTP + Offline buffer
│   └── schemas/              # Contract Pydantic models (ViolationEvent)
├── models/                   # .pt model weights (không commit lên Git)
├── tests/                    # Unit + contract tests
├── docs/                     # PROJECT_OVERVIEW.md và tài liệu nội bộ
├── requirements.txt
├── .env.example
├── Dockerfile
└── README.md
```

---

## 3. Thiết lập môi trường

```bash
# Tạo virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# Cài thư viện
pip install -r requirements.txt

# Cấu hình môi trường
cp .env.example .env
# Sửa .env: điền MINIO_SECRET_KEY và các giá trị cụ thể
```

---

## 4. Chạy pipeline

```bash
python -m app.main
```

---

## 5. Chạy tests

```bash
pytest tests/ -v
pytest tests/test_contract.py -v   # Chỉ chạy contract tests
pytest tests/test_constants.py -v  # Chỉ chạy constant tests
```

---

## 6. Integration với Backend

### Kafka
```bash
# Kiểm tra event đã vào topic
kcat -b localhost:9092 -t ppe.violation.detected -C -o beginning -e
```

### HTTP (Sync mode)
```bash
curl -X POST http://localhost:8893/api/v1/integration/ai-vision/events?sync=true \
  -H "Content-Type: application/json" \
  -d '{
    "cameraCode": "CAM-GATE-01",
    "timestamp": "2026-10-13T08:30:00.000Z",
    "evidence": {
      "snapshotUrl": "http://localhost:9000/ppe-evidences/evidences/SITE-01/CAM-GATE-01/1234567890_track12.jpg",
      "frameResolution": { "width": 1920, "height": 1080 }
    },
    "detections": [
      { "class": "PERSON", "confidence": 0.96, "box": { "x": 420, "y": 180, "width": 240, "height": 620 }, "trackId": 12 }
    ],
    "violations": [
      { "violationType": "NO_HELMET", "severity": "HIGH", "confidence": 0.94, "description": "Không đội mũ bảo hộ", "box": { "x": 470, "y": 180, "width": 110, "height": 95 } }
    ]
  }'
```

**Expected response:**
```json
{
  "success": true,
  "data": {
    "eventId": "...",
    "incidentId": "...",
    "status": "INCIDENT_CREATED"
  }
}
```

---

## 7. Hằng số & cấu hình động

- Mọi magic string phải dùng từ [`app/constants.py`](app/constants.py)
- Giá trị `confirmationDuration`, `cooldownDuration`, `confidenceThreshold` được lấy động từ:
  - `GET /api/v1/safety-rules/system-default`
  - Fallback: giá trị trong `constants.py`

---

## 8. Changelog

| Phiên bản | Ngày | Mô tả |
|-----------|------|-------|
| `2.1.0-PPE-STABLE` | 2026-10-13 | Cấu trúc repo chuẩn hóa theo Integration Contract; thêm `constants.py`, `backend_client.py`, Pydantic schema, offline buffer SQLite |
