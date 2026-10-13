# Hướng Dẫn Tích Hợp AI-Vision & Backend (Contract 2.1.0-PPE-STABLE)

Tài liệu kỹ thuật mô tả kiến trúc, hợp đồng dữ liệu, các lệnh kiểm thử và quy trình đồng bộ giữa **smart-ppe-ai-service** (Edge AI) và hệ thống Backend Microservices.

---

## 1. Kiến Trúc Luồng Dữ Liệu (High-Level Dataflow)

```
[Camera RTSP / Video Stream]
         │
         ▼
[1. VideoReader] (OpenCV, robust frame decode)
         │
         ▼
[2. Detector] (YOLOv8 / YOLO11) -> [3. Tracker] (ByteTrack per camera)
         │
         ▼
[4. Associator] (Ghép Person ↔ Helmet / Vest theo mỏ neo hình học & IoU)
         │
         ▼
[5. Safety Rules Engine] (Lọc kích thước, kiểm tra mép khung, đánh giá vi phạm)
         │
         ▼
[6. Temporal Confirmation] (Xác nhận vi phạm liên tục theo timestamp — mặc định 3.0s)
         │
         ▼
[7. Deduplicator] (Chống phát trùng trong cooldown 30s & nhận diện id-switch track)
         │
         ▼
[8. Evidence Snapshot] (Vẽ translucent red overlay, lưu local & đẩy MinIO S3)
         │
         ▼
[9. Schema Validator] (Pydantic Contract 2.1.0-PPE-STABLE)
         │
         ▼
[10. Resilient Multi-Transport Dispatcher]
    ├── (1st) Kafka Topic: ppe.violation.detected
    ├── (2nd) HTTP Fallback: POST /api/v1/integration/ai-vision/events
    └── (3rd) Local SQLite Offline Buffer -> Batch Flush (/events/batch)
```

---

## 2. Hợp Đồng Dữ Liệu (Contract Schema Payload)

Mọi sự kiện phát hiện vi phạm PPE gửi về Backend qua Kafka hoặc HTTP đều phải tuân thủ schema JSON sau:

```json
{
  "cameraCode": "CAM-GATE-01",
  "timestamp": "2026-10-13T08:30:00.000Z",
  "evidence": {
    "snapshotUrl": "http://localhost:9000/ppe-evidences/evidences/SITE-01/CAM-GATE-01/1760344200_track12.jpg",
    "clipUrl": null,
    "frameResolution": {
      "width": 1920,
      "height": 1080
    }
  },
  "detections": [
    {
      "class": "PERSON",
      "confidence": 0.96,
      "box": {
        "x": 420.0,
        "y": 180.0,
        "width": 240.0,
        "height": 620.0
      },
      "trackId": 12
    }
  ],
  "violations": [
    {
      "violationType": "NO_HELMET",
      "severity": "HIGH",
      "confidence": 0.94,
      "description": "Không đội mũ bảo hộ lao động",
      "box": {
        "x": 470.0,
        "y": 180.0,
        "width": 110.0,
        "height": 95.0
      }
    }
  ],
  "summary": {
    "totalPersons": 1,
    "totalViolations": 1,
    "highestSeverity": "HIGH"
  },
  "metadata": {
    "confirmedDurationMs": 3000,
    "cooldownDurationMs": 30000,
    "model": "best.pt",
    "contractVersion": "2.1.0-PPE-STABLE"
  }
}
```

---

## 3. Lệnh Kiểm Thử Tích Hợp (curl & kcat)

### 3.1. Lấy Cấu Hình Luật An Toàn Động Từ Backend (Dynamic Safety Rules)

```bash
# Lấy cấu hình mặc định toàn hệ thống:
curl -X GET "http://localhost:3000/api/v1/safety-rules/system-default" \
     -H "Accept: application/json"

# Lấy cấu hình hiệu lực theo camera và công trường cụ thể:
curl -X GET "http://localhost:3000/api/v1/safety-rules/effective?cameraId=CAM-01&siteId=SITE-01" \
     -H "Accept: application/json"
```

### 3.2. Gửi Sự Kiện Vi Phạm Qua HTTP (Single Event)

```bash
curl -X POST "http://localhost:3000/api/v1/integration/ai-vision/events" \
     -H "Content-Type: application/json" \
     -d '{
       "cameraCode": "CAM-01",
       "timestamp": "2026-10-13T10:00:00.000Z",
       "evidence": {
         "snapshotUrl": "http://localhost:9000/ppe-evidences/evidences/SITE-01/CAM-01/demo.jpg",
         "frameResolution": {"width": 1920, "height": 1080}
       },
       "detections": [
         {
           "class": "PERSON",
           "confidence": 0.95,
           "box": {"x": 100.0, "y": 100.0, "width": 80.0, "height": 220.0},
           "trackId": 1
         }
       ],
       "violations": [
         {
           "violationType": "NO_HELMET",
           "severity": "HIGH",
           "confidence": 0.93,
           "description": "Không đội mũ bảo hộ",
           "box": {"x": 110.0, "y": 100.0, "width": 60.0, "height": 50.0}
         }
       ]
     }'
```

### 3.3. Nạp Lại Sự Kiện Từ Offline Buffer Qua Batch Endpoint

```bash
curl -X POST "http://localhost:3000/api/v1/integration/ai-vision/events/batch" \
     -H "Content-Type: application/json" \
     -d '{
       "events": [
         {
           "cameraCode": "CAM-01",
           "timestamp": "2026-10-13T09:58:00.000Z",
           "evidence": {
             "snapshotUrl": "http://localhost:9000/ppe-evidences/evidences/SITE-01/CAM-01/buffered_1.jpg",
             "frameResolution": {"width": 1920, "height": 1080}
           },
           "detections": [
             {
               "class": "PERSON",
               "confidence": 0.92,
               "box": {"x": 200.0, "y": 150.0, "width": 70.0, "height": 200.0},
               "trackId": 2
             }
           ],
           "violations": [
             {
               "violationType": "NO_VEST",
               "severity": "MEDIUM",
               "confidence": 0.90,
               "description": "Không mặc áo phản quang",
               "box": {"x": 200.0, "y": 180.0, "width": 70.0, "height": 100.0}
             }
           ]
         }
       ]
     }'
```

### 3.4. Kiểm Tra & Giám Sát Kafka Topic (kcat / kafkacat)

```bash
# Lắng nghe các sự kiện vi phạm được publish vào Kafka:
kcat -b localhost:9092 -t ppe.violation.detected -C -f 'Partition: %p | Offset: %o\nPayload: %s\n---\n'

# Publish thử nghiệm 1 message mẫu vào Kafka:
echo '{"topic":"ppe.violation.detected","producer":"smart-ppe-ai-service","payload":{"cameraCode":"CAM-01","timestamp":"2026-10-13T10:00:00.000Z","evidence":{"snapshotUrl":"http://localhost:9000/ppe-evidences/test.jpg"},"detections":[],"violations":[{"violationType":"NO_HELMET","severity":"HIGH","confidence":0.95}]}}' | kcat -b localhost:9092 -t ppe.violation.detected -P
```

---

## 4. Nhật Ký Thay Đổi (Changelog & Migration)

### Phiên bản 2.1.0-PPE-STABLE
- **Hợp nhất Repository**: Hoàn tất di chuyển toàn bộ logic phát hiện, tracking, ghép PPE, rule engine và evidence snapshot từ `video-intelligence/` vào cấu trúc chuẩn microservice `smart-ppe-ai-service/`.
- **Loại bỏ Magic Strings**: Tập trung hóa toàn bộ định danh trạng thái (`COMPLIANT`, `VIOLATION_CANDIDATE`, `UNKNOWN`), loại vi phạm (`NO_HELMET`, `NO_VEST`), lớp nhãn (`PERSON`, `HARD_HAT`, `SAFETY_VEST`), và topics vào module `app/constants.py`.
- **Cơ chế Khả Dụng Cao 3 Tầng**:
  - Gửi sự kiện ưu tiên qua Kafka broker (`ppe.violation.detected`).
  - Tự động fallback sang HTTP REST API khi Kafka gián đoạn.
  - Lưu trữ sự kiện vào SQLite buffer cục bộ (`offline_buffer.db`) khi mất mạng hoàn toàn và tự động đồng bộ (flush) theo lô (tối đa 50 sự kiện/lần) khi kết nối phục hồi.
- **Tích Hợp Cấu Hình Động (Dynamic Config)**: Tự động lấy ngưỡng confidence, thời gian xác nhận (`confirmationDuration`), và thời gian cooldown (`cooldownDuration`) từ backend `/api/v1/safety-rules/*` khi khởi động.
- **Bảo Vệ Bí Mật Môi Trường**: Loại bỏ mọi hardcode secret, cập nhật `.env.example` và thiết lập `.gitignore` loại trừ file weights, cache, buffer.
