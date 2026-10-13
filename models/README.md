# Model Weights

Thư mục này chứa file trọng số `.pt` của YOLO. **Không commit lên Git (đã khai báo trong `.gitignore`).**

## Mô hình đang sử dụng: YOLO11s (Fine-tuned)

File: `models/yolo11s-best.pt` (hoặc alias `models/yolo11m-best.pt`)  
Kích thước file: **19.18 MB** (9.4M params, 21.5 GFLOPs).

### Thông số kỹ thuật & Đánh giá thực nghiệm

| Model               | Params | GFLOPs | mAP@50     | mAP@50-95  | Precision  | Recall     | AP@50 (person) | AP@50 (helmet) | AP@50 (safety_vest) |
| ------------------- | ------ | ------ | ---------- | ---------- | ---------- | ---------- | -------------- | -------------- | ------------------- |
| **yolo11s-best.pt** | 9.4M   | 21.5   | **0.9307** | **0.6236** | **0.9337** | **0.8792** | 0.8894         | 0.9626         | 0.9400              |

### Cách cấu hình trong `.env`

```env
YOLO_MODEL_PATH=models/yolo11s-best.pt
MODEL_IMGSZ=640
```
