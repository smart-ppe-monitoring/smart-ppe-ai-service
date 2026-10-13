# Tổng quan đồ án DATN — Phát hiện đồ bảo hộ lao động (PPE) bằng YOLO

> File này mô tả hiện trạng thư mục `DATN` tại thời điểm 12/10/2026, giúp nắm nhanh đồ án trước khi bắt tay vào làm việc.

## 1. Đồ án nói về điều gì

- **Bài toán:** Phát hiện đồ bảo hộ lao động cá nhân (PPE — Personal Protective Equipment) trong hình ảnh công trường xây dựng, phục vụ giám sát an toàn lao động.
- **3 lớp nhận diện:** `person` (người), `helmet` (mũ bảo hộ), `safety_vest` (áo phản quang).
- **Phương pháp:** Fine-tune (tinh chỉnh) các mô hình thuộc họ YOLO (YOLOv8, YOLO11) trên bộ dữ liệu `dataset_ppe_3class`.
- **Môi trường huấn luyện:** Google Colab + Google Drive (đường dẫn gốc trong `args.yaml` là `/content/ppe_training/...` và `/content/drive/MyDrive/DATN/...`).
- **Trạng thái hiện tại:** Thư mục chứa **kết quả huấn luyện** (số liệu, biểu đồ, weights, ảnh demo) và **1 notebook huấn luyện** (bản YOLO11s/Kaggle). Dataset và các bản notebook chạy trên Colab vẫn chưa có trên máy.

## 2. Cấu trúc thư mục

```
DATN/
├── notebooks/
│   └── 02_ppe3_css_yolo11.ipynb         # Notebook huấn luyện (bản YOLO11s, viết cho Kaggle)
└── training_results/
    └── runs/
        ├── model_comparison_summary.csv   # Bảng so sánh 4 mô hình đã fine-tune
        ├── per_class_ap50.csv             # AP@50 chi tiết theo từng lớp
        ├── Gợi ý lựa chọn mô hình.docx    # Tài liệu phân tích + gợi ý chọn mô hình
        ├── yolo11m_ppe_finetune/          # Kết quả huấn luyện YOLO11m (86 MB)
        │   ├── args.yaml                  # Cấu hình huấn luyện
        │   ├── results.csv                # Log metrics từng epoch (100 epochs)
        │   ├── results.png                # Biểu đồ tổng hợp loss & metrics
        │   ├── BoxP/BoxR/BoxF1/BoxPR_curve.png   # Các đường cong P, R, F1, PR
        │   ├── confusion_matrix(_normalized).png # Ma trận nhầm lẫn
        │   ├── labels.jpg                 # Ảnh minh họa các lớp + màu nhãn
        │   ├── train_batch*.jpg           # Ảnh batch huấn luyện (kèm augmentation)
        │   ├── val_batch*_labels/pred.jpg # Ảnh batch validation (ground truth vs dự đoán)
        │   └── weights/
        │       ├── best.pt                # Weights tốt nhất (≈ 40 MB)
        │       └── last.pt                # Weights epoch cuối (≈ 40 MB)
        ├── yolo11m_test_eval/             # Kết quả đánh giá trên tập TEST (5 MB)
        │   ├── BoxP/BoxR/BoxF1/BoxPR_curve.png
        │   ├── confusion_matrix(_normalized).png
        │   └── val_batch*_labels/pred.jpg # Ảnh dự đoán trên tập test
        └── demo_yolov8n/ demo_yolo11n/ demo_yolo11s/ demo_yolo11m/
                                            # Ảnh demo inference của 4 mô hình (3 ảnh/mô hình)
```

## 3. Kết quả thực nghiệm

### 3.1 Bảng so sánh 4 mô hình (`model_comparison_summary.csv`)

| Mô hình (Fine-tuned) | Params | GFLOPs | mAP@50 | mAP@50-95 | Precision | Recall | Thời gian train |
|---|---|---|---|---|---|---|---|
| YOLOv8n | 3.2M | 8.7 | 0.9055 | 0.5861 | 0.9230 | 0.8361 | N/A |
| YOLO11n | 2.6M | 6.5 | 0.8864 | 0.5818 | 0.8996 | 0.8050 | N/A |
| YOLO11s | 9.4M | 21.5 | 0.9307 | 0.6236 | 0.9337 | 0.8792 | N/A |
| **YOLO11m** | **20.1M** | **68.0** | **0.9317** | **0.6541** | **0.9827** | **0.8893** | **125.4 phút** |

### 3.2 AP@50 theo từng lớp (`per_class_ap50.csv`)

| Mô hình | person | helmet | safety_vest |
|---|---|---|---|
| YOLOv8n | 0.8510 | 0.9652 | 0.9004 |
| YOLO11n | 0.8495 | 0.9346 | 0.8751 |
| YOLO11s | 0.8894 | 0.9626 | 0.9400 |
| YOLO11m | 0.8871 | 0.9639 | 0.9441 |

Nhận xét: lớp `helmet` và `safety_vest` được nhận diện tốt (AP ≥ 0.93), lớp `person` khó hơn (AP ~ 0.85–0.89).

### 3.3 Tài liệu "Gợi ý lựa chọn mô hình.docx"

Tài liệu phân tích sẵn để phục vụ bảo vệ trước hội đồng, đưa ra 2 phương án:

- **Phương án 1 — YOLO11n (khuyên dùng cho hệ thống Edge/Real-time):** siêu nhẹ (2.6M params, 6.5 GFLOPs — nhẹ hơn YOLOv8n 19% tham số, 25% tính toán) nhưng vẫn đạt mAP@50 88.64%. Phù hợp camera giám sát 24/7, thiết bị biên (Jetson Nano, Raspberry Pi), tiết kiệm điện và băng thông.
- **Phương án 2 — YOLO11s (cân bằng hoàn hảo):** mAP@50 93.07%, mAP@50-95 62.36%, 9.4M params / 21.5 GFLOPs — chạy mượt trên PC/GPU phổ thông.
- **Không nên chọn YOLO11m cho ứng dụng thực tế:** độ chính xác bão hòa (chỉ +0.1% mAP@50 so với YOLO11s) trong khi params gấp hơn 2 lần và GFLOPs gấp hơn 3 lần.

Tài liệu cũng gợi ý sẵn một đoạn văn mẫu để viết vào báo cáo.

### 3.4 Chi tiết huấn luyện YOLO11m (`yolo11m_ppe_finetune/args.yaml`)

- Model: `yolo11m.pt` (pretrained), task detect, mode train
- Data: `/content/ppe_training/dataset_ppe_3class/data.yaml`
- **100 epochs**, imgsz **640**, batch **16**, patience 20, optimizer auto, amp bật
- Augmentation: mosaic 1.0 (tắt sau 10 epoch cuối — close_mosaic=10), hsv, translate 0.1, scale 0.5, fliplr 0.5, erasing 0.4, auto_augment randaugment
- Loss weights: box 7.5, cls 0.5, dfl 1.5

Kết quả epoch cuối (epoch 100): mAP50 **0.908**, mAP50-95 **0.620**, Precision 0.955, Recall 0.828.

### 3.5 Đánh giá trên tập test (`yolo11m_test_eval/`)

Chỉ chứa các biểu đồ (đường cong P/R/F1/PR, ma trận nhầm lẫn) và ảnh dự đoán trên tập test — không có `results.csv`, tức đây là kết quả của một lần chạy đánh giá (val/test) riêng.

### 3.6 Ảnh demo inference (`demo_*/`)

Mỗi mô hình (yolov8n, yolo11n, yolo11s, yolo11m) có 3 ảnh demo kết quả dự đoán (bounding box + nhãn lớp). Tên file dạng `*.rf.<hash>.jpg` cho thấy ảnh gốc lấy từ dataset xuất bởi **Roboflow**.

## 4. Code huấn luyện — đã tìm thấy trên máy

### 4.1 Notebook `notebooks/02_ppe3_css_yolo11.ipynb`

Tìm thấy ở `~/Downloads` (12/10/2026) và đã copy vào dự án. Đây là notebook huấn luyện hoàn chỉnh, viết cho **Kaggle** (đường dẫn dataset trong notebook là `/kaggle/input/...`), gồm 8 phần:

1. **Cài đặt** ultralytics + kiểm tra CUDA
2. **Cấu hình `CFG`**: 3 lớp mới (0=Person, 1=Helmet, 2=Safety-Vest), seed 42, imgsz 640, 100 epochs, patience 25, batch 16, base model **yolo11s.pt**, tỷ lệ ảnh nền tối đa 10%
3. **Đổi nhãn từ dataset gốc**: giữ Person(5)→0, Hardhat(0)→1, Safety Vest(7)→2; **xóa** các lớp Mask, NO-Mask, NO-Hardhat, NO-Safety Vest, Safety Cone, machinery, vehicle
4. **Chống rò rỉ dữ liệu**: gộp ảnh theo ảnh nguồn gốc (trước khi Roboflow augment, tách theo phần trước `.rf.`), chọn ≤10% ảnh nền, chia train/valid/test không trùng ảnh nguồn
5. **Kiểm tra nhãn bằng mắt** + thống kê instance theo lớp
6. **Huấn luyện** YOLO11 với `cos_lr`, đánh giá valid + test từng lớp
7. **Ảnh dự đoán mẫu** + đo tốc độ suy luận (ms/ảnh)
8. **Lưu model** (.pt, .onnx) + metadata.json

> **Lưu ý quan trọng:**
> - Notebook này là bản dùng **YOLO11s trên Kaggle**. Các kết quả trong `training_results` (YOLOv8n, YOLO11n, YOLO11s, YOLO11m) được tạo bởi **các bản đã chỉnh sửa của notebook này chạy trên Google Colab** (đường dẫn `/content/...` trong `args.yaml`). Các bản đó hiện chưa có trên máy — nằm trên Colab/Drive.
> - Dataset huấn luyện thực tế trên Colab là `/content/ppe_training/dataset_ppe_3class/` (cũng chưa có trên máy).
> - Notebook còn có mục "Bước tiếp theo": bổ sung dữ liệu Safety-Vest (yếu nhất) từ dataset khác, gộp dataset, xây bộ test cố định, và hướng phát triển là **Tracking + Association + Rule Engine** để đo Violation Precision/Recall, False Alert Rate.

### 4.2 Nguồn dataset

- Dataset gốc: **"Construction Site Safety Image Dataset"** (Roboflow Universe, license **CC BY 4.0** — phải ghi nguồn khi dùng), có trên Kaggle: `snehilsanyal/construction-site-safety-image-dataset-roboflow`
- Bản Roboflow v27 có 2801 ảnh nhưng tập train đã được augment sẵn — số ảnh gốc độc lập chỉ **~717** (train 521, valid 114, test 82). Không nên báo cáo "2.8k ảnh".
- Chưa có trên máy → cần tải về (qua Kaggle API hoặc tải tay).

## 5. Những gì CHƯA có trên máy

- Dataset `css-data` / `dataset_ppe_3class` (trên Kaggle/Drive)
- Các bản notebook đã chỉnh sửa chạy trên Colab (YOLOv8n, YOLO11n, YOLO11m)
- Weights của YOLOv8n, YOLO11n, YOLO11s (máy chỉ có `best.pt`/`last.pt` của YOLO11m)
- Mã nguồn hệ thống/ứng dụng demo (web app, camera real-time...)
- Báo cáo / slide đồ án

## 6. Gợi ý việc cần làm tiếp

1. **Chốt mô hình** dựa trên tài liệu gợi ý (YOLO11n cho real-time/edge, YOLO11s cho độ chính xác cao).
2. **Xây dựng app demo** (web/camera) sử dụng weights có sẵn — hiện chỉ có `best.pt` của YOLO11m trong máy; nếu chọn YOLO11n/s cần tải weights tương ứng từ Colab/Drive về.
3. **Hoàn thiện báo cáo & slide** với các số liệu và biểu đồ đã có sẵn trong thư mục.
4. Nếu cần thêm thực nghiệm (ví dụ đo FPS, latency trên thiết bị thực), thực hiện và bổ sung vào bảng so sánh.

## 7. Cách chạy dịch vụ video-intelligence (Phase 2 — cập nhật 13/10/2026)

Dịch vụ `video-intelligence/` (của SV2) đã xong **VI-01 → VI-08**: đọc video → detect → tracking (ByteTrack) → ghép PPE–Person → luật an toàn → xác nhận 3 giây → chống trùng 30 giây → snapshot bằng chứng (người vi phạm được tô đỏ). Chi tiết đầy đủ: `video-intelligence/docs/BAO-CAO-VI01-VI08.md`.

Mã nguồn đẩy lên: https://github.com/smart-ppe-monitoring/smart-ppe-ai-service.git

### 7.1 Cài đặt (1 lần)

```bash
cd ~/Downloads/DATN/video-intelligence
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
# Test UI cần thêm:
pip install -r test-ui/requirements.txt
```

### 7.2 Chạy pipeline đầu cuối (terminal)

```bash
cd ~/Downloads/DATN/video-intelligence
source .venv/bin/activate
python -m app.main --source videos/video2.mp4
```

Kết quả: in tiến độ theo frame; khi có vi phạm đủ 3 giây + không trùng 30 giây → dòng `🚨 VI PHẠM` kèm ảnh bằng chứng lưu tại `evidences/<eventId>.jpg`; video có vẽ box ghi tại `outputs/`.

### 7.3 Chạy Test UI (xem kết quả trên trình duyệt)

```bash
cd ~/Downloads/DATN/video-intelligence
source .venv/bin/activate
python -m test-ui.server
# Mở trình duyệt: http://127.0.0.1:8000
# Upload video (vd videos/video2.mp4) → chờ xử lý → xem video có box + log đồng bộ theo frame
```

> Test UI **tách hoàn toàn khỏi core logic** (`app/`) — chỉ để xem kết quả, chỉ dùng lại API công khai của core.

### 7.4 Chạy unit test (54 test)

```bash
cd ~/Downloads/DATN/video-intelligence
source .venv/bin/activate
.venv/bin/python tests/test_association.py     # VI-04 — 10 test
.venv/bin/python tests/test_safety_rules.py    # VI-05 — 13 test
.venv/bin/python tests/test_confirmation.py    # VI-06 — 12 test
.venv/bin/python tests/test_deduplication.py   # VI-07 — 9 test
.venv/bin/python tests/test_evidence.py        # VI-08 — 10 test
```

### 7.5 Đổi model không sửa code

```bash
export MODEL_PATH=models/best.pt      # model hiện tại (YOLO11m tạm của SV1)
# Khi có model cuối (YOLO11n/s): copy file vào models/ rồi đổi MODEL_PATH là xong
```

Các ngưỡng 3s/30s/60px/0.5 nằm trong `app/config.py`, ghi đè được qua biến môi trường.
