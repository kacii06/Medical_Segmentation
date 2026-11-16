# Dự án Phân đoạn Y tế

Kho mã nguồn nhẹ dùng để thử nghiệm các kiến trúc U‑Net cho phân đoạn ảnh y tế. Dự án cung cấp:

- **Tạo dữ liệu tổng hợp** (`download_sample_data.py`) để nhanh chóng sinh một bộ dữ liệu mẫu dạng y tế mà không cần tải lên internet.
- **Các triển khai U‑Net**:
  - `unet_scratch.py` – U‑Net thuần từ đầu.
  - `unet_pretrained.py` – U‑Net với encoder ResNet được pretrained (ResNet‑34 hoặc ResNet‑50).
  - Hàm `build_unet` giúp chuyển đổi dễ dàng giữa hai phiên bản trên.
- **Tiện ích huấn luyện** (thư mục `utils/`) để load dữ liệu, hiển thị kết quả và tính toán chỉ số Dice / IoU.

## Bắt đầu nhanh

1. **Cài đặt phụ thuộc**
   ```bash
   pip install -r requirements.txt
   # tùy chọn: pip install segmentation_models_pytorch   # để sử dụng wrapper pretrained
   ```

2. **Tạo bộ dữ liệu tổng hợp** (không cần tải dữ liệu bên ngoài)
   ```bash
   python download_sample_data.py --dir data/sample_medical --num_samples 50 --img_size 256
   ```
   > Script hiện chỉ tạo các thư mục `images/` và `masks/` bên trong thư mục chỉ định. Thư mục `sample` cũ đã được xóa.

3. **Huấn luyện mô hình**
   ```bash
   # ví dụ: huấn luyện U‑Net với encoder ResNet‑34
   python train.py --config configs/resnet34.yaml
   ```
   Bạn có thể sửa file cấu hình để thay đổi siêu tham số, backbone hoặc chuyển sang phiên bản scratch.

4. **Đánh giá / Hiển thị**
   ```bash
   python evaluate.py --model_path outputs/best.pth --data_dir data/sample_medical
   ```

## Cấu trúc dự án

```
Medical_Segmentation/
├─ data/                 # Thư mục chứa ảnh và mask được sinh ra (ví dụ: sample_medical/)
├─ models/               # Các mô hình U‑Net
│   ├─ unet_scratch.py
│   └─ unet_pretrained.py
├─ utils/                # Bộ loader dữ liệu, tính chỉ số, visualisation
├─ download_sample_data.py   # Công cụ tạo dữ liệu tổng hợp (đã dọn sạch các print không cần)
├─ train.py / evaluate.py    # Script mẫu huấn luyện và đánh giá
├─ requirements.txt
└─ README.md              # Tài liệu này
```


