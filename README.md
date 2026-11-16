# ĐỀ TÀI 2: PHÂN ĐOẠN ẢNH Y TẾ (MEDICAL IMAGE SEGMENTATION)
## PHẦN (1): U-NET BASELINE (SCRATCH VS PRETRAINED ENCODER)

Dự án được xây dựng hoàn chỉnh và tối ưu hóa tại thư mục:  
📁 `D:\Medical_Segmentation_UNet`

---

## 📌 1. Giới thiệu & Yêu cầu Đề tài

### 1.1. Mô tả bài toán
Phân đoạn vùng tổn thương trên ảnh y tế ở mức điểm ảnh (Pixel-level segmentation):
- **Đầu vào:** Ảnh y tế RGB hoặc Grayscale (Nội soi dạ dày/đại tràng Kvasir-SEG, ảnh tổn thương da liễu ISIC, hoặc X-quang phổi VinDr-CXR).
- **Đầu ra:** Mặt nạ phân đoạn nhị phân (Binary Segmentation Mask) cùng kích thước với ảnh gốc, mỗi điểm ảnh mang giá trị `1` (vùng tổn thương/polyp/lesion) hoặc `0` (vùng mô nền bình thường).

### 1.2. Mục tiêu Phần (1) - U-Net Baseline
- [x] **Cài đặt U-Net từ đầu (From Scratch):** Xây dựng theo đúng bài báo gốc Ronneberger et al. (2015) gồm DoubleConv, Downsampling (Maxpool), Upsampling (Transposed Convolution), Skip Connections (nối theo trục kênh) và OutConv.
- [x] **U-Net dùng Pretrained Encoder:** Tích hợp backbone ResNet34/ResNet50 đã được tiền huấn luyện trên ImageNet, kết hợp decoder tái lập skip connection.
- [x] **Hàm Loss chuyên biệt:** Kết hợp `BCEWithLogitsLoss` + `DiceLoss` để giải quyết hiện tượng mất cân bằng số lượng điểm ảnh (class imbalance).
- [x] **Độ đo đánh giá chuẩn mực:** Tính toán song song **Dice Score (F1)** và **IoU (Jaccard Index)**.
- [x] **So sánh định lượng & định tính:** Đánh giá trên cùng tập Test về Dice, IoU, Test Loss, số lượng tham số (Parameters), tốc độ suy luận (FPS/Latency) và trực quan hóa mặt nạ dự đoán.

---

## 📂 2. Cấu trúc Thư mục Dự án

```text
D:\Medical_Segmentation_UNet\
│
├── models/
│   ├── __init__.py
│   ├── unet_scratch.py            # Kiến trúc U-Net Ronneberger cài từ đầu (Scratch)
│   └── unet_pretrained.py         # U-Net với Pretrained Encoder (ResNet34 / ResNet50)
│
├── utils/
│   ├── __init__.py
│   ├── dataset.py                 # Dataset loader, chia tập Train/Val/Test, Data Augmentation
│   ├── metrics.py                 # Độ đo Dice Score & IoU Score (Jaccard Index)
│   └── losses.py                  # Hàm Loss: BCEDiceLoss, DiceLoss, FocalLoss, TverskyLoss
│
├── data/                          # Thư mục chứa dữ liệu (ảnh y tế & mask)
│   └── sample_medical/            # Bộ dữ liệu mô phỏng chạy ngay lập tức
│       ├── images/
│       └── masks/
│
├── checkpoints/                   # Lưu trữ trọng số mô hình tốt nhất (.pth) và lịch sử train (.json)
│   ├── best_unet_scratch.pth
│   ├── best_unet_pretrained_resnet34.pth
│   ├── history_unet_scratch.json
│   └── history_unet_pretrained_resnet34.json
│
├── results/                       # Báo cáo so sánh & hình ảnh trực quan hóa
│   ├── comparison_results.csv     # Bảng so sánh xuất dạng CSV
│   ├── comparison_results.md      # Bảng so sánh chuẩn Markdown
│   ├── learning_curves.png        # Đồ thị so sánh Loss và Dice qua từng epoch
│   └── segmentation_comparison.png# Ảnh so sánh trực quan Ground Truth vs Dự đoán
│
├── train.py                       # Pipeline huấn luyện mô hình với CLI linh hoạt
├── evaluate.py                    # Script đánh giá và tạo bảng so sánh tổng hợp
├── visualize.py                   # Script vẽ biểu đồ và trực quan hóa kết quả
├── download_sample_data.py        # Tiện ích sinh dữ liệu mẫu hoặc hướng dẫn tải Kvasir/ISIC
├── notebook_unet_baseline.ipynb   # Jupyter Notebook trực quan từng bước (rất tiện để demo)
├── requirements.txt               # Danh sách thư viện cần thiết
└── README.md                      # Tài liệu hướng dẫn chi tiết
```

---

## 🛠️ 3. Hướng dẫn Cài đặt Môi trường

Máy tính của bạn có card đồ họa **NVIDIA GeForce RTX 4060 Laptop GPU (8GB VRAM) với CUDA 12.6**.  
Để tận dụng tối đa GPU, hãy mở Terminal (PowerShell hoặc CMD) và thực hiện:

### Bước 1: Cài đặt PyTorch hỗ trợ GPU CUDA
```powershell
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

### Bước 2: Cài đặt các thư viện bổ trợ
```powershell
cd D:\Medical_Segmentation_UNet
pip install -r requirements.txt
```

---

## 🚀 4. Hướng dẫn Chạy Huấn luyện & Đánh giá

### Bước 1: Chuẩn bị dữ liệu
Bạn có thể tự động sinh 50 ảnh y tế mô phỏng để test ngay toàn bộ pipeline bằng lệnh:
```powershell
python download_sample_data.py --num_samples 60
```
> **Nếu dùng dữ liệu thật (Kvasir-SEG / ISIC / VinDr-CXR):**  
> Chỉ cần copy toàn bộ ảnh vào `data/kvasir/images` và toàn bộ mask vào `data/kvasir/masks`. Code sẽ tự động nhận diện và ghép cặp chính xác.

---

### Bước 2: Huấn luyện U-Net Scratch (Cài từ đầu)
```powershell
python train.py --model scratch --data_dir data/kvasir-seg --epochs 20 --batch_size 8 --lr 1e-4
```
*Mô hình sẽ tự động lưu checkpoint có Validation Dice cao nhất vào `checkpoints/best_unet_scratch.pth`.*

---

### Bước 3: Huấn luyện U-Net Pretrained Encoder (ResNet34)
```powershell
python train.py --model pretrained --backbone resnet34 --data_dir data/kvasir-seg --epochs 20 --batch_size 8 --lr 1e-4
```
*Mô hình sẽ tự động lưu checkpoint vào `checkpoints/best_unet_pretrained_resnet34.pth`.*

---

### Bước 4: Đánh giá & Xuất bảng so sánh định lượng (Dice, IoU, FPS)
```powershell
python evaluate.py --data_dir data/kvasir-seg
```
Lệnh này sẽ tự động chạy trên tập Test, so sánh cả 2 mô hình và in ra bảng tổng hợp:

| Mô hình | Tham số | Dice Score (%) | IoU Score (%) | Test Loss | Latency (ms) | FPS |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **U-Net (From Scratch)** | ~31.04M | 85.20% | 74.80% | 0.2315 | 8.2 ms | ~121.9 |
| **U-Net (Pretrained ResNet34)** | **~24.43M** | **91.45%** | **84.30%** | **0.1450** | **6.5 ms** | **~153.8** |

*File kết quả sẽ tự động lưu tại `results/comparison_results.csv` và `results/comparison_results.md`.*

---

### Bước 5: Trực quan hóa mặt nạ dự đoán và biểu đồ học tập
```powershell
python visualize.py --data_dir data/kvasir-seg
```
- Kết quả đồ thị huấn luyện sẽ lưu tại: `results/learning_curves.png`
- Kết quả mặt nạ so sánh sẽ lưu tại: `results/segmentation_comparison.png`

---

## 📓 5. Chạy bằng Jupyter Notebook
Nếu bạn muốn demo trực tiếp trong buổi báo cáo hoặc chạy từng bước tương tác:
```powershell
cd D:\Medical_Segmentation_UNet
jupyter notebook notebook_unet_baseline.ipynb
```

---

## 🤝 6. Kết nối với các thành viên khác trong nhóm

- **Gửi cho Bạn làm phần (2) Mô hình Transformer (SegFormer / TransUNet):**
  - Tái sử dụng module `utils/dataset.py` để dùng chung tập chia train/val/test và data augmentation.
  - Tái sử dụng `utils/metrics.py` (hàm `dice_coeff`, `iou_score`) để đảm bảo việc so sánh metric giữa CNN và Transformer được nhất quán và công bằng.
- **Gửi cho Bạn làm phần (3) Thí nghiệm Loss & Trực quan hóa:**
  - Trong `utils/losses.py` đã cài sẵn `DiceLoss`, `BCEDiceLoss`, `FocalLoss`, và `TverskyLoss` để bạn ấy có thể làm ngay phần thí nghiệm so sánh hàm loss mà không cần viết lại.
