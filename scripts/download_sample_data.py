"""
=============================================================================
Script chuẩn bị dữ liệu mẫu cho bài toán phân đoạn ảnh y tế:
1. Tự động sinh tập dữ liệu mô phỏng tổn thương y tế (Synthetic Medical Dataset)
   để người dùng có thể chạy thử pipeline huấn luyện ngay lập tức mà không cần mạng.
2. Cung cấp hướng dẫn và liên kết tải các bộ dữ liệu y tế chuẩn:
   - Kvasir-SEG (Polyp nội soi đường tiêu hóa)
   - ISIC (Tổn thương da liễu Melanoma)
   - VinDr-CXR (X-quang lồng ngực)
=============================================================================
"""

import os
import argparse
import numpy as np
from PIL import Image, ImageDraw, ImageFilter


def create_synthetic_dataset(output_dir: str = "data/samples/sample_medical", num_samples: int = 50, img_size: int = 256):
    """
    Tạo bộ dữ liệu mô phỏng ảnh y tế (Polyp nội soi / Tổn thương da) và mặt nạ Ground Truth:
    - Background: Mô phỏng bề mặt mô/niêm mạc với gradient màu hồng/đỏ da và nhiễu vi mô.
    - Lesion: Vùng tổn thương hình bầu dục/bất quy tắc với độ tương phản màu và biên mờ.
    - Mask: Mặt nạ nhị phân 0 (nền) và 255 (tổn thương).
    """
    images_dir = os.path.join(output_dir, "images")
    masks_dir = os.path.join(output_dir, "masks")
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(masks_dir, exist_ok=True)

    print(f"[*] Đang tạo {num_samples} ảnh y tế mẫu tại: {output_dir}")

    rng = np.random.default_rng(42)

    for i in range(num_samples):
        # 1. Tạo nền (Background mô phỏng niêm mạc/da y tế)
        base_color = rng.integers(140, 200, size=3)
        # Giảm kênh Blue để tạo tông đỏ/hồng/nâu đặc trưng của nội soi và da liễu
        base_color[2] = int(base_color[2] * 0.6)

        bg = np.ones((img_size, img_size, 3), dtype=np.uint8) * base_color
        # Thêm nhiễu kết cấu (texture noise)
        noise = rng.normal(0, 15, (img_size, img_size, 3)).astype(np.int16)
        bg = np.clip(bg.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        img = Image.fromarray(bg)

        # 2. Tạo mặt nạ nhãn (Mask ground truth)
        mask = Image.new("L", (img_size, img_size), 0)
        draw_mask = ImageDraw.Draw(mask)
        draw_img = ImageDraw.Draw(img)

        # Tọa độ và kích thước vùng tổn thương (Polyp / Lesion)
        cx = rng.integers(int(img_size * 0.3), int(img_size * 0.7))
        cy = rng.integers(int(img_size * 0.3), int(img_size * 0.7))
        rx = rng.integers(int(img_size * 0.12), int(img_size * 0.28))
        ry = rng.integers(int(img_size * 0.12), int(img_size * 0.28))

        bbox = [cx - rx, cy - ry, cx + rx, cy + ry]

        # Vẽ tổn thương lên mask (giá trị 255)
        draw_mask.ellipse(bbox, fill=255)

        # Vẽ tổn thương lên ảnh gốc với màu sắc sậm/khác biệt
        lesion_color = (
            int(np.clip(base_color[0] * 0.65, 0, 255)),
            int(np.clip(base_color[1] * 0.50, 0, 255)),
            int(np.clip(base_color[2] * 0.40, 0, 255))
        )
        draw_img.ellipse(bbox, fill=lesion_color)

        # Làm mờ nhẹ ảnh để tạo sự chuyển tiếp tự nhiên giữa tổn thương và mô lành
        img = img.filter(ImageFilter.GaussianBlur(radius=1.5))

        # Lưu ảnh và mask
        sample_id = f"sample_{i+1:04d}"
        img.save(os.path.join(images_dir, f"{sample_id}.jpg"), quality=95)
        mask.save(os.path.join(masks_dir, f"{sample_id}.png"))

    print(f"[✓] Đã tạo thành công {num_samples} mẫu tại:")
    print(f"    - Ảnh gốc : {images_dir}")
    print(f"    - Mặt nạ   : {masks_dir}")


def print_dataset_instructions():
    print("""
=============================================================================
HƯỚNG DẪN TẢI CÁC BỘ DỮ LIỆU CHUẨN TRONG ĐỀ BÀI:
=============================================================================
1. Kvasir-SEG (Polyp nội soi đường tiêu hóa - Khuyên dùng cho U-Net):
   - Link tải: https://datasets.simula.no/kvasir-seg/
   - Giải nén: Đặt ảnh vào `data/kvasir/images` và mask vào `data/kvasir/masks`

2. ISIC Skin Lesion (Tổn thương da liễu):
   - Link tải: https://challenge.isic-archive.com/data/
   - Bộ ISIC 2018 Task 1 có kích thước vừa phải, rất phù hợp để so sánh.

3. VinDr-CXR (X-quang lồng ngực Việt Nam):
   - Link tải: https://vindr.ai/datasets/cxr
   - Chứa ảnh X-quang phổi kèm nhãn phân đoạn tổn thương (pneumothorax, effusion).
=============================================================================
""")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Chuẩn bị dữ liệu mẫu cho U-Net")
    parser.add_argument("--dir", type=str, default="data/samples/sample_medical", help="Thư mục lưu dữ liệu mẫu hoặc dữ liệu thật Kvasir‑SEG")
    parser.add_argument("--num_samples", type=int, default=50, help="Số lượng ảnh mẫu sinh tự động")
    parser.add_argument("--img_size", type=int, default=256, help="Kích thước ảnh vuông")
    args = parser.parse_args()

    create_synthetic_dataset(args.dir, args.num_samples, args.img_size)
    print_dataset_instructions()
