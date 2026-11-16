import os
import argparse
import numpy as np
from PIL import Image, ImageDraw, ImageFilter


def create_synthetic_dataset(output_dir: str = "data/sample_medical", num_samples: int = 50, img_size: int = 256):

    images_dir = os.path.join(output_dir, "images")
    masks_dir = os.path.join(output_dir, "masks")
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(masks_dir, exist_ok=True)

    rng = np.random.default_rng(42)

    for i in range(num_samples):
        base_color = rng.integers(140, 200, size=3)
        base_color[2] = int(base_color[2] * 0.6)

        bg = np.ones((img_size, img_size, 3), dtype=np.uint8) * base_color
        noise = rng.normal(0, 15, (img_size, img_size, 3)).astype(np.int16)
        bg = np.clip(bg.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        img = Image.fromarray(bg)

        mask = Image.new("L", (img_size, img_size), 0)
        draw_mask = ImageDraw.Draw(mask)
        draw_img = ImageDraw.Draw(img)

        cx = rng.integers(int(img_size * 0.3), int(img_size * 0.7))
        cy = rng.integers(int(img_size * 0.3), int(img_size * 0.7))
        rx = rng.integers(int(img_size * 0.12), int(img_size * 0.28))
        ry = rng.integers(int(img_size * 0.12), int(img_size * 0.28))

        bbox = [cx - rx, cy - ry, cx + rx, cy + ry]

        draw_mask.ellipse(bbox, fill=255)

        lesion_color = (
            int(np.clip(base_color[0] * 0.65, 0, 255)),
            int(np.clip(base_color[1] * 0.50, 0, 255)),
            int(np.clip(base_color[2] * 0.40, 0, 255))
        )
        draw_img.ellipse(bbox, fill=lesion_color)

        img = img.filter(ImageFilter.GaussianBlur(radius=1.5))

        sample_id = f"sample_{i+1:04d}"
        img.save(os.path.join(images_dir, f"{sample_id}.jpg"), quality=95)
        mask.save(os.path.join(masks_dir, f"{sample_id}.png"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Chuẩn bị dữ liệu mẫu cho U-Net")
    parser.add_argument("--dir", type=str, default="data/kvasir", help="Thư mục lưu dữ liệu mẫu hoặc dữ liệu thật Kvasir‑SEG")
    parser.add_argument("--num_samples", type=int, default=50, help="Số lượng ảnh mẫu sinh tự động")
    parser.add_argument("--img_size", type=int, default=256, help="Kích thước ảnh vuông")
    args = parser.parse_args()

    create_synthetic_dataset(args.dir, args.num_samples, args.img_size)
