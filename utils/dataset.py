"""
=============================================================================
Dataset & DataLoader cho Phân đoạn ảnh y tế:
- Tương thích tốt với Kvasir-SEG, ISIC (Skin Lesion), VinDr-CXR hoặc dataset bất kỳ
- Hỗ trợ ghép cặp linh hoạt giữa ảnh gốc (images/) và mặt nạ (masks/)
- Tích hợp Data Augmentation (lật ngang, lật dọc, xoay góc, chuẩn hóa)
=============================================================================
"""

import os
import glob
from pathlib import Path
from typing import List, Tuple, Optional, Callable
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms.functional as TF
import random


class MedicalSegmentationDataset(Dataset):
    """
    Dataset chuẩn cho bài toán phân đoạn ảnh y tế nhị phân:
    - Đọc ảnh RGB (hoặc Grayscale) và mặt nạ nhãn nhị phân (0 hoặc 255/1).
    - Chuẩn hóa mặt nạ về tensor [1, H, W] với giá trị 0.0 hoặc 1.0.
    """

    def __init__(
        self,
        image_paths: List[str],
        mask_paths: List[str],
        img_size: Tuple[int, int] = (256, 256),
        is_train: bool = True,
        augment: bool = True
    ):
        super().__init__()
        assert len(image_paths) == len(mask_paths), f"Số lượng ảnh ({len(image_paths)}) và mask ({len(mask_paths)}) phải bằng nhau!"
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.img_size = img_size
        self.is_train = is_train
        self.augment = augment and is_train

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img_path = self.image_paths[idx]
        mask_path = self.mask_paths[idx]

        # Đọc ảnh gốc (RGB)
        image = Image.open(img_path).convert("RGB")
        # Đọc mask nhãn (Grayscale 'L')
        mask = Image.open(mask_path).convert("L")

        # Resize ảnh và mask về kích thước mục tiêu
        image = image.resize(self.img_size, Image.BILINEAR)
        mask = mask.resize(self.img_size, Image.NEAREST)

        # Chuyển đổi sang Tensor
        image_tensor = TF.to_tensor(image)  # [3, H, W], giá trị [0.0, 1.0]

        # Mask: ngưỡng hóa để đảm bảo nhị phân 0.0 hoặc 1.0
        mask_np = np.array(mask, dtype=np.float32)
        mask_np = (mask_np > 127.5).astype(np.float32)
        mask_tensor = torch.from_numpy(mask_np).unsqueeze(0)  # [1, H, W]

        # Data Augmentation đồng thời cho cả ảnh và mask (chỉ khi train)
        if self.augment:
            # Lật ngang (Horizontal Flip) với xác suất 50%
            if random.random() > 0.5:
                image_tensor = TF.hflip(image_tensor)
                mask_tensor = TF.hflip(mask_tensor)

            # Lật dọc (Vertical Flip) với xác suất 50%
            if random.random() > 0.5:
                image_tensor = TF.vflip(image_tensor)
                mask_tensor = TF.vflip(mask_tensor)

            # Xoay góc ngẫu nhiên (90, 180, 270 độ)
            angle = random.choice([0, 90, 180, 270])
            if angle > 0:
                image_tensor = TF.rotate(image_tensor, angle)
                mask_tensor = TF.rotate(mask_tensor, angle)

        # Chuẩn hóa ảnh theo ImageNet Mean/Std (rất tốt khi dùng Pretrained Backbone)
        normalize = TF.normalize(
            image_tensor,
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )

        return normalize, mask_tensor


def find_image_mask_pairs(data_dir: str) -> Tuple[List[str], List[str]]:
    """
    Tự động tìm kiếm và ghép cặp ảnh gốc và mask từ thư mục dữ liệu:
    Hỗ trợ các cấu trúc thư mục:
    - data_dir/images và data_dir/masks
    - data_dir/train/images và data_dir/train/masks
    - hoặc file cùng tên với hậu tố mask/segmentation
    """
    valid_exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif", "*.tiff")

    images_dir = os.path.join(data_dir, "images")
    masks_dir = os.path.join(data_dir, "masks")

    if not os.path.exists(images_dir) or not os.path.exists(masks_dir):
        # Thử tìm trực tiếp trong data_dir
        images_dir = data_dir
        masks_dir = data_dir

    all_images = []
    for ext in valid_exts:
        all_images.extend(glob.glob(os.path.join(images_dir, ext)))
        all_images.extend(glob.glob(os.path.join(images_dir, ext.upper())))
    all_images = sorted(list(set(all_images)))

    # Lọc bỏ các file mask nằm chung thư mục nếu có
    image_files = [f for f in all_images if "_mask" not in f.lower() and "_segmentation" not in f.lower()]
    image_files.sort()

    paired_images = []
    paired_masks = []

    for img_path in image_files:
        stem = Path(img_path).stem

        # Danh sách các tên mask khả dĩ
        candidate_names = [
            f"{stem}.png", f"{stem}.jpg", f"{stem}.jpeg",
            f"{stem}_mask.png", f"{stem}_mask.jpg",
            f"{stem}_segmentation.png", f"{stem}_segmentation.jpg"
        ]

        found_mask = None
        for c_name in candidate_names:
            c_path = os.path.join(masks_dir, c_name)
            if os.path.exists(c_path):
                found_mask = c_path
                break

        if found_mask is not None:
            paired_images.append(img_path)
            paired_masks.append(found_mask)

    return paired_images, paired_masks


def get_dataloaders(
    data_dir: str,
    img_size: Tuple[int, int] = (256, 256),
    batch_size: int = 8,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    num_workers: int = 0,
    seed: int = 42
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Tạo 3 DataLoaders: Train, Validation và Test
    """
    image_paths, mask_paths = find_image_mask_pairs(data_dir)
    total_samples = len(image_paths)

    if total_samples == 0:
        raise FileNotFoundError(
            f"Không tìm thấy cặp ảnh và mask nào trong thư mục '{data_dir}'!\n"
            f"Hãy đảm bảo thư mục có chứa 'images/' và 'masks/' với các file ảnh hợp lệ."
        )

    # Xáo trộn có kiểm soát bằng seed
    rng = random.Random(seed)
    indices = list(range(total_samples))
    rng.shuffle(indices)

    n_train = int(total_samples * train_ratio)
    n_val = int(total_samples * val_ratio)
    n_test = total_samples - n_train - n_val

    # Đảm bảo ít nhất 1 sample cho val và test nếu tổng số lượng nhỏ
    if n_val == 0 and total_samples >= 2:
        n_val = 1
        n_train -= 1
    if n_test == 0 and total_samples >= 3:
        n_test = 1
        n_train -= 1

    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]

    train_imgs = [image_paths[i] for i in train_idx]
    train_masks = [mask_paths[i] for i in train_idx]

    val_imgs = [image_paths[i] for i in val_idx]
    val_masks = [mask_paths[i] for i in val_idx]

    test_imgs = [image_paths[i] for i in test_idx]
    test_masks = [mask_paths[i] for i in test_idx]

    # Khởi tạo datasets
    train_dataset = MedicalSegmentationDataset(train_imgs, train_masks, img_size=img_size, is_train=True, augment=True)
    val_dataset = MedicalSegmentationDataset(val_imgs, val_masks, img_size=img_size, is_train=False, augment=False)
    test_dataset = MedicalSegmentationDataset(test_imgs, test_masks, img_size=img_size, is_train=False, augment=False)

    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    print(f"Dataset split: Train = {len(train_dataset)}, Val = {len(val_dataset)}, Test = {len(test_dataset)} (Total: {total_samples})")
    return train_loader, val_loader, test_loader
