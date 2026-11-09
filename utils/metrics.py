"""
=============================================================================
Đánh giá độ đo phân đoạn ảnh y tế: Dice Score (F1) & IoU (Jaccard Index)
Theo đúng yêu cầu tối thiểu của đề tài: "đánh giá bằng Dice và IoU".
=============================================================================
"""

import torch
import numpy as np
from typing import Dict, Tuple


def dice_coeff(
    pred: torch.Tensor,
    target: torch.Tensor,
    smooth: float = 1e-6,
    threshold: float = 0.5
) -> float:
    """
    Tính Dice Coefficient (F1 Score) giữa mặt nạ dự đoán và ground truth:
        Dice = (2 * |X ∩ Y| + smooth) / (|X| + |Y| + smooth)

    Tham số:
        pred: Tensor dự đoán (Logits hoặc Probabilities), shape (B, 1, H, W) hoặc (B, H, W)
        target: Tensor nhãn thực tế ground truth (0 hoặc 1), cùng shape với pred
        smooth: Hằng số làm trơn để tránh chia cho 0
        threshold: Ngưỡng phân loại nhị phân (mặc định 0.5)

    Trả về:
        Điểm Dice trung bình trên batch (float từ 0.0 đến 1.0)
    """
    # Nếu đầu ra là logits chưa qua sigmoid, áp dụng sigmoid
    if pred.is_floating_point():
        if (pred < 0).any() or (pred > 1).any():
            pred = torch.sigmoid(pred)
        pred = (pred > threshold).float()

    target = (target > 0.5).float()

    # Flatten tensors để tính toán theo từng mẫu trong batch
    pred = pred.contiguous().view(pred.size(0), -1)
    target = target.contiguous().view(target.size(0), -1)

    intersection = (pred * target).sum(dim=1)
    union = pred.sum(dim=1) + target.sum(dim=1)

    dice = (2.0 * intersection + smooth) / (union + smooth)
    return dice.mean().item()


def iou_score(
    pred: torch.Tensor,
    target: torch.Tensor,
    smooth: float = 1e-6,
    threshold: float = 0.5
) -> float:
    """
    Tính IoU (Intersection over Union / Jaccard Index):
        IoU = (|X ∩ Y| + smooth) / (|X ∪ Y| + smooth)
            = (|X ∩ Y| + smooth) / (|X| + |Y| - |X ∩ Y| + smooth)
    """
    if pred.is_floating_point():
        if (pred < 0).any() or (pred > 1).any():
            pred = torch.sigmoid(pred)
        pred = (pred > threshold).float()

    target = (target > 0.5).float()

    pred = pred.contiguous().view(pred.size(0), -1)
    target = target.contiguous().view(target.size(0), -1)

    intersection = (pred * target).sum(dim=1)
    total = pred.sum(dim=1) + target.sum(dim=1)
    union = total - intersection

    iou = (intersection + smooth) / (union + smooth)
    return iou.mean().item()


def calculate_metrics(pred: torch.Tensor, target: torch.Tensor, threshold: float = 0.5) -> Dict[str, float]:
    """
    Hàm tiện ích tính toán đồng thời cả Dice Score và IoU Score.
    """
    dice = dice_coeff(pred, target, threshold=threshold)
    iou = iou_score(pred, target, threshold=threshold)
    return {
        "dice": dice,
        "iou": iou
    }


if __name__ == "__main__":
    # Test tính toán metric với ground truth và prediction giả lập
    gt = torch.zeros((2, 1, 64, 64))
    gt[:, :, 20:40, 20:40] = 1.0  # Vùng tổn thương

    # Dự đoán hoàn hảo
    perfect_pred = gt.clone()
    print("Perfect Match ->", calculate_metrics(perfect_pred, gt))

    # Dự đoán lệch 1 nửa
    half_pred = torch.zeros((2, 1, 64, 64))
    half_pred[:, :, 20:40, 20:30] = 1.0
    print("50% Overlap  ->", calculate_metrics(half_pred, gt))

    # Dự đoán lệch hoàn toàn
    zero_pred = torch.zeros((2, 1, 64, 64))
    zero_pred[:, :, 0:10, 0:10] = 1.0
    print("No Overlap   ->", calculate_metrics(zero_pred, gt))
