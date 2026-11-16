"""
Metrics for binary medical image segmentation: Dice and IoU.
"""

import torch
from typing import Dict


def dice_coeff(
    pred: torch.Tensor,
    target: torch.Tensor,
    smooth: float = 1e-6,
    threshold: float = 0.5,
    from_logits: bool = True,
) -> float:
    """
    Compute the mean Dice score over a batch.

    Args:
        pred: Prediction tensor with shape (B, 1, H, W) or (B, H, W).
        target: Binary ground-truth tensor with the same shape.
        smooth: Smoothing constant.
        threshold: Binary prediction threshold.
        from_logits: Apply sigmoid when pred contains logits.
    """
    if from_logits:
        pred = torch.sigmoid(pred)
    pred = (pred > threshold).float()
    target = (target > 0.5).float()

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
    threshold: float = 0.5,
    from_logits: bool = True,
) -> float:
    """
    Compute the mean Intersection over Union over a batch.
    """
    if from_logits:
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


def calculate_metrics(
    pred: torch.Tensor,
    target: torch.Tensor,
    threshold: float = 0.5,
    from_logits: bool = True,
) -> Dict[str, float]:
    """Compute Dice and IoU together."""
    return {
        "dice": dice_coeff(
            pred, target, threshold=threshold, from_logits=from_logits
        ),
        "iou": iou_score(
            pred, target, threshold=threshold, from_logits=from_logits
        ),
    }


if __name__ == "__main__":
    gt = torch.zeros((2, 1, 64, 64))
    gt[:, :, 20:40, 20:40] = 1.0

    perfect_pred = gt.clone()
    print("Perfect Match ->", calculate_metrics(perfect_pred, gt, from_logits=False))

    half_pred = torch.zeros((2, 1, 64, 64))
    half_pred[:, :, 20:40, 20:30] = 1.0
    print("50% Overlap  ->", calculate_metrics(half_pred, gt, from_logits=False))

    zero_pred = torch.zeros((2, 1, 64, 64))
    zero_pred[:, :, 0:10, 0:10] = 1.0
    print("No Overlap   ->", calculate_metrics(zero_pred, gt, from_logits=False))
