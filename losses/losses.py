"""
=============================================================================
Hàm Loss cho Phân đoạn ảnh y tế:
1. DiceLoss: Tối ưu trực tiếp độ đo Dice giữa mask thực tế và dự đoán.
2. BCEDiceLoss: Kết hợp Binary Cross Entropy (phân loại từng pixel) và Dice Loss
   (tối ưu overlap), giải quyết triệt để vấn đề mất cân bằng mẫu (class imbalance).
3. FocalLoss & TverskyLoss: Hỗ trợ mở rộng cho thí nghiệm loss.
=============================================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """
    Dice Loss: 1 - Dice Coefficient
    Dùng xác suất mềm (Sigmoid probabilities) để tính đạo hàm ngược (Backpropagation).
    """

    def __init__(self, smooth: float = 1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)

        probs = probs.contiguous().view(probs.size(0), -1)
        targets = targets.contiguous().view(targets.size(0), -1)

        intersection = (probs * targets).sum(dim=1)
        total = probs.sum(dim=1) + targets.sum(dim=1)

        dice = (2.0 * intersection + self.smooth) / (total + self.smooth)
        return 1.0 - dice.mean()


class BCEDiceLoss(nn.Module):
    """
    Hàm Loss tiêu chuẩn kết hợp:
        Total Loss = alpha * BCE_with_logits + beta * Dice_Loss
    Mặc định: alpha = 1.0, beta = 1.0
    """

    def __init__(self, bce_weight: float = 1.0, dice_weight: float = 1.0, smooth: float = 1e-6):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.dice_loss = DiceLoss(smooth=smooth)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = self.bce_loss(logits, targets)
        dice = self.dice_loss(logits, targets)
        return self.bce_weight * bce + self.dice_weight * dice


class FocalLoss(nn.Module):
    """
    Focal Loss (Lin et al.): Giảm trọng số của các điểm ảnh dễ phân loại (background),
    tập trung gradient vào các điểm ảnh khó (vùng biên tổn thương).
    """

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        probs = torch.sigmoid(logits)
        p_t = probs * targets + (1 - probs) * (1 - targets)
        loss = bce * ((1 - p_t) ** self.gamma)

        if self.alpha >= 0:
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            loss = alpha_t * loss

        return loss.mean()


class TverskyLoss(nn.Module):
    """
    Tversky Loss: Tổng quát hóa của Dice Loss, cho phép điều chỉnh trọng số giữa
    False Positives (alpha) và False Negatives (beta) - rất quan trọng trong y tế
    khi cần tránh bỏ sót vùng tổn thương (ưu tiên giảm False Negatives).
    """

    def __init__(self, alpha: float = 0.7, beta: float = 0.3, smooth: float = 1e-6):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        probs = probs.contiguous().view(probs.size(0), -1)
        targets = targets.contiguous().view(targets.size(0), -1)

        tp = (probs * targets).sum(dim=1)
        fp = (probs * (1 - targets)).sum(dim=1)
        fn = ((1 - probs) * targets).sum(dim=1)

        tversky = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)
        return 1.0 - tversky.mean()
