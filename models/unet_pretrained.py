"""
=============================================================================
U-Net với Pretrained Encoder (ResNet Backbone)
So sánh với U-Net Scratch theo đúng yêu cầu đề tài:
- Encoder dùng weights pretrained trên ImageNet (torchvision hoặc segmentation_models_pytorch)
- Decoder tái cấu trúc skip connections từ các tầng của ResNet
=============================================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from typing import Optional


class DecoderBlock(nn.Module):
    """
    Khối giải mã (Decoder Block) cho U-Net có Pretrained Encoder:
    Upsample -> Concatenate với Skip Feature từ ResNet -> 2x Conv-BN-ReLU
    """

    def __init__(self, in_channels: int, skip_channels: int, out_channels: int):
        super().__init__()
        self.upsample = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        total_in = in_channels // 2 + skip_channels

        self.conv = nn.Sequential(
            nn.Conv2d(total_in, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor, skip: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = self.upsample(x)

        if skip is not None:
            # Căn chỉnh kích thước nếu có sự lệch lẻ do phép làm tròn kích thước
            if x.shape[2:] != skip.shape[2:]:
                x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=True)
            x = torch.cat([skip, x], dim=1)

        return self.conv(x)


class UNetResNet(nn.Module):
    """
    U-Net kiến trúc hoàn chỉnh sử dụng ResNet34/ResNet50 làm Pretrained Encoder:
    - Trích xuất 5 tầng đặc trưng từ ResNet để làm skip connection:
      * s0: H, W (input)
      * s1: H/2, W/2 (conv1 + bn1 + relu)
      * s2: H/4, W/4 (maxpool + layer1)
      * s3: H/8, W/8 (layer2)
      * s4: H/16, W/16 (layer3)
      * s5: H/32, W/32 (layer4 - Bottleneck)
    - Decoder khôi phục độ phân giải từng bước và kết nối đặc trưng tương ứng.
    """

    def __init__(
        self,
        backbone: str = "resnet34",
        pretrained: bool = True,
        in_channels: int = 3,
        num_classes: int = 1
    ):
        super().__init__()
        self.backbone_name = backbone
        self.num_classes = num_classes

        # Khởi tạo backbone ResNet
        if backbone == "resnet34":
            weights = models.ResNet34_Weights.DEFAULT if pretrained else None
            resnet = models.resnet34(weights=weights)
            filters = [64, 64, 128, 256, 512]
        elif backbone == "resnet50":
            weights = models.ResNet50_Weights.DEFAULT if pretrained else None
            resnet = models.resnet50(weights=weights)
            filters = [64, 256, 512, 1024, 2048]
        else:
            raise ValueError(f"Backbone '{backbone}' chưa được hỗ trợ. Hãy chọn 'resnet34' hoặc 'resnet50'.")

        # Điều chỉnh số kênh đầu vào nếu khác 3 (ví dụ ảnh grayscale 1 channel)
        if in_channels != 3:
            resnet.conv1 = nn.Conv2d(
                in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
            )

        # Trích xuất các tầng Encoder
        self.encoder_conv1 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu)
        self.encoder_maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

        # Decoder Blocks
        self.dec4 = DecoderBlock(filters[4], filters[3], 256)
        self.dec3 = DecoderBlock(256, filters[2], 128)
        self.dec2 = DecoderBlock(128, filters[1], 64)
        self.dec1 = DecoderBlock(64, filters[0], 32)

        # Final Upsample khôi phục lại độ phân giải gốc H x W
        self.final_up = nn.Sequential(
            nn.ConvTranspose2d(32, 32, kernel_size=2, stride=2),
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, num_classes, kernel_size=1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder forward
        s1 = self.encoder_conv1(x)               # H/2, W/2
        s2 = self.layer1(self.encoder_maxpool(s1)) # H/4, W/4
        s3 = self.layer2(s2)                     # H/8, W/8
        s4 = self.layer3(s3)                     # H/16, W/16
        s5 = self.layer4(s4)                     # H/32, W/32 (Bottleneck)

        # Decoder forward kết hợp skip connections
        d4 = self.dec4(s5, s4)                   # H/16, W/16
        d3 = self.dec3(d4, s3)                   # H/8, W/8
        d2 = self.dec2(d3, s2)                   # H/4, W/4
        d1 = self.dec1(d2, s1)                   # H/2, W/2

        out = self.final_up(d1)                  # H, W
        return out

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class UNetPretrained(nn.Module):
    """
    Wrapper linh hoạt:
    - Ưu tiên dùng segmentation_models_pytorch nếu có (chuẩn công nghiệp, hỗ trợ đa dạng backbone)
    - Tự động fallback sang UNetResNet (thuần torchvision) nếu chưa cài SMP.
    """

    def __init__(
        self,
        backbone: str = "resnet34",
        pretrained: bool = True,
        in_channels: int = 3,
        num_classes: int = 1
    ):
        super().__init__()
        self.backbone = backbone
        self.use_smp = False

        try:
            import segmentation_models_pytorch as smp
            weights = "imagenet" if pretrained else None
            self.model = smp.Unet(
                encoder_name=backbone,
                encoder_weights=weights,
                in_channels=in_channels,
                classes=num_classes
            )
            self.use_smp = True
        except ImportError:
            # Fallback sang kiến trúc tự xây dựng bằng torchvision
            self.model = UNetResNet(
                backbone=backbone,
                pretrained=pretrained,
                in_channels=in_channels,
                num_classes=num_classes
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def build_unet(model_type: str = "scratch", backbone: str = "resnet34", pretrained: bool = True, in_channels: int = 3, num_classes: int = 1):
    """
    Factory function tiện lợi để khởi tạo mô hình:
    - 'scratch': UNet thuần cài từ đầu
    - 'pretrained': UNet với encoder pretrained (ResNet34, ResNet50)
    """
    from .unet_scratch import UNetScratch

    if model_type.lower() == "scratch":
        return UNetScratch(in_channels=in_channels, num_classes=num_classes)
    elif model_type.lower() in ["pretrained", "resnet", "unet_pretrained"]:
        return UNetPretrained(
            backbone=backbone,
            pretrained=pretrained,
            in_channels=in_channels,
            num_classes=num_classes
        )
    else:
        raise ValueError(f"model_type '{model_type}' không hợp lệ. Chọn 'scratch' hoặc 'pretrained'.")


if __name__ == "__main__":
    print("=" * 60)
    print("Kiểm tra mô hình UNetPretrained...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNetPretrained(backbone="resnet34", pretrained=False, in_channels=3, num_classes=1).to(device)
    dummy_input = torch.randn(2, 3, 256, 256).to(device)
    output = model(dummy_input)

    print(f"Device: {device}")
    print(f"Sử dụng SMP: {model.use_smp}")
    print(f"Input shape : {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Tổng số tham số: {model.count_parameters():,}")
    assert output.shape == (2, 1, 256, 256), "Output shape không khớp!"
    print("=> UNetPretrained hoạt động chính xác!")
    print("=" * 60)
