"""
=============================================================================
U-Net Baseline cài đặt từ đầu (From Scratch) bằng PyTorch
Dựa trên bài báo gốc: "U-Net: Convolutional Networks for Biomedical Image Segmentation"
(Olaf Ronneberger, Philipp Fischer, Thomas Brox - MICCAI 2015)
=============================================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """
    Khối tích chập kép (Double Convolution):
    [Conv2D (3x3) -> BatchNorm -> ReLU] x 2
    Padding=1 để giữ nguyên kích thước không gian (same padding).
    """

    def __init__(self, in_channels: int, out_channels: int, mid_channels: int = None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels

        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.double_conv(x)


class Down(nn.Module):
    """
    Khối Downsampling (Encoder block):
    Max Pooling 2x2 để giảm 1/2 kích thước không gian, sau đó đi qua DoubleConv
    để tăng gấp đôi số kênh đặc trưng.
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.maxpool_conv(x)


class Up(nn.Module):
    """
    Khối Upsampling (Decoder block):
    1. Phóng đại kích thước x2 bằng Transposed Convolution (hoặc Bilinear Interpolation).
    2. Ghép nối (Concatenate) với đặc trưng từ Encoder tương ứng (Skip Connection).
    3. Đi qua DoubleConv để tinh chỉnh thông tin đặc trưng.
    """

    def __init__(self, in_channels: int, out_channels: int, bilinear: bool = False):
        super().__init__()
        self.bilinear = bilinear

        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            # Transposed Convolution chuẩn theo bài báo gốc
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        """
        x1: Tensor từ decoder tầng dưới đi lên (cần upsample)
        x2: Tensor từ encoder tầng ngang truyền sang (skip connection)
        """
        x1 = self.up(x1)

        # Xử lý trường hợp kích thước chiều cao/rộng bị lệch (odd dimensions)
        diff_y = x2.size()[2] - x1.size()[2]
        diff_x = x2.size()[3] - x1.size()[3]

        if diff_x > 0 or diff_y > 0:
            x1 = F.pad(
                x1,
                [diff_x // 2, diff_x - diff_x // 2,
                 diff_y // 2, diff_y - diff_y // 2]
            )

        # Skip Connection: Ghép nối theo chiều channels (dim=1)
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    """
    Khối phân loại điểm ảnh cuối cùng (Final 1x1 Convolution):
    Chuyển đổi số kênh đặc trưng thành số lớp phân đoạn (num_classes).
    Với bài toán phân đoạn tổn thương y tế nhị phân: num_classes = 1 (Logits).
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class UNetScratch(nn.Module):
    """
    Kiến trúc hoàn chỉnh U-Net cài đặt từ đầu (Scratch):
    - Khởi tạo số kênh cơ sở (base_c): mặc định 64 (hoặc 32 để nhẹ hơn).
    - Encoder: 4 lần Downsampling (giảm kích thước 1/16).
    - Bottleneck: Tầng đáy chứa thông tin ngữ nghĩa sâu nhất.
    - Decoder: 4 lần Upsampling kết hợp Skip Connection.
    """

    def __init__(self, in_channels: int = 3, num_classes: int = 1, base_c: int = 64, bilinear: bool = False):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.bilinear = bilinear

        # Encoder (Contracting Path)
        self.inc = DoubleConv(in_channels, base_c)          # e.g., 3 -> 64
        self.down1 = Down(base_c, base_c * 2)               # 64 -> 128
        self.down2 = Down(base_c * 2, base_c * 4)           # 128 -> 256
        self.down3 = Down(base_c * 4, base_c * 8)           # 256 -> 512
        factor = 2 if bilinear else 1
        self.down4 = Down(base_c * 8, (base_c * 16) // factor)  # 512 -> 1024 (hoặc 512 nếu bilinear)

        # Decoder (Expanding Path)
        self.up1 = Up(base_c * 16, (base_c * 8) // factor, bilinear)
        self.up2 = Up(base_c * 8, (base_c * 4) // factor, bilinear)
        self.up3 = Up(base_c * 4, (base_c * 2) // factor, bilinear)
        self.up4 = Up(base_c * 2, base_c, bilinear)

        # Final Classifier
        self.outc = OutConv(base_c, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder forward & lưu lại feature map làm skip connection
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        # Decoder forward với skip connections
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits

    def count_parameters(self) -> int:
        """Đếm tổng số tham số có thể huấn luyện của mô hình"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Test nhanh kiến trúc U-Net Scratch
    print("=" * 60)
    print("Kiểm tra mô hình UNetScratch...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNetScratch(in_channels=3, num_classes=1, base_c=64, bilinear=False).to(device)
    dummy_input = torch.randn(2, 3, 256, 256).to(device)
    output = model(dummy_input)

    print(f"Device: {device}")
    print(f"Input shape : {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Tổng số tham số: {model.count_parameters():,}")
    assert output.shape == (2, 1, 256, 256), "Output shape không khớp!"
    print("=> UNetScratch hoạt động chính xác!")
    print("=" * 60)
