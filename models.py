import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# --------------------------
# 无透镜成像核心模块
# --------------------------

class PhysicalConv(nn.Module):
    """物理卷积核处理模块"""

    def __init__(self, kernel, kernel_size=510):
        super().__init__()
        self.conv = nn.Conv2d(1, 1, kernel_size=kernel_size,
                              padding=kernel_size // 2, bias=False)
        self.conv.weight = nn.Parameter(kernel.flip(0, 1), requires_grad=False)

    def forward(self, x):
        return self.conv(x)


def wiener_filter(x, psf, K=100):
    """维纳滤波重建"""
    device = x.device
    # 创建扩展PSF矩阵
    psf_matrix = torch.zeros(621, 621, device=device)
    psf_matrix[55:565, 55:565] = psf.flip(2, 3)

    # 频域计算
    input_fft = torch.fft.fftn(x, dim=(2, 3))
    psf_fft = torch.fft.fft2(psf_matrix)

    # 维纳逆滤波
    psf_spectrum = psf_fft.abs().square() + K
    filtered = (psf_fft.conj() / psf_spectrum) * input_fft

    # 逆变换与后处理
    result = torch.fft.ifftn(filtered, dim=(2, 3)).abs()
    result = torch.fft.fftshift(result, dim=(2, 3))
    return result[:, :, 256:368, 256:368]  # 中心裁剪


# --------------------------
# U-Net 重建网络
# --------------------------

class DoubleConv(nn.Module):
    """双卷积块"""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class Downsample(nn.Module):
    """下采样模块"""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, stride=2, padding=1),
            DoubleConv(out_ch, out_ch)
        )

    def forward(self, x):
        return self.block(x)


class Upsample(nn.Module):
    """上采样模块"""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch // 2, in_ch // 2, 2, stride=2)
        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # 自动对齐尺寸
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class ReconstructionNet(nn.Module):
    """完整的U-Net重建网络"""

    def __init__(self, in_ch=1, out_ch=3):
        super().__init__()
        self.inc = DoubleConv(in_ch, 64)
        self.down1 = Downsample(64, 128)
        self.down2 = Downsample(128, 256)
        self.down3 = Downsample(256, 512)
        self.up1 = Upsample(512, 256)
        self.up2 = Upsample(256, 128)
        self.up3 = Upsample(128, 64)
        self.outc = nn.Conv2d(64, out_ch, 3, padding=1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        return torch.sigmoid(self.outc(x))


# --------------------------
# HRNet 特征提取网络
# --------------------------

class HRBlock(nn.Module):
    """高分辨率残差块"""

    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1, stride=stride)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_ch)

        self.shortcut = nn.Sequential()
        if in_ch != out_ch or stride != 1:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride),
                nn.BatchNorm2d(out_ch)
            )

    def forward(self, x):
        residual = self.shortcut(x)
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return F.relu(x + residual)


class HRNet(nn.Module):
    """高分辨率特征提取网络"""

    def __init__(self, in_ch=3, feature_dim=512):
        super().__init__()
        # 初始特征提取
        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            HRBlock(64, 64, stride=1)
        )

        # 多分辨率分支
        self.stage1 = self._make_stage(64, 64, 4)
        self.stage2 = self._make_stage(64, 128, 4, stride=2)
        self.stage3 = self._make_stage(128, 256, 4, stride=2)

        # 特征融合
        self.fusion = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(256, feature_dim)
        )

    def _make_stage(self, in_ch, out_ch, blocks, stride=1):
        layers = [HRBlock(in_ch, out_ch, stride)]
        for _ in range(1, blocks):
            layers.append(HRBlock(out_ch, out_ch))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        return self.fusion(x)


# --------------------------
# 完整成像系统
# --------------------------

class ImagingSystem(nn.Module):
    """端到端成像系统"""

    def __init__(self, psf_kernel):
        super().__init__()
        # 物理成像模块
        self.physical_conv = PhysicalConv(psf_kernel)

        # 数字重建模块
        self.recon_net = ReconstructionNet(in_ch=1, out_ch=3)

        # 特征提取模块
        self.feature_net = HRNet(in_ch=3)

    def forward(self, x):
        # 物理成像过程
        raw = self.physical_conv(x)

        # 数字重建
        recon = self.recon_net(raw)

        # 特征提取
        features = self.feature_net(recon)
        return features