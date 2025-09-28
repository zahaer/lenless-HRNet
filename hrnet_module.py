import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50  # 作为基础backbone


class HRNetFeatureExtractor(nn.Module):
    def __init__(self, pretrained=True, feature_dim=128):
        super().__init__()
        # 实现简化的HRNet结构
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        # Stage配置
        self.stage1 = self._make_stage(64, 64, 2)
        self.stage2 = self._make_stage(64, 128, 3)
        self.stage3 = self._make_stage(128, 256, 4)
        self.stage4 = self._make_stage(256, 512, 3)

        # 自适应池化
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512, feature_dim)

        # 加载预训练
        if pretrained:
            self._load_pretrained_weights()

    def _make_stage(self, in_channels, out_channels, num_blocks):
        layers = []
        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1))
        layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.ReLU(inplace=True))

        for _ in range(1, num_blocks):
            layers.append(nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1))
            layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ReLU(inplace=True))

        return nn.Sequential(*layers)

    def _load_pretrained_weights(self):
        # 加载ResNet部分权重作为示例
        pretrained = resnet50(pretrained=True)
        self.stem[0].weight.data = pretrained.conv1.weight.data[:, :3]
        self.stem[3].weight.data = pretrained.layer1[0].conv1.weight.data

    def forward(self, x):
        x = self.stem(x)  # 1/4
        x = self.stage1(x)  # 1/8
        x = self.stage2(x)  # 1/16
        x = self.stage3(x)  # 1/32
        x = self.stage4(x)  # 1/64

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return F.normalize(self.fc(x), p=2, dim=1)