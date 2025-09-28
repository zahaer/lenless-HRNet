import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50  # 作为基础backbone


class HRNet(nn.Module):
    def __init__(self, in_ch=128, feature_dim=128):  # 将 in_ch 修改为 128
        super(HRNet, self).__init__()

        # Stage 1: 高分辨率主干
        self.stage1 = nn.Sequential(
            nn.Conv2d(in_ch, 64, 3, stride=2, padding=1),  # 修改此行
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            self._make_layer(64, 64, 4)
        )

        # Stage 2: 多分辨率融合
        self.stage2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            self._make_layer(128, 128, 4)
        )

        # Stage 3: 特征增强
        self.stage3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            self._make_layer(256, 256, 4)
        )

        # 特征后处理
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(256, feature_dim)
        self.bn = nn.BatchNorm1d(feature_dim, eps=0.001)

        # 初始化权重
        self._init_weights()

    def _make_layer(self, in_ch, out_ch, blocks):
        layers = []
        for _ in range(blocks):
            layers.append(ResidualBlock(in_ch, out_ch))
            in_ch = out_ch
        return nn.Sequential(*layers)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # 特征提取流程
        x = self.stage1(x)  # /2
        x = self.stage2(x)  # /4
        x = self.stage3(x)  # /8

        # 全局特征生成
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return F.normalize(self.bn(x), p=2, dim=1)




    def load_pretrained_weights(self, pretrained_path):
        # 加载预训练模型的权重
        pretrained_dict = torch.load(pretrained_path)
        model_dict = self.state_dict()

        # 过滤掉预训练模型中不需要加载的权重
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}

        # 更新模型的权重
        model_dict.update(pretrained_dict)
        self.load_state_dict(model_dict)


class ResidualBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)

        self.shortcut = nn.Sequential()
        if in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, bias=False),
                nn.BatchNorm2d(out_ch)
            )

    def forward(self, x):
        residual = self.shortcut(x)
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x += residual
        return F.relu(x)
