import torch.nn as nn
import torch.nn.functional as F


class MobileNetV1(nn.Module):
    def __init__(self):
        super(MobileNetV1, self).__init__()
        self.stage1 = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU6(inplace=True),

            nn.Conv2d(32, 32, 3, groups=32, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU6(inplace=True),
            nn.Conv2d(32, 64, 1, stride=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU6(inplace=True)
        )
        self.stage2 = nn.Sequential(
            # 20,20,256 -> 10,10,512
            conv_dw(256, 512, 2),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
        )
        self.stage3 = nn.Sequential(
            # 10,10,512 -> 5,5,1024
            conv_dw(512, 1024, 2),
            conv_dw(1024, 1024, 1),
        )

        self.avg = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(1024, 1000)

    def forward(self, x):
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.avg(x)
        # x = self.model(x)
        x = x.view(-1, 1024)
        x = self.fc(x)
        return x


class HRNet(nn.Module):
    def __init__(self, input_size=112, feature_dim=128):
        super(HRNet, self).__init__()
        # Stage 1
        self.stage1 = nn.Sequential(
            nn.Conv2d(3, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            self._make_layer(64, 64, 4)
        )

        # Stage 2
        self.stage2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            self._make_layer(128, 128, 4)
        )

        # Stage 3
        self.stage3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            self._make_layer(256, 256, 4)
        )

        # Feature processing
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(256, feature_dim)
        self.bn = nn.BatchNorm1d(feature_dim, eps=0.001)

    def _make_layer(self, in_ch, out_ch, blocks):
        layers = []
        for _ in range(blocks):
            layers.append(ResidualBlock(in_ch, out_ch))
            in_ch = out_ch
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return F.normalize(self.bn(x), p=2, dim=1)


class ResidualBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
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