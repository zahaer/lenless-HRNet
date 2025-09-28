import torch.nn as nn
from hrnet_impl import HRNet  # 复用HRNet主实现


class HRNetWrapper(nn.Module):
    """兼容原有InceptionResnetV1接口的HRNet封装"""

    def __init__(self, pretrained=True):
        super().__init__()
        self.model = HRNet()

        if pretrained:
            self._load_pretrained()

    def _load_pretrained(self):
        # 这里可以加载预训练权重
        pass

    def forward(self, x):
        return self.model(x)

    @property
    def num_ftrs(self):
        return 128  # 特征维度保持统一