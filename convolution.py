import torch
import torch.nn as nn

class convolution(nn.Module):
    def __init__(self, kernel, kernel_conv=63):
        super(convolution, self).__init__()
        # 无透镜成像核心卷积模块
        self.conv1 = nn.Conv2d(1, 1, stride=1, kernel_size=kernel_conv,
                             padding=kernel_conv//2, bias=False)
        self.conv1.weight = nn.Parameter(kernel.flip(0,1), requires_grad=True)

    def forward(self, x):
        # 执行物理成像卷积操作
        return self.conv1(x)