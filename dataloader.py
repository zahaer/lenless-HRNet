import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms


class DatasetFromFilenames(Dataset):
    def __init__(self, filenames_loc_meas, input_size=112):
        # 初始化HRNet输入参数
        self.input_size = input_size
        self.paths_meas = self._get_paths(filenames_loc_meas)

        # 数据预处理流程
        self.transform = transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

    def _get_paths(self, fname):
        with open(fname, 'r') as f:
            return [line.strip() for line in f]

    def __len__(self):
        return len(self.paths_meas)

    def __getitem__(self, index):
        # 数据加载适配HRNet输入
        img_path = self.paths_meas[index]
        label = img_path.split('/')[-2]  # 假设路径中包含类别信息

        # 图像加载和预处理
        img = Image.open(img_path).convert('RGB')
        img = self.transform(img)

        return img, label


# 辅助函数保持兼容
def get_paths(fname):
    with open(fname, 'r') as f:
        return [line.strip() for line in f]