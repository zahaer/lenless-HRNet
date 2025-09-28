import torch
import torch.nn as nn
import torch.nn.functional as F
from hrnet_module import HRNetFeatureExtractor  # 导入自定义HRNet模块


class HRNetFace(nn.Module):
    def __init__(self,
                 backbone="hrnet",
                 embedding_size=128,
                 num_classes=None,
                 mode="train"):
        super(HRNetFace, self).__init__()

        # 初始化HRNet主干网络
        self.backbone = HRNetFeatureExtractor(pretrained=True)

        # 特征处理模块
        self.embedding = nn.Linear(2048, embedding_size)
        self.bn = nn.BatchNorm1d(embedding_size, eps=0.001, momentum=0.1)

        # 训练模式添加分类头
        if mode == "train":
            self.classifier = nn.Linear(embedding_size, num_classes)

    def forward(self, x):
        # HRNet特征提取流程
        features = self.backbone(x)
        x = self.embedding(features)
        x = self.bn(x)
        return F.normalize(x, p=2, dim=1)

    def forward_feature(self, x):
        # 分离特征和归一化输出
        features = self.backbone(x)
        embeddings = self.embedding(features)
        before_norm = self.bn(embeddings)
        return before_norm, F.normalize(before_norm, p=2, dim=1)

    def forward_classifier(self, x):
        return self.classifier(x)


# 保持兼容性的辅助函数
def load_pretrained(model, ckpt_path):
    state_dict = torch.load(ckpt_path)
    model.load_state_dict(state_dict, strict=False)