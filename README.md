无透镜成像系统项目

快速开始

环境要求

• Python 3.8+

• PyTorch 1.10+

• CUDA 11.3+ (推荐)

• 其他依赖见 requirements.txt

安装步骤

1. 创建并激活conda环境：
conda create -n lensless python=3.9 -y
conda activate lensless


2. 安装依赖：
pip install -r requirements.txt


3. 安装PyTorch（根据您的CUDA版本）：
# 例如 CUDA 11.3
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch


配置文件

创建 config.yaml：
data:
  train_path: "./data/raw/train"
  val_path: "./data/raw/val"
  psf_path: "./data/psf.npy"

model:
  input_channels: 1
  output_channels: 3
  feature_dim: 512

training:
  batch_size: 16
  epochs: 100
  lr: 1e-4
  checkpoint_dir: "./checkpoints"


使用指南

训练模型

python train.py --config config.yaml


测试模型

python test.py --checkpoint checkpoints/best_model.pth --input data/test_sample.npy


可视化工具

from visualize import ImagingVisualizer

# 初始化可视化工具
visualizer = ImagingVisualizer(psf_path="data/psf.npy")

# 可视化PSF
visualizer.visualize_psf(save_path="results/psf_visualization.png")

# 可视化处理流程
visualizer.visualize_processing(input_signal, save_path="results/processing_pipeline.png")

# 可视化特征图
visualizer.visualize_feature_maps(model, input_signal, save_path="results/feature_maps.png")


使用预训练模型

1. 下载预训练模型：
wget https://example.com/pretrained_model.pth -O checkpoints/pretrained.pth


2. 加载模型：
from models import ImagingSystem
import torch

model = ImagingSystem(psf)
model.load_state_dict(torch.load("checkpoints/pretrained.pth"))
model.eval()


