import torch
import numpy as np
import skimage.io
from torchvision import transforms
from models import FlatNet, HRNetFace


def test(test_loader, recon_model, face_model, device):
    recon_model.eval()
    face_model.eval()

    with torch.no_grad():
        for batch_idx, (meas_data, labels) in enumerate(test_loader):
            # 物理成像重建
            meas_data = meas_data.to(device)
            recon_imgs, _ = recon_model(meas_data)

            # HRNet特征提取
            features = face_model(recon_imgs)

            # 保存重建结果
            save_images(recon_imgs, labels, batch_idx)


def save_images(images, labels, batch_idx):
    for i in range(images.size(0)):
        img = images[i].cpu().numpy().transpose(1, 2, 0)
        img = (img * 255).astype(np.uint8)
        skimage.io.imsave(f"results/{labels[i]}_{batch_idx}_{i}.png", img)

# 在测试脚本中添加
def generate_report(model, test_loader, psf):
    visualizer = ImagingVisualizer(psf)

    # 1. PSF可视化
    visualizer.visualize_psf('report/psf.png')

    # 2. 随机样本处理流程
    sample = next(iter(test_loader))[0]
    visualizer.visualize_processing(sample, 'report/sample_process.png')

    # 3. 特征图可视化
    visualizer.visualize_feature_maps(model, sample, 'report/features.png')

    # 4. 重建对比
    batch = next(iter(test_loader))[:4]
    visualizer.visualize_reconstruction_comparison(batch, 'report/comparison.png')


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 加载物理成像模型
    psf = torch.randn(1, 1, 510, 510)  # 示例PSF
    recon_net = FlatNet(psf).to(device)
    recon_net.load_state_dict(torch.load("recon_model.pth"))

    # 加载HRNet人脸识别模型
    face_net = HRNetFace(num_classes=None).to(device)
    face_net.load_state_dict(torch.load("hrnet_face.pth"))

    # 运行测试
    test(test_loader, recon_net, face_net, device)

    # 在train.py中添加
    if epoch % 10 == 0:
        visualizer = ImagingVisualizer(psf)
        visualizer.visualize_processing(
            test_batch[0],
            save_path=f'results/epoch_{epoch}_process.png'
        )
