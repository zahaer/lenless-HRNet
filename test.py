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