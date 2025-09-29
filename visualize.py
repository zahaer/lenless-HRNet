import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision.utils import make_grid
from models import PhysicalConv, wiener_filter, ReconstructionNet


class ImagingVisualizer:
    """无透镜成像系统可视化工具"""

    def __init__(self, psf_kernel):
        self.psf_kernel = psf_kernel
        self.physical_conv = PhysicalConv(psf_kernel)
        self.recon_net = ReconstructionNet()

    def visualize_psf(self, save_path=None):
        """可视化点扩散函数(PSF)"""
        plt.figure(figsize=(10, 8))

        # 原始PSF
        plt.subplot(221)
        plt.imshow(self.psf_kernel.squeeze().cpu().numpy(), cmap='viridis')
        plt.title('原始PSF')
        plt.colorbar()

        # PSF中心区域放大
        plt.subplot(222)
        center_psf = self.psf_kernel.squeeze().cpu().numpy()
        h, w = center_psf.shape
        plt.imshow(center_psf[h // 2 - 50:h // 2 + 50, w // 2 - 50:w // 2 + 50], cmap='viridis')
        plt.title('PSF中心区域')
        plt.colorbar()

        # PSF的3D可视化
        ax = plt.subplot(223, projection='3d')
        x = np.arange(0, 100)
        y = np.arange(0, 100)
        X, Y = np.meshgrid(x, y)
        Z = center_psf[h // 2 - 50:h // 2 + 50, w // 2 - 50:w // 2 + 50][:100, :100]
        ax.plot_surface(X, Y, Z, cmap='viridis')
        plt.title('PSF 3D视图')

        # PSF的频域表示
        plt.subplot(224)
        psf_fft = np.fft.fft2(center_psf)
        psf_fft_shift = np.fft.fftshift(psf_fft)
        magnitude = np.log(np.abs(psf_fft_shift) + 1)
        plt.imshow(magnitude, cmap='viridis')
        plt.title('PSF频域幅度')
        plt.colorbar()

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300)
        plt.show()

    def visualize_processing(self, input_signal, save_path=None):
        """可视化整个处理流程"""
        # 确保输入是单通道
        if input_signal.dim() == 3:
            input_signal = input_signal.unsqueeze(0).unsqueeze(0)
        elif input_signal.dim() == 4 and input_signal.shape[1] != 1:
            input_signal = input_signal.mean(dim=1, keepdim=True)

        # 物理卷积过程
        with torch.no_grad():
            raw = self.physical_conv(input_signal)
            wiener_recon = wiener_filter(raw, self.psf_kernel)
            unet_recon = self.recon_net(raw)

        # 转换为numpy用于可视化
        input_np = input_signal.squeeze().cpu().numpy()
        raw_np = raw.squeeze().cpu().numpy()
        wiener_np = wiener_recon.squeeze().cpu().numpy()
        unet_np = unet_recon.squeeze().permute(1, 2, 0).cpu().numpy()

        plt.figure(figsize=(15, 10))

        # 原始输入信号
        plt.subplot(231)
        plt.imshow(input_np, cmap='gray')
        plt.title('原始输入信号')
        plt.axis('off')

        # 物理卷积结果
        plt.subplot(232)
        plt.imshow(raw_np, cmap='gray')
        plt.title('物理卷积结果')
        plt.axis('off')

        # 维纳滤波重建
        plt.subplot(233)
        plt.imshow(wiener_np, cmap='gray')
        plt.title('维纳滤波重建')
        plt.axis('off')

        # U-Net重建结果
        plt.subplot(234)
        plt.imshow(unet_np)
        plt.title('U-Net重建')
        plt.axis('off')

        # 重建结果对比
        plt.subplot(235)
        plt.imshow(np.clip(wiener_np, 0, 1), cmap='gray')
        plt.title('维纳滤波重建 (对比)')
        plt.axis('off')

        plt.subplot(236)
        plt.imshow(unet_np)
        plt.title('U-Net重建 (对比)')
        plt.axis('off')

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300)
        plt.show()

        return wiener_recon, unet_recon

    def visualize_feature_maps(self, model, input_tensor, layer_names=None, save_path=None):
        """可视化网络中间特征图"""
        # 获取中间层输出
        activations = {}

        def get_activation(name):
            def hook(model, input, output):
                activations[name] = output.detach()

            return hook

        # 注册钩子
        hooks = []
        if layer_names is None:
            # 默认选择关键层
            layer_names = [
                'recon_net.inc.conv.1',  # U-Net第一层输出
                'recon_net.down1.block.1.conv.5',  # U-Net下采样输出
                'recon_net.up3.conv.5',  # U-Net上采样输出
                'feature_net.stem.1',  # HRNet主干输出
                'feature_net.stage3.3.bn2'  # HRNet最后层输出
            ]

        for name, layer in model.named_modules():
            if name in layer_names:
                hooks.append(layer.register_forward_hook(get_activation(name)))

        # 前向传播
        with torch.no_grad():
            model(input_tensor)

        # 移除钩子
        for hook in hooks:
            hook.remove()

        # 可视化特征图
        plt.figure(figsize=(15, 10))
        for i, (name, act) in enumerate(activations.items()):
            # 处理不同形状的特征图
            if act.dim() == 4:
                # 特征图可视化
                num_feat = min(8, act.shape[1])
                feat_grid = make_grid(act[0, :num_feat].unsqueeze(1), nrow=4, normalize=True)
                feat_np = feat_grid.permute(1, 2, 0).cpu().numpy()

                plt.subplot(2, 3, i + 1)
                plt.imshow(feat_np)
                plt.title(f'{name}\n{act.shape[1]}通道')
                plt.axis('off')
            elif act.dim() == 2:
                # 特征向量可视化
                plt.subplot(2, 3, i + 1)
                plt.bar(range(len(act[0])), act[0].cpu().numpy())
                plt.title(f'{name}\n特征向量')

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300)
        plt.show()

    def visualize_reconstruction_comparison(self, input_signals, save_path=None):
        """多输入重建结果对比"""
        num_samples = min(4, input_signals.shape[0])
        fig, axes = plt.subplots(num_samples, 4, figsize=(16, 4 * num_samples))

        if num_samples == 1:
            axes = [axes]

        for i in range(num_samples):
            input_signal = input_signals[i:i + 1]

            with torch.no_grad():
                raw = self.physical_conv(input_signal)
                wiener_recon = wiener_filter(raw, self.psf_kernel)
                unet_recon = self.recon_net(raw)

            # 转换为numpy用于可视化
            input_np = input_signal.squeeze().cpu().numpy()
            raw_np = raw.squeeze().cpu().numpy()
            wiener_np = wiener_recon.squeeze().cpu().numpy()
            unet_np = unet_recon.squeeze().permute(1, 2, 0).cpu().numpy()

            # 原始输入
            axes[i][0].imshow(input_np, cmap='gray')
            axes[i][0].set_title(f'样本 {i + 1} 输入')
            axes[i][0].axis('off')

            # 物理卷积结果
            axes[i][1].imshow(raw_np, cmap='gray')
            axes[i][1].set_title(f'样本 {i + 1} 卷积结果')
            axes[i][1].axis('off')

            # 维纳滤波重建
            axes[i][2].imshow(wiener_np, cmap='gray')
            axes[i][2].set_title(f'样本 {i + 1} 维纳重建')
            axes[i][2].axis('off')

            # U-Net重建
            axes[i][3].imshow(unet_np)
            axes[i][3].set_title(f'样本 {i + 1} U-Net重建')
            axes[i][3].axis('off')

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300)
        plt.show()