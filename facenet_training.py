import os
import numpy as np
import scipy.signal
import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib import pyplot as plt


class HRNetWithHead(nn.Module):
    def __init__(self, base_model, num_classes):
        super().__init__()
        self.hrnet = base_model
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, x):
        features = self.hrnet(x)
        return self.classifier(features)


class TripletLoss(nn.Module):
    def __init__(self, margin=0.2):
        super().__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        pos_dist = F.pairwise_distance(anchor, positive)
        neg_dist = F.pairwise_distance(anchor, negative)
        losses = F.relu(pos_dist - neg_dist + self.margin)
        return torch.mean(losses)


def weights_init(net, init_type='kaiming', init_gain=0.02):
    def init_func(m):
        class_name = m.__class__.__name__
        if hasattr(m, 'weight') and ('Conv' in class_name or 'Linear' in class_name):
            if init_type == 'normal':
                nn.init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                nn.init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                nn.init.orthogonal_(m.weight.data, gain=init_gain)
        elif 'BatchNorm' in class_name:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0.0)

    print(f'Initialize network with {init_type} type')
    net.apply(init_func)


class LossHistory:
    def __init__(self, log_dir):
        from datetime import datetime
        curr_time = datetime.now()
        self.time_str = curr_time.strftime('%Y_%m_%d_%H_%M_%S')
        self.save_path = os.path.join(log_dir, f"loss_{self.time_str}")
        os.makedirs(self.save_path, exist_ok=True)

        self.acc = []
        self.losses = []
        self.val_loss = []

    def append(self, acc, loss, val_loss):
        self.acc.append(acc)
        self.losses.append(loss)
        self.val_loss.append(val_loss)

        # Save to files
        for data, name in zip([self.acc, self.losses, self.val_loss],
                              ['acc', 'loss', 'val_loss']):
            np.savetxt(os.path.join(self.save_path, f'epoch_{name}_{self.time_str}.txt'),
                       data, fmt='%.4f')

        self._plot_loss()
        self._plot_acc()

    def _plot_loss(self):
        plt.figure(figsize=(10, 6))
        x = np.arange(len(self.losses))

        plt.plot(x, self.losses, 'r-', label='Train Loss')
        plt.plot(x, self.val_loss, 'coral', label='Val Loss')

        try:
            window = min(15, len(x) // 4)
            if window > 1:
                smooth_train = scipy.signal.savgol_filter(self.losses, window, 3)
                smooth_val = scipy.signal.savgol_filter(self.val_loss, window, 3)
                plt.plot(x, smooth_train, 'g--', label='Smooth Train')
                plt.plot(x, smooth_val, '#8B4513--', label='Smooth Val')
        except Exception as e:
            print(f"Smoothing failed: {str(e)}")

        plt.title('Training Metrics')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(self.save_path, f'loss_plot_{self.time_str}.png'))
        plt.close()

    def _plot_acc(self):
        plt.figure(figsize=(10, 6))
        x = np.arange(len(self.acc))

        plt.plot(x, self.acc, 'b-', label='Accuracy')

        try:
            window = min(15, len(x) // 4)
            if window > 1:
                smooth_acc = scipy.signal.savgol_filter(self.acc, window, 3)
                plt.plot(x, smooth_acc, 'g--', label='Smooth Acc')
        except Exception as e:
            print(f"Smoothing failed: {str(e)}")

        plt.title('Accuracy Metrics')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(self.save_path, f'acc_plot_{self.time_str}.png'))
        plt.close()