import torch
import torch.nn as nn
import torch.optim as optim
from hrnet_impl import HRNet
from torchvision import datasets, transforms
import argparse
from torch.utils import data
import sys
import os
from datetime import datetime
from pytz import timezone
import torch.nn.functional as F
from losses import TripletLoss

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
datapath = '/home/wzp/root01/lensless/'
savedir = os.path.join(datapath, "flatnet_new")


class Logger(object):
    def __init__(self, save_dir):
        self.terminal = sys.stdout
        self.log = open(os.path.join(save_dir, "log.txt"), "a+")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()


if not os.path.exists(savedir):
    os.makedirs(savedir)
sys.stdout = Logger(savedir)

print('======== Log ========')
print(datetime.now(timezone('Asia/Shanghai')))
print("\nCommand ran:\n%s\n" % " ".join(sys.argv))


class PairMNIST(datasets.MNIST):
    def __getitem__(self, index):
        img, _ = super().__getitem__(index)
        return img, img  # 返回相同图像作为输入和目标


parser = argparse.ArgumentParser()
parser.add_argument('--modelRoot', default='flatnet_new')
parser.add_argument('--checkpoint', default='')
parser.add_argument('--wtp', default=1, type=float)
parser.add_argument('--wtmse', default=1, type=float)
parser.add_argument('--generatorLR', default=1e-4, type=float)
parser.add_argument('--hrnetLR', default=1e-5, type=float)
parser.add_argument('--num_classes', default=10, type=int)
parser.add_argument('--numEpoch1', default=50, type=int)
parser.add_argument('--numEpoch2', default=100, type=int)
parser.add_argument('--valFreq', default=200, type=int)
parser.add_argument('--pretrain', dest='pretrain', action='store_true')
parser.set_defaults(pretrain=True)
opt = parser.parse_args()


def main():
    transform = transforms.Compose([  # 数据预处理
        transforms.Resize((112, 112)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    train_dataset = PairMNIST(root='./data', train=True, download=True, transform=transform)
    val_dataset = PairMNIST(root='./data', train=False, download=True, transform=transform)

    train_loader = data.DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=16)
    val_loader = data.DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=4)

    hrnet = HRNet(in_ch=1, feature_dim=1).to(device)

    # 冻结HRNet的所有层，但确保在后续部分仍然计算梯度
    for param in hrnet.parameters():
        param.requires_grad = False

    # 解冻 fc 层（全连接层），或者根据需求解冻其他层
    for param in hrnet.fc.parameters():  # 解冻全连接层
        param.requires_grad = True

    gen_criterion = nn.MSELoss()
    hrnet_criterion = TripletLoss(margin=1.0)
    optim_gen = optim.Adam(filter(lambda p: p.requires_grad, hrnet.parameters()), lr=opt.hrnetLR)

    if opt.checkpoint:
        checkpoint = torch.load(os.path.join(datapath, opt.checkpoint, 'latest.tar'))
        hrnet.load_state_dict(checkpoint['hrnet_state_dict'])
        optim_gen.load_state_dict(checkpoint['optim_gen_state_dict'])
        print(f"Loaded checkpoint from {opt.checkpoint}")

    for epoch in range(opt.numEpoch1 + opt.numEpoch2):
        hrnet.train()
        for batch_idx, (meas, orig) in enumerate(train_loader):
            meas = meas.to(device).float()  # shape: [4, 1, 112, 112]
            orig = orig.to(device).float()

            # 前向传播
            gen_output = hrnet(meas)

            # 维度修正
            if gen_output.dim() == 2:
                gen_output = gen_output.unsqueeze(-1).unsqueeze(-1)  # [4, 1] => [4, 1, 1, 1]
            elif gen_output.dim() == 3:
                gen_output = gen_output.unsqueeze(-1)  # [4, 1, H] => [4, 1, H, 1]

            resized_output = F.interpolate(gen_output, size=(112, 112), mode='bilinear', align_corners=False)

            # 特征提取
            features_gen = hrnet(resized_output)
            features_orig = hrnet(orig)

            # 计算损失
            negative = features_orig[torch.randperm(features_orig.size(0))]
            loss_triplet = hrnet_criterion(features_gen, features_orig, negative)

            loss_mse = gen_criterion(resized_output, orig)
            total_loss = opt.wtmse * loss_mse + opt.wtp * loss_triplet

            # 反向传播
            optim_gen.zero_grad()
            total_loss.backward()
            optim_gen.step()

            if batch_idx % 100 == 0:
                print(f'Epoch: {epoch} | Batch: {batch_idx} | Loss: {total_loss.item():.4f}')

        # 验证
        if epoch % opt.valFreq == 0:
            hrnet.eval()
            val_loss = 0.0
            with torch.no_grad():
                for meas_val, orig_val in val_loader:
                    meas_val = meas_val.to(device)
                    orig_val = orig_val.to(device)

                    gen_output = hrnet(meas_val)
                    if gen_output.dim() == 2:
                        gen_output = gen_output.unsqueeze(-1).unsqueeze(-1)
                    resized_output = F.interpolate(gen_output, size=(112, 112), mode='bilinear', align_corners=False)

                    loss_mse = gen_criterion(resized_output, orig_val)
                    val_loss += loss_mse.item()

                avg_val_loss = val_loss / len(val_loader)
                print(f'Validation MSE: {avg_val_loss:.4f}')

        # 保存模型
        torch.save({
            'hrnet_state_dict': hrnet.state_dict(),
            'optim_gen_state_dict': optim_gen.state_dict(),
            'epoch': epoch,
            'loss': total_loss.item()
        }, os.path.join(savedir, 'latest.tar'))


if __name__ == '__main__':
    main()
