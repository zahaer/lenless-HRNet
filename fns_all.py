import torch
import numpy as np
from models import HRNet  # 导入新的HRNet实现


def validate(gen, hrnet, wts, val_loader, gen_criterion, facenet_criterion, device):
    gen.eval()
    hrnet.eval()
    tloss = 0
    with torch.no_grad():
        for X_val, Y_val in val_loader:
            X_val, Y_val = X_val.to(device), Y_val.to(device)

            # 生成重建图像
            X_valout, _ = gen(X_val)

            # HRNet特征提取
            val_features1 = hrnet(X_valout)  # [N, 128]
            val_features2 = hrnet(Y_val)  # [N, 128]

            # 计算联合损失
            recon_loss = gen_criterion(Y_val, X_valout)
            feat_loss = facenet_criterion(val_features2, val_features1)
            total_loss = wts[0] * recon_loss + wts[1] * feat_loss

            tloss += total_loss.item()

        return X_valout, tloss / len(val_loader)


def train_frozen_epoch(gen, hrnet, wts, optim_gen, train_loader, val_loader,
                       gen_criterion, facenet_criterion, device, valFreq):
    gen.train()
    hrnet.eval()

    for batch_idx, (X_train, Y_train) in enumerate(train_loader):
        X_train, Y_train = X_train.to(device), Y_train.to(device)

        # 冻结HRNet参数
        for param in hrnet.parameters():
            param.requires_grad = False

        # 生成器前向
        optim_gen.zero_grad()
        Xout, _ = gen(X_train)

        # 特征提取
        hr_features = hrnet(Xout)  # [N, 128]
        gt_features = hrnet(Y_train)

        # 计算双损失
        recon_loss = gen_criterion(Y_train, Xout)
        feat_loss = facenet_criterion(gt_features, hr_features)
        total_loss = wts[0] * recon_loss + wts[1] * feat_loss

        # 反向传播
        total_loss.backward()
        optim_gen.step()

        # 验证环节
        if batch_idx % valFreq == 0:
            val_out, val_loss = validate(gen, hrnet, wts, val_loader,
                                         gen_criterion, facenet_criterion, device)

    return total_loss.item(), val_loss