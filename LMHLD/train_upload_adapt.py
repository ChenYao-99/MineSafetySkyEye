# -*- coding: utf-8 -*-
"""
train_upload_adapt.py — 上传影像域适配微调
============================================
问题: 上传的普通影像(0-255)与训练数据(原始DN值 1000-16000)分布不匹配,
模型预测偏差大。解法: 在"逐样本 min-max 归一化到 0-255"的数据上
对先验加权模型做低学习率微调, 使模型适配网络影像风格输入。

流程:
  1. Emilia 训练/验证/测试集 → 逐样本 4 通道 min-max 归一化到 [0,255]
  2. 从 λ=1.0 模型加载权重, 40 epochs 低学习率(0.001)微调, 先验加权损失
  3. 测试集验证: 上传风格输入下 F1 应与原始域同量级
输出: models_prior/Emilia-Romagna_Italy_32_upload_seed42/best_model.pth
"""
import os
import sys
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import segmentation_models_pytorch as smp
import albumentations as A
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.stdout.reconfigure(encoding="utf-8")
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False

os.environ["SM_FRAMEWORK"] = "torch"
os.environ["SMP_DISABLE_VERSION_CHECK"] = "1"

from train_prior import compute_prior_map

DATA_ROOT = r"F:\LMHLD\LMHLD_dataset_different_patch_sizes"
MODELS_ROOT = r"D:\LMHLD\models_prior"
SOURCE = "Emilia-Romagna_Italy_32"
SRC_WEIGHTS = os.path.join(MODELS_ROOT, f"{SOURCE}_prior1.0_seed42", "best_model.pth")
OUT_TAG = f"{SOURCE}_upload_seed42"
EPOCHS = 40
LR = 0.001
LAMBDA = 1.0
SEED = 42


def normalize_01_to_255(image_hwc):
    """逐样本逐通道 min-max 归一化到 [0,255] (模拟上传图预处理)"""
    out = np.zeros_like(image_hwc, dtype=np.float32)
    for c in range(image_hwc.shape[-1]):
        ch = image_hwc[..., c].astype(np.float32)
        lo, hi = ch.min(), ch.max()
        out[..., c] = (ch - lo) / (hi - lo + 1e-6) * 255.0
    return out


class UploadAdaptDataset(Dataset):
    def __init__(self, root_dir, region, split, transform=None):
        self.transform = transform
        self.images = np.load(os.path.join(root_dir, region, f"{split}_images.npy"))
        self.labels = np.load(os.path.join(root_dir, region, f"{split}_labels.npy"))
        self.labels[self.labels > 0] = 1
        print(f"加载 {region}/{split}: {len(self.images)} 样本")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        mask = self.labels[idx]
        if image.ndim == 3 and image.shape[0] == 4:
            image = np.transpose(image, (1, 2, 0))
        if mask.ndim == 3 and mask.shape[0] == 1:
            mask = mask[0]
        image = normalize_01_to_255(image)      # 关键: 模拟上传预处理
        if self.transform:
            aug = self.transform(image=image, mask=mask)
            image, mask = aug["image"], aug["mask"]
        image, mask = image.copy(), mask.copy()
        prior = compute_prior_map(image)
        return (torch.from_numpy(image).permute(2, 0, 1).float(),
                torch.from_numpy(mask).unsqueeze(0).float(),
                torch.from_numpy(prior).unsqueeze(0).float())


def evaluate(model, loader, device):
    model.eval()
    preds, gts = [], []
    with torch.no_grad():
        for images, masks, _ in loader:
            preds.append((torch.sigmoid(model(images.to(device))) > 0.5).cpu().numpy())
            gts.append(masks.numpy())
    preds = np.concatenate(preds)[:, 0]
    gts = np.concatenate(gts)[:, 0].astype(np.uint8)
    tp = int(((preds == 1) & (gts == 1)).sum())
    fp = int(((preds == 1) & (gts == 0)).sum())
    fn = int(((preds == 0) & (gts == 1)).sum())
    prec = tp / (tp + fp + 1e-6)
    rec = tp / (tp + fn + 1e-6)
    f1 = 2 * prec * rec / (prec + rec + 1e-6)
    blds = (prec * rec * f1) ** (1 / 3)
    return {"precision": prec, "recall": rec, "f1": f1, "blds": blds}


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--pretrained", action="store_true", help="使用 ImageNet 预训练骨干")
    ap.add_argument("--loss", default="bce", choices=["bce", "bce_dice"], help="损失函数")
    ap.add_argument("--tag", default="", help="输出目录后缀")
    ap.add_argument("--src", default="", help="源权重路径(默认用 Emilia 原版)")
    args = ap.parse_args()
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    aug = A.Compose([A.HorizontalFlip(p=0.5), A.VerticalFlip(p=0.5), A.RandomRotate90(p=0.5)], is_check_shapes=False)
    no_aug = A.Compose([], is_check_shapes=False)
    global OUT_TAG
    if args.tag:
        OUT_TAG = OUT_TAG + args.tag
    print(f"===== 上传域适配微调: {SOURCE} (0-255 归一化, {EPOCHS} epochs, lr={LR}, pretrained={args.pretrained}, loss={args.loss}) =====")

    train_ds = UploadAdaptDataset(DATA_ROOT, SOURCE, "train", aug)
    val_ds = UploadAdaptDataset(DATA_ROOT, SOURCE, "val", no_aug)
    test_ds = UploadAdaptDataset(DATA_ROOT, SOURCE, "test", no_aug)
    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=16, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=16, shuffle=False, num_workers=0)

    model = smp.UnetPlusPlus(encoder_name="resnet34",
                             encoder_weights=("imagenet" if args.pretrained else None),
                             in_channels=4, classes=1).to(device)
    src = args.src or SRC_WEIGHTS
    model.load_state_dict(torch.load(src, weights_only=True, map_location=device))
    print(f"源权重: {src}")
    print("源权重加载完成, 开始微调...")

    criterion = nn.BCEWithLogitsLoss(reduction="none")
    optimizer = optim.SGD(model.parameters(), lr=LR, momentum=0.9, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.98)

    best_f1 = 0.0
    for ep in range(EPOCHS):
        model.train()
        total = 0.0
        for images, masks, priors in train_loader:
            images, masks, priors = images.to(device), masks.to(device), priors.to(device)
            optimizer.zero_grad()
            out = model(images)
            loss_map = criterion(out, masks)
            if LAMBDA > 0:
                loss_map = loss_map * (1.0 + LAMBDA * priors)
            loss = loss_map.mean()
            if args.loss == "bce_dice":
                probs = torch.sigmoid(out)
                num = 2 * (probs * masks).sum(dim=(2, 3)) + 1e-6
                den = probs.sum(dim=(2, 3)) + masks.sum(dim=(2, 3)) + 1e-6
                loss = loss + (1 - num / den).mean()
            loss.backward()
            optimizer.step()
            total += loss.item() * images.size(0)
        scheduler.step()
        # 验证 (归一化域)
        model.eval()
        with torch.no_grad():
            preds, gts = [], []
            for images, masks, _ in val_loader:
                preds.append((torch.sigmoid(model(images.to(device))) > 0.5).cpu().numpy())
                gts.append(masks.numpy())
        preds = np.concatenate(preds)[:, 0]
        gts = np.concatenate(gts)[:, 0].astype(np.uint8)
        tp = int(((preds == 1) & (gts == 1)).sum())
        fp = int(((preds == 1) & (gts == 0)).sum())
        fn = int(((preds == 0) & (gts == 1)).sum())
        prec = tp / (tp + fp + 1e-6)
        rec = tp / (tp + fn + 1e-6)
        vf1 = 2 * prec * rec / (prec + rec + 1e-6)
        if vf1 > best_f1:
            best_f1 = vf1
            out_dir = os.path.join(MODELS_ROOT, OUT_TAG)
            os.makedirs(out_dir, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(out_dir, "best_model.pth"))
        if (ep + 1) % 10 == 0:
            print(f"epoch {ep+1}/{EPOCHS} loss {total/len(train_loader.dataset):.4f} val_f1 {vf1:.4f}")

    # 测试集评估 (归一化域)
    model.load_state_dict(torch.load(os.path.join(MODELS_ROOT, OUT_TAG, "best_model.pth"), weights_only=True))
    test_metrics = evaluate(model, test_loader, device)
    print(f"\n[上传域] 测试集: F1={test_metrics['f1']:.4f} BLDS={test_metrics['blds']:.4f} "
          f"Prec={test_metrics['precision']:.4f} Rec={test_metrics['recall']:.4f}")
    with open(os.path.join(MODELS_ROOT, OUT_TAG, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump({"test": test_metrics}, f, ensure_ascii=False, indent=2)
    print(f"模型已保存: {os.path.join(MODELS_ROOT, OUT_TAG)}")


if __name__ == "__main__":
    main()
