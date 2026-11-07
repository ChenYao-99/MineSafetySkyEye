# -*- coding: utf-8 -*-
"""
train_unified.py — 多区域统一模型训练 (跨域泛化优化)
======================================================
思路: 把 6 个 4 通道区域的 32px 瓦片合并训练"一个统一模型",
      配合 ImageNet 预训练骨干与(可选)域随机化增强, 检验跨区域泛化能力。

协议与 train_prior.py 保持一致 (可比): UnetPlusPlus/resnet34, SGD lr=0.01,
ExponentialLR(0.99), λ=1.0 先验加权损失, seed 42, 按 BLDS 选优。

用法:
  python train_unified.py --pretrained --epochs 100            # 统一模型(预训练)
  python train_unified.py --pretrained --dr --epochs 100       # +域随机化增强
输出: models_prior/unified_{pre|scratch}[_dr]_seed42/
"""
import os
import sys
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import segmentation_models_pytorch as smp
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

NATIVE = r"F:\LMHLD\LMHLD_dataset_different_patch_sizes"
TILED = r"F:\LMHLD\LMHLD_tiled"
OUT_ROOT = r"D:\LMHLD\models_prior"
PS = 32

# 区域 → (数据根目录, 文件夹名)
REGIONS = [
    ("Emilia（意大利）", NATIVE, "Emilia-Romagna_Italy_32"),
    ("Gorkha（尼泊尔）", NATIVE, "Gorkha_Nepal_32"),
    ("Hokkaido（日本）", TILED, "Hokkaido_Japan_32"),
    ("Jiuzhaigou（九寨沟）", TILED, "Jiuzhaigou_China_32"),
    ("Wenchuan（汶川）", TILED, "Wenchuan_China_32"),
    ("Taiwan（台湾）", TILED, "Taiwan_China_32"),
]
CAP_TRAIN = 4000    # 每区域训练瓦片上限
CAP_VAL = 600       # 每区域验证瓦片上限


# ── 域随机化增强 (4 通道安全, 自研) ─────────────────────
def dr_augment(img, mask, rng):
    """img: (H,W,4) float(原始DN)  mask: (H,W)"""
    # 1. 几何: 翻转/旋转
    if rng.random() < 0.5:
        img, mask = img[:, ::-1], mask[:, ::-1]
    if rng.random() < 0.5:
        img, mask = img[::-1, :], mask[::-1, :]
    k = int(rng.integers(0, 4))
    if k:
        img, mask = np.rot90(img, k), np.rot90(mask, k)
    # 2. 随机放大+裁剪 (模拟分辨率/尺度差)
    if rng.random() < 0.5:
        s = float(rng.uniform(1.0, 1.35))
        H = int(round(PS * s))
        img = cv2.resize(img, (H, H), interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask.astype(np.uint8), (H, H), interpolation=cv2.INTER_NEAREST)
        y0 = int(rng.integers(0, H - PS + 1)); x0 = int(rng.integers(0, H - PS + 1))
        img, mask = img[y0:y0+PS, x0:x0+PS], mask[y0:y0+PS, x0:x0+PS]
    img = np.ascontiguousarray(img, dtype=np.float32)
    mask = np.ascontiguousarray(mask, dtype=np.float32)
    # 3. 光度: 逐通道增益/偏置 (模拟传感器色彩响应差异)
    gain = rng.uniform(0.80, 1.25, size=(1, 1, 4)).astype(np.float32)
    mean = float(img.mean()) + 1e-6
    bias = (rng.uniform(-0.10, 0.10, size=(1, 1, 4)) * mean).astype(np.float32)
    img = img * gain + bias
    # 4. 模糊 (模拟不同空间分辨率/成像质量)
    if rng.random() < 0.4:
        sig = float(rng.uniform(0.4, 1.3))
        img = cv2.GaussianBlur(img, (0, 0), sig)
    # 5. JPEG 压缩 (模拟不同产品级别)
    if rng.random() < 0.4:
        q = int(rng.integers(40, 96))
        ok, buf = cv2.imencode(".jpg", img[..., :3].astype(np.uint8), [cv2.IMWRITE_JPEG_QUALITY, q])
        if ok:
            dec = cv2.imdecode(buf, cv2.IMREAD_COLOR).astype(np.float32)
            img = np.concatenate([dec, img[..., 3:4]], axis=-1)
    # 6. 高斯噪声 (模拟信噪比差异)
    if rng.random() < 0.3:
        sd = float(rng.uniform(0.03, 0.12)) * (float(img.std()) + 1e-6)
        img = img + rng.normal(0, sd, img.shape).astype(np.float32)
    return np.clip(img, 0, None), (mask > 0.5).astype(np.float32)


class MultiRegionDS(Dataset):
    """多区域瓦片数据集: 各区域分别加载并采样, 合并为一个训练池"""

    def __init__(self, split, cap, use_dr=False, seed=42):
        self.use_dr = use_dr
        self.augment = (split == "train")
        self.items = []   # (images_mmap, labels, idx)  - 按需读取, 降低内存
        self.pools = []
        rng = np.random.default_rng(seed)
        for disp, root, folder in REGIONS:
            f_img = os.path.join(root, folder, f"{split}_images.npy")
            f_lbl = os.path.join(root, folder, f"{split}_labels.npy")
            if not os.path.exists(f_img):
                print(f"  [跳过] {disp} 无 {split} 数据")
                continue
            imgs = np.load(f_img, mmap_mode="r")
            lbls = np.load(f_lbl, mmap_mode="r")
            n = len(imgs)
            idx = np.arange(n) if n <= cap else rng.choice(n, cap, replace=False)
            self.pools.append((disp, imgs, lbls, np.sort(idx)))
            print(f"  {disp}: {split} 采 {len(idx)}/{n}")
        self.offsets = np.cumsum([0] + [len(p[3]) for p in self.pools])

    def __len__(self):
        return int(self.offsets[-1])

    def __getitem__(self, i):
        pool_i = int(np.searchsorted(self.offsets, i, side="right") - 1)
        _, imgs, lbls, idx = self.pools[pool_i]
        j = idx[i - self.offsets[pool_i]]
        image = np.asarray(imgs[j])
        mask = np.asarray(lbls[j])
        if image.ndim == 3 and image.shape[0] == 4:
            image = np.transpose(image, (1, 2, 0))
        if mask.ndim == 3 and mask.shape[0] == 1:
            mask = mask[0]
        mask = (mask > 0).astype(np.float32)
        if self.use_dr:
            rng = np.random.default_rng()
            image, mask = dr_augment(image, mask, rng)
        elif self.augment:   # 仅训练集做随机翻转 (验证集保持确定性)
            if np.random.random() < 0.5:
                image, mask = image[:, ::-1], mask[:, ::-1]
            if np.random.random() < 0.5:
                image, mask = image[::-1, :], mask[::-1, :]
        # 关键: 翻转会产生负步长, 必须连续化拷贝后才能转 tensor
        image = np.ascontiguousarray(image, dtype=np.float32)
        mask = np.ascontiguousarray(mask, dtype=np.float32)
        prior = compute_prior_map(image)
        return (torch.from_numpy(image).permute(2, 0, 1).float(),
                torch.from_numpy(mask).unsqueeze(0).float(),
                torch.from_numpy(prior).unsqueeze(0).float())



def dice_loss(logits, targets, eps=1e-6):
    """Dice 损失(逐样本计算后取均值) — 应对滑坡类像素不平衡"""
    probs = torch.sigmoid(logits)
    num = 2 * (probs * targets).sum(dim=(2, 3)) + eps
    den = probs.sum(dim=(2, 3)) + targets.sum(dim=(2, 3)) + eps
    return (1 - num / den).mean()


def validate(model, loader, device):
    model.eval()
    preds_all, masks_all, loss_tot = [], [], 0.0
    crit = nn.BCEWithLogitsLoss()
    with torch.no_grad():
        for images, masks, _ in loader:
            images, masks = images.to(device), masks.to(device)
            out = model(images)
            loss_tot += crit(out, masks).item() * images.size(0)
            preds_all.append((torch.sigmoid(out) > 0.5).cpu().numpy())
            masks_all.append(masks.cpu().numpy())
    p = np.concatenate(preds_all); m = np.concatenate(masks_all)
    tp = int(((p == 1) & (m == 1)).sum()); fp = int(((p == 1) & (m == 0)).sum())
    fn = int(((p == 0) & (m == 1)).sum())
    pr = tp / (tp + fp + 1e-6); rc = tp / (tp + fn + 1e-6)
    f1 = 2 * pr * rc / (pr + rc + 1e-6)
    return loss_tot / len(loader.dataset), (pr * rc * f1) ** (1 / 3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pretrained", action="store_true")
    ap.add_argument("--dr", action="store_true", help="域随机化增强")
    ap.add_argument("--loss", default="bce", choices=["bce", "bce_dice"], help="损失函数")
    ap.add_argument("--dice-w", type=float, default=1.0, help="Dice 损失权重")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--cap", type=int, default=CAP_TRAIN)
    args = ap.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tag = f"unified_{'pre' if args.pretrained else 'scratch'}{'_dr' if args.dr else ''}_seed{args.seed}" + ("_dice" if args.loss == "bce_dice" else "")
    out_dir = os.path.join(OUT_ROOT, tag)
    os.makedirs(out_dir, exist_ok=True)
    print(f"\n===== 统一模型训练: {tag} | 设备 {device} =====")

    print("[训练集]")
    train_ds = MultiRegionDS("train", args.cap, use_dr=args.dr, seed=args.seed)
    print("[验证集]")
    val_ds = MultiRegionDS("val", CAP_VAL, use_dr=False, seed=args.seed)
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False, num_workers=0)
    print(f"训练池 {len(train_ds)} | 验证池 {len(val_ds)}")

    _enc = "imagenet" if args.pretrained else None
    model = smp.UnetPlusPlus(encoder_name="resnet34", encoder_weights=_enc,
                             in_channels=4, classes=1).to(device)
    criterion = nn.BCEWithLogitsLoss(reduction="none")
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)

    best, history = 0.0, {"val_blds": [], "val_loss": []}
    for ep in range(args.epochs):
        model.train()
        tot = 0.0
        for images, masks, priors in train_loader:
            images, masks, priors = images.to(device), masks.to(device), priors.to(device)
            optimizer.zero_grad()
            out = model(images)
            loss_map = criterion(out, masks)
            loss_map = loss_map * (1.0 + 1.0 * priors)     # λ=1.0 先验加权(同协议)
            loss = loss_map.mean()
            if args.loss == "bce_dice":
                loss = loss + args.dice_w * dice_loss(out, masks)
            loss.backward(); optimizer.step()
            tot += loss.item() * images.size(0)
        scheduler.step()
        vloss, vblds = validate(model, val_loader, device)
        history["val_blds"].append(vblds); history["val_loss"].append(vloss)
        if vblds > best:
            best = vblds
            torch.save(model.state_dict(), os.path.join(out_dir, "best_model.pth"))
        if (ep + 1) % 10 == 0 or ep == 0:
            print(f"epoch {ep+1:3d}/{args.epochs} | train {tot/len(train_ds):.4f} | "
                  f"val_loss {vloss:.4f} | val_BLDS {vblds:.4f} (best {best:.4f})")

    with open(os.path.join(out_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump({"tag": tag, "best_val_blds": best, "history": history,
                   "n_train": len(train_ds), "n_val": len(val_ds),
                   "pretrained": args.pretrained, "dr": args.dr}, f, ensure_ascii=False, indent=2)
    print(f"完成: {out_dir} (best val BLDS {best:.4f})")


if __name__ == "__main__":
    main()
