# -*- coding: utf-8 -*-
"""
train_prior.py — 第二阶段实验: 地质先验加权训练损失
====================================================
核心: 在训练阶段将地质先验图作为空间权重融入 BCE Loss:
      loss = BCEWithLogitsLoss(reduction='none') * (1 + λ·prior) 取均值
      (λ=0 时退化为原训练, 作为公平基线)

先验图与第一组实验完全一致(裸模型无需外部数据):
      prior = 0.5 * bare(NDVI反推) + 0.5 * rough(NIR局部标准差)  ∈ [0,1]

实验协议(保证可比性):
  - 与 train.py 相同的模型/优化器/增强/调度器, 仅损失函数不同
  - 固定随机种子(42), 同一训练/验证/测试划分
  - 按 BLDS 保存最优权重, 测试集输出像素级指标 + 分尺度区域召回

用法:
  python train_prior.py --region Emilia-Romagna_Italy_32 --lambda 1.0 --epochs 100 --seed 42
  python train_prior.py --region Gorkha_Nepal_32 --lambda 0 --epochs 100   # 基线复训

输出: D:\\LMHLD\\models_prior\\{region}_prior{λ}_seed{s}\\ 下
      best_model.pth / metrics.json / loss_curves.png
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

DATA_ROOT = r"F:\LMHLD\LMHLD_dataset_different_patch_sizes"
OUT_ROOT = r"D:\LMHLD\models_prior"

NUM_CHANNELS = 4      # B,G,R,NIR
NUM_CLASSES = 1
BATCH_SIZE = 16
LEARNING_RATE = 0.01
IMAGE_SIZE = 32
SMALL, MID = 25, 100  # 分桶阈值(与实验一一致)


def compute_prior_map(image_hwc):
    """(H,W,4)[B,G,R,NIR] → (H,W) float [0,1] 地质先验图(与实验一一致)"""
    b, g, r, nir = (image_hwc[..., i] for i in range(4))
    ndvi = (nir.astype(np.float32) - r.astype(np.float32)) / (nir + r + 1e-6)
    ndvi_mm = (ndvi - ndvi.min()) / (ndvi.max() - ndvi.min() + 1e-6)
    bare = 1.0 - ndvi_mm
    nir_f = nir.astype(np.float32)
    mean = cv2.blur(nir_f, (3, 3))
    sq_mean = cv2.blur(nir_f * nir_f, (3, 3))
    std = np.sqrt(np.maximum(sq_mean - mean * mean, 0))
    rough = (std - std.min()) / (std.max() - std.min() + 1e-6)
    return (0.5 * bare + 0.5 * rough).astype(np.float32)


class LMHLDPriorDataset(Dataset):
    """与 train.py 相同的数据加载, 额外返回增强后的先验图"""

    def __init__(self, root_dir, region, split="train", transform=None,
                 norm_scale=1.0):
        self.transform = transform
        self.norm_scale = float(norm_scale)
        self.images = np.load(os.path.join(root_dir, region, f"{split}_images.npy"))
        self.labels = np.load(os.path.join(root_dir, region, f"{split}_labels.npy"))
        self.labels[self.labels > 0] = 1
        print(f"加载 {region} {split} 集: {len(self.images)} 个样本")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        mask = self.labels[idx]
        if image.ndim == 3 and image.shape[0] == 4:
            image = np.transpose(image, (1, 2, 0))
        if mask.ndim == 3 and mask.shape[0] == 1:
            mask = mask[0]
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"]
        image = image.copy()
        mask = mask.copy()
        # 增强后计算先验(与增强后的影像对齐: 翻转/旋转同步)
        prior = compute_prior_map(image)
        # 输入尺度归一化（第十九组 W2: 单因素对照实验）
        # ⚠ 放在**先验计算之后** —— 先验本身不变, 只改送进网络的影像尺度,
        #   这样测的就是"输入尺度"这一个因素。默认 1.0 = 不缩放(既有行为逐位不变)。
        #   LMHLD 的 Sentinel-2 区域是 BOA 反射率×10000, 值域可达 ~8600;
        #   而送进网络的是原始量级 —— 对 ImageNet 预训练骨干尤其不友好。
        if self.norm_scale != 1.0:
            image = image / self.norm_scale
        image_tensor = torch.from_numpy(image).permute(2, 0, 1).float()
        mask_tensor = torch.from_numpy(mask).unsqueeze(0).float()
        prior_tensor = torch.from_numpy(prior).unsqueeze(0).float()
        return image_tensor, mask_tensor, prior_tensor


def pixel_metrics(pred, gt):
    tp = int(((pred == 1) & (gt == 1)).sum())
    fp = int(((pred == 1) & (gt == 0)).sum())
    fn = int(((pred == 0) & (gt == 1)).sum())
    prec = tp / (tp + fp + 1e-6)
    rec = tp / (tp + fn + 1e-6)
    f1 = 2 * prec * rec / (prec + rec + 1e-6)
    blds = (prec * rec * f1) ** (1 / 3)
    return {"precision": prec, "recall": rec, "f1": f1, "blds": blds}


def region_recall_buckets(gt, pred):
    buckets = {"small": [0, 0], "mid": [0, 0], "large": [0, 0]}
    n, lab, st, _ = cv2.connectedComponentsWithStats(gt.astype(np.uint8), 8)
    for i in range(1, n):
        area = st[i, cv2.CC_STAT_AREA]
        key = "small" if area <= SMALL else ("mid" if area <= MID else "large")
        buckets[key][1] += 1
        if pred[lab == i].any():
            buckets[key][0] += 1
    return buckets



def dice_loss(logits, targets, eps=1e-6):
    """Dice 损失(逐样本计算后取均值) — 应对滑坡类像素不平衡"""
    probs = torch.sigmoid(logits)
    num = 2 * (probs * targets).sum(dim=(2, 3)) + eps
    den = probs.sum(dim=(2, 3)) + targets.sum(dim=(2, 3)) + eps
    return (1 - num / den).mean()


def train_one_epoch(model, loader, criterion, optimizer, device, lambda_prior,
                    loss_kind="bce", dice_w=1.0):
    model.train()
    total = 0.0
    for images, masks, priors in loader:
        images, masks, priors = images.to(device), masks.to(device), priors.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss_map = criterion(outputs, masks)            # BCE(reduction=none)
        if lambda_prior > 0:
            loss_map = loss_map * (1.0 + lambda_prior * priors)
        loss = loss_map.mean()
        if loss_kind == "bce_dice":
            loss = loss + dice_w * dice_loss(outputs, masks)
        loss.backward()
        optimizer.step()
        total += loss.item() * images.size(0)
    return total / len(loader.dataset)


def validate(model, loader, criterion, device):
    model.eval()
    total = 0.0
    preds_all, masks_all = [], []
    with torch.no_grad():
        for images, masks, _ in loader:
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            total += criterion(outputs, masks).mean().item() * images.size(0)
            preds_all.append((torch.sigmoid(outputs) > 0.5).cpu().numpy())
            masks_all.append(masks.cpu().numpy())
    preds = np.concatenate(preds_all)
    masks = np.concatenate(masks_all)
    tp = int(((preds == 1) & (masks == 1)).sum())
    fp = int(((preds == 1) & (masks == 0)).sum())
    fn = int(((preds == 0) & (masks == 1)).sum())
    prec = tp / (tp + fp + 1e-6)
    rec = tp / (tp + fn + 1e-6)
    f1 = 2 * prec * rec / (prec + rec + 1e-6)
    blds = (prec * rec * f1) ** (1 / 3)
    return total / len(loader.dataset), prec, rec, f1, blds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--region", default="Emilia-Romagna_Italy_32")
    ap.add_argument("--lambda", dest="lambda_prior", type=float, default=1.0)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--batch", type=int, default=BATCH_SIZE)
    ap.add_argument("--data-root", default=DATA_ROOT)
    ap.add_argument("--pretrained", action="store_true", help="使用 ImageNet 预训练 ResNet34 骨干")
    ap.add_argument("--loss", default="bce", choices=["bce", "bce_dice"], help="损失函数")
    ap.add_argument("--arch", default="unetplusplus", choices=["unetplusplus", "unet"],
                    help="主干架构。用于验证先验加权机制是否**可跨架构迁移**"
                         "（默认 unetplusplus = 既有行为）")
    ap.add_argument("--dice-w", type=float, default=1.0, help="Dice 损失权重")
    ap.add_argument("--norm-scale", type=float, default=1.0,
                    help="输入尺度归一化: 影像除以该值后送网络(1.0=不缩放, 既有行为)")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n===== 训练: {args.region} | λ={args.lambda_prior} | epochs={args.epochs} | seed={args.seed} | norm={args.norm_scale} | device={device} =====")

    train_transform = A.Compose([
        A.HorizontalFlip(p=0.5), A.VerticalFlip(p=0.5), A.RandomRotate90(p=0.5),
    ], is_check_shapes=False)
    val_transform = A.Compose([], is_check_shapes=False)

    train_ds = LMHLDPriorDataset(args.data_root, args.region, "train", train_transform,
                                    norm_scale=args.norm_scale)
    val_ds = LMHLDPriorDataset(args.data_root, args.region, "val", val_transform,
                                    norm_scale=args.norm_scale)
    test_ds = LMHLDPriorDataset(args.data_root, args.region, "test", val_transform,
                                    norm_scale=args.norm_scale)
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=args.batch, shuffle=False, num_workers=0)

    _enc_w = "imagenet" if args.pretrained else None
    _ARCH = {"unetplusplus": smp.UnetPlusPlus, "unet": smp.Unet}
    model = _ARCH[args.arch](
        encoder_name="resnet34", encoder_weights=_enc_w,
        in_channels=NUM_CHANNELS, classes=NUM_CLASSES,
    ).to(device)
    print(f"架构: {args.arch} | 编码器权重: {_enc_w or '从头训练'}")
    criterion = nn.BCEWithLogitsLoss(reduction="none")
    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)

    tag = f"{args.region}_prior{args.lambda_prior}_seed{args.seed}" + ("_pre" if args.pretrained else "") + ("_dice" if args.loss == "bce_dice" else "") + ("" if args.arch == "unetplusplus" else f"_{args.arch}")
    out_dir = os.path.join(OUT_ROOT, tag)
    os.makedirs(out_dir, exist_ok=True)

    best_blds, best_epoch = 0.0, -1
    history = {"train_loss": [], "val_loss": [], "val_f1": [], "val_blds": []}

    for epoch in range(args.epochs):
        tl = train_one_epoch(model, train_loader, criterion, optimizer, device, args.lambda_prior,
                            loss_kind=args.loss, dice_w=args.dice_w)
        vl, vp, vr, vf1, vb = validate(model, val_loader, criterion, device)
        history["train_loss"].append(tl)
        history["val_loss"].append(vl)
        history["val_f1"].append(vf1)
        history["val_blds"].append(vb)
        if vb > best_blds:
            best_blds, best_epoch = vb, epoch
            torch.save(model.state_dict(), os.path.join(out_dir, "best_model.pth"))
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"epoch {epoch+1:3d}/{args.epochs} | train_loss {tl:.4f} | val_f1 {vf1:.4f} | val_blds {vb:.4f}")
        scheduler.step()
    print(f"最优: epoch {best_epoch+1} BLDS={best_blds:.4f}")

    # ── 测试集评估(像素级 + 分尺度区域召回) ──
    model.load_state_dict(torch.load(os.path.join(out_dir, "best_model.pth"), weights_only=True))
    model.eval()
    preds_all, gt_all = [], []
    with torch.no_grad():
        for images, masks, _ in test_loader:
            preds_all.append((torch.sigmoid(model(images.to(device))) > 0.5).cpu().numpy())
            gt_all.append(masks.numpy())
    preds = np.concatenate(preds_all)[:, 0]
    gt = np.concatenate(gt_all)[:, 0].astype(np.uint8)

    pm = pixel_metrics(preds, gt)
    buckets = aggregate([region_recall_buckets(gt[i], preds[i]) for i in range(len(gt))])

    result = {
        "region": args.region, "lambda": args.lambda_prior, "seed": args.seed,
        "best_epoch": best_epoch + 1, "best_val_blds": best_blds,
        "test": {"pixel": pm, "region_recall": buckets},
    }
    with open(os.path.join(out_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"\n测试集: F1={pm['f1']:.4f} BLDS={pm['blds']:.4f} "
          f"Prec={pm['precision']:.4f} Rec={pm['recall']:.4f}")
    print(f"区域召回: 小={buckets['small']:.4f} 中={buckets['mid']:.4f} 大={buckets['large']:.4f}")
    print(f"结果已保存: {out_dir}")

    # ── 损失曲线 ──
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].plot(history["train_loss"], label="train loss")
    ax[0].plot(history["val_loss"], label="val loss")
    ax[0].set_title(f"{tag} 损失曲线"); ax[0].legend(); ax[0].grid(alpha=.3)
    ax[1].plot(history["val_f1"], label="val F1")
    ax[1].plot(history["val_blds"], label="val BLDS")
    ax[1].set_title("验证指标"); ax[1].legend(); ax[1].grid(alpha=.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "loss_curves.png"), dpi=130, bbox_inches="tight")
    plt.close()


def aggregate(buckets_list):
    out = {}
    for key in ("small", "mid", "large"):
        hit = sum(b[key][0] for b in buckets_list)
        tot = sum(b[key][1] for b in buckets_list)
        out[key] = hit / tot if tot else 0.0
    return out


if __name__ == "__main__":
    main()
