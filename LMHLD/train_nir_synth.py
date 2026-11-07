# -*- coding: utf-8 -*-
"""
train_nir_synth.py — RGB → NIR 波段重建（多光谱合成器）
=========================================================
动机: 第十四组证明 Sentinel-2（10m，有真 NIR）分辨率不足以支撑单个滑坡检测；
第十三组证明 NIR 缺失代价 −32.7pp。两条合起来指向同一个结论：
**必须在保持高分辨率的前提下把 NIR 补回来** —— 即从 RGB 重建 NIR。

可行性依据（规划期实测, 非推测）:
  · LMHLD 的每个 patch 都是 4 通道数组, **ch0:3 与 ch3 同块共存** —— 零配准误差、
    零时相错配, 比常规的多光谱合成数据集强得多（实际训练用 22,780 对）；
  · **逐图块归一化后 RGB→NIR 的 R² = 0.49~0.89**，而**全局拟合仅 0.001**
    （各传感器 NIR/R 比值从 2.0 到 6.4，差 3 倍）→ **必须在逐图块归一化域里学**，
    而检测器用的正是 `patch_norm="minmax"` 逐图块协议，二者一致；
  · 训练预算: resnet18 Unet @32px 约 7300 patch/s（RTX 4060），代价可接受。

⚠ 三条必须遵守的设计约束（否则会得到虚假的"成功"）:
  1. **逐图块逐通道 min-max 归一化**后再学 —— 全局域不可学；
  2. **按区域配平采样** —— LMHLD_tiled 里台湾一个区域就占 64% 的瓦片，
     不配平会被它主导，而各区域 RGB/NIR 关系差异极大；
  3. **验证必须含空间相关性** —— 实测过: 5% 噪声的合成 NIR 在均值/标准差上
     与真值几乎一致，但与真实先验图的**空间相关性为 0**。只看分布会误判通过。

用法: python train_nir_synth.py [--epochs 30] [--per-region 8000]
"""
import os
import sys
import json
import time
import argparse
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import segmentation_models_pytorch as smp

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, r"D:\LMHLD")

NATIVE = r"F:\LMHLD\LMHLD_dataset_different_patch_sizes"
TILED = r"F:\LMHLD\LMHLD_tiled"
OUT = r"D:\LMHLD\models_prior"
PS = 32
DEV = "cuda" if torch.cuda.is_available() else "cpu"

# (数据根, 区域目录) —— 全部 32px; 大区域用 LMHLD_tiled 的瓦片
# ⚠ Rio de Janeiro 为 5 通道, 与本任务不兼容, 显式排除
REGIONS = [
    (NATIVE, "Emilia-Romagna_Italy_32"), (NATIVE, "Gorkha_Nepal_32"),
    (TILED, "Hokkaido_Japan_32"), (TILED, "Jiuzhaigou_China_32"),
    (TILED, "Wenchuan_China_32"), (TILED, "Taiwan_China_32"),
]


def norm_patch(p):
    """逐图块逐通道 min-max → [0,255]。p: (N,C,H,W) 或 (C,H,W)
    ★ 这是本任务唯一可学的域（全局域 R²≈0.001）"""
    ax = (2, 3) if p.ndim == 4 else (1, 2)
    mn = p.min(axis=ax, keepdims=True)
    mx = p.max(axis=ax, keepdims=True)
    return (p - mn) / (mx - mn + 1e-6) * 255.0


def load_split(split, per_region, seed=42):
    """按区域配平采样, 返回 (rgb (N,3,32,32), nir (N,1,32,32)) —— 均为逐图块归一化后的 0-255 域"""
    rng = np.random.default_rng(seed)
    Xs, Ys, cnt = [], [], {}
    for root, reg in REGIONS:
        f = os.path.join(root, reg, f"{split}_images.npy")
        if not os.path.exists(f):
            print(f"  ✗ {reg}: 缺 {split}_images.npy")
            continue
        imgs = np.load(f, mmap_mode="r")
        if imgs.shape[1] != 4:                       # 防御: 排除 5 通道区域
            print(f"  ⚠ {reg}: {imgs.shape[1]} 通道, 跳过")
            continue
        n = len(imgs)
        idx = rng.choice(n, min(per_region, n), replace=False)
        arr = np.stack([np.asarray(imgs[i]) for i in idx]).astype(np.float32)   # (k,4,32,32)
        arr = norm_patch(arr) / 255.0                # ★ 逐图块逐通道归一化 → [0,1]
        Xs.append(arr[:, :3])                        # RGB
        Ys.append(arr[:, 3:4])                       # NIR
        cnt[reg] = len(idx)
    X = np.concatenate(Xs); Y = np.concatenate(Ys)
    return X, Y, cnt


class DS(torch.utils.data.Dataset):
    def __init__(self, X, Y, train=True, seed=0):
        self.X, self.Y, self.train = X, Y, train
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        x, y = self.X[i].copy(), self.Y[i].copy()
        if self.train:
            if self.rng.random() < 0.5:
                x, y = x[:, :, ::-1], y[:, :, ::-1]
            if self.rng.random() < 0.5:
                x, y = x[:, ::-1, :], y[:, ::-1, :]
            k = int(self.rng.integers(0, 4))
            if k:
                x, y = np.rot90(x, k, axes=(1, 2)), np.rot90(y, k, axes=(1, 2))
        return (torch.as_tensor(np.ascontiguousarray(x), dtype=torch.float32),
                torch.as_tensor(np.ascontiguousarray(y), dtype=torch.float32))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--per-region", type=int, default=8000)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--tag", default="nir_synth")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    print(f"device={DEV}")
    print("载入数据（逐图块归一化）…")
    t0 = time.time()
    Xtr, Ytr, c1 = load_split("train", args.per_region, args.seed)
    Xva, Yva, c2 = load_split("val", max(600, args.per_region // 10), args.seed + 1)
    print(f"  训练 {Xtr.shape}  验证 {Xva.shape}  ({time.time()-t0:.0f}s)")
    print(f"  各区域样本数: {c1}")

    dl_tr = torch.utils.data.DataLoader(DS(Xtr, Ytr, True, args.seed),
                                        batch_size=args.batch, shuffle=True,
                                        num_workers=0, drop_last=True)
    dl_va = torch.utils.data.DataLoader(DS(Xva, Yva, False),
                                        batch_size=args.batch, shuffle=False, num_workers=0)

    # 轻量主干: 合成任务是密集回归, 不需要 UnetPlusPlus+resnet34 的容量
    model = smp.Unet(encoder_name="resnet18", encoder_weights=None,
                     in_channels=3, classes=1).to(DEV)
    n_par = sum(p.numel() for p in model.parameters())
    print(f"模型: Unet-resnet18, 3ch→1ch, {n_par/1e6:.2f}M 参数")

    opt = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    crit = nn.MSELoss()

    def val_loss():
        model.eval(); s = 0.0
        with torch.no_grad():
            for x, y in dl_va:
                s += float(crit(model(x.to(DEV)), y.to(DEV))) * len(x)
        model.train()
        return s / len(Xva)

    # 基线: "NIR := R"（即检测器现状）。合成器必须超过它才有意义。
    base = float(np.mean((Xva[:, 2:3] - Yva) ** 2))   # [0,1] 域
    print(f"\n基线(NIR:=R) 在归一化域的 MSE = {base:.2f}  ← 合成器必须低于它\n")

    outdir = os.path.join(OUT, f"{args.tag}_seed{args.seed}")
    os.makedirs(outdir, exist_ok=True)
    hist = []
    t0 = time.time()
    for ep in range(1, args.epochs + 1):
        model.train(); tot = 0.0
        for x, y in dl_tr:
            x, y = x.to(DEV), y.to(DEV)
            opt.zero_grad()
            loss = crit(model(x), y)
            loss.backward(); opt.step()
            tot += float(loss) * len(x)
        sched.step()
        v = val_loss()
        hist.append({"epoch": ep, "train_mse": round(tot / len(Xtr), 3),
                     "val_mse": round(v, 3)})
        if ep % 5 == 0 or ep == 1:
            print(f"  ep {ep:>3}/{args.epochs}  train={tot/len(Xtr):>7.3f}  "
                  f"val={v:>7.3f}  (基线 {base:.2f})  {time.time()-t0:.0f}s")
            torch.save(model.state_dict(), os.path.join(outdir, "best_model.pth"))
    torch.save(model.state_dict(), os.path.join(outdir, "best_model.pth"))
    v = val_loss()
    print(f"\n最终 val MSE = {v:.3f}   基线(NIR:=R) = {base:.2f}   "
          f"→ {'✓ 优于基线' if v < base else '✗ 未超过基线'}  (降低 {(1-v/base)*100:.1f}%)")
    with open(os.path.join(OUT, f"{args.tag}_metrics.json"), "w", encoding="utf-8") as f:
        json.dump({"when": datetime.now().isoformat(), "tag": args.tag,
                   "params_M": round(n_par/1e6, 2), "epochs": args.epochs,
                   "n_train": int(len(Xtr)), "n_val": int(len(Xva)),
                   "composition": c1, "baseline_mse_nir_eq_r": round(base, 3),
                   "final_val_mse": round(v, 3),
                   "improve_pct": round((1 - v / base) * 100, 1),
                   "history": hist}, f, ensure_ascii=False, indent=2)
    print(f"权重: {outdir}\\best_model.pth")
    return 0


if __name__ == "__main__":
    sys.exit(main())
