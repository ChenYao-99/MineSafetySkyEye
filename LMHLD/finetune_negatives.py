# -*- coding: utf-8 -*-
"""
finetune_negatives.py — 真实影像域适配微调 (负样本 + NIR 鲁棒性)
=================================================================
依据第十三组诊断结论设计:

  · 基率/先验失配已被推翻 → 本脚本的目的**不是**"补无滑坡先验",
    而是**域适配**: 在真实影像的成像域里重新校准模型;
  · **NIR 缺失是最大的单项代价(-32.7pp)**, 且 Esri 无近红外波段。
    对策: 训练时以概率 p 把 NIR 通道替换为 R(`--nir-dropout`),
    使模型在"无 NIR"条件下也能工作 —— 这是唯一能打的点。
  · 防遗忘: 训练集**混合 LMHLD 数据**, 而非只训目标域
    (adapt_baige 的教训: 只训目标域会使源域 F1 从 0.728 崩到 0.242)。

三重评估(缺一不可):
  ① LMHLD 测试 F1(正常协议)      —— 源域是否退化
  ② LMHLD 测试 F1(NIR:=R)        —— NIR 鲁棒性(真实影像的条件)  ← 本轮重点
  ③ 真实负样本误报率             —— 误报是否下降(基线 88.8%/55.3%)

用法:
  python finetune_negatives.py --tag v1 --nir-dropout 0.5 --epochs 30
  python finetune_negatives.py --tag ctrl --nir-dropout 0.0 --epochs 30   # 对照组
"""
import os
import sys
import json
import argparse
import time
from datetime import datetime

import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
import segmentation_models_pytorch as smp

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, r"D:\LMHLD")

from data_paths import resolve_data_root
DATA_ROOT = resolve_data_root("native")
MODELS_ROOT = r"D:\LMHLD\models_prior"
OUT = r"D:\LMHLD\experiments\domain_gap"
OUT16 = r"D:\LMHLD\experiments\nir_train"          # 第十六组产物
NEG_NPY = os.path.join(OUT, "negatives.npy")
# 合成器共用实现(第十六组抽库; 见 nir_synth_utils 的模块说明)
from nir_synth_utils import load_synth, synth_nir   # noqa: E402
REAL_IMG = r"D:\竞赛材料\真实场景验证\影像"
PS = 32
DEV = "cuda" if torch.cuda.is_available() else "cpu"

# 白格真实滑坡 ROI (与 adapt_baige.py 一致) —— 唯一的真实影像正样本来源
BAIGE_POS = (420, 300, 800, 440)

# 各区域原生图块尺寸不同(32/128/224), 必须统一到 32px 才能混合训练:
# Emilia/Gorkha 原生即 32; 其余用 LMHLD_tiled 的 32px 瓦片(与 train_unified.py 一致)
NATIVE = DATA_ROOT
TILED = resolve_data_root("tiled")
LMHLD_REGIONS = [
    (NATIVE, "Emilia-Romagna_Italy_32"), (NATIVE, "Gorkha_Nepal_32"),
    (TILED, "Jiuzhaigou_China_32"), (TILED, "Wenchuan_China_32"),
    (TILED, "Taiwan_China_32"),
]


def norm_patch(p):
    """逐图块逐通道 min-max → [0,255] (基座上传模型的训练协议)。
    p: (K,4,H,W) 或 (4,H,W)"""
    ax = (2, 3) if p.ndim == 4 else (1, 2)
    mn = p.min(axis=ax, keepdims=True)
    mx = p.max(axis=ax, keepdims=True)
    return (p - mn) / (mx - mn + 1e-6) * 255.0


def _read_4ch(path):
    img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    return np.concatenate([img, img[..., 2:3]], axis=-1).astype(np.float32)


def build_dataset(n_pos=2000, n_neg=2000, n_esri_neg=2500, seed=42, neg_idx=None,
                  neg_extra=None, n_extra=0, return_provenance=False):
    """返回 (imgs (N,4,32,32) float32 0-255域, masks (N,32,32) float32, meta, neg_sel)

    设计要点: 正负样本在**两个域都出现** —— 否则模型可能只学会"分域"
    (LMHLD 影像 vs Esri 影像) 而非"分滑坡", 那是虚假的泛化。
      LMHLD 域: 正(占比≥10%) + 负(占比≤1%)
      Esri 域 : 正(白格滑坡 ROI, 弱标注) + 负(城区/农田/港口/林地/裸地)

    第十六组: `neg_idx` 显式指定用哪些负样本(由 --dump-split 产出的 train_idx);
    为 None 时退回原来的 rng.choice 行为 —— 保证旧实验可原样复现。
    `neg_sel` 返回实际用到的 negatives.npy 行号, 供留出集取补集。
    """
    rng = np.random.default_rng(seed)
    X, Y = [], []
    provenance = []
    cnt = {"lmhld_pos": 0, "lmhld_neg": 0, "esri_neg": 0, "esri_pos": 0,
           "mining_neg": 0}

    # ── LMHLD 域 ──
    per_p = max(1, n_pos // len(LMHLD_REGIONS))
    per_n = max(1, n_neg // len(LMHLD_REGIONS))
    for root, reg in LMHLD_REGIONS:
        f = os.path.join(root, reg, "train_images.npy")
        lf = os.path.join(root, reg, "train_labels.npy")
        if not (os.path.exists(f) and os.path.exists(lf)):
            continue
        imgs = np.load(f, mmap_mode="r"); lbls = np.load(lf, mmap_mode="r")
        n = len(imgs)
        P, N = [], []
        for i in rng.choice(n, min(1500, n), replace=False):
            r = float((np.asarray(lbls[i, 0]) > 0).mean())
            if r >= 0.10:
                P.append(i)
            elif r <= 0.01:
                N.append(i)
        for i in (rng.choice(len(P), min(per_p, len(P)), replace=False) if P else []):
            X.append(np.asarray(imgs[P[i]])); Y.append((np.asarray(lbls[P[i], 0]) > 0))
            provenance.append({"kind": "lmhld", "region": reg, "split": "train", "index": int(P[i]), "label": "positive"})
            cnt["lmhld_pos"] += 1
        for i in (rng.choice(len(N), min(per_n, len(N)), replace=False) if N else []):
            X.append(np.asarray(imgs[N[i]])); Y.append(np.zeros((PS, PS), bool))
            provenance.append({"kind": "lmhld", "region": reg, "split": "train", "index": int(N[i]), "label": "background"})
            cnt["lmhld_neg"] += 1

    # ── Esri 域负样本 ──
    sel = np.array([], dtype=np.int64)
    if os.path.exists(NEG_NPY):
        neg = np.load(NEG_NPY)          # 全量入内存(90MB), 避免 memmap 花式索引的病态开销
        if neg_idx is None:
            sel = rng.choice(len(neg), min(n_esri_neg, len(neg)), replace=False)
        else:
            sel = np.asarray(neg_idx, dtype=np.int64)
        for i in sel:
            X.append(np.asarray(neg[i])); Y.append(np.zeros((PS, PS), bool))
            provenance.append({"kind": "generic_negative", "array": NEG_NPY, "index": int(i)})
            cnt["esri_neg"] += 1

    # ── 矿区负样本（第十九组）──
    # ⚠ 必须放在 Esri 负样本**之后**: rng 是顺序消费的, 提前插会打乱上面 sel 的选取,
    #   使既有实验无法复现。
    if neg_extra and n_extra > 0 and os.path.exists(neg_extra):
        ex = np.load(neg_extra)
        selx = rng.choice(len(ex), min(n_extra, len(ex)), replace=False)
        for i in selx:
            X.append(np.asarray(ex[i])); Y.append(np.zeros((PS, PS), bool))
            provenance.append({"kind": "mining_negative", "array": str(neg_extra), "index": int(i)})
            cnt["mining_neg"] += 1

    # ── Esri 域正样本 (白格滑坡 ROI, 弱标注: 整块同标签; 步长 16 重叠取样) ──
    bp = os.path.join(REAL_IMG, "baige_landslide.png")
    if os.path.exists(bp):
        img = _read_4ch(bp)
        x0, y0, x1, y1 = BAIGE_POS
        for y in range(y0, y1 - PS + 1, 16):
            for x in range(x0, x1 - PS + 1, 16):
                X.append(np.transpose(img[y:y+PS, x:x+PS], (2, 0, 1)))
                Y.append(np.ones((PS, PS), bool))
                provenance.append({"kind": "baige_weak_positive", "path": bp, "box": [x, y, x+PS, y+PS], "label": "weak_whole_patch"})
                cnt["esri_pos"] += 1

    X = norm_patch(np.stack(X).astype(np.float32))   # 统一为基座训练协议
    Y = np.stack(Y).astype(np.float32)
    # 返回**未排序**的 sel（即追加进 X 的顺序）。调用方若需比较集合用 np.sort。
    # 第十六组: 顺序会影响训练轨迹 —— 原训练按 rng.choice 的置换序追加，若改用
    # 排序序，样本集合相同但 batch 组成不同（实测对白格分离度有可观影响），
    # 故 --dump-split 同时落盘 perm 序以便做复现对照。
    if return_provenance:
        return X, Y, cnt, sel, provenance
    return X, Y, cnt, sel


class DS(torch.utils.data.Dataset):
    """第十六组: NIR 通道改为**三路互斥**。

    原来只有二元(NIR:=R / 真 NIR)。现在:
        合成 NIR(p=nir_synth) —— 有信息但带噪的估计, 模型须学会"信多少"
        NIR:=R  (p=nir_dropout) —— 零信息, 模型学到"忽略它"
        真 NIR  (其余)          —— 完整信息
    ⚠ 反证(第十三/十五组): realfit 练过 NIR:=R dropout, 但在矿山场景用合成 NIR
      时误报反而升 —— 说明"见过 NIR 缺失"不能推出"会用合成 NIR", 故需真做实验。

    `XS` 是预计算的合成 NIR (N,PS,PS) [0,255] 域。协议天然对齐: X 进本类前
    已被 norm_patch 归一化到 [0,255], 正是合成器的训练域, 无需额外归一化。
    几何增强时对 x/y/xs 同步翻转旋转(见 __getitem__ 注释中的非等变性说明)。
    """

    def __init__(self, X, Y, nir_dropout=0.0, nir_synth=0.0, XS=None, train=True, seed=0):
        self.X, self.Y = X, Y
        self.XS = XS
        self.nir_dropout = nir_dropout
        self.nir_synth = nir_synth
        self.train = train
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        x = self.X[i].copy()
        y = self.Y[i].copy()
        xs = self.XS[i].copy() if self.XS is not None else None
        if self.train:
            if self.rng.random() < 0.5:
                x, y = x[:, :, ::-1], y[:, ::-1]
                if xs is not None:
                    xs = xs[:, ::-1]
            if self.rng.random() < 0.5:
                x, y = x[:, ::-1, :], y[::-1, :]
                if xs is not None:
                    xs = xs[::-1, :]
            k = int(self.rng.integers(0, 4))
            if k:
                x, y = np.rot90(x, k, axes=(1, 2)), np.rot90(y, k)
                if xs is not None:
                    xs = np.rot90(xs, k)
            # ★ NIR 三路互斥。rng 调用位置与次数与改造前**完全一致**
            #   (nir_synth=0 时退化为原 `r < nir_dropout → R`), 故对照臂可复现 v1。
            r = self.rng.random()
            if xs is not None and r < self.nir_synth:
                x = np.concatenate([x[:3], xs[None]], axis=0)
            elif r < self.nir_synth + self.nir_dropout:
                x = np.concatenate([x[:3], x[2:3]], axis=0)
        return (torch.as_tensor(np.ascontiguousarray(x), dtype=torch.float32),
                torch.as_tensor(np.ascontiguousarray(y), dtype=torch.float32))


def evaluate(model, esri_neg=None, holdout_idx=None, synth_model=None, tag=""):
    """四重评估(第十六组扩入合成 NIR 变体):
      ① LMHLD F1(orig)      —— 源域是否退化
      ② LMHLD F1(nir_r)     —— 无 NIR 鲁棒性(真实影像的默认条件)
      ⑤ LMHLD F1(nir_synth) —— 合成器在**本模型**上的可用性
      ③ 真实负样本误报率     —— **只在留出集上算**(第十六组修正 45% 泄漏)
      ④ 白格真实滑坡分离度   —— 防"全说没有"; 分 nir_r / synth 两种推理
    键名约定: 无后缀 = NIR:=R(与第十三组口径一致, 保持历史可比);
              _synth 后缀 = 用合成 NIR 替换第 4 通道。
    """
    model.eval()
    out = {}
    vs = ["orig", "nir_r"] + (["nir_synth"] if synth_model is not None else [])
    # ①②⑤ LMHLD 测试集(Emilia, 与基座同源)
    f = os.path.join(DATA_ROOT, "Emilia-Romagna_Italy_32", "test_images.npy")
    lf = os.path.join(DATA_ROOT, "Emilia-Romagna_Italy_32", "test_labels.npy")
    if os.path.exists(f):
        imgs = np.load(f, mmap_mode="r"); lbls = np.load(lf, mmap_mode="r")
        n = min(len(imgs), 800)
        acc = {k: [0, 0, 0] for k in vs}
        with torch.no_grad():
            for i in range(0, n, 64):
                raw = norm_patch(np.asarray(imgs[i:i+64]).astype(np.float32))
                gt = (np.asarray(lbls[i:i+64])[:, 0] > 0).astype(np.uint8)
                b = torch.as_tensor(raw, device=DEV)
                xs = synth_nir(synth_model, raw) if synth_model is not None else None
                for k in vs:
                    if k == "orig":
                        x = b
                    elif k == "nir_r":
                        x = torch.cat([b[:, :3], b[:, 2:3]], 1)
                    else:   # nir_synth
                        x = torch.cat([b[:, :3],
                                       torch.as_tensor(xs, device=DEV).unsqueeze(1)], 1)
                    pr = (torch.sigmoid(model(x))[:, 0].cpu().numpy() >= 0.5).astype(np.uint8)
                    a = acc[k]
                    a[0] += int((pr & (gt == 1)).sum()); a[1] += int((pr & (gt == 0)).sum())
                    a[2] += int(((pr == 0) & (gt == 1)).sum())
        for k, (tp, fp, fn) in acc.items():
            p = tp / max(tp + fp, 1e-9); r = tp / max(tp + fn, 1e-9)
            out[f"lmhld_F1_{k}"] = round(2 * p * r / max(p + r, 1e-9), 4)
            out[f"lmhld_P_{k}"] = round(p, 4)
            out[f"lmhld_R_{k}"] = round(r, 4)   # ★ 必须看召回: 误报下降若以漏检为代价则无意义
    # ③ 真实负样本误报
    #    ⚠ 第十六组修正: 旧版评估**全部 5501 块**, 而训练用了其中 2500 块 → 45% 泄漏,
    #      "5.78%" 实为含训练样本的表现。现在只评估 holdout_idx 指定的补集。
    if esri_neg is not None and len(esri_neg):
        idx = (np.arange(len(esri_neg)) if holdout_idx is None
               else np.asarray(holdout_idx, dtype=np.int64))
        out["esri_n_eval"] = int(len(idx))
        out["esri_n_total"] = int(len(esri_neg))
        for key, use_synth in (("", False), ("_synth", True)):
            if use_synth and synth_model is None:
                continue
            fp_px = tot = blk = 0
            with torch.no_grad():
                for i in range(0, len(idx), 256):
                    raw = norm_patch(np.asarray(esri_neg[idx[i:i+256]]).astype(np.float32))
                    b = torch.as_tensor(raw, device=DEV)
                    if use_synth:
                        xs = torch.as_tensor(synth_nir(synth_model, raw),
                                             device=DEV).unsqueeze(1)
                        b = torch.cat([b[:, :3], xs], 1)
                    pr = (torch.sigmoid(model(b))[:, 0].cpu().numpy() >= 0.5).astype(np.uint8)
                    fp_px += int(pr.sum()); tot += pr.size
                    blk += int((pr.reshape(len(pr), -1).sum(1) >= 10).sum())
            out[f"esri_pixel_alarm{key}"] = round(fp_px / max(tot, 1), 5)
            out[f"esri_block_alarm{key}"] = round(blk / max(len(idx), 1), 4)
    # ④ 真实滑坡阳性检验: 白格滑坡区响应 vs 背景响应(分离度)
    #    —— 防止"误报下降"其实来自"模型学会了全说没有"
    bp = os.path.join(REAL_IMG, "baige_landslide.png")
    if os.path.exists(bp):
        img = _read_4ch(bp)          # 注意: _read_4ch 出来的第 4 通道已是 NIR:=R
        ps = img.shape[0] // (img.shape[0] // PS)      # 保持 32
        yy = list(range(0, img.shape[0] - PS + 1, PS))
        xx = list(range(0, img.shape[1] - PS + 1, PS))
        patches = norm_patch(np.stack([np.transpose(img[y:y+PS, x:x+PS], (2, 0, 1))
                                       for y in yy for x in xx]).astype(np.float32))
        xs_all = synth_nir(synth_model, patches) if synth_model is not None else None
        for key, mode in (("", "nir_r"), ("_synth", "synth")):
            if mode == "synth" and xs_all is None:
                continue
            prob = np.zeros((len(yy) * PS, len(xx) * PS), np.float32)
            with torch.no_grad():
                for i in range(0, len(patches), 256):
                    b = torch.as_tensor(patches[i:i+256], device=DEV)
                    if mode == "synth":
                        b = torch.cat([b[:, :3], torch.as_tensor(
                            xs_all[i:i+256], device=DEV).unsqueeze(1)], 1)
                    else:
                        b = torch.cat([b[:, :3], b[:, 2:3]], 1)
                    pr = torch.sigmoid(model(b))[:, 0].cpu().numpy()
                    for t, v in enumerate(pr):
                        r_, c_ = divmod(i + t, len(xx))
                        prob[r_*PS:(r_+1)*PS, c_*PS:(c_+1)*PS] = v

            def _m(box, _p=prob):
                x0_, y0_, x1_, y1_ = box
                return float(_p[y0_:y1_, x0_:x1_].mean())

            ls = _m((420, 300, 800, 440))
            bg = np.mean([_m((1030, 780, 1270, 1240)), _m((420, 620, 680, 860))])
            out[f"baige_landslide{key}"] = round(ls, 4)
            out[f"baige_background{key}"] = round(float(bg), 4)
            out[f"baige_separation{key}"] = round(ls - float(bg), 4)
    model.train()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=os.path.join(
        MODELS_ROOT, "Emilia-Romagna_Italy_32_upload_seed42", "best_model.pth"))
    ap.add_argument("--tag", default="v1")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--nir-dropout", type=float, default=0.5)
    ap.add_argument("--n-pos", type=int, default=2000)
    ap.add_argument("--n-neg", type=int, default=2000)
    ap.add_argument("--n-esri-neg", type=int, default=2500)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--eval-only", action="store_true")
    # ── 第十六组新增 ──
    ap.add_argument("--nir-synth", type=float, default=0.0,
                    help="以概率 p 把 NIR 换成**合成 NIR**(有信息带噪的估计)。与 "
                         "--nir-dropout 构成三路互斥: 合成 p / NIR:=R p_dropout / 真 NIR 其余")
    ap.add_argument("--synth-tag", default="nir_synth")
    ap.add_argument("--synth-seed", type=int, default=42)
    ap.add_argument("--split-json", default=os.path.join(OUT16, "neg_split_seed42.json"),
                    help="训练/留出负样本划分文件; 须先跑 --dump-split 生成")
    ap.add_argument("--dump-split", action="store_true",
                    help="复现原 rng 选区并写出划分文件后退出(不训练)")
    ap.add_argument("--use-perm", action="store_true",
                    help="用置换序（原训练的顺序）而非排序序取负样本; 集合相同, 仅顺序不同")
    ap.add_argument("--neg-extra", default=None,
                    help="额外的负样本 npy（第十九组: 矿区安全层 + 天然干旱地貌）")
    ap.add_argument("--n-mining-neg", type=int, default=0,
                    help="从 --neg-extra 取多少个（0 = 不用）")
    args = ap.parse_args()

    torch.manual_seed(args.seed)

    # ── 负样本划分 ──
    # ⚠ 防呆: --neg-extra 路径不存在时必须**响亮报错**。
    #   曾经因为 runner 里单反斜杠把 `\$f` 写成字面量, build_dataset 静默跳过,
    #   导致整轮对照实验的两个臂都没用上矿区负样本(日志里 mining_neg=0), 白跑一次。
    if args.neg_extra and not os.path.exists(args.neg_extra):
        print(f"!!! --neg-extra 指定的文件不存在: {args.neg_extra}")
        print("    (build_dataset 会静默跳过, 故在此硬拦截)")
        return 2
    if args.neg_extra and args.n_mining_neg <= 0:
        print("!!! 给了 --neg-extra 但 --n-mining-neg <= 0, 等于没用。请显式指定数量。")
        return 2

    if args.dump_split:
        X, Y, meta, sel = build_dataset(args.n_pos, args.n_neg, args.n_esri_neg, args.seed)
        n_tot = int(len(np.load(NEG_NPY, mmap_mode="r")))
        hold = np.setdiff1d(np.arange(n_tot), sel)
        assert len(np.intersect1d(sel, hold)) == 0 and len(sel) + len(hold) == n_tot
        os.makedirs(OUT16, exist_ok=True)
        rec = {"when": datetime.now().isoformat(), "seed": args.seed,
               "n_total": n_tot, "n_train": int(len(sel)), "n_holdout": int(len(hold)),
               "source": "build_dataset 的 rng 原样复现; holdout = 全集减 train",
               "train_idx": [int(v) for v in np.sort(sel)],
               "train_idx_perm": [int(v) for v in sel],
               "holdout_idx": [int(v) for v in hold]}
        with open(args.split_json, "w", encoding="utf-8") as fh:
            json.dump(rec, fh, ensure_ascii=False, indent=1)
        print(f"划分已写出 {args.split_json}: train={len(sel)} holdout={len(hold)} "
              f"(总 {n_tot})")
        return 0

    if not os.path.exists(args.split_json):
        print(f"!!! 缺少划分文件 {args.split_json}; 请先跑 --dump-split")
        return 2
    with open(args.split_json, encoding="utf-8") as fh:
        sp = json.load(fh)
    tr_idx = np.asarray(sp["train_idx"], dtype=np.int64)
    if args.use_perm:
        # 复现对照用: 采用 rng.choice 的**置换序**（原训练即如此）。集合完全相同，
        # 仅追加顺序不同 —— 用于判定"排序序"是否引入了系统偏差。
        tr_idx = np.asarray(sp["train_idx_perm"], dtype=np.int64)
        print("  [--use-perm] 采用置换序（与原训练一致）")
    ho_idx = np.asarray(sp["holdout_idx"], dtype=np.int64)
    assert len(np.intersect1d(tr_idx, ho_idx)) == 0, "训练集与留出集有交集!"
    print(f"划分载入: train={len(tr_idx)} holdout={len(ho_idx)} (总 {sp['n_total']})")

    # 合成器: 即使 nir_synth=0 也载入, 以便评估期算出 nir_synth 变体会话
    synth_model = None
    try:
        synth_model = load_synth(args.synth_tag, args.synth_seed)
        print(f"合成器已载入: {args.synth_tag}_seed{args.synth_seed}")
    except FileNotFoundError as e:
        print(f"合成器不可用({e}) -> 只评估 orig / nir_r 两个变体")

    model = smp.UnetPlusPlus(encoder_name="resnet34", encoder_weights=None,
                             in_channels=4, classes=1).to(DEV)
    if os.path.exists(args.base):
        model.load_state_dict(torch.load(args.base, weights_only=True, map_location=DEV))
        print(f"基座加载: {os.path.basename(os.path.dirname(args.base))}")
    outdir = os.path.join(MODELS_ROOT, f"real_ft_{args.tag}_seed{args.seed}")
    os.makedirs(outdir, exist_ok=True)

    esri_neg = np.load(NEG_NPY) if os.path.exists(NEG_NPY) else None   # 全量入内存
    if args.eval_only:
        r = evaluate(model, esri_neg, ho_idx, synth_model)
        print(json.dumps(r, ensure_ascii=False, indent=1))
        return 0

    print(f"构建数据集 (nir_dropout={args.nir_dropout} nir_synth={args.nir_synth}) …")
    X, Y, meta, sel = build_dataset(args.n_pos, args.n_neg, args.n_esri_neg, args.seed,
                                    neg_idx=tr_idx, neg_extra=args.neg_extra,
                                    n_extra=args.n_mining_neg)
    assert np.array_equal(np.sort(sel), np.sort(tr_idx)), "实际用到的负样本与划分文件不一致"
    print(f"  总样本 {len(X)} | " + " ".join(f"{k}={v}" for k, v in meta.items()))
    print(f"  正样本占比 {Y.mean():.3f}")

    # ── 预计算合成 NIR(而非逐样本前向: 后者 18 万次小前向 + 每次 GPU 同步, 训练翻倍) ──
    XS = None
    if args.nir_synth > 0 and synth_model is not None:
        print("  预计算合成 NIR …")
        XS = synth_nir(synth_model, X)
        _s = X[:64]
        _d = float(np.abs(synth_nir(synth_model, np.ascontiguousarray(_s[:, :, :, ::-1]))
                          - synth_nir(synth_model, _s)[:, :, ::-1]).mean())
        print(f"  增强非等变性 |synth(flip)-flip(synth)| = {_d:.3f} (0-255 域)")
    n_val = max(200, len(X) // 10)
    idx = np.random.default_rng(0).permutation(len(X))
    tr, va = idx[n_val:], idx[:n_val]
    ds_tr = DS(X[tr], Y[tr], nir_dropout=args.nir_dropout, nir_synth=args.nir_synth,
               XS=(XS[tr] if XS is not None else None), train=True, seed=args.seed)
    ds_va = DS(X[va], Y[va], 0.0, 0.0, None, train=False)
    # drop_last: 32px 输入经 resnet34 编码器后最深层为 1x1, 若末尾 batch 只剩 1 个样本
    # 会触发 BatchNorm "Expected more than 1 value per channel"(实测)
    dl_tr = torch.utils.data.DataLoader(ds_tr, batch_size=args.batch, shuffle=True,
                                        num_workers=0, drop_last=True)
    dl_va = torch.utils.data.DataLoader(ds_va, batch_size=args.batch, shuffle=False, num_workers=0)

    base_eval = evaluate(model, esri_neg, ho_idx, synth_model)
    print(f"\n微调前: {base_eval}")

    crit = nn.BCEWithLogitsLoss()
    opt = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)
    sched = optim.lr_scheduler.ExponentialLR(opt, gamma=0.96)
    # ★ 第十六组: 按 val_loss 选优。val split(599 样本, 混合源域+目标域)完全在
    #   **留出集之外**, 用它做选择不污染留出集 —— 留出集只用于报告。
    #   改造前是"每 5 轮覆盖存、实存末轮", 做 A/B 对比时机与过拟合程度混淆。
    best, best_ep, hist = float("inf"), -1, []
    t0 = time.time()
    for ep in range(1, args.epochs + 1):
        model.train(); tot = 0.0
        for x, y in dl_tr:
            x, y = x.to(DEV), y.to(DEV).unsqueeze(1)
            opt.zero_grad()
            loss = crit(model(x), y)
            loss.backward(); opt.step()
            tot += float(loss) * len(x)
        sched.step()
        model.eval(); v = 0.0
        with torch.no_grad():
            for x, y in dl_va:
                v += float(crit(model(x.to(DEV)), y.to(DEV).unsqueeze(1))) * len(x)
        v /= len(va)
        rec_ep = {"epoch": ep, "train_loss": round(tot / len(tr), 5), "val_loss": round(v, 5)}
        if v < best:
            best, best_ep = v, ep
            torch.save(model.state_dict(), os.path.join(outdir, "best_model.pth"))
        if ep % 5 == 0 or ep == args.epochs:
            ev = evaluate(model, esri_neg, ho_idx, synth_model)
            rec_ep["eval"] = ev
            print(f"  ep {ep:>3}/{args.epochs}  train={tot/len(tr):.5f}  val={v:.5f}  "
                  f"F1orig={ev.get('lmhld_F1_orig')} F1r={ev.get('lmhld_F1_nir_r')} "
                  f"holdFPR={ev.get('esri_block_alarm')} sep={ev.get('baige_separation')} "
                  f"sepS={ev.get('baige_separation_synth')}  ({time.time()-t0:.0f}s)")
        hist.append(rec_ep)

    torch.save(model.state_dict(), os.path.join(outdir, "last_model.pth"))
    print(f"\nbest_ep={best_ep} (val_loss={best:.5f}); 顺带存末轮 last_model.pth")

    after = evaluate(model, esri_neg, ho_idx, synth_model)      # 末轮
    print(f"\n微调后(末轮): {after}")
    if best_ep != args.epochs:
        model.load_state_dict(torch.load(os.path.join(outdir, "best_model.pth"),
                                         weights_only=True, map_location=DEV))
        best_eval = evaluate(model, esri_neg, ho_idx, synth_model)
        print(f"微调后(最优轮 ep{best_ep}): {best_eval}")
    else:
        best_eval = dict(after)

    rec = {"when": datetime.now().isoformat(), "tag": args.tag, "base": args.base,
           "nir_dropout": args.nir_dropout, "nir_synth": args.nir_synth,
           "synth_tag": args.synth_tag, "synth_seed": args.synth_seed,
           "epochs": args.epochs, "seed": args.seed,
           "n_samples": int(len(X)), "composition": meta,
           "pos_ratio": round(float(Y.mean()), 4),
           "neg_split": args.split_json, "n_holdout": int(len(ho_idx)),
           "neg_extra": args.neg_extra, "n_mining_neg": args.n_mining_neg,
           "best_ep": best_ep, "best_val_loss": round(best, 5),
           "before": base_eval, "after_last": after, "after_best": best_eval,
           "history": hist}
    os.makedirs(OUT16, exist_ok=True)
    p_rec = os.path.join(OUT16, f"finetune_{args.tag}.json")
    with open(p_rec, "w", encoding="utf-8") as f:
        json.dump(rec, f, ensure_ascii=False, indent=2)
    print(f"\n权重: {outdir}\\best_model.pth (+ last_model.pth)")
    print(f"记录: {p_rec}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
