# -*- coding: utf-8 -*-
"""
eval_ds.py — D-S 证据理论多源融合实验 (创新点二核心)
======================================================
识别框架 Θ = {滑坡(L), 非滑坡(N), 不确定(T)}
证据源 (Hokkaido 区域, 同一影像):
  E1: 128px 大尺度检测器 (全局上下文)
  E2: 32px  细粒度检测器 (细节信息)
  E3: 地质先验图 (NDVI裸露度+粗糙度)
组合规则: Dempster 组合 (含冲突因子 K)
决策:   m(L) > 0.5 → 预警; 冲突拒绝: K > θ → 标为不确定, 不触发预警

核心假设: 单一检测器的误报在证据组合中被抑制 → 系统误报率下降
对比: 单证据 / 平均融合 / D-S / D-S+冲突拒绝

用法: python eval_ds.py
输出: experiments/ds_fusion/
"""
import os
import sys
import json
import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.stdout.reconfigure(encoding="utf-8")
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False

from detector import LandslideDetector
from train_prior import compute_prior_map

DATA_ROOT = r"F:\LMHLD\LMHLD_dataset_different_patch_sizes"
MODELS_ROOT = r"D:\LMHLD\models_prior"
OUT_ROOT = r"D:\LMHLD\experiments\ds_fusion"
REGION = "Hokkaido_Japan_128"
SMALL, MID = 25, 100
UNCERTAINTY = 0.05   # 每证据的固有不确定度 m(Θ)
DECISION_THR = 0.5


# ── D-S 组合 (向量化, 逐像素) ──────────────────────────
def mass_from_prob(p, unc=UNCERTAINTY):
    """概率 → mass 函数 (mL, mN, mT), 保持总质量=1 (支持数组)"""
    mL = p * (1 - unc)
    mN = (1 - p) * (1 - unc)
    return mL, mN, np.full_like(p, unc)


def combine_two(m1, m2):
    """Dempster 组合两 mass 函数 (逐像素)"""
    mL1, mN1, mT1 = m1
    mL2, mN2, mT2 = m2
    K = mL1 * mN2 + mN1 * mL2              # 冲突因子
    denom = 1 - K + 1e-9
    mL = (mL1 * mL2 + mL1 * mT2 + mT1 * mL2) / denom
    mN = (mN1 * mN2 + mN1 * mT2 + mT1 * mN2) / denom
    mT = (mT1 * mT2) / denom
    return mL, mN, mT


def ds_combine(masses):
    """组合任意多个 mass 函数 (逐像素)"""
    result = masses[0]
    for m in masses[1:]:
        result = combine_two(result, m)
    return result


def pixel_metrics(pred, gt):
    tp = int(((pred == 1) & (gt == 1)).sum())
    fp = int(((pred == 1) & (gt == 0)).sum())
    fn = int(((pred == 0) & (gt == 1)).sum())
    tn = int(((pred == 0) & (gt == 0)).sum())
    prec = tp / (tp + fp + 1e-6)
    rec = tp / (tp + fn + 1e-6)
    f1 = 2 * prec * rec / (prec + rec + 1e-6)
    blds = (prec * rec * f1) ** (1 / 3)
    fpr = fp / (fp + tn + 1e-6)            # 误报率 (False Positive Rate)
    return {"precision": prec, "recall": rec, "f1": f1, "blds": blds, "fpr": fpr}


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


def aggregate(bl):
    return {k: (sum(b[k][0] for b in bl) / max(sum(b[k][1] for b in bl), 1)) for k in ("small", "mid", "large")}


def eval_mask(mask, gt):
    return {"pixel": pixel_metrics(mask, gt),
            "region_recall": aggregate([region_recall_buckets(gt[i], mask[i]) for i in range(len(gt))])}


def main():
    region_dir = os.path.join(DATA_ROOT, REGION)
    images = np.load(os.path.join(region_dir, "test_images.npy"))
    labels = np.load(os.path.join(region_dir, "test_labels.npy"))
    gt = (labels[:, 0] > 0).astype(np.uint8)
    N = len(images)
    print(f"===== {REGION} D-S 证据融合 | {N} 样本 =====")

    det128 = LandslideDetector(os.path.join(MODELS_ROOT, f"{REGION}_prior1.0_seed42", "best_model.pth"), patch_size=128)
    det32 = LandslideDetector(os.path.join(MODELS_ROOT, "Hokkaido_Japan_32_prior1.0_seed42", "best_model.pth"), patch_size=32)

    probs128 = det128.predict_batch(images)
    probs32 = np.stack([det32.detect_image(images[i].transpose(1, 2, 0), stride=32) for i in range(N)])
    priors = np.stack([compute_prior_map(images[i].transpose(1, 2, 0)) for i in range(N)])
    # 先验转为概率: 用 logistic 校准, 中位先验 → 0.5
    prior_logit = 1 / (1 + np.exp(-6 * (priors - np.median(priors))))

    results = {}
    # A. 单证据
    results["E1_128px"] = eval_mask((probs128 >= DECISION_THR).astype(np.uint8), gt)
    results["E2_32px"] = eval_mask((probs32 >= DECISION_THR).astype(np.uint8), gt)
    results["E3_prior"] = eval_mask((prior_logit >= DECISION_THR).astype(np.uint8), gt)
    # B. 平均融合
    avg = (probs128 + probs32) / 2
    results["avg_fusion"] = eval_mask((avg >= DECISION_THR).astype(np.uint8), gt)
    # C. D-S: 两证据
    mL2, _, _ = ds_combine([mass_from_prob(probs128), mass_from_prob(probs32)])
    results["DS_128+32"] = eval_mask((mL2 >= DECISION_THR).astype(np.uint8), gt)
    # D. D-S: 三证据
    mL3, _, _ = ds_combine([mass_from_prob(probs128), mass_from_prob(probs32),
                            mass_from_prob(prior_logit)])
    results["DS_128+32+prior"] = eval_mask((mL3 >= DECISION_THR).astype(np.uint8), gt)
    # 冲突因子 K (128px vs 32px 两两冲突)
    k_map = probs128 * (1 - probs32) + (1 - probs128) * probs32
    # E. D-S + 冲突拒绝 (K>θ → 不确定, 不预警)
    for theta in (0.3, 0.5, 0.7):
        mask = (mL3 >= DECISION_THR).astype(np.uint8)
        mask[k_map > theta] = 0
        results[f"DS+reject_θ={theta}"] = eval_mask(mask, gt)

    # ── 打印 ──
    print(f"\n{'方案':<20}{'F1':<8}{'BLDS':<8}{'Prec':<8}{'Rec':<8}{'误报率FPR':<10}{'小区召回':<9}{'中区召回'}")
    for k, r in results.items():
        p, rr = r["pixel"], r["region_recall"]
        print(f"{k:<20}{p['f1']:<8.4f}{p['blds']:<8.4f}{p['precision']:<8.4f}{p['recall']:<8.4f}"
              f"{p['fpr']:<10.4f}{rr['small']:<9.4f}{rr['mid']:.4f}")

    base = results["E1_128px"]
    print(f"\n相对单检测器(128px) FPR={base['pixel']['fpr']:.4f}:")
    for k in ["avg_fusion", "DS_128+32", "DS_128+32+prior", "DS+reject_θ=0.5", "DS+reject_θ=0.7"]:
        p, rr = results[k]["pixel"], results[k]["region_recall"]
        print(f"  {k:<20} FPR {p['fpr']-base['pixel']['fpr']:+.4f} ({p['fpr']/base['pixel']['fpr']*100:.0f}%) | F1 {p['f1']-base['pixel']['f1']:+.4f} | 小区召回 {rr['small']-base['region_recall']['small']:+.4f}")

    out_dir = os.path.join(OUT_ROOT, REGION)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "ds_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    # ── 可视化 ──
    i = int(np.argmax(gt.sum(axis=(1, 2))))
    fig, axes = plt.subplots(1, 5, figsize=(18, 4))
    plots = [
        ("原图", det128.to_rgb(images[i].transpose(1, 2, 0)), False),
        ("真实标签", (gt[i] * 255).astype(np.uint8), True),
        ("128px检测", (probs128[i] >= 0.5).astype(np.uint8) * 255, True),
        ("D-S融合", (mL3[i] >= 0.5).astype(np.uint8) * 255, True),
        ("D-S+冲突拒绝", ((mL3[i] >= 0.5) & (k_map[i] <= 0.7)).astype(np.uint8) * 255, True),
    ]
    for ax, (t, img, gray) in zip(axes, plots):
        ax.imshow(img, cmap="gray" if gray else None)
        ax.set_title(t, fontsize=10); ax.axis("off")
    plt.tight_layout()
    fig_path = os.path.join(out_dir, "ds_visual.png")
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n可视化已保存: {fig_path}")


if __name__ == "__main__":
    main()
