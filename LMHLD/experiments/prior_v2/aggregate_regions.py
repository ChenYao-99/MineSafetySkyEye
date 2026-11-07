# -*- coding: utf-8 -*-
"""
aggregate_regions.py — 6 区域一致性汇总 (先验加权训练损失)
============================================================
读取 models_prior 下所有 {region}_prior{0,1.0}_seed42 的 metrics.json,
生成统一对比表 + 跨区域柱状图 + markdown 汇总。

用法: python aggregate_regions.py
输出: experiments/prior_v2/区域一致性汇总/
"""
import os
import sys
import json
import glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.stdout.reconfigure(encoding="utf-8")
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False

MODELS_ROOT = r"D:\LMHLD\models_prior"
OUT_ROOT = r"D:\LMHLD\experiments\prior_v2\区域一致性汇总"

# 区域显示名 (中英对照)
REGION_NAMES = {
    "Emilia-Romagna_Italy_32": "意大利 Emilia",
    "Gorkha_Nepal_32": "尼泊尔 Gorkha",
    "Hokkaido_Japan_128": "日本 Hokkaido",
    "Hokkaido_Japan_64": "日本 Hokkaido(64px)",
    "Jiuzhaigou_China_128": "中国 九寨沟",
    "Wenchuan_China_224": "中国 汶川",
    "Taiwan_China_224": "中国 台湾",
}


def load_pair(region):
    """加载 region 的 λ=0 与 λ=1.0 指标, 缺任一返回 None"""
    pair = {}
    for lam in (0.0, 1.0):
        d = os.path.join(MODELS_ROOT, f"{region}_prior{lam}_seed42", "metrics.json")
        if not os.path.exists(d):
            return None
        with open(d, encoding="utf-8") as f:
            m = json.load(f)
        pair[lam] = m["test"]
    return pair


def main():
    pairs = {}
    for region in REGION_NAMES:
        p = load_pair(region)
        if p:
            pairs[region] = p

    print(f"汇总 {len(pairs)} 个区域 (λ=0 vs λ=1.0 配对):")
    print(f"\n{'区域':<14}{'基线F1':<9}{'先验F1':<9}{'ΔF1':<8}{'基线小区':<9}{'先验小区':<9}{'Δ小区':<9}{'基线BLDS':<10}{'先验BLDS'}")
    rows = []
    for region, p in pairs.items():
        b, t = p[0.0], p[1.0]
        bf, tf = b["pixel"]["f1"], t["pixel"]["f1"]
        bs, ts = b["region_recall"]["small"], t["region_recall"]["small"]
        bb, tb = b["pixel"]["blds"], t["pixel"]["blds"]
        rows.append((region, bf, tf, tf - bf, bs, ts, ts - bs, bb, tb))
        print(f"{REGION_NAMES[region]:<14}{bf:<9.4f}{tf:<9.4f}{tf-bf:<+8.4f}"
              f"{bs:<9.4f}{ts:<9.4f}{ts-bs:<+9.4f}{bb:<10.4f}{tb:.4f}")

    n_win = sum(1 for r in rows if r[3] > 0)
    print(f"\nλ=1.0 F1 优于基线的区域: {n_win}/{len(rows)}")
    print(f"平均 ΔF1: {np.mean([r[3] for r in rows]):+.4f} | 平均 Δ小区召回: {np.mean([r[6] for r in rows]):+.4f}")

    # ── 图: 分组柱状 (F1) + Δ小区召回折线 ──
    os.makedirs(OUT_ROOT, exist_ok=True)
    names = [REGION_NAMES[r[0]] for r in rows]
    fig, axes = plt.subplots(1, 2, figsize=(15, 4.8))
    x = np.arange(len(rows))
    w = 0.36
    axes[0].bar(x - w / 2, [r[1] for r in rows], w, label="基线 λ=0", color="#B0B0B0")
    axes[0].bar(x + w / 2, [r[2] for r in rows], w, label="先验加权 λ=1.0", color="#D92B2B")
    for i, r in enumerate(rows):
        axes[0].text(i - w / 2, r[1] + 0.004, f"{r[1]:.3f}", ha="center", fontsize=7.5)
        axes[0].text(i + w / 2, r[2] + 0.004, f"{r[2]:.3f}", ha="center", fontsize=7.5, color="#D92B2B")
    axes[0].set_xticks(x); axes[0].set_xticklabels(names, fontsize=8)
    axes[0].set_ylabel("整体 F1"); axes[0].set_title("先验加权训练损失: 6 区域一致性验证")
    axes[0].legend(); axes[0].grid(alpha=.3, axis="y")
    axes[0].set_ylim(0.65, 0.82)

    axes[1].bar(x, [r[6] for r in rows], 0.5, color=["#0B8A2F" if r[6] > 0 else "#E06A1E" for r in rows])
    for i, r in enumerate(rows):
        axes[1].text(i, r[6] + (0.001 if r[6] >= 0 else -0.006), f"{r[6]:+.3f}", ha="center", fontsize=9)
    axes[1].axhline(0, color="#666", lw=0.8)
    axes[1].set_xticks(x); axes[1].set_xticklabels(names, fontsize=8)
    axes[1].set_ylabel("Δ 小尺度裂缝区域召回"); axes[1].set_title("先验加权的尺度增益")
    axes[1].grid(alpha=.3, axis="y")
    plt.tight_layout()
    fig_path = os.path.join(OUT_ROOT, "region_consistency.png")
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n图已保存: {fig_path}")

    # ── markdown ──
    md = ["# 先验加权训练损失 · 跨区域一致性汇总", "",
          "实验协议: 各区域 λ=0(基线) vs λ=1.0(先验加权), 同 seed 42 / 同模型结构 / 同数据划分", "",
          "| 区域 | 基线F1 | 先验F1 | ΔF1 | 基线小区召回 | 先验小区召回 | Δ小区召回 | 基线BLDS | 先验BLDS |",
          "|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        md.append(f"| {REGION_NAMES[r[0]]} | {r[1]:.4f} | {r[2]:.4f} | {r[3]:+.4f} | "
                  f"{r[4]:.4f} | {r[5]:.4f} | {r[6]:+.4f} | {r[7]:.4f} | {r[8]:.4f} |")
    md += ["", f"结论: λ=1.0 在 {n_win}/{len(rows)} 个区域 F1 优于基线, "
               f"平均 ΔF1 {np.mean([r[3] for r in rows]):+.4f}, 平均 Δ小区召回 {np.mean([r[6] for r in rows]):+.4f}",
            "", f"图: region_consistency.png", ""]
    md_path = os.path.join(OUT_ROOT, "汇总.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md))
    print(f"汇总已保存: {md_path}")


if __name__ == "__main__":
    main()
