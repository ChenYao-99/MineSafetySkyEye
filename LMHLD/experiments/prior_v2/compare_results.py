# -*- coding: utf-8 -*-
"""
compare_results.py — 汇总第二阶段实验(先验加权训练损失)全部结果
用法: python compare_results.py [region]
读取 D:\\LMHLD\\models_prior\\{region}_prior{λ}_seed42\\metrics.json
输出: experiments\\prior_v2\\{region}\\summary.md + comparison.png
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
OUT_ROOT = r"D:\LMHLD\experiments\prior_v2"


def load_results(region):
    """加载该区域所有 λ 的实验结果"""
    results = {}
    for d in sorted(glob.glob(os.path.join(MODELS_ROOT, f"{region}_prior*_seed*.json".replace(".json", ""))) or []):
        pass
    # glob 目录
    dirs = sorted(glob.glob(os.path.join(MODELS_ROOT, f"{region}_prior*_seed*")))
    for d in dirs:
        m_path = os.path.join(d, "metrics.json")
        if not os.path.exists(m_path):
            continue
        with open(m_path, encoding="utf-8") as f:
            m = json.load(f)
        lam = m.get("lambda", float(os.path.basename(d).split("prior")[1].split("_")[0]))
        results[lam] = m
    return results


def main():
    region = sys.argv[1] if len(sys.argv) > 1 else "Emilia-Romagna_Italy_32"
    results = load_results(region)
    if not results:
        print(f"未找到 {region} 的训练结果, 请先运行 train_prior.py")
        return

    out_dir = os.path.join(OUT_ROOT, region)
    os.makedirs(out_dir, exist_ok=True)

    lams = sorted(results.keys())
    print(f"\n===== {region} 先验加权训练损失对比 ({len(lams)} 组) =====\n")
    print(f"{'λ':<6}{'Epoch':<8}{'F1':<8}{'BLDS':<8}{'Prec':<8}{'Rec':<8}{'小区召回':<9}{'中区召回':<9}{'大区召回'}")
    rows = []
    for lam in lams:
        r = results[lam]
        p, rr = r["test"]["pixel"], r["test"]["region_recall"]
        rows.append((lam, r.get("best_epoch", "-"), p, rr))
        print(f"{lam:<6}{r.get('best_epoch','-'):<8}{p['f1']:<8.4f}{p['blds']:<8.4f}"
              f"{p['precision']:<8.4f}{p['recall']:<8.4f}{rr['small']:<9.4f}{rr['mid']:<9.4f}{rr['large']:.4f}")

    # 找最佳配置
    best_f1 = max(rows, key=lambda x: x[2]["f1"])
    best_small = max(rows, key=lambda x: x[3]["small"])
    best_bal = max(rows, key=lambda x: (x[2]["f1"] - abs(x[3]["small"] - rows[0][3]["small"]) * 0))
    print(f"\n最优整体 F1: λ={best_f1[0]} ({best_f1[2]['f1']:.4f})")
    print(f"最优小区召回: λ={best_small[0]} ({best_small[3]['small']:.4f})")
    base = rows[0]
    print(f"\n相对基线(λ=0)变化:")
    for lam, ep, p, rr in rows[1:]:
        print(f"  λ={lam}: F1 {p['f1']-base[2]['f1']:+.4f} | 小区召回 {rr['small']-base[3]['small']:+.4f} | BLDS {p['blds']-base[2]['blds']:+.4f}")

    # ── 汇总图 ──
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.5))
    f1s = [r[2]["f1"] for r in rows]
    blds = [r[2]["blds"] for r in rows]
    smalls = [r[3]["small"] for r in rows]
    mids = [r[3]["mid"] for r in rows]
    xs = [str(l) for l in lams]

    axes[0].plot(xs, f1s, "o-", color="#D92B2B", label="F1")
    axes[0].plot(xs, blds, "s-", color="#E06A1E", label="BLDS")
    axes[0].set_xlabel("先验权重 λ"); axes[0].set_ylabel("整体指标")
    axes[0].set_title(f"{region} 不同λ的整体精度"); axes[0].legend(); axes[0].grid(alpha=.3)

    axes[1].plot(xs, smalls, "o-", color="#1E6FD9", label="小尺度(≤25px)")
    axes[1].plot(xs, mids, "s-", color="#0B8A2F", label="中尺度(26-100px)")
    axes[1].set_xlabel("先验权重 λ"); axes[1].set_ylabel("区域级召回率")
    axes[1].set_title("分尺度区域召回"); axes[1].legend(); axes[1].grid(alpha=.3)

    axes[2].plot(smalls, f1s, "o-", color="#7B57C9")
    for lam, ep, p, rr in rows:
        axes[2].annotate(f"λ={lam}", (rr["small"], p["f1"]), textcoords="offset points",
                         xytext=(6, 4), fontsize=9)
    axes[2].set_xlabel("小尺度裂缝区域召回率"); axes[2].set_ylabel("整体 F1")
    axes[2].set_title("小尺度召回 vs 整体精度"); axes[2].grid(alpha=.3)

    plt.tight_layout()
    fig_path = os.path.join(out_dir, "comparison.png")
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"对比图已保存: {fig_path}")

    # ── markdown 汇总 ──
    md = [f"# {region} · 先验加权训练损失对比", "",
          f"实验日期: 2026-09-01 | 模型: UnetPlusPlus/resnet34 | 协议与 train.py 一致, 固定种子 42", "",
          "| λ | 最优Epoch | F1 | BLDS | Precision | Recall | 小区召回 | 中区召回 | 大区召回 |",
          "|---|---|---|---|---|---|---|---|---|"]
    for lam, ep, p, rr in rows:
        md.append(f"| {lam} | {ep} | {p['f1']:.4f} | {p['blds']:.4f} | {p['precision']:.4f} | "
                  f"{p['recall']:.4f} | {rr['small']:.4f} | {rr['mid']:.4f} | {rr['large']:.4f} |")
    md += ["", f"图见: comparison.png", ""]
    md += ["## 结论", ""]
    md += [f"- 整体最优: λ={best_f1[0]} (F1={best_f1[2]['f1']:.4f})"]
    md += [f"- 小尺度最优: λ={best_small[0]} (召回={best_small[3]['small']:.4f})"]
    md += [f"- 基线λ=0: F1={base[2]['f1']:.4f}, 小区召回={base[3]['small']:.4f}"]
    md_path = os.path.join(out_dir, "summary.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md))
    print(f"汇总已保存: {md_path}")


if __name__ == "__main__":
    main()
