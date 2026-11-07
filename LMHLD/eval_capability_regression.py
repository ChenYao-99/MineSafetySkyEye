# -*- coding: utf-8 -*-
"""
eval_capability_regression.py — 第十九组: 上线前的**能力回归测试**
================================================================
背景: 第十九组的矿区负样本把误报降了 4~7 倍, 但**模型整体变保守**。
用户 2026-09-30 明确要求: **若精度下降导致项目能力弱化, 必须立即反馈**。
故在上线推荐之前, 对三个候选模型做全面能力对照, 判据写死在本文件里。

★ 关键补充指标: `滑坡区风险面积占比@0.5` —— 操作性的"到底有没有检出来"。
  只报 mean_prob 会掩盖"均值降了但仍高于阈值"与"掉到阈值以下"的巨大差别。

判据（什么算**能力弱化**）:
  · A 源域:   F1(真NIR) 保留 < 85%                        → 源域退化
  · B 检出:   滑坡区风险面积占比@0.5 明显低于现役          → **检出能力下降**
  · C 上传域: 留出负样本误报显著变差                       → 上传域退化

用法: python eval_capability_regression.py
输出: experiments/mining_neg/capability_regression.json
"""
import json
import os
import sys
from datetime import datetime

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, r"D:\LMHLD")
import numpy as np
import cv2
import torch

from detector import LandslideDetector
from finetune_negatives import evaluate, NEG_NPY, _read_4ch, REAL_IMG, PS
from nir_synth_utils import load_synth

OUT = r"D:\LMHLD\experiments\mining_neg"
MODELS_ROOT = r"D:\LMHLD\models_prior"
BASE_F1 = 0.7372                      # 微调前基座源域 F1
BAIGE_LS = (420, 300, 800, 440)
BAIGE_BG = [(1030, 780, 1270, 1240), (420, 620, 680, 860)]

CANDS = {
    "realfit(现役)": os.path.join(MODELS_ROOT, "real_ft_v1_seed42", "best_model.pth"),
    "MIN(2000)": os.path.join(MODELS_ROOT, "real_ft_MIN_s42_seed42", "best_model.pth"),
    "MIN8(800)": os.path.join(MODELS_ROOT, "real_ft_MIN8_s42_seed42", "best_model.pth"),
    # 第二十组 T1b 的 ⑥：矿区负样本 2000 = 干旱600 + 工业1400（安全层）
    # 实测矿区误报 2.55/4.24/4.39%、白格分离度 0.2076，两项均优于已部署的 MIN8
    "T1b_ours(2000)": os.path.join(MODELS_ROOT, "real_ft_T1b_ours_seed42",
                                   "best_model.pth"),
}


def baige_probs(w, ps):
    """白格真实影像的整幅概率图（与 evaluate 同口径: 逐图块 minmax）"""
    img = _read_4ch(os.path.join(REAL_IMG, "baige_landslide.png"))
    yy = list(range(0, img.shape[0] - PS + 1, PS))
    xx = list(range(0, img.shape[1] - PS + 1, PS))
    pat = np.stack([np.transpose(img[y:y + PS, x:x + PS], (2, 0, 1))
                    for y in yy for x in xx]).astype(np.float32)
    mn = pat.min(axis=(1, 2, 3), keepdims=True)
    mx = pat.max(axis=(1, 2, 3), keepdims=True)
    pat = (pat - mn) / (mx - mn + 1e-6) * 255.0
    det = LandslideDetector(w, patch_size=PS)
    prob = det.detect_image(img, patch_norm="minmax")
    del det
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return prob


def main():
    os.makedirs(OUT, exist_ok=True)
    sp = json.load(open(os.path.join(r"D:\LMHLD\experiments\nir_train",
                                     "neg_split_seed42.json"), encoding="utf-8"))
    ho = np.asarray(sp["holdout_idx"], dtype=np.int64)
    neg = np.load(NEG_NPY)
    synth = load_synth("nir_synth", 42)
    res = {"when": datetime.now().isoformat(), "base_f1": BASE_F1, "models": {}}

    for name, w in CANDS.items():
        if not os.path.exists(w):
            print(f"缺权重: {name}")
            continue
        print(f"\n{'='*72}\n{name}\n{'='*72}")
        det = LandslideDetector(w, patch_size=32)
        ev = evaluate(det.model, neg, ho, synth)
        del det
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # A. 源域
        f1o = ev["lmhld_F1_orig"]; f1r = ev["lmhld_F1_nir_r"]
        # B. 白格检出（含**风险面积占比**——操作性的"有没有检出来"）
        prob = baige_probs(w, PS)
        x0, y0, x1, y1 = BAIGE_LS
        ls = prob[y0:y1, x0:x1]
        bgm = np.concatenate([prob[b[1]:b[3], b[0]:b[2]].ravel() for b in BAIGE_BG])
        binfo = {
            "ls_mean": round(float(ls.mean()), 4),
            "bg_mean": round(float(bgm.mean()), 4),
            "sep_abs": round(float(ls.mean() - bgm.mean()), 4),
            "sep_rel": round(float((ls.mean() - bgm.mean()) /
                                   ((ls.mean() + bgm.mean()) / 2 + 1e-9)), 4),
            "ls_risk@0.5": round(float((ls >= 0.5).mean()), 4),
            "bg_risk@0.5": round(float((bgm >= 0.5).mean()), 4),
        }
        row = {"lmhld_F1_orig": f1o, "lmhld_F1_nir_r": f1r,
               "retention": round(f1o / BASE_F1, 4) if f1o else None,
               "baige": binfo,
               "holdout_block_alarm": ev["esri_block_alarm"],
               "holdout_pixel_alarm": ev["esri_pixel_alarm"]}
        # D. 真实矿区（复用第十九组已测结果）
        mp = os.path.join(OUT, "eval_mining_neg.json")
        pre = os.path.join(OUT, "_mining_by_model.json")
        res["models"][name] = row
        print(f"  源域   F1(真NIR) {f1o}  保留 {row['retention']:.1%}   "
              f"F1(NIR:=R) {f1r}")
        print(f"  白格滑坡区 均值 {binfo['ls_mean']:.4f}  "
              f"**风险面积@0.5 {binfo['ls_risk@0.5']:.1%}**   "
              f"背景 {binfo['bg_mean']:.4f} / {binfo['bg_risk@0.5']:.1%}")
        print(f"  分离度 绝对 {binfo['sep_abs']:+.4f}   相对 {binfo['sep_rel']:+.4f}")
        print(f"  留出Esri负样本  图块误报 {ev['esri_block_alarm']:.2%}  "
              f"像素误报 {ev['esri_pixel_alarm']:.2%}")

    # ── 判读 ──
    print("\n" + "=" * 72 + "\n能力回归判读（相对现役）\n" + "=" * 72)
    ref = res["models"].get("realfit(现役)")
    verdict = {}
    if ref:
        for name, r in res["models"].items():
            if name == "realfit(现役)":
                continue
            d = {
                "retention_vs_base": r["retention"],
                "A_source_ok": (r["retention"] or 0) >= 0.85,
                "B_detect_ls_risk": r["baige"]["ls_risk@0.5"],
                "B_detect_vs_ref": round(r["baige"]["ls_risk@0.5"] -
                                         ref["baige"]["ls_risk@0.5"], 4),
                "B_detect_ok": r["baige"]["ls_risk@0.5"] >=
                               ref["baige"]["ls_risk@0.5"] - 0.02,
                "C_fp_vs_ref": round(r["holdout_block_alarm"] -
                                     ref["holdout_block_alarm"], 4),
                "C_fp_ok": r["holdout_block_alarm"] <= ref["holdout_block_alarm"] + 0.01,
            }
            verdict[name] = d
            ok = d["A_source_ok"] and d["B_detect_ok"] and d["C_fp_ok"]
            print(f"  {name}")
            print(f"    A 源域保留 {d['retention_vs_base']:.1%} "
                  f"{'✓' if d['A_source_ok'] else '✗ 源域退化'}")
            print(f"    B 检出(滑坡区风险面积) {d['B_detect_ls_risk']:.1%} "
                  f"vs 现役 {ref['baige']['ls_risk@0.5']:.1%} "
                  f"({d['B_detect_vs_ref']:+.2%}) "
                  f"{'✓' if d['B_detect_ok'] else '✗ **检出能力下降**'}")
            print(f"    C 上传域误报 {r['holdout_block_alarm']:.2%} "
                  f"vs 现役 {ref['holdout_block_alarm']:.2%} "
                  f"({d['C_fp_vs_ref']:+.2%}) {'✓' if d['C_fp_ok'] else '✗'}")
            print(f"    >>> {'**无能力弱化**' if ok else '**存在能力弱化, 需人工判断**'}")
    res["verdict"] = verdict
    with open(os.path.join(OUT, "capability_regression.json"), "w",
              encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)
    print(f"\n结果: {os.path.join(OUT, 'capability_regression.json')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
