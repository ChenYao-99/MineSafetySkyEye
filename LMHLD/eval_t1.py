# -*- coding: utf-8 -*-
"""
eval_t1.py — 第二十组 T1: 负样本选法对照的评估
================================================================
⚠ 本实验的**测量陷阱**：我们没有矿山滑坡正样本，只能测误报。
  而 ③（把采坑当负样本）被训练成"采场/排土场不是滑坡" → 它的误报**可能反而更低**，
  但那恰恰是**以牺牲真实检出为代价**换来的。**只看误报会把"把矿区学成负类"误读成"方法更好"。**

故本脚本除误报外，另设**三个反向探针**：

  P1 抚顺·矿坑 ROI 响应  —— ③ 若**专门**压低矿坑响应远超过 ④，即证据表明它把
                             "采场形态"当成了负类特征（过度压制）。
  P2 白格真实滑坡分离度  —— 已见训练场景响应检查，不能视为独立未见正样本。
  P3 LMHLD 源域 F1       —— 防遗忘（从训练 json 读）。

用法: python eval_t1.py
输出: experiments/mining_neg/t1_eval.json
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
from finetune_negatives import _read_4ch, REAL_IMG, PS
from eval_capability_regression import baige_probs

OUT = r"D:\LMHLD\experiments\mining_neg"
MODELS_ROOT = r"D:\LMHLD\models_prior"
DEV = "cuda" if torch.cuda.is_available() else "cpu"

# (显示名, 权重路径, 训练记录 tag / None 表示无)
MODELS = [
    ("① base(源域)", os.path.join(MODELS_ROOT, "Emilia-Romagna_Italy_32_upload_seed42",
                                "best_model.pth"), None),
    ("② realfit(现行)", os.path.join(MODELS_ROOT, "real_ft_v1_seed42", "best_model.pth"), "v1"),
    # T1 第一版（干旱1000+第二半1000）—— 两组都落进过度压制区，保留作对照
    ("③ naive(1000+1000)",
     os.path.join(MODELS_ROOT, "real_ft_T1_naive_seed42", "best_model.pth"), "T1_naive"),
    ("④ ours(1000+1000)",
     os.path.join(MODELS_ROOT, "real_ft_T1_ours_seed42", "best_model.pth"), "T1_ours"),
    # T1b 修正版（干旱600+第二半1400，对齐 MIN8 配比）
    ("⑤ naive(600+1400)",
     os.path.join(MODELS_ROOT, "real_ft_T1b_naive_seed42", "best_model.pth"), "T1b_naive"),
    ("⑥ ours(600+1400)",
     os.path.join(MODELS_ROOT, "real_ft_T1b_ours_seed42", "best_model.pth"), "T1b_ours"),
    # 第十九组 MIN8 作健康区间的参照（配比≈干旱571+工业1429）
    ("★ MIN8(第十九组参照)",
     os.path.join(MODELS_ROOT, "real_ft_MIN8_s42_seed42", "best_model.pth"), "MIN8_s42"),
]

# 留出测试矿（两组训练都没用过）：阿拉善 / 黑岱沟 / 霍林河
SITES = [("阿拉善(干旱)", "alxa_xinjing.png", (380, 300, 2160, 2160)),
         ("黑岱沟", "mine_heidaigou.png", None),
         ("霍林河", "mine_huolinhe2.png", None)]
# P1: 抚顺矿坑 ROI（第十二/十三组口径）
FUSHUN = "fushun_west_mine.png"
FUSHUN_PIT = (150, 550, 1200, 1050)
BAIGE_LS = (420, 300, 800, 440)
BAIGE_BG = [(1030, 780, 1270, 1240), (420, 620, 680, 860)]


def risk(prob, box=None):
    m = (prob >= 0.5)
    if box is None:
        return float(m.mean()), None
    x0, y0, x1, y1 = box
    inner = np.zeros_like(m); inner[y0:y1, x0:x1] = True
    return float(m[inner].mean()), float(m[~inner].mean())


def safe_detect(det, img, chunk=1024):
    """大图分块推理 —— detect_image 会把**全部滑窗一次性批量**送入网络
    （2304²/32px = 5184 窗口），在本机 8GB 显存 + 检测服务常驻的情况下会 OOM
    （实测 torch.OutOfMemoryError）。切成 ≤1024² 的子图分别推理再拼回。
    仅影响块边界处的极少数像素，对 ROI 均值统计无实质影响。"""
    H, W = img.shape[:2]
    if H <= chunk and W <= chunk:
        return det.detect_image(img, patch_norm="minmax")
    out = np.zeros((H, W), np.float32)
    for y in range(0, H, chunk):
        for x in range(0, W, chunk):
            sub = img[y:y + chunk, x:x + chunk]
            p = det.detect_image(sub, patch_norm="minmax")
            out[y:y + p.shape[0], x:x + p.shape[1]] = p
    return out


def main():
    res = {"when": datetime.now().isoformat(),
           "design": "两组负样本总量相同(2000)、第一半相同(准东干旱1000); "
                     "仅第二半不同 —— 工业设施(④) vs 采坑(③)",
           "probes": {"P1": "抚顺矿坑 ROI 响应 —— 过度压制探针",
                      "P2": "白格真实滑坡分离度 —— 全局检出能力",
                      "P3": "LMHLD 源域 F1 —— 防遗忘"},
           "models": {}}
    print(f"{'模型':<26}{'阿拉善':<10}{'黑岱沟':<10}{'霍林河':<10}"
          f"{'抚顺矿坑':<11}{'白格分离':<10}{'矿坑均值'}")
    for name, w, tag in MODELS:
        if not os.path.exists(w):
            print(f"  缺权重: {name}")
            continue
        det = LandslideDetector(w, patch_size=PS)
        row = {"fp": {}}
        # 误报（留出矿）
        for sname, fn, box in SITES:
            img = _read_4ch(os.path.join(REAL_IMG, fn))
            prob = safe_detect(det, img)
            a, b = risk(prob, box)
            row["fp"][sname] = round(b if b is not None else a, 4)
        # P1: 抚顺矿坑 ROI
        img = _read_4ch(os.path.join(REAL_IMG, FUSHUN))
        prob = safe_detect(det, img)
        pit = prob[FUSHUN_PIT[1]:FUSHUN_PIT[3], FUSHUN_PIT[0]:FUSHUN_PIT[2]]
        row["P1_fushun_pit_mean"] = round(float(pit.mean()), 4)
        row["P1_fushun_pit_risk"] = round(float((pit >= 0.5).mean()), 4)
        del det
        if DEV == "cuda":
            torch.cuda.empty_cache()
        # P2: 白格分离度
        prob = baige_probs(w, PS)
        ls = prob[BAIGE_LS[1]:BAIGE_LS[3], BAIGE_LS[0]:BAIGE_LS[2]]
        bg = np.concatenate([prob[b[1]:b[3], b[0]:b[2]].ravel() for b in BAIGE_BG])
        row["P2_baige_sep"] = round(float(ls.mean() - bg.mean()), 4)
        row["P2_baige_sep_rel"] = round(
            float((ls.mean() - bg.mean()) / ((ls.mean() + bg.mean()) / 2 + 1e-9)), 4)
        # P3: 源域（从该模型的训练记录 json 读）
        row["P3_lmhld_F1"] = None
        row["P3_lmhld_F1_nir_r"] = None
        if tag:
            for base in (r"D:\LMHLD\experiments\nir_train", r"D:\LMHLD\experiments\domain_gap"):
                cp = os.path.join(base, f"finetune_{tag}.json")
                if os.path.exists(cp):
                    d = json.load(open(cp, encoding="utf-8"))
                    ab = d.get("after_best") or d.get("after") or {}
                    row["P3_lmhld_F1"] = ab.get("lmhld_F1_orig")
                    row["P3_lmhld_F1_nir_r"] = ab.get("lmhld_F1_nir_r")
                    break
        res["models"][name] = row
        fp = row["fp"]
        print(f"{name:<26}{fp['阿拉善(干旱)']*100:<10.2f}"
              f"{fp['黑岱沟']*100:<10.2f}{fp['霍林河']*100:<10.2f}"
              f"{row['P1_fushun_pit_risk']*100:<11.2f}"
              f"{row['P2_baige_sep']:<10.4f}{row['P1_fushun_pit_mean']:.4f}")

    # ── 核心判读：每一对 naive vs ours ──
    print("\n" + "=" * 78 + "\n核心判读：naive(采坑也作负样本) vs ours(只作安全层)\n" + "=" * 78)
    pairs = [("③ naive(1000+1000)", "④ ours(1000+1000)", "T1 第一版(干旱1000)"),
             ("⑤ naive(600+1400)", "⑥ ours(600+1400)", "T1b 修正版(干旱600)")]
    vd = {}
    for kn, ko, label in pairs:
        if kn not in res["models"] or ko not in res["models"]:
            continue
        n, o = res["models"][kn], res["models"][ko]
        print(f"\n  【{label}】")
        for k in n["fp"]:
            print(f"    误报 {k:<14} naive {n['fp'][k]*100:5.2f}%  vs  "
                  f"ours {o['fp'][k]*100:5.2f}%   ({(n['fp'][k]-o['fp'][k])*100:+.2f}pp)")
        print(f"    P1 抚顺矿坑均值  naive {n['P1_fushun_pit_mean']:.4f}  vs  "
              f"ours {o['P1_fushun_pit_mean']:.4f}")
        print(f"    P2 白格分离度    naive {n['P2_baige_sep']:+.4f}  vs  "
              f"ours {o['P2_baige_sep']:+.4f}")
        print(f"    P3 LMHLD F1      naive {n.get('P3_lmhld_F1')}  vs  "
              f"ours {o.get('P3_lmhld_F1')}")
        vd[label] = {
            "pit_probe_ratio": round(n["P1_fushun_pit_mean"] /
                                     max(o["P1_fushun_pit_mean"], 1e-9), 3),
            "fp_delta_pp": {k: round((n["fp"][k] - o["fp"][k]) * 100, 3) for k in n["fp"]},
            "baige_sep": [n["P2_baige_sep"], o["P2_baige_sep"]],
            "lmhld_f1": [n.get("P3_lmhld_F1"), o.get("P3_lmhld_F1")]}
    # 健康区间检查：MIN8 参照
    ref = res["models"].get("★ MIN8(第十九组参照)")
    if ref:
        print(f"\n  【健康区间参照】★ MIN8: 误报 {ref['fp']['阿拉善(干旱)']*100:.2f}%  "
              f"P2 白格 {ref['P2_baige_sep']:+.4f}  P3 F1 {ref.get('P3_lmhld_F1')}")
        vd["reference_MIN8"] = {"fp_alxa": ref["fp"]["阿拉善(干旱)"],
                                "baige_sep": ref["P2_baige_sep"],
                                "lmhld_f1": ref.get("P3_lmhld_F1")}
    vd["note"] = ("P1 比值 <1 表示 naive 比 ours 更压低矿坑响应 → 过度压制采场形态的证据; "
                  "两组都需与 MIN8 参照比对, 若白格分离度远低于 MIN8 则已落入退化区")
    res["verdict"] = vd
    with open(os.path.join(OUT, "t1_eval.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)
    print(f"\n结果: {os.path.join(OUT, 't1_eval.json')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
