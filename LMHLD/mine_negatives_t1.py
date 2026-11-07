# -*- coding: utf-8 -*-
"""
mine_negatives_t1.py — 第二十组 T1: 负样本「选法」的单因素对照池
================================================================
问题（用户 2026-09-30 拍板）：**采场台阶不算负样本**。那么"排除可能正在变形的部位"
这条边界纪律，**到底是不是一个方法**？还是"只要往矿区加负样本"就行？

★ 严格单因素设计 —— 两组**总量相同(2000)、第一半完全相同**，只差第二半：

    组 ④ 「我们的」 : 准东干旱地貌 1000 + **准东工业设施 1000**   （安全层）
    组 ③ 「朴素」   : 准东干旱地貌 1000 + **准东采坑影像 1000**   （含采场台阶/排土场）

  两组用**同一个地区（准东）**、**同一批传感器/分辨率**的影像，唯一差别是
  "第二半取的是不可能变形的工业设施，还是可能正在变形的采场"。
  → 任何性能差异只能归因于**选法**，与数据量、地区、分辨率无关。

⚠ 已知的测量局限（必须如实标注）：
  我们没有**矿山滑坡正样本**，所以只能测「误报」这一侧。
  ③ 因为被训练成"采场/排土场不是滑坡"，其误报**可能反而更低** —— 但那恰恰是
  **它以牺牲真实检出为代价**换来的（逻辑必然，不需测量）。
  本实验因此在报告**误报的同时，必须一并给出"矿区内响应结构"**，避免把
  "把矿区学成负类" 误读为 "方法更好"。

用法: python mine_negatives_t1.py
输出: experiments/mining_neg/negatives_t1_ours.npy / negatives_t1_naive.npy
"""
import json
import os
import sys
from datetime import datetime

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, r"D:\LMHLD")
import numpy as np
import cv2

REAL = r"D:\竞赛材料\真实场景验证\影像"
OUT = r"D:\LMHLD\experiments\mining_neg"
PS = 32
# 两半各取多少（默认 1000/1000；T1b 用 600/1400 对齐 MIN8 的配比）

ARID = ["mine_zd_a.png"]                                   # 干旱戈壁（两组共用）
SAFE = ["mine_zd_b.png", "mine_zhundong.png"]              # 工业设施（安全层）
PIT = ["mine_zd_pit1.png", "mine_zd_pit2.png"]             # 采坑（含采场台阶/排土场）


def read_4ch(path):
    img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    return np.concatenate([img, img[..., 2:3]], axis=-1).astype(np.float32)


def tile_ok(p):
    """剔除纯黑/过亮/无纹理/大面积饱和的图块（两组用同一套判据，不构成差异来源）"""
    g = p[..., :3].mean(-1)
    return not (g.mean() < 25 or g.mean() > 235 or g.std() < 6 or (g > 250).mean() > 0.2)


def tiles(files):
    out = []
    for fn in files:
        p = os.path.join(REAL, fn)
        if not os.path.exists(p):
            print(f"  缺 {fn}")
            continue
        img = read_4ch(p)
        H, W = img.shape[:2]
        for y in range(0, H - PS + 1, PS):
            for x in range(0, W - PS + 1, PS):
                pt = img[y:y + PS, x:x + PS]
                if tile_ok(pt):
                    out.append(np.transpose(pt, (2, 0, 1)))
    return out


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--arid", type=int, default=1000, help="第一半(干旱戈壁)取多少")
    ap.add_argument("--second", type=int, default=1000, help="第二半取多少")
    ap.add_argument("--suffix", default="", help="输出文件后缀, 如 _b")
    args = ap.parse_args()
    N_ARID, N_SECOND = args.arid, args.second
    os.makedirs(OUT, exist_ok=True)
    rng = np.random.default_rng(42)
    print(f"共用地基（干旱戈壁）: {ARID}")
    base = tiles(ARID)
    print(f"  → 合格 {len(base)} 块，取 {N_ARID}")
    sel_base = rng.choice(len(base), min(N_ARID, len(base)), replace=False)
    half_base = [base[i] for i in sel_base]

    print(f"\n④ 第二半 = 工业设施（安全层）: {SAFE}")
    safe = tiles(SAFE)
    print(f"  → 合格 {len(safe)} 块，取 {N_SECOND}")
    sel_safe = rng.choice(len(safe), min(N_SECOND, len(safe)), replace=False)
    half_safe = [safe[i] for i in sel_safe]

    print(f"\n③ 第二半 = 采坑（含采场台阶/排土场）: {PIT}")
    pit = tiles(PIT)
    print(f"  → 合格 {len(pit)} 块，取 {N_SECOND}")
    sel_pit = rng.choice(len(pit), min(N_SECOND, len(pit)), replace=False)
    half_pit = [pit[i] for i in sel_pit]

    ours = np.stack(half_base + half_safe).astype(np.float32)
    naive = np.stack(half_base + half_pit).astype(np.float32)
    np.save(os.path.join(OUT, f"negatives_t1{args.suffix}_ours.npy"), ours)
    np.save(os.path.join(OUT, f"negatives_t1{args.suffix}_naive.npy"), naive)

    meta = {"when": datetime.now().isoformat(),
            "design": f"总量相同({N_ARID}+{N_SECOND}); 第一半均为准东干旱戈壁; "
                      "仅第二半不同 → 单因素比较「负样本选法」",
            "ours": {"file": f"negatives_t1{args.suffix}_ours.npy", "n": int(len(ours)),
                     "half2": "工业设施(安全层)", "half2_sources": SAFE,
                     "half2_avail": len(safe)},
            "naive": {"file": f"negatives_t1{args.suffix}_naive.npy", "n": int(len(naive)),
                      "half2": "采坑(含采场台阶/排土场)", "half2_sources": PIT,
                      "half2_avail": len(pit)},
            "half1": {"desc": "准东干旱戈壁", "sources": ARID, "avail": len(base)},
            "caveat": "无矿山滑坡正样本, 只能测误报侧; ③ 的低误报可能来自"
                      "「把矿区学成负类」, 须同时看矿区内响应结构",
            "channel_order": "B,G,R,NIR(=R)"}
    with open(os.path.join(OUT, "negatives_t1_meta.json"), "w", encoding="utf-8") as f:
        meta["arid_n"] = N_ARID; meta["second_n"] = N_SECOND
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"\n④ 我们的 : {ours.shape}  -> negatives_t1_ours.npy")
    print(f"③ 朴素   : {naive.shape}")
    print(f"元数据   : {os.path.join(OUT, 'negatives_t1_meta.json')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
