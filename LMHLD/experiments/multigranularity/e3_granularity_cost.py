# -*- coding: utf-8 -*-
"""
e3_granularity_cost.py — E3: 三种粒度表达的信息损失 (RemoteReasoner Tab.6 本地化)
================================================================================
问题: 同一次分割结果, 用三种粒度表达后再还原成掩膜, 各损失多少?

  · Mask       : 原始掩膜 (像素级, 上界)
  · Bbox2Mask  : 每个实例降级成外接框 → 回栅格  (区域级)
  · Polygon2Mask: 每个实例降级成 eps=0.5 简化多边形 → 回栅格  (对象级)

⚠ 口径声明: 这是 **GT→GT 的任务变换代价上界**, 衡量"粒度本身能表达多少信息",
  不是模型精度, 也不涉及任何模型。材料引用时必须标注。

产出: experiments/multigranularity/e3_granularity_cost.json + 控制台表
用法: python e3_granularity_cost.py [--max-per-region N]
"""
import os
import sys
import json
import argparse

import numpy as np
import cv2

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, r"D:\LMHLD")
from geo_geometry import mask_to_regions, polygon_to_mask, DEFAULT_SIMPLIFY_EPS  # noqa: E402

DATA_ROOT = r"F:\LMHLD\LMHLD_dataset_different_patch_sizes"
OUT = r"D:\LMHLD\experiments\multigranularity\e3_granularity_cost.json"

REGIONS = ["Emilia-Romagna_Italy_32", "Gorkha_Nepal_32", "Hokkaido_Japan_128",
           "Jiuzhaigou_China_128", "Wenchuan_China_224", "Taiwan_China_224"]


def bbox_to_mask(regions, shape):
    """实例外接框 → 回栅格 (区域级表达的信息量)。"""
    out = np.zeros(shape, np.uint8)
    for r in regions:
        x, y, w, h = r["bbox"]
        out[y:y + h, x:x + w] = 1
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-per-region", type=int, default=0,
                    help="每区域最多评估的 patch 数 (0=全部)")
    args = ap.parse_args()

    all_res = {}
    for region in REGIONS:
        lp = os.path.join(DATA_ROOT, region, "test_labels.npy")
        if not os.path.exists(lp):
            print(f"  ✗ {region}: 缺少 {lp}")
            continue
        labels = np.load(lp, mmap_mode="r")
        n_total = labels.shape[0]
        idxs = range(n_total) if not args.max_per_region else \
            range(min(n_total, args.max_per_region))

        acc = {"n_patch": 0, "n_patch_pos": 0, "n_inst": 0,
               "inter_poly": 0, "union_poly": 0, "inter_bbox": 0, "union_bbox": 0,
               "inter_mask": 0, "union_mask": 0,
               "iou_poly_per_img": [], "iou_bbox_per_img": [],
               "d_eq": [], "n_vertices": [], "eps_used": {},
               "n_inst_per_patch": []}

        for i in idxs:
            gt = (np.asarray(labels[i, 0]) > 0).astype(np.uint8)
            acc["n_patch"] += 1
            if gt.sum() == 0:
                continue
            acc["n_patch_pos"] += 1
            res = mask_to_regions(gt, min_area=10, max_regions=10_000,
                                  simplify_eps=DEFAULT_SIMPLIFY_EPS, max_vertices=10_000)
            regs = res["regions"]
            if not regs:
                continue
            acc["n_inst"] += len(regs)
            acc["n_inst_per_patch"].append(len(regs))
            for r in regs:
                acc["d_eq"].append(2.0 * np.sqrt(r["area_px"] / np.pi))
                acc["n_vertices"].append(len(r["contour"]))
                e = str(r.get("simplify_eps_used", DEFAULT_SIMPLIFY_EPS))
                acc["eps_used"][e] = acc["eps_used"].get(e, 0) + 1

            poly = polygon_to_mask(regs, gt.shape)
            bbox = bbox_to_mask(regs, gt.shape)
            for key, pred in (("poly", poly), ("bbox", bbox), ("mask", gt)):
                acc[f"inter_{key}"] += int((pred & gt).sum())
                acc[f"union_{key}"] += int((pred | gt).sum())
            # 逐图 IoU (gIoU 口径: 对 GT 非空图块求平均)
            ip = (poly & gt).sum() / max((poly | gt).sum(), 1)
            ib = (bbox & gt).sum() / max((bbox | gt).sum(), 1)
            acc["iou_poly_per_img"].append(float(ip))
            acc["iou_bbox_per_img"].append(float(ib))

        if acc["n_patch_pos"] == 0:
            continue
        d = np.array(acc["d_eq"])
        row = {
            "n_patch": acc["n_patch"], "n_patch_pos": acc["n_patch_pos"],
            "n_instances": acc["n_inst"],
            "instances_per_patch_mean": round(float(np.mean(acc["n_inst_per_patch"])), 2),
            "d_eq_median_px": round(float(np.median(d)), 2),
            "d_eq_p90_px": round(float(np.percentile(d, 90)), 2),
            "frac_d_eq_ge20px": round(float((d >= 20).mean()), 3),
            # E3 主结果: 粒度代价 (累积口径 cIoU 与逐图平均口径 gIoU)
            "cIoU_mask": round(acc["inter_mask"] / max(acc["union_mask"], 1), 4),
            "cIoU_polygon": round(acc["inter_poly"] / max(acc["union_poly"], 1), 4),
            "cIoU_bbox": round(acc["inter_bbox"] / max(acc["union_bbox"], 1), 4),
            "gIoU_polygon": round(float(np.mean(acc["iou_poly_per_img"])), 4),
            "gIoU_bbox": round(float(np.mean(acc["iou_bbox_per_img"])), 4),
            "vertices_median": int(np.median(acc["n_vertices"])),
            "eps_used_hist": acc["eps_used"],
            # 累积口径的原始计数 (汇总合并多区域时必需)
            "_sums": {k: acc[k] for k in
                      ("inter_mask", "union_mask", "inter_poly",
                       "union_poly", "inter_bbox", "union_bbox")},
        }
        row["polygon_recovery_vs_mask"] = round(
            row["cIoU_polygon"] / max(row["cIoU_mask"], 1e-9), 4)
        row["bbox_recovery_vs_mask"] = round(
            row["cIoU_bbox"] / max(row["cIoU_mask"], 1e-9), 4)
        all_res[region] = row

        print(f"  {region:<26} 实例={row['n_instances']:>5} "
              f"d_eq中位={row['d_eq_median_px']:>6}px  "
              f"cIoU: mask={row['cIoU_mask']:.4f} poly={row['cIoU_polygon']:.4f} "
              f"bbox={row['cIoU_bbox']:.4f}  顶点中位={row['vertices_median']}")

    # ── 汇总 ──
    if all_res:
        tot_i = {k: sum(r["_sums"][k] for r in all_res.values())
                 for k in ("inter_mask", "union_mask", "inter_poly",
                           "union_poly", "inter_bbox", "union_bbox")}
        summary = {
            "when": __import__("datetime").datetime.now().isoformat(),
            "protocol": "GT→GT 任务变换代价上界 (非模型精度)",
            "simplify_eps": DEFAULT_SIMPLIFY_EPS,
            "overall": {
                "cIoU_mask": round(tot_i["inter_mask"] / max(tot_i["union_mask"], 1), 4),
                "cIoU_polygon": round(tot_i["inter_poly"] / max(tot_i["union_poly"], 1), 4),
                "cIoU_bbox": round(tot_i["inter_bbox"] / max(tot_i["union_bbox"], 1), 4),
            },
            "per_region": all_res,
        }
        o = summary["overall"]
        print(f"\n══ 全体 (6 区域合并) ══")
        print(f"  Mask   cIoU = {o['cIoU_mask']:.4f}  (上界)")
        print(f"  Polygon cIoU = {o['cIoU_polygon']:.4f} "
              f"→ 还原 {(o['cIoU_polygon'] / o['cIoU_mask']) * 100:.1f}% 的掩膜信息")
        print(f"  Bbox    cIoU = {o['cIoU_bbox']:.4f} "
              f"→ 还原 {(o['cIoU_bbox'] / o['cIoU_mask']) * 100:.1f}% 的掩膜信息")
        os.makedirs(os.path.dirname(OUT), exist_ok=True)
        with open(OUT, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        print(f"\n结果: {OUT}")


if __name__ == "__main__":
    main()
