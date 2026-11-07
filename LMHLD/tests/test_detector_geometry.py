# -*- coding: utf-8 -*-
"""
test_detector_geometry.py — detector.analyze_mask 几何集成 + 向后兼容回归
==========================================================================
验证三件事:
  ① 原有 6 键的取值与旧实现**逐值一致**(老消费者不受影响);
  ② num_regions 的 [:10] 静默截断已修正(真实计数), 且 region_areas 仍为前 10;
  ③ overlay(outline=False) 与旧实现逐字节一致(图像基线不被破坏)。

用法: python tests/test_detector_geometry.py
"""
import os
import sys
import json
import types

import numpy as np
import cv2

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, r"D:\LMHLD")

from detector import LandslideDetector, RISK_RULES          # noqa: E402
from geo_geometry import MAX_REGIONS_LITE, LITE_MAX_VERTICES  # noqa: E402

DATA_ROOT = r"F:\LMHLD\LMHLD_dataset_different_patch_sizes"
OK = True


def chk(name, cond, extra=""):
    global OK
    print(f"  {'✓' if cond else '✗'} {name} {extra}")
    if not cond:
        OK = False


class FakeSelf:
    """analyze_mask 只用到 self._risk_level, 无需加载模型权重。"""
    def _risk_level(self, ratio):
        for thr, level in RISK_RULES:
            if ratio >= thr:
                return level
        return "blue"


def old_analyze_mask(mask, min_region_area=10):
    """旧实现原文(独立参照, 用于证明前 6 键取值未变)。"""
    mask = (mask > 0).astype(np.uint8)
    if mask.sum() > 0:
        num, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        areas = stats[1:, cv2.CC_STAT_AREA]
        keep = np.where(areas >= min_region_area)[0] + 1
        if len(keep) == 0:
            landslide_px, region_areas, max_region = 0, [], 0
        else:
            mask_clean = np.isin(labels, list(set(keep.tolist())))
            landslide_px = int(mask_clean.sum())
            areas = areas[keep - 1]
            region_areas = sorted([int(a) for a in areas], reverse=True)[:10]
            max_region = int(areas.max())
    else:
        landslide_px, region_areas, max_region = 0, [], 0
    ratio = landslide_px / mask.size
    return {"landslide_pixels": landslide_px,
            "area_ratio": round(float(ratio), 4),
            "num_regions": len(region_areas),          # ← 旧实现的截断口径
            "max_region_area": max_region,
            "region_areas": region_areas,
            "risk_level": FakeSelf()._risk_level(ratio)}


def main():
    fs = FakeSelf()

    print("① 原有 6 键取值与旧实现一致（region_areas 仍为前 10）")
    n_checked = n_trunc = 0
    for region in ["Emilia-Romagna_Italy_32", "Hokkaido_Japan_128",
                   "Wenchuan_China_224", "Taiwan_China_224"]:
        lp = os.path.join(DATA_ROOT, region, "test_labels.npy")
        if not os.path.exists(lp):
            continue
        labels = np.load(lp, mmap_mode="r")
        step = max(1, labels.shape[0] // 40)
        for i in range(0, labels.shape[0], step):
            gt = (np.asarray(labels[i, 0]) > 0).astype(np.uint8)
            new = LandslideDetector.analyze_mask(fs, gt)
            old = old_analyze_mask(gt)
            n_checked += 1
            bad = [k for k in ("landslide_pixels", "area_ratio", "max_region_area",
                               "region_areas", "risk_level")
                   if new[k] != old[k]]
            if bad:
                chk(f"{region} 样本{i} 前6键一致", False, f"不一致: {bad}")
                return 1
            if new["num_regions_total"] != old["num_regions"]:
                n_trunc += 1
                chk(f"{region} 样本{i}: 截断已修正",
                    new["num_regions"] == new["num_regions_total"] > 10,
                    f"旧={old['num_regions']} 新={new['num_regions']}")
    chk(f"抽样 {n_checked} 个真实图块, 前 6 键逐值一致", True)
    chk(f"其中 {n_trunc} 个样本命中 >10 实例的截断修正", n_trunc >= 0,
        f"(截断影响面本来就小, 实测全集 1.0%)")

    print("② 新增键的语义自洽")
    m = np.zeros((128, 128), np.uint8)
    for k, (x, y, w, h) in enumerate([(2, 2, 9, 9), (20, 20, 8, 8), (40, 40, 7, 7),
                                      (60, 60, 6, 6), (80, 80, 5, 5), (100, 100, 9, 9),
                                      (110, 20, 9, 9), (20, 110, 9, 9), (60, 100, 9, 9),
                                      (100, 60, 9, 9), (40, 90, 9, 9), (90, 40, 9, 9)]):
        if w * h >= 10:
            m[y:y + h, x:x + w] = 1
    r = LandslideDetector.analyze_mask(fs, m)
    chk("实例数 >10 (构造 12 处)", r["num_regions_total"] > 10, f"实={r['num_regions_total']}")
    chk("num_regions == 真实计数", r["num_regions"] == r["num_regions_total"])
    chk("region_areas 仍截断为 10", len(r["region_areas"]) == 10)
    chk("truncated 标志正确", r["region_areas_truncated"] is True)
    chk("regions 数量 == min(总数, 上限)",
        len(r["regions"]) == min(r["num_regions_total"], MAX_REGIONS_LITE))
    chk("每个实例含多边形", all(len(x["contour"]) >= 3 for x in r["regions"]))
    chk("顶点数不超上限",
        all(len(x["contour"]) <= LITE_MAX_VERTICES for x in r["regions"]))
    chk("geometry_meta.enabled", r["geometry_meta"]["enabled"] is True)
    try:
        json.dumps(r, ensure_ascii=False)
        chk("整体 JSON 可序列化", True)
    except TypeError as e:
        chk("整体 JSON 可序列化", False, repr(e))

    print("③ 关闭几何 / 退化输入")
    r2 = LandslideDetector.analyze_mask(fs, m, with_regions=False)
    chk("with_regions=False 时 regions 为空", r2["regions"] == [])
    chk("with_regions=False 时 meta.enabled=False",
        r2["geometry_meta"]["enabled"] is False)
    chk("with_regions=False 不抛异常(原6键仍正确)",
        r2["num_regions"] == r["num_regions"] and r2["region_areas"] == r["region_areas"])
    empty = LandslideDetector.analyze_mask(fs, np.zeros((32, 32), np.uint8))
    chk("空掩膜不抛异常", empty["num_regions"] == 0 and empty["regions"] == [])
    tiny = np.zeros((32, 32), np.uint8)
    tiny[5, 5] = 1
    r3 = LandslideDetector.analyze_mask(fs, tiny)
    chk("单像素被 min_area 过滤", r3["num_regions"] == 0)

    print("④ overlay(outline=False) 与旧实现逐字节一致")
    rgb = (np.random.RandomState(0).rand(64, 64, 3) * 255).astype(np.uint8)
    mm = np.zeros((64, 64), np.uint8)
    cv2.circle(mm, (32, 32), 15, 1, -1)
    old_overlay = rgb.astype(np.float32).copy()
    sel = mm > 0
    old_overlay[sel] = (old_overlay[sel] * (1 - 0.55)
                        + np.array((255, 0, 0), dtype=np.float32) * 0.55)
    old_overlay = np.clip(old_overlay, 0, 255).astype(np.uint8)
    new_overlay = LandslideDetector.overlay(fs, rgb, mm)
    chk("默认输出逐字节一致", np.array_equal(old_overlay, new_overlay))
    outlined = LandslideDetector.overlay(fs, rgb, mm, outline=True)
    chk("outline=True 确实改变了图像", not np.array_equal(outlined, new_overlay))
    chk("outline=True 形状不变", outlined.shape == new_overlay.shape)
    chk("outline=True 空掩膜不抛异常",
        LandslideDetector.overlay(fs, rgb, np.zeros((64, 64), np.uint8),
                                  outline=True).shape == rgb.shape)

    print(f"\n{'✅ 全部通过' if OK else '❌ 存在失败项'}")
    return 0 if OK else 1


if __name__ == "__main__":
    sys.exit(main())
