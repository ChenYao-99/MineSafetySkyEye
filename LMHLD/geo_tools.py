# -*- coding: utf-8 -*-
"""
geo_tools.py — 矿安天眼 · 定量地理分析工具集
==============================================
对标 Earth-Agent (ICLR 2026) 的 Index / Analysis / Statistics 工具包，
为多智能体提供**定量分析能力**（使预警报告从定性描述升级为定量引用）。

四个工具:
  1. compute_spectral_index  — NDVI/NDWI 光谱指数统计（植被/水体定量）
  2. compute_region_stats    — 风险区面积分级统计（封装连通域分析）
  3. analyze_temporal_trend  — 数值序列趋势斜率 + CUSUM 变点检测
  4. compare_periods         — 两期统计对比量化（变化量/变化率/等级迁移）

设计: 纯函数、零外部依赖(除 numpy/cv2)、既可被 pipeline 直接调用注入提示词,
也可注册为智能体工具。
"""
from __future__ import annotations

import numpy as np
import cv2

# 指数定义: (A通道idx, B通道idx, 名称)  — 通道序 [B,G,R,NIR] = [0,1,2,3]
INDEX_DEFS = {
    "ndvi": (3, 2, "归一化植被指数 (NIR-R)/(NIR+R)), 越高植被越茂密"),
    "ndwi": (1, 3, "归一化水体指数 (G-NIR)/(G+NIR)), 越高水体越显著"),
    "bsi": (2, 3, "裸土指数近似 (R-NIR)/(R+NIR)), 越高裸土/岩土越显著"),
}
RISK_NAMES_CN = {"blue": "蓝色-未检出明显异常", "yellow": "黄色-少量疑似异常",
                 "orange": "橙色-较多疑似异常", "red": "红色-大量疑似异常",
                 "unknown":"待复核-证据冲突"}


def _load_hwc4(image):
    """接受: (H,W,4) 数组 / npy 路径 / 普通图像路径(3通道则复制R为NIR) → (H,W,4) float"""
    if isinstance(image, np.ndarray):
        arr = image
    elif isinstance(image, str):
        if image.lower().endswith(".npy"):
            arr = np.load(image)
            if arr.ndim == 3 and arr.shape[0] in (1, 3, 4) and arr.shape[0] < arr.shape[-1]:
                arr = np.transpose(arr, (1, 2, 0))     # CHW → HWC
        else:
            raw = np.fromfile(image, dtype=np.uint8) if not image.startswith("http") else None
            arr = cv2.imdecode(raw, cv2.IMREAD_UNCHANGED) if raw is not None else None
            if arr is None:
                raise ValueError(f"无法读取影像: {image}")
    else:
        raise TypeError("image 需为数组或路径")
    arr = arr.astype(np.float32)
    if arr.ndim == 2:
        arr = np.stack([arr] * 4, axis=-1)
    if arr.shape[2] == 3:
        arr = np.concatenate([arr, arr[..., 2:3]], axis=-1)   # NIR ≈ R
    return arr[..., :4]


def compute_spectral_index(image, index="ndvi", nir_source=None) -> dict:
    """
    计算光谱指数统计 (NDVI/NDWI/BSI)
    image: (H,W,4)[B,G,R,NIR] 数组 或 npy/png 路径
    返回: {index, mean, std, min, max, 高值占比, 低值占比, 描述}
    """
    if str(index).lower()=='bsi':
        return {'index':'BSI','available':False,'reason':'标准BSI需要真实SWIR；BGRN四波段不能用负NDVI代替BSI'}
    if str(index).lower() not in INDEX_DEFS:
        raise ValueError('不支持的光谱指数')
    if isinstance(image,str) and image.lower().endswith(('.tif','.tiff')):
        from pathlib import Path
        import base64
        from upload_utils import decode_upload
        from raster_upload import calibrated_spectral_image,observed_mask
        x,meta=decode_upload(base64.b64encode(Path(image).read_bytes()).decode(),'AUTO')
        if not meta.get('spectral_indices_available'):
            return {'index':index.upper(),'available':False,'reason':'原始栅格缺少真实NIR或可核对的辐射定标'}
        valid=observed_mask(meta,x.shape[:2])
        return compute_spectral_index(calibrated_spectral_image(x,meta)[valid][:,None,:],index,nir_source=meta['nir_source'])
    if nir_source is None:
        # Four numeric channels alone do not prove NIR: the fourth could be alpha.
        nir_source = 'unknown'
    if nir_source != "measured":
        return {"index": index.upper(), "available": False, "nir_source": nir_source,
                "reason": "未提供真实近红外，不能计算有物理意义的NDVI/NDWI"}
    img = _load_hwc4(image)
    ia, ib, desc = INDEX_DEFS[index.lower()]
    a = img[..., ia]; b = img[..., ib]
    valid=np.isfinite(a)&np.isfinite(b)&(a>=0)&(b>=0)&(a+b>1e-8)
    if not valid.any():
        return {'index':index.upper(),'available':False,'nir_source':'measured',
                'reason':'所选光谱波段没有可用于指数的非负有限反射率'}
    idx_map = (a[valid]-b[valid])/(a[valid]+b[valid])
    hi = float((idx_map > 0.3).mean())
    lo = float((idx_map < 0.1).mean())
    return {
        "index": index.upper(),
        "available": True,
        "nir_source": "measured",
        'index_valid_pixels':int(valid.sum()),'index_excluded_pixels':int((~valid).sum()),
        "mean": round(float(idx_map.mean()), 4),
        "std": round(float(idx_map.std()), 4),
        "min": round(float(idx_map.min()), 4),
        "max": round(float(idx_map.max()), 4),
        "high_ratio": round(hi, 4),      # NDVI: 茂密植被占比 / NDWI: 水体占比
        "low_ratio": round(lo, 4),       # 低值占比 (NDVI: 裸土/岩土)
        "description": desc,
    }


def compute_region_stats(mask) -> dict:
    """
    风险区连通域分级统计 (mask: (H,W) 0/1 或概率图+阈值)
    返回: {num_regions, area_ratio, max_region_area, region_areas(前5), risk_level}
    """
    if mask.dtype != np.uint8:
        mask = (mask > 0.5).astype(np.uint8)
    mask = (mask > 0).astype(np.uint8)
    total = mask.size
    px = int(mask.sum())
    num, areas, max_area = 0, [], 0
    if px > 0:
        n, lab, st, _ = cv2.connectedComponentsWithStats(mask, 8)
        areas_all = [int(a) for a in st[1:, cv2.CC_STAT_AREA] if a >= 10]   # 去噪
        areas = sorted(areas_all, reverse=True)[:5]
        num = len(areas_all)
        max_area = max(areas_all) if areas_all else 0
        px = sum(areas_all) if areas_all else 0
    ratio = px / total
    level = ("red" if ratio >= 0.15 else "orange" if ratio >= 0.05
             else "yellow" if ratio >= 0.01 else "blue")
    return {"num_regions": num, "area_ratio": round(float(ratio), 4),
            "max_region_area": int(max_area), "region_areas": areas,
            "risk_level": level, "risk_name": RISK_NAMES_CN[level]}


def analyze_temporal_trend(values, timestamps=None) -> dict:
    """
    数值序列的趋势与变点分析
    values: 数值列表(如多期风险面积占比)
    返回: {slope, slope_per_step, direction, change_point_index, change_magnitude, volatility}
    """
    v = np.asarray(values, dtype=np.float64)
    n = len(v)
    if n < 3:
        return {"error": "序列过短(至少 3 期)"}
    x = np.arange(n)
    slope = float(np.polyfit(x, v, 1)[0])
    # CUSUM 变点检测: 找到均值偏差累积最大的位置
    mean_all = v.mean()
    cusum = np.cumsum(v - mean_all)
    cp = int(np.argmax(np.abs(cusum)))
    before = v[:cp + 1].mean() if cp > 0 else v[0]
    after = v[cp + 1:].mean() if cp + 1 < n else v[-1]
    return {
        "n_periods": n,
        "slope": round(slope, 5),
        "direction": "上升" if slope > 1e-6 else ("下降" if slope < -1e-6 else "平稳"),
        "change_point_index": cp,
        "change_point_magnitude": round(float(after - before), 4),
        "volatility": round(float(v.std()), 4),
        "values": [round(float(t), 4) for t in v],
    }


def compare_periods(stats_t1: dict, stats_t2: dict) -> dict:
    """
    两期风险统计对比量化
    stats: compute_region_stats 的输出(或含 area_ratio/num_regions/risk_level 的字典)
    返回: {delta_ratio, change_rate, direction, level_transition, added_regions}
    """
    r1 = float(stats_t1.get("area_ratio", 0)); r2 = float(stats_t2.get("area_ratio", 0))
    delta = r2 - r1
    rate = delta / r1 if r1 > 0 else None
    return {
        "ratio_t1": round(r1, 4), "ratio_t2": round(r2, 4),
        "delta_ratio": round(delta, 4),
        "change_rate": round(float(rate), 4) if rate is not None else None,
        "direction": "掩膜占比增加" if delta > 0 else ("掩膜占比减少" if delta < 0 else "掩膜占比持平"),
        "level_t1": stats_t1.get("risk_name", "?"),
        "level_t2": stats_t2.get("risk_name", "?"),
        "added_regions": int(stats_t2.get("num_regions", 0)) - int(stats_t1.get("num_regions", 0)),
    }


def format_quant_block(index_stats=None, region_stats=None, trend=None,
                       compare=None, geometry=None) -> str:
    """把定量分析结果格式化为可注入提示词的文本块

    geometry: 多粒度几何块 (dict: {regions, num_regions_total, meters_per_pixel?})
              文本渲染委托给 geo_geometry.format_geometry_block —— 单一实现,
              避免同一份措辞在两处维护而漂移。
              注意: 该块有硬长度约束(≤5 行/200 字), 因为提示词此前已多次触及
              CAMEL 的摘要阈值, 注入过长文本会诱发记忆压缩。
    """
    lines = ["【定量分析工具输出】(请在你的分析中引用这些数值)"]
    if index_stats and index_stats.get("available", True):
        s = index_stats
        lines.append(
            f"· {s['index']} 指数: 均值 {s['mean']} (高值占比 {s['high_ratio']*100:.1f}%, "
            f"低值占比 {s['low_ratio']*100:.1f}%) — {s['description']}")
    if region_stats:
        s = region_stats
        extra = ""
        if s.get("num_regions_total") and s["num_regions_total"] != s.get("num_regions"):
            extra = f"(实例总数 {s['num_regions_total']})"
        lines.append(
            f"· 风险区统计: {s['num_regions']} 处连通区域{extra}, 面积占比 {s['area_ratio']*100:.1f}%, "
            f"最大区域 {s['max_region_area']} px, 判定 {s['risk_name']}")
    if geometry:
        try:
            from geo_geometry import format_geometry_block
            blk = format_geometry_block(
                geometry.get("regions") or [],
                num_regions_total=geometry.get("num_regions_total"),
                meters_per_pixel=geometry.get("meters_per_pixel"),
                image_shape=geometry.get("image_shape"))
            if blk:
                lines.append(blk)
        except Exception as e:
            print(f"[geo_tools] 几何块渲染失败: {e}")
    if trend and "error" not in trend:
        lines.append(
            f"· {trend['n_periods']} 期趋势: 斜率 {trend['slope']} ({trend['direction']}), "
            f"变点在第 {trend['change_point_index']+1} 期 (幅度 {trend['change_point_magnitude']})")
    if compare:
        c = compare
        lines.append(
            f"· 两期对比: 面积占比 {c['ratio_t1']*100:.2f}% → {c['ratio_t2']*100:.2f}%，"
            f"净变化 {c['delta_ratio']*100:+.2f} 个百分点（{c['direction']}）；原型筛查 {c['level_t1']} → {c['level_t2']}")
    return "\n".join(lines)


if __name__ == "__main__":
    # 自测
    import sys
    sys.stdout.reconfigure(encoding="utf-8")
    rng = np.random.default_rng(0)
    img = rng.integers(1000, 8000, (64, 64, 4)).astype(np.float32)
    print("NDVI:", compute_spectral_index(img, "ndvi"))
    m = (rng.random((64, 64)) > 0.9).astype(np.uint8)
    print("区域统计:", compute_region_stats(m))
    print("趋势:", analyze_temporal_trend([0.01, 0.012, 0.035, 0.05, 0.062]))
    print("对比:", compare_periods({"area_ratio": 0.02, "num_regions": 2, "risk_name": "黄色-中风险"},
                                   {"area_ratio": 0.06, "num_regions": 4, "risk_name": "橙色-高风险"}))
