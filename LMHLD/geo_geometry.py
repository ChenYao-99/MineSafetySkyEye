# -*- coding: utf-8 -*-
"""
geo_geometry.py — 滑坡掩膜的多粒度几何变换引擎
==================================================
设计来源: RemoteReasoner (AAAI-26) 的"决策级任务变换"思想 ——
  单次分割推理的掩膜, 经一次几何变换同时产出三种粒度结果:
    · 区域级 (region) : 外接框 bbox + 质心     —— 快报/计数口径
    · 对象级 (object) : 简化多边形轮廓 + 形态量 —— 矢量制图/方量测算口径
    · 像素级 (pixel)  : 原始掩膜                —— 面积占比/风险分级口径
  无需任何任务专属解码器, 不引入新的模型权重。

关键设计约束 (均为规划期实测/核实的结论):
  · 纯 numpy + cv2, 不 import torch —— 需同时被 lmhld(numpy 2.0.1) 与
    GeoMMA(numpy 1.26.4) 两个环境 import, 故禁用已废弃别名(np.float_ 等)。
  · 所有返回值必须是 python 原生 int/float/list —— numpy 2.0 下 jsonify 对
    np.float32 直接抛 TypeError, 会让整个 /api/detect 请求 500。
  · 全链路不得有未捕获异常: 32px 图块里的滑坡常只有几个像素,
    任何"点数不足"类异常(如 cv2.fitEllipse 需 ≥5 点)都会打挂请求,
    故形态量一律用 cv2.moments 的矩特征值计算。
  · 简化容差 eps=0.5 为实测最优 (保真 IoU ≥0.985, 224px 顶点数中位 108)。
  · 面积/周长一律由实例多边形而非外接框给出 —— 实测外接框只能还原
    46%~54% 的滑坡面积, 而 eps=0.5 多边形可还原 97.8%~99.7%。

自测: python geo_geometry.py --selftest   (两个 conda 环境都要跑一遍)
"""
import sys
import json
import math

import numpy as np
import cv2

# ── 配置常量 ─────────────────────────────────────────────
DEFAULT_MIN_AREA = 10          # 与 detector.analyze_mask 的 min_region_area 对齐
DEFAULT_SIMPLIFY_EPS = 0.5     # 实测最优
EPS_LADDER = (0.5, 1.0, 2.0)   # 顶点超限时的升级阶梯
MAX_VERTICES = 128             # 导出档单轮廓顶点上限
LITE_MAX_VERTICES = 64         # 响应内嵌档单轮廓顶点上限
MAX_REGIONS_LITE = 32          # 响应内嵌档实例数上限 (实测最大 17, 基本不截断)
MAX_REGIONS_FULL = 500         # 导出档实例数上限 (上传影像可达上百实例)

# 形态学后处理算子 (消融实验与可选服务开关共用)
MORPH_OPS = ("none", "fill_holes", "close3", "open3",
             "area10", "area25", "area50", "shape_filter",
             "close_then_area", "largest_only")

EARTH_R = 6378137.0            # WGS84 长半轴 (m)


# ══════════════════════════════════════════════════════════
# 一、掩膜 → 实例几何
# ══════════════════════════════════════════════════════════
def _shape_metrics(cnt, area_px):
    """由轮廓计算形态量。cnt: (N,1,2) int32 轮廓; area_px: 像素面积。

    长度量均以"像素"为单位; 轴长采用二阶中心矩特征值法:
      均匀椭圆半轴 a,b 的二阶中心矩为 a²/4, b²/4,
      故 4*sqrt(λ) 恰为整轴长度 (长轴 2a)。
    ⚠ 不用 cv2.fitEllipse —— 它要求 ≥5 点, 32px 小区的滑坡会不足而抛异常。
    """
    m = cv2.moments(cnt)
    out = {"perimeter_px": 0.0, "circularity": 0.0, "aspect_ratio": 1.0,
           "orientation_deg": 0.0, "major_axis_px": 0.0, "minor_axis_px": 0.0,
           "solidity": 0.0}
    per = float(cv2.arcLength(cnt, True))
    out["perimeter_px"] = round(per, 2)
    # 紧致度 4πA/P²: 理论上界为 1(圆)。数字轮廓的周长因阶梯效应略偏大
    # (r=20 圆盘实测 131.88px vs 理论 125.66px, 偏大 ~5%), 故圆盘实测约 0.91。
    # ⚠ 极小实例(如 32px 的碎块)上 CHAIN_APPROX_SIMPLE 会低估周长, 使该值
    #   超过 1 —— 实测出现过 1.1056。此处按理论界截断到 1.0, 并靠 area_px
    #   保留"该实例很小、形态量不可靠"的信息。
    if per > 1e-6:
        out["circularity"] = round(min(float(4 * math.pi * area_px / (per * per)), 1.0), 4)

    if m["m00"] > 1e-9:
        mu20 = m["mu20"] / m["m00"]
        mu02 = m["mu02"] / m["m00"]
        mu11 = m["mu11"] / m["m00"]
        common = math.sqrt(max(4 * mu11 * mu11 + (mu20 - mu02) ** 2, 0.0))
        lam1 = (mu20 + mu02) / 2 + common / 2      # 较大特征值
        lam2 = (mu20 + mu02) / 2 - common / 2      # 较小特征值
        lam1, lam2 = max(lam1, 0.0), max(lam2, 0.0)
        out["major_axis_px"] = round(4 * math.sqrt(lam1), 2)
        out["minor_axis_px"] = round(4 * math.sqrt(lam2), 2)
        if lam2 > 1e-9:
            out["aspect_ratio"] = round(math.sqrt(lam1 / lam2), 3)
        out["orientation_deg"] = round(
            math.degrees(0.5 * math.atan2(2 * mu11, mu20 - mu02)), 1)
    # 凸度 = 面积/凸包面积
    hull = cv2.convexHull(cnt)
    ha = abs(float(cv2.contourArea(hull)))
    if ha > 1e-6:
        out["solidity"] = round(min(float(area_px) / ha, 1.0), 4)
    return out


def _simplify(cnt, eps=DEFAULT_SIMPLIFY_EPS, max_vertices=LITE_MAX_VERTICES):
    """简化轮廓到顶点上限内, 返回 (顶点列表, 实际使用的 eps)。"""
    used = eps
    for e in (eps,) + tuple(x for x in EPS_LADDER if x > eps):
        approx = cv2.approxPolyDP(cnt, e, True)
        used = e
        if len(approx) <= max_vertices:
            break
    pts = [[int(p[0][0]), int(p[0][1])] for p in approx]
    return pts, round(float(used), 2)


def region_record(labels, label_id, mask_shape, simplify_eps=DEFAULT_SIMPLIFY_EPS,
                  max_vertices=LITE_MAX_VERTICES, with_contours=True):
    """生成单个连通域的实例记录; 无效实例返回 None。

    labels: connectedComponentsWithStats 的标签图
    返回键: id/bbox/centroid/area_px/touches_border/n_holes/contour/
            perimeter_px/circularity/aspect_ratio/orientation_deg/
            major_axis_px/minor_axis_px/solidity/simplify_eps_used
    """
    H, W = mask_shape
    comp = (labels == label_id).astype(np.uint8)
    area_px = int(comp.sum())
    if area_px <= 0:
        return None

    ys, xs = np.nonzero(comp)
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    # 贴边实例的几何量被图幅裁剪而失真 —— 标记出来供汇总时排除
    touches = bool(x0 == 0 or y0 == 0 or x1 == W - 1 or y1 == H - 1)

    rec = {"id": int(label_id),
           "bbox": [x0, y0, x1 - x0 + 1, y1 - y0 + 1],
           "centroid": [round(float(xs.mean()), 2), round(float(ys.mean()), 2)],
           "area_px": area_px,
           "touches_border": touches,
           "n_holes": 0,
           "contour": []}

    # RETR_CCOMP: 外层轮廓 + 空洞两层结构
    cnts, hier = cv2.findContours(comp, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return rec
    hier = hier[0] if hier is not None else None
    # 取面积最大的外层轮廓作为该实例主体 (hier[i][3] == -1 表示无父轮廓)
    outer_idx = [i for i in range(len(cnts))
                 if hier is None or hier[i][3] == -1]
    if not outer_idx:
        outer_idx = list(range(len(cnts)))
    main_i = max(outer_idx, key=lambda i: abs(cv2.contourArea(cnts[i])))
    main = cnts[main_i]
    # 空洞数 = 父轮廓为该主体的子轮廓个数
    if hier is not None:
        rec["n_holes"] = int(sum(1 for i in range(len(cnts)) if hier[i][3] == main_i))

    rec.update(_shape_metrics(main, area_px))
    if with_contours:
        pts, used = _simplify(main, simplify_eps, max_vertices)
        rec["contour"] = pts
        rec["simplify_eps_used"] = used
        children=[i for i in range(len(cnts)) if hier is not None and hier[i][3]==main_i]
        rec['hole_contours']=[_simplify(cnts[i],simplify_eps,max_vertices)[0] for i in children[:100]]
        rec['hole_contours_truncated']=len(children)>100
    return rec


def mask_to_regions(mask, min_area=DEFAULT_MIN_AREA, max_regions=MAX_REGIONS_LITE,
                    simplify_eps=DEFAULT_SIMPLIFY_EPS,
                    max_vertices=LITE_MAX_VERTICES, connectivity=8,
                    with_contours=True):
    """掩膜 → 实例几何列表 (按面积降序)。这是三粒度输出的唯一入口。

    返回: {"regions":[...], "num_regions_total":int, "num_regions_returned":int,
           "truncated":bool, "mask_area_px":int, "min_region_area":int,
           "simplify_eps":float}
    全部值为 python 原生类型, 可直接 json.dumps。
    """
    m = (np.asarray(mask) > 0).astype(np.uint8)
    out = {"regions": [], "num_regions_total": 0, "num_regions_returned": 0,
           "truncated": False, "mask_area_px": int(m.sum()),
           "min_region_area": int(min_area),
           "simplify_eps": float(simplify_eps)}
    if m.sum() == 0:
        return out

    n, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity)
    recs = []
    for i in range(1, n):
        if int(stats[i, cv2.CC_STAT_AREA]) < min_area:
            continue
        r = region_record(labels, i, m.shape, simplify_eps,
                          max_vertices, with_contours)
        if r is not None:
            recs.append(r)
    recs.sort(key=lambda r: -r["area_px"])

    total = len(recs)
    out["num_regions_total"] = total
    out["regions"] = recs[:max_regions]
    out["num_regions_returned"] = len(out["regions"])
    out["truncated"] = bool(total > max_regions)
    return out


# ══════════════════════════════════════════════════════════
# 二、形态学后处理 (消融实验 / 可选服务开关)
# ══════════════════════════════════════════════════════════
def _drop_small(m, min_area):
    n, lab, st, _ = cv2.connectedComponentsWithStats(m, 8)
    keep = [i for i in range(1, n) if int(st[i, cv2.CC_STAT_AREA]) >= min_area]
    return np.isin(lab, keep).astype(np.uint8) if keep else np.zeros_like(m)


def apply_morphology(mask, op="none", **kw):
    """对二值掩膜做形态学后处理。未知 op 抛 ValueError (便于评测脚本早失败)。"""
    if op not in MORPH_OPS:
        raise ValueError(f"未知形态学算子: {op}; 可选 {MORPH_OPS}")
    m = (np.asarray(mask) > 0).astype(np.uint8)
    if op == "none" or m.sum() == 0:
        return m
    k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    if op == "fill_holes":
        # 从边界泛洪填充背景, 未被触及的即为内部空洞。
        # ⚠ 必须先补一圈零边界再泛洪: 若影像角落本身是前景, 直接以 (0,0) 为种子
        #   会把这块前景当作"待填充"→ 之后 m | ~ff 使整幅图变为全 1。
        #   实测在汶川/台湾的碎裂掩膜上, 该缺陷曾导致 F1 骤降 27pp(0.836→0.564)。
        h, w = m.shape
        pad = np.zeros((h + 2, w + 2), np.uint8)
        pad[1:-1, 1:-1] = m
        ff = pad.copy()
        cv2.floodFill(ff, np.zeros((h + 4, w + 4), np.uint8), (0, 0), 1)
        return (pad | (1 - ff))[1:-1, 1:-1].astype(np.uint8)
    if op == "close3":
        return cv2.morphologyEx(m, cv2.MORPH_CLOSE, k3)
    if op == "open3":
        return cv2.morphologyEx(m, cv2.MORPH_OPEN, k3)
    if op in ("area10", "area25", "area50"):
        return _drop_small(m, int(op[4:]))
    if op == "close_then_area":
        return _drop_small(cv2.morphologyEx(m, cv2.MORPH_CLOSE, k3), 10)
    if op == "largest_only":
        n, lab, st, _ = cv2.connectedComponentsWithStats(m, 8)
        if n <= 1:
            return m
        areas = st[1:, cv2.CC_STAT_AREA]
        keep = int(np.argmax(areas)) + 1
        return (lab == keep).astype(np.uint8)
    if op == "shape_filter":
        # 丢弃"面积小且凸度低"的碎片 (纯噪声形态), 保留细长的小滑坡
        n, lab, st, _ = cv2.connectedComponentsWithStats(m, 8)
        keep = []
        for i in range(1, n):
            a = int(st[i, cv2.CC_STAT_AREA])
            if a >= 100:
                keep.append(i)
                continue
            comp = (lab == i).astype(np.uint8)
            cnts, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not cnts:
                continue
            c = max(cnts, key=lambda x: abs(cv2.contourArea(x)))
            ha = abs(float(cv2.contourArea(cv2.convexHull(c))))
            if ha > 1e-6 and (a / ha) >= 0.3:
                keep.append(i)
        return np.isin(lab, keep).astype(np.uint8) if keep else np.zeros_like(m)
    return m


def instance_prob_filter(mask, prob, tau=0.0, min_area=DEFAULT_MIN_AREA):
    """按**实例平均概率**丢弃虚假实例（第十七组；与 apply_morphology 并列的后处理）。

    为什么不复用 `area*` 系列: 实例级多报的 68.3% 是"与任何真值都无重叠的虚假实例",
    但**面积不是好判别轴** —— 真实滑坡也可以很小（第十一组实测 area25/area50 使
    Acc@0.5 掉 0.062/0.149，正是删掉了真实小滑坡）。实测判别力:
        实例内平均概率 AUC **0.869**  vs  面积 0.747  vs  紧致度 0.681（且方向相反）
    本函数因此用**概率**而非几何量。

    与"逐像素阈值"的本质区别: 逐像素阈值会切碎或抹掉"整体强响应、但边缘弥散"的
    真实大滑坡; 本函数只丢弃**整体弱响应**的整实例。

    实测（LMHLD 6 区域, val 选 tau / test 汇报, tau=0.6）:
        实例 F1@0.5 **+1.38pp**、计数偏差 **+0.329→+0.065**、像素 F1 不变;
        5/6 区域改善, 增益落在"过报型"区域（台湾 +5.25pp）。
    详见 experiments/instance_diag/第十七组实验报告_实例级误差归因与概率过滤.md

    tau <= 0 时**原样返回**（默认关, 保证既有行为逐字节不变）。
    mask: (H,W) 二值; prob: 同尺寸概率图。形状不符时不动 —— 全链路不得抛异常。
    """
    m = (np.asarray(mask) > 0).astype(np.uint8)
    if tau <= 0 or m.sum() == 0:
        return m
    p = np.asarray(prob, dtype=np.float32)
    if p.shape != m.shape:
        return m
    lab, areas = label_instances(m, min_area)
    if not areas:
        return m
    keep = np.zeros_like(m)
    for i in range(1, len(areas) + 1):
        sel = (lab == i)
        if float(p[sel].mean()) >= tau:
            keep[sel] = 1
    return keep


# ══════════════════════════════════════════════════════════
# 三、尺度换算 (仅当有 GSD / 影像元数据时启用)
# ══════════════════════════════════════════════════════════
def attach_scale(regions, meters_per_pixel):
    """就地补充米制量。无 GSD 时一律不臆造米制数字 —— 这是诚实性红线。"""
    if not meters_per_pixel or meters_per_pixel <= 0:
        return regions
    mpp = float(meters_per_pixel)
    for r in regions:
        r["area_m2"] = round(r["area_px"] * mpp * mpp, 1)
        r["perimeter_m"] = round(r["perimeter_px"] * mpp, 2)
        r["major_axis_m"] = round(r["major_axis_px"] * mpp, 2)
        r["minor_axis_m"] = round(r["minor_axis_px"] * mpp, 2)
        r["bbox_m"] = [round(v * mpp, 2) for v in r["bbox"][2:]]
    return regions


def pixel_to_lonlat(x, y, origin_latlon, meters_per_pixel):
    """像素 → 经纬度 (局部等距近似)。origin_latlon = 影像左上角 (lat, lon)。

    几公里范围内经度方向按 cos(lat) 缩放, 误差 <0.1%, 对矿区尺度足够;
    GeoJSON properties 中会写明该近似。
    """
    lat0, lon0 = float(origin_latlon[0]), float(origin_latlon[1])
    mpp = float(meters_per_pixel)
    lon = lon0 + (x * mpp) / (EARTH_R * math.cos(math.radians(lat0))) * (180.0 / math.pi)
    lat = lat0 - (y * mpp) / EARTH_R * (180.0 / math.pi)
    return round(lon, 7), round(lat, 7)


# ══════════════════════════════════════════════════════════
# 四、矢量导出
# ══════════════════════════════════════════════════════════
def _ring(contour, origin, mpp):
    if origin and mpp:
        pts = [pixel_to_lonlat(x, y, origin, mpp) for x, y in contour]
    else:
        pts = [[float(x), float(y)] for x, y in contour]
    if not pts:
        return []
    if pts[0] != pts[-1]:          # GeoJSON 要求环闭合
        pts.append(list(pts[0]))
    return pts


def regions_to_geojson(regions, meters_per_pixel=None, origin_latlon=None,
                       image_shape=None, name="landslide",georeference=None):
    """实例几何 → GeoJSON FeatureCollection (RFC 7946)。

    有 origin_latlon + meters_per_pixel → 输出经纬度坐标(局部等距近似);
    否则输出像素坐标, 并在 properties.crs 标明 "pixel"(诚实口径)。
    传入 meters_per_pixel 时, 若实例尚未带米制量则就地补算 —— 避免出现
    "坐标是经纬度、面积却只有像素"的不自洽(不改动调用方的原对象)。
    """
    if meters_per_pixel and meters_per_pixel > 0:
        regions = [dict(r) for r in regions]
        attach_scale(regions, meters_per_pixel)
    geo_ok = bool(georeference or (origin_latlon and meters_per_pixel))
    def ring_for(points):
        if not georeference: return _ring(points,origin_latlon if geo_ok else None,meters_per_pixel if geo_ok else None)
        from rasterio.transform import Affine
        from rasterio.warp import transform
        affine=Affine(*georeference['transform']); native=[affine*(float(x)+.5,float(y)+.5) for x,y in points]
        east,north=zip(*native); lon,lat=transform(georeference['crs'],'EPSG:4326',list(east),list(north))
        if not all(math.isfinite(x) and math.isfinite(y) and -180<=x<=180 and -90<=y<=90 for x,y in zip(lon,lat)):
            raise ValueError('栅格地理参考不能转换为有效经纬度')
        ring=[[round(x,7),round(y,7)] for x,y in zip(lon,lat)]
        if ring and ring[0]!=ring[-1]: ring.append(ring[0].copy())
        return ring
    feats = []
    for r in regions:
        if not r.get("contour"):
            continue
        ring = ring_for(r['contour'])
        if len(ring) < 4:
            continue
        props = {"id": r["id"], "area_px": r["area_px"],
                 "perimeter_px": r["perimeter_px"],
                 "major_axis_px": r["major_axis_px"],
                 "minor_axis_px": r["minor_axis_px"],
                 "circularity": r["circularity"],
                 "aspect_ratio": r["aspect_ratio"],
                 "orientation_deg": r["orientation_deg"],
                 "solidity": r["solidity"],
                 "n_holes": r["n_holes"],
                 "touches_border": r["touches_border"],
                 "bbox": r["bbox"],
                 "simplify_eps": r.get("simplify_eps_used", DEFAULT_SIMPLIFY_EPS),
                 "crs": "EPSG:4326(文件仿射/投影转换)" if georeference else "EPSG:4326(局部等距近似)" if geo_ok else "pixel",
                 'hole_contours_truncated':bool(r.get('hole_contours_truncated',False))}
        for k in ("area_m2", "perimeter_m", "major_axis_m", "minor_axis_m"):
            if k in r:
                props[k] = r[k]
        if geo_ok and meters_per_pixel is not None:
            props["meters_per_pixel"] = float(meters_per_pixel)
        if georeference:
            props['coordinate_source']='uploaded_raster_header'; props['independent_ground_accuracy_verified']=False
        rings=[ring]+[ring_for(h) for h in r.get('hole_contours',[]) if len(h)>=3]
        feats.append({"type": "Feature", "properties": props,
                      "geometry": {"type": "Polygon", "coordinates": rings}})
    out = {"type": "FeatureCollection",
           "name": name,
           "properties": {"source": "矿安天眼 多粒度几何引擎",
                          "n_features": len(feats),
                          "meters_per_pixel": (float(meters_per_pixel) if geo_ok and meters_per_pixel is not None else None),
                          'source_crs':georeference.get('crs') if georeference else None}}
    if image_shape:
        out["properties"]["image_shape"] = [int(image_shape[0]), int(image_shape[1])]
    out["features"] = feats
    return out


def regions_to_csv(regions, meters_per_pixel=None) -> str:
    """实例几何 → CSV 文本 (面积/周长口径与 GeoJSON 一致)。

    同 regions_to_geojson: 给了 meters_per_pixel 就补算米制列。
    """
    if meters_per_pixel and meters_per_pixel > 0:
        regions = [dict(r) for r in regions]
        attach_scale(regions, meters_per_pixel)
    cols = ["id", "x", "y", "w", "h", "area_px", "perimeter_px",
            "major_axis_px", "minor_axis_px", "circularity", "aspect_ratio",
            "orientation_deg", "solidity", "n_holes", "touches_border"]
    if meters_per_pixel:
        cols += ["area_m2", "perimeter_m", "major_axis_m"]
    lines = [",".join(cols)]
    for r in regions:
        row = [r["id"]] + list(r["bbox"]) + [r["area_px"], r["perimeter_px"],
              r["major_axis_px"], r["minor_axis_px"], r["circularity"],
              r["aspect_ratio"], r["orientation_deg"], r["solidity"],
              r["n_holes"], int(r["touches_border"])]
        if meters_per_pixel:
            row += [r.get("area_m2", ""), r.get("perimeter_m", ""),
                    r.get("major_axis_m", "")]
        lines.append(",".join(str(v) for v in row))
    return "\n".join(lines)


def instance_overlap(label_a, label_b, n_a, n_b):
    """两组实例的重叠矩阵 (供实例匹配使用)。

    用联合直方图一次算完, 避免逐对比较的 O(n·m·HW)。
    返回 dict:
      inter       : (n_a, n_b) 交集像素数
      area_a/b    : (n_a,) / (n_b,) 各自面积
      iou         : 交并比
      containment : |A∩B| / min(|A|,|B|)  —— 单向包含度

    为什么需要 containment: 判断"同一处滑坡是否扩大"时, 区域扩大会使
    外接/掩膜 IoU 必然下降, 用 IoU 匹配会把「扩大」误判成「删除+新增」;
    而包含度对单向扩张是稳定的 (小者几乎完全落在扩大后的区域里)。
    """
    inter = np.zeros((max(n_a, 0), max(n_b, 0)))
    area_a = np.zeros(max(n_a, 0))
    area_b = np.zeros(max(n_b, 0))
    if n_a == 0 or n_b == 0:
        return {"inter": inter, "area_a": area_a, "area_b": area_b,
                "iou": inter, "containment": inter}
    flat = (label_a.ravel().astype(np.int64) * (n_b + 1)
            + label_b.ravel().astype(np.int64))
    joint = np.bincount(flat, minlength=(n_a + 1) * (n_b + 1))
    inter = joint.reshape(n_a + 1, n_b + 1)[1:, 1:].astype(np.float64)
    area_a = np.bincount(label_a.ravel(), minlength=n_a + 1)[1:].astype(np.float64)
    area_b = np.bincount(label_b.ravel(), minlength=n_b + 1)[1:].astype(np.float64)
    union = area_a[:, None] + area_b[None, :] - inter
    iou = inter / np.maximum(union, 1e-9)
    min_a = np.maximum(area_a[:, None], 1e-9)
    containment = inter / np.minimum(min_a, np.maximum(area_b[None, :], 1e-9))
    return {"inter": inter, "area_a": area_a, "area_b": area_b,
            "iou": iou, "containment": containment}


def label_instances(mask, min_area=DEFAULT_MIN_AREA):
    """连通域标注 (过滤面积 < min_area 的噪点)。

    返回 (label_map, areas) —— label_map 为 0(背景)/1..n 的 int32, areas 为各实例面积。
    """
    m = (np.asarray(mask) > 0).astype(np.uint8)
    if m.sum() == 0:
        return np.zeros(m.shape, np.int32), []
    n, lab, st, _ = cv2.connectedComponentsWithStats(m, 8)
    keep = [i for i in range(1, n) if int(st[i, cv2.CC_STAT_AREA]) >= min_area]
    if not keep:
        return np.zeros(m.shape, np.int32), []
    relabel = np.zeros(n, np.int32)
    for new, old in enumerate(keep, start=1):
        relabel[old] = new
    return relabel[lab], [int(st[i, cv2.CC_STAT_AREA]) for i in keep]


def polygon_to_mask(regions, shape) -> np.ndarray:
    """多边形回栅格化 (评测 E3 与前端预览共用)。shape = (H, W)。"""
    out = np.zeros(shape, np.uint8)
    for r in regions:
        if not r.get("contour"):
            continue
        cv2.fillPoly(out, [np.asarray(r["contour"], np.int32)], 1)
    return out


# ══════════════════════════════════════════════════════════
# 五、文本块与数字白名单 (提示词注入 / 报告一致性校验)
# ══════════════════════════════════════════════════════════
def format_geometry_block(regions, num_regions_total=None, meters_per_pixel=None,
                          topk=2, image_shape=None, max_chars=200) -> str:
    """生成注入提示词的中文几何块 —— 硬约束 ≤5 行 / ≤max_chars 字。

    规划期实测: 现有 prompt 已在 16384 token 阈值边缘(出现过 Token count
    20953 触发摘要并 400 失败), 故本块必须极短。
    """
    if not regions:
        return "【几何形态工具输出】未检出有效实例。"
    lines = ["【几何形态工具输出】(由实例多边形直接计算, 请只引用以下数值, 勿自行估计)"]
    for i, r in enumerate(regions[:topk], 1):
        # Sorting by area must not change the identity used by maps and archives.
        instance_id = r.get('id', i)
        seg = (f"· 实例#{instance_id}: 外接框 {r['bbox'][2]}×{r['bbox'][3]}px, "
               f"面积 {r['area_px']}px, 周长 {r['perimeter_px']}px, "
               f"长轴 {r['major_axis_px']}px, 紧致度 {r['circularity']}")
        if meters_per_pixel and "area_m2" in r:
            seg += f", 折合面积 {r['area_m2']}m²、长轴 {r['major_axis_m']}m"
        lines.append(seg)
    total = num_regions_total if num_regions_total is not None else len(regions)
    biggest = regions[0]["area_px"] / max(sum(x["area_px"] for x in regions), 1)
    tail = f"· 共 {total} 处实例, 最大实例占检出面积 {biggest * 100:.0f}%"
    if image_shape:
        tail += f"; 影像 {int(image_shape[1])}×{int(image_shape[0])}px"
    lines.append(tail)
    txt = "\n".join(lines)
    return txt if len(txt) <= max_chars else txt[:max_chars - 1] + "…"


def _fmt_num(v):
    """生成一个数字的多种常见写法, 供白名单宽松匹配 (含四舍五入变体)。"""
    out = set()
    for nd in (0, 1, 2):
        try:
            s = f"{float(v):.{nd}f}".rstrip("0").rstrip(".")
            if s:
                out.add(s)
        except (TypeError, ValueError):
            pass
    return out


def geometry_number_whitelist(regions, num_regions_total=None,
                              meters_per_pixel=None) -> set:
    """返回报告"允许引用"的几何数字集合 (字符串形式)。

    供 backend.validate_report 做跨模块一致性校验: 报告里凡出现
    「长度/宽度/周长/长轴/外接框 + 数字」这类表述, 其数字应落在白名单内,
    否则说明模型在编造几何 (已存档的报告曾凭空写出"长度约 180 像素")。
    """
    wl = set()
    for r in regions:
        for k in ("area_px", "perimeter_px", "major_axis_px", "minor_axis_px",
                  "area_m2", "perimeter_m", "major_axis_m", "minor_axis_m"):
            if k in r:
                wl |= _fmt_num(r[k])
        for v in r["bbox"]:
            wl |= _fmt_num(v)
        wl |= _fmt_num(r["circularity"])
        wl |= _fmt_num(r["aspect_ratio"])
    if num_regions_total is not None:
        wl |= _fmt_num(num_regions_total)
    return {s for s in wl if s}


# ══════════════════════════════════════════════════════════
# 六、自测
# ══════════════════════════════════════════════════════════
def selftest():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    ok = True

    def chk(name, cond, extra=""):
        nonlocal ok
        print(f"  {'✓' if cond else '✗'} {name} {extra}")
        if not cond:
            ok = False

    print(f"numpy {np.__version__} / cv2 {cv2.__version__}")

    # 1) 圆盘: 形态量应有正确的量级
    m = np.zeros((128, 128), np.uint8)
    cv2.circle(m, (64, 64), 20, 1, -1)
    res = mask_to_regions(m)
    r = res["regions"][0]
    print(f"  圆盘 r=20: area={r['area_px']} major={r['major_axis_px']} "
          f"minor={r['minor_axis_px']} circ={r['circularity']} "
          f"perp={r['perimeter_px']} solidity={r['solidity']}")
    chk("圆盘只有 1 个实例", res["num_regions_total"] == 1)
    chk("圆盘面积 ≈1257", abs(r["area_px"] - 1257) < 30, f"实={r['area_px']}")
    chk("长轴≈短轴≈40px", abs(r["major_axis_px"] - 40) < 4 and
        abs(r["minor_axis_px"] - 40) < 4)
    chk("长宽比≈1", 0.9 < r["aspect_ratio"] < 1.1)
    chk("凸度≈1", r["solidity"] > 0.95)
    chk("紧致度在合理区间(数字周长偏大导致 <1)", 0.55 < r["circularity"] < 1.0)

    # 2) 圆环: 空洞计数 + 填洞正确性
    ring = np.zeros((128, 128), np.uint8)
    cv2.circle(ring, (64, 64), 20, 1, -1)
    cv2.circle(ring, (64, 64), 8, 0, -1)
    rr = mask_to_regions(ring)["regions"][0]
    chk("圆环空洞数 = 1", rr["n_holes"] == 1, f"实={rr['n_holes']}")
    fh = apply_morphology(ring, "fill_holes")
    chk("填洞后空洞消失", int((fh & (1 - ring)).sum()) > 100
        and int(fh.sum()) > int(ring.sum()),
        f"原={int(ring.sum())} 填后={int(fh.sum())}")
    chk("填洞不影响外部背景", int(fh[0, 0]) == 0 and int(fh[5, 5]) == 0)

    # 2b) 角落即前景的掩膜 —— 曾经因泛洪种子取 (0,0) 而使整幅图变全 1
    corner = np.zeros((64, 64), np.uint8)
    corner[0:20, 0:20] = 1
    corner[10:12, 10:12] = 0        # 该前景块内部的小洞
    cf = apply_morphology(corner, "fill_holes")
    chk("角落前景: 不把整幅图填满", int(cf.sum()) < corner.size * 0.5,
        f"填后前景={int(cf.sum())}/{corner.size}")
    chk("角落前景: 内部小洞被填上", int(cf[10, 10]) == 1)

    # 3) 双实例 + 小噪声过滤
    two = np.zeros((128, 128), np.uint8)
    cv2.rectangle(two, (5, 5), (25, 30), 1, -1)      # 面积 21*26=546
    cv2.rectangle(two, (60, 60), (75, 70), 1, -1)    # 面积 16*11=176
    two[100, 100] = 1                                 # 单像素噪声
    t2 = mask_to_regions(two)
    chk("小噪声被 min_area 过滤, 剩 2 实例", t2["num_regions_total"] == 2,
        f"实={t2['num_regions_total']}")
    chk("按面积降序", t2["regions"][0]["area_px"] > t2["regions"][1]["area_px"])

    # 4) 退化输入不抛异常
    for nm, arr in [("空掩膜", np.zeros((32, 32), np.uint8)),
                    ("全掩膜", np.ones((32, 32), np.uint8)),
                    ("单点", np.zeros((32, 32), np.uint8)),
                    ("3x3", np.zeros((32, 32), np.uint8))]:
        if nm == "单点":
            arr[16, 16] = 1
        if nm == "3x3":
            arr[10:13, 10:13] = 1
        try:
            mask_to_regions(arr)
            chk(f"退化输入[{nm}] 不抛异常", True)
        except Exception as e:
            chk(f"退化输入[{nm}] 不抛异常", False, repr(e))

    # 5) 贴边标记
    edge = np.zeros((32, 32), np.uint8)
    cv2.rectangle(edge, (0, 0), (20, 20), 1, -1)
    chk("贴边实例被标记", mask_to_regions(edge)["regions"][0]["touches_border"])

    # 6) JSON 可序列化 (numpy 2.0 下这是 500 错误的高发点)
    for nm, arr in [("圆盘", m), ("圆环", ring), ("双实例", two)]:
        try:
            json.dumps(mask_to_regions(arr), ensure_ascii=False)
            chk(f"JSON 序列化[{nm}]", True)
        except TypeError as e:
            chk(f"JSON 序列化[{nm}]", False, repr(e))
    try:
        json.dumps(regions_to_geojson(mask_to_regions(m)["regions"]), ensure_ascii=False)
        chk("GeoJSON 序列化", True)
    except TypeError as e:
        chk("GeoJSON 序列化", False, repr(e))

    # 7) 尺度换算与经纬度
    regs = attach_scale(mask_to_regions(m)["regions"], 2.046)
    chk("面积换算 m²", abs(regs[0]["area_m2"] - 1257 * 2.046 ** 2) < 300,
        f"实={regs[0]['area_m2']}")
    lon, lat = pixel_to_lonlat(640, 640, (31.09057409495418, 98.690185546875), 2.046)
    chk("经纬度换算在东经/北纬合理范围", 98 < lon < 99 and 31 < lat < 32,
        f"({lon},{lat})")

    # 8) 形态学算子全部可执行且不改变尺寸
    for op in MORPH_OPS:
        try:
            o = apply_morphology(two, op)
            chk(f"形态学[{op}]", o.shape == two.shape and set(np.unique(o)) <= {0, 1})
        except Exception as e:
            chk(f"形态学[{op}]", False, repr(e))
    try:
        apply_morphology(two, "bogus")
        chk("未知算子抛 ValueError", False)
    except ValueError:
        chk("未知算子抛 ValueError", True)

    # 9) 几何文本块的硬长度约束
    blk = format_geometry_block(mask_to_regions(m)["regions"], 1, max_chars=200)
    chk("几何块 ≤5 行", len(blk.split("\n")) <= 5, f"实={len(blk.split(chr(10)))}行")
    chk("几何块 ≤200 字", len(blk) <= 200, f"实={len(blk)}字")
    wl = geometry_number_whitelist(mask_to_regions(m)["regions"], 1)
    chk("数字白名单非空", len(wl) > 0, f"例: {sorted(wl)[:5]}")

    # 10) 多边形回栅格化保真度 (E3 实验的基础)
    back = polygon_to_mask(mask_to_regions(m)["regions"], m.shape)
    inter = int((back & m).sum())
    union = int((back | m).sum())
    iou = inter / max(union, 1)
    chk("多边形回栅格 IoU >0.97", iou > 0.97, f"实={iou:.4f}")

    print(f"\n{'✅ 自测全部通过' if ok else '❌ 存在失败项'}")
    return 0 if ok else 1


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        sys.exit(selftest())
    print(__doc__)
