# -*- coding: utf-8 -*-
"""
fetch_sentinel2.py — Sentinel-2 真实多光谱影像获取
====================================================
动机(第十三组诊断结论): 真实影像泛化的主因是 **NIR 通道缺失(-32.7pp)**,
而 Esri World Imagery 只有 RGB。Sentinel-2 提供 B02/B03/B04/**B08(NIR)** 四个
10m 波段 —— 这是本项目首个**真实多光谱**数据源。

数据源选型(实测):
  · AWS Earth Search (Element 84 镜像 sentinel-cogs) —— **零门槛, 无需注册**
    STAC: https://earth-search.aws.element84.com/v1/search
  · 实测不可达: ESA SciHub / Google Earth Engine / 地理空间数据云 /
    国家地对地观测中心 / 中国资源卫星中心
    (意味着高分系列多光谱暂时拿不到)

为什么用 Range 分块读取而非整景下载:
  实测整波段(85MB)下载**会被中断** —— 只下到 2.8MB 就断(与 GitHub 同一现象);
  而 Range 分块请求稳定(单次约 3s)。故只读目标窗口所在的分块。
  另: 本地无 GDAL/rasterio 时曾尝试 tifffile+自建 Range 文件对象, 其分块解码
  API 不稳定, 最终采用 rasterio(GDAL 打包在 wheel 内, pip 即装)。

用法:
  python fetch_sentinel2.py --lat 31.081 --lon 98.705 --name baige_s2 --half-km 2
  python fetch_sentinel2.py --lat 41.833 --lon 123.905 --name fushun_s2 --half-km 4
  python fetch_sentinel2.py --name baige_s2 --list-dates   # 只看可用日期
"""
import os
import io
import sys
import json
import time
import argparse
import urllib.request
from datetime import datetime

import numpy as np

sys.stdout.reconfigure(encoding="utf-8")

STAC = "https://earth-search.aws.element84.com/v1/search"
COLLECTION = "sentinel-2-l2a"
# S2 波段资产名 → 本系统通道序 [B, G, R, NIR]
BANDS = [("blue", "B02 蓝"), ("green", "B03 绿"), ("red", "B04 红"), ("nir", "B08 近红外")]
DEFAULT_OUT = r"D:\竞赛材料\真实场景验证\影像\sentinel2"

# GDAL 的 HTTP 优化: 关闭目录列举、HTTP/2 多路复用、启用 VSICACHE
# (实测: 首次窗口 ~90s, 之后同数据集的窗口因缓存几乎瞬时返回)
GDAL_ENV = dict(
    GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR",
    CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tif",
    GDAL_HTTP_MULTIPLEX="YES",
    GDAL_HTTP_VERSION="2",
    VSI_CACHE="TRUE",
    VSI_CACHE_SIZE=str(128 * 1024 * 1024),
    GDAL_HTTP_MAX_RETRY="1",
    GDAL_HTTP_TIMEOUT="60",
)


def stac_search(lon, lat, date_from, date_to, max_cloud=30, limit=10):
    """查询覆盖该点的 S2 L2A 影像, 按云量升序返回"""
    body = {
        "collections": [COLLECTION],
        "limit": limit,
        "query": {"eo:cloud_cover": {"lt": max_cloud}},
        "datetime": f"{date_from}T00:00:00Z/{date_to}T23:59:59Z",
        "intersects": {"type": "Point", "coordinates": [lon, lat]},
        "sortby": [{"field": "properties.eo:cloud_cover", "direction": "asc"}],
    }
    req = urllib.request.Request(STAC, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=90) as r:
        return json.loads(r.read().decode()).get("features", [])


def read_window(url, lon, lat, half_px):
    """用 rasterio 从 HTTP 上的 COG 读一个窗口(经纬度为中心)"""
    import rasterio
    from rasterio.windows import Window
    from rasterio.warp import transform as rio_transform
    from rasterio.env import Env
    with Env(**GDAL_ENV):
        with rasterio.open(url) as src:
            xs, ys = rio_transform("EPSG:4326", src.crs, [lon], [lat])
            row, col = src.index(xs[0], ys[0])
            w = h = 2 * half_px
            col0, row0 = col - half_px, row - half_px
            # 越界裁剪: 窗口必须完全落在影像内
            col0 = max(0, min(col0, src.width - w))
            row0 = max(0, min(row0, src.height - h))
            arr = src.read(1, window=Window(col0, row0, w, h))
            return arr, {"crs": str(src.crs), "res": src.res,
                         "row0": int(row0), "col0": int(col0)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lat", type=float)
    ap.add_argument("--lon", type=float)
    ap.add_argument("--name", required=True)
    ap.add_argument("--half-km", type=float, default=2.0, help="窗口半边长(公里)")
    ap.add_argument("--date-from", default="2026-06-01")
    ap.add_argument("--date-to", default=datetime.now().strftime("%Y-%m-%d"))
    ap.add_argument("--max-cloud", type=float, default=20.0)
    ap.add_argument("--outdir", default=DEFAULT_OUT)
    ap.add_argument("--list-dates", action="store_true")
    args = ap.parse_args()

    if args.lat is None or args.lon is None:
        print("✗ 需要 --lat/--lon")
        return 1

    print(f"STAC 查询 覆盖 ({args.lon}, {args.lat}) 的 Sentinel-2 L2A …")
    feats = stac_search(args.lon, args.lat, args.date_from, args.date_to, args.max_cloud)
    if not feats:
        print("✗ 无可用影像(可放宽 --max-cloud 或 --date-from)")
        return 1
    print(f"  命中 {len(feats)} 景:")
    for k, f in enumerate(feats[:6]):
        p = f["properties"]
        print(f"    [{k}] {p['datetime'][:10]}  云量 {p.get('eo:cloud_cover', -1):>5.1f}%  {f['id']}")
    if args.list_dates:
        return 0

    f = feats[0]
    p = f["properties"]
    print(f"\n选用: {f['id']}  ({p['datetime'][:10]}, 云量 {p.get('eo:cloud_cover'):.1f}%)")

    half_px = int(args.half_km * 1000 / 10)          # S2 为 10m/px
    os.makedirs(args.outdir, exist_ok=True)
    bands, geo = [], None
    t0 = time.time()
    for key, label in BANDS:
        if key not in f["assets"]:
            print(f"  ✗ 缺波段资产 {key}")
            return 1
        url = f["assets"][key]["href"]
        arr, meta = read_window(url, args.lon, args.lat, half_px)
        bands.append(arr.astype(np.float32))
        geo = meta
        print(f"  {label:<10} {arr.shape} 均值 {arr.mean():>6.0f}  ({time.time()-t0:.0f}s)")

    # 通道序 [B, G, R, NIR] —— 与 detector.py 的 NUM_CHANNELS/注释一致
    img = np.stack(bands)                              # (4, H, W)
    npy = os.path.join(args.outdir, args.name + ".npy")
    np.save(npy, img)

    meta = {
        "name": args.name, "center": [args.lat, args.lon],
        "source": "Sentinel-2 L2A (AWS Earth Search / sentinel-cogs)",
        "scene_id": f["id"], "datetime": p["datetime"],
        "cloud_cover": round(float(p.get("eo:cloud_cover", -1)), 2),
        "epsg": p.get("proj:epsg"), "crs": geo["crs"],
        "meters_per_pixel": 10.0,
        "size_px": [int(img.shape[2]), int(img.shape[1])],
        "extent_km": [round(img.shape[2] * 10 / 1000, 2), round(img.shape[1] * 10 / 1000, 2)],
        "channels": [b[0] for b in BANDS],
        "channel_order": "B,G,R,NIR",
        "values": "BOA 反射率 ×10000 (uint16 原始量纲)",
        "license": "Copernicus Sentinel Data License (需署名 ESA/Copernicus)",
    }
    with open(os.path.join(args.outdir, args.name + "_meta.json"), "w", encoding="utf-8") as fp:
        json.dump(meta, fp, ensure_ascii=False, indent=2)

    print(f"\n影像: {npy}  形状 {img.shape}  值域 {img.min():.0f}~{img.max():.0f}")
    print(f"元数据: {os.path.join(args.outdir, args.name + '_meta.json')}")
    print(f"总计 {time.time()-t0:.0f}s")
    # 物理合理性自检: NIR 应显著高于 R(植被) —— 这是强的正确性判据
    rat = float(img[3].mean() / max(img[2].mean(), 1))
    ndvi = float(((img[3] - img[2]) / (img[3] + img[2] + 1e-6)).mean())
    print(f"自检: NIR/R = {rat:.2f},  平均 NDVI = {ndvi:+.3f}  "
          f"({'✓ 符合植被区预期' if ndvi > 0.1 else '⚠ NDVI 偏低, 检查是否云/水体覆盖'})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
