# -*- coding: utf-8 -*-
"""
fetch_mine_imagery.py — 真实矿区卫星影像下载与拼接（用于场景验证）
================================================================
数据源: Esri World Imagery (公开可访问的免费底图瓦片, 无需密钥)
用法:
  python fetch_mine_imagery.py --lat 41.833 --lon 123.905 --zoom 15 --tiles 9 --name fushun_west
参数:
  --lat/--lon  中心坐标;  --zoom 层级(15≈3.6m/px, 16≈1.8m/px, 17≈0.9m/px)
  --tiles      拼接瓦片边长(9 = 9×9 瓦片 = 2304×2304 px)
  --name       输出文件名前缀
输出: <outdir>/<name>.png (拼接影像) + <name>_meta.json (坐标/分辨率信息)
"""
import os
import sys
import math
import time
import json
import argparse
import urllib.request

import numpy as np
from PIL import Image

sys.stdout.reconfigure(encoding="utf-8")

TILE_URL = ("https://server.arcgisonline.com/ArcGIS/rest/services/"
            "World_Imagery/MapServer/tile/{z}/{y}/{x}")
DEFAULT_OUT = r"D:\竞赛材料\真实场景验证\影像"


def deg2tile(lat, lon, z):
    n = 2 ** z
    x = int((lon + 180.0) / 360.0 * n)
    lat_rad = math.radians(lat)
    y = int((1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n)
    return x, y


def tile2deg(x, y, z):
    n = 2 ** z
    lon = x / n * 360.0 - 180.0
    lat = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * y / n))))
    return lat, lon


def fetch_tile(z, x, y, retries=3):
    url = TILE_URL.format(z=z, x=x, y=y)
    for i in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=20) as r:
                return r.read()
        except Exception as e:
            if i == retries - 1:
                print(f"    瓦片失败 {z}/{x}/{y}: {e}")
                return None
            time.sleep(1.5)
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lat", type=float, required=True)
    ap.add_argument("--lon", type=float, required=True)
    ap.add_argument("--zoom", type=int, default=16)
    ap.add_argument("--tiles", type=int, default=9)
    ap.add_argument("--name", required=True)
    ap.add_argument("--outdir", default=DEFAULT_OUT)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    cx, cy = deg2tile(args.lat, args.lon, args.zoom)
    half = args.tiles // 2
    x0, y0 = cx - half, cy - half

    print(f"[{args.name}] zoom={args.zoom} 瓦片范围 x:{x0}..{x0+args.tiles-1} y:{y0}..{y0+args.tiles-1}")
    canvas = Image.new("RGB", (args.tiles * 256, args.tiles * 256))
    ok = 0
    for dy in range(args.tiles):
        for dx in range(args.tiles):
            data = fetch_tile(args.zoom, x0 + dx, y0 + dy)
            if data is None:
                continue
            import io
            tile = Image.open(io.BytesIO(data)).convert("RGB")
            canvas.paste(tile, (dx * 256, dy * 256))
            ok += 1
            time.sleep(0.08)   # 温和限速
        print(f"    行 {dy+1}/{args.tiles} 完成 ({ok} 瓦片)")

    out_png = os.path.join(args.outdir, f"{args.name}.png")
    canvas.save(out_png)

    # 元数据: 左上/右下角坐标 + 地面分辨率
    lat_tl, lon_tl = tile2deg(x0, y0, args.zoom)
    lat_br, lon_br = tile2deg(x0 + args.tiles, y0 + args.tiles, args.zoom)
    mpp = 156543.03 * math.cos(math.radians(args.lat)) / (2 ** args.zoom)
    meta = {
        "name": args.name, "center": [args.lat, args.lon], "zoom": args.zoom,
        "tiles": args.tiles, "size_px": [canvas.width, canvas.height],
        "bbox": [[lat_tl, lon_tl], [lat_br, lon_br]],
        "meters_per_pixel": round(mpp, 3),
        "extent_km": [round(canvas.width * mpp / 1000, 2), round(canvas.height * mpp / 1000, 2)],
        "source": "Esri World Imagery",
    }
    with open(os.path.join(args.outdir, f"{args.name}_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"[完成] {out_png} ({canvas.width}×{canvas.height}px, {mpp:.2f} m/px, "
          f"覆盖 {meta['extent_km'][0]}×{meta['extent_km'][1]} km)")


if __name__ == "__main__":
    main()
