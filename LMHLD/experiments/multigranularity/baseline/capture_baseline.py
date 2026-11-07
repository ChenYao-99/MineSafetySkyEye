# -*- coding: utf-8 -*-
"""
capture_baseline.py — 抓取改造前的服务响应"黄金基线"
=====================================================
用途: 在改动任何代码之前, 把 live 服务(:5100 -> :5101)的确定性响应落盘,
      作为改造后 A/B 比对的唯一客观判据。只读, 不修改服务状态。

设计要点:
  · 只抓确定性端点 (health/regions/detect/detect_upload/compare)
    不含 /api/pipeline —— 它走 LLM, 输出不确定且消耗额度。
  · 响应里的 base64 图像不落盘原图, 只存 md5 + 长度 —— 既能逐字节比对,
    又能把 Golden 文件控制在 KB 级。图像一旦变化 md5 立刻不同。

用法: python capture_baseline.py            # 抓取并写入本目录
     python capture_baseline.py --check    # 与已有基线比对 (改造后用)
"""
import os
import sys
import json
import base64
import hashlib
import urllib.request
from datetime import datetime

import numpy as np
import cv2

sys.stdout.reconfigure(encoding="utf-8")

API = "http://localhost:5100/api"
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "golden_baseline.json")
P_IMG = r"D:\竞赛材料\真实场景验证\影像"


def post(path, payload, timeout=300):
    req = urllib.request.Request(API + path,
                                 data=json.dumps(payload).encode("utf-8"),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def get(path, timeout=60):
    with urllib.request.urlopen(API + path, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def b64_of(name):
    img = cv2.imdecode(np.fromfile(os.path.join(P_IMG, name + ".png"),
                                   dtype=np.uint8), cv2.IMREAD_COLOR)
    return base64.b64encode(cv2.imencode(".png", img)[1].tobytes()).decode()


def digest(obj):
    """把响应中所有 base64 图像换成 md5+长度; 其余数值保留。
    返回可直接 JSON 序列化的 (摘要对象, 图像指纹字典)。"""
    imgs = {}

    def walk(o, path=""):
        if isinstance(o, dict):
            return {k: walk(v, f"{path}.{k}") for k, v in o.items()}
        if isinstance(o, list):
            return [walk(v, f"{path}[{i}]") for i, v in enumerate(o)]
        if isinstance(o, str) and len(o) > 512 and path.endswith("_b64"):
            b = base64.b64decode(o)
            imgs[path] = {"md5": hashlib.md5(b).hexdigest(), "bytes": len(b)}
            return f"<b64:{imgs[path]['md5'][:12]}:{len(b)}B>"
        return o

    return walk(obj), imgs


# ── 抓取用例: 覆盖三条检测链路 + 健康检查 ──────────────────
CASES = [
    ("health", "GET", "/health", None),
    # 注: 后端未代理 /regions (404), 该端点只在检测服务 :5101 上, 故不纳入基线
    ("detect_emilia_n6", "POST", "/detect",
     {"region": "Emilia-Romagna_Italy_32", "num": 6}),
    ("detect_taiwan_n4", "POST", "/detect",
     {"region": "Taiwan_China_224", "num": 4}),
    ("detect_emilia_fuse_ds", "POST", "/detect",
     {"region": "Hokkaido_Japan_128", "num": 3, "fuse": True, "ds_reject": True}),
    ("detect_upload_baige_base", "POST", "/detect_upload",
     {"image": None, "model_tag": "base"}),          # image 运行时填充
    ("compare_baige", "POST", "/compare",
     {"image_t1": None, "image_t2": None}),
]


def run_capture():
    print(f"抓取黄金基线 <- {API}")
    out = {"when": datetime.now().isoformat(), "api": API, "cases": {}}
    for name, method, path, payload in CASES:
        try:
            if name == "detect_upload_baige_base":
                payload = dict(payload, image=b64_of("baige_landslide"))
            elif name == "compare_baige":
                img = b64_of("baige_landslide")
                payload = {"image_t1": img, "image_t2": img}
            raw = get(path) if method == "GET" else post(path, payload)
            summary, imgs = digest(raw)
            out["cases"][name] = {"path": path, "summary": summary, "images": imgs}
            n_img = len(imgs)
            print(f"  ✓ {name:<28} JSON键={len(summary) if isinstance(summary, dict) else '-'} 图像={n_img}")
        except Exception as e:
            out["cases"][name] = {"path": path, "error": str(e)}
            print(f"  ✗ {name:<28} {e}")
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\n基线已写入: {OUT}")
    return out


def run_check():
    if not os.path.exists(OUT):
        print(f"未找到基线文件: {OUT}\n请先不带 --check 运行一次。")
        return 1
    base = json.load(open(OUT, encoding="utf-8"))
    print(f"与基线比对 (基线抓取于 {base['when']})\n")
    now = {"cases": {}}
    for name, method, path, payload in CASES:
        if name == "detect_upload_baige_base":
            payload = dict(payload, image=b64_of("baige_landslide"))
        elif name == "compare_baige":
            img = b64_of("baige_landslide")
            payload = {"image_t1": img, "image_t2": img}
        try:
            raw = get(path) if method == "GET" else post(path, payload)
            summary, imgs = digest(raw)
            now["cases"][name] = {"summary": summary, "images": imgs}
        except Exception as e:
            now["cases"][name] = {"error": str(e)}

    bad = 0
    for name in base["cases"]:
        b, n = base["cases"].get(name, {}), now["cases"].get(name, {})
        if "error" in n:
            print(f"  ✗ {name:<28} 请求失败: {n['error']}")
            bad += 1
            continue
        # 1) 图像逐字节比对
        bi, ni = b.get("images", {}), n.get("images", {})
        img_diff = [k for k in bi if k in ni and bi[k]["md5"] != ni[k]["md5"]]
        if img_diff:
            print(f"  ⚠ {name:<28} {len(img_diff)} 张图像发生变化: {img_diff[:3]}")
            bad += 1
        # 2) 数值字段比对 (新键允许增加, 旧键不允许改)
        bs, ns = b.get("summary", {}), n.get("summary", {})
        changed = []
        for k, v in bs.items():
            if isinstance(v, (int, float, str)) and ns.get(k) != v:
                changed.append(f"{k}: {v} -> {ns.get(k)}")
        if changed:
            print(f"  ⚠ {name:<28} 字段变化: {changed[:4]}")
            bad += 1
        if not img_diff and not changed:
            print(f"  ✓ {name:<28} 与基线一致")
    print(f"\n{'✅ 全部一致, 无回归' if bad == 0 else f'⚠ {bad} 处差异, 请逐条确认是否有意为之'}")
    return 0 if bad == 0 else 2


if __name__ == "__main__":
    if "--check" in sys.argv:
        sys.exit(run_check())
    run_capture()
