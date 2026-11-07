# -*- coding: utf-8 -*-
"""
sync_frontend.py — 前端演示页三份副本同步
==========================================
前端有同一个文件的三份副本(规范源 + 两份分发副本), 手工维护极易漂移:

  规范源:   D:\\GeoMMAgent\\矿安天眼_前端演示(后端版).html
  副本 1:   D:\\GeoMMAgent\\kat_demo.html
  副本 2:   桌面\\矿安天眼-前端演示(打开这个).html

只改规范源, 然后运行本脚本分发。改完务必核对三份 MD5 一致。

用法:
  python sync_frontend.py --check     # 只检查是否一致(退出码 1 表示不一致)
  python sync_frontend.py             # 从规范源同步到两份副本
  python sync_frontend.py --to-src     # 反向: 若只有副本被改过, 用它回填规范源
  python sync_frontend.py --workspace-only  # 只同步本仓库副本
"""
import os
import sys
import hashlib
import shutil
import argparse

sys.stdout.reconfigure(encoding="utf-8")

BASE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(BASE, "矿安天眼_前端演示(后端版).html")
COPIES = [
    os.path.join(BASE, "kat_demo.html"),
    os.path.join(os.path.expanduser("~"), "Desktop", "矿安天眼-前端演示(打开这个).html"),
]


def md5(path):
    if not os.path.exists(path):
        return None
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="只检查一致性")
    ap.add_argument("--to-src", action="store_true", help="由副本回填规范源")
    ap.add_argument("--workspace-only", action="store_true", help="只同步仓库内副本，不访问桌面路径")
    args = ap.parse_args()
    copies = COPIES[:1] if args.workspace_only else COPIES

    paths = [SRC] + copies
    hashes = {p: md5(p) for p in paths}

    print(f"{len(paths)}份副本指纹:")
    for p in paths:
        h = hashes[p]
        tag = "规范源" if p == SRC else "副本  "
        print(f"  [{tag}] {h or '(缺失)':<34} {os.path.basename(p)}")

    present = {p: h for p, h in hashes.items() if h}
    if not present:
        print("✗ 找不到任何前端文件")
        return 1

    if args.check:
        if len(set(present.values())) == 1 and len(present) == len(paths):
            print(f"\n✅ {len(paths)}份一致")
            return 0
        print("\n⚠ 不一致(或缺失), 请运行 python sync_frontend.py 同步")
        return 1

    # 选定源
    if args.to_src:
        src_path = max(present, key=lambda p: os.path.getmtime(p))
        if src_path == SRC:
            print("\n最近修改的已是规范源, 无需回填")
            return 0
        print(f"\n由最近修改的副本回填规范源: {os.path.basename(src_path)}")
        shutil.copyfile(src_path, SRC)
    else:
        if not hashes.get(SRC):
            print("\n✗ 规范源缺失, 无法同步")
            return 1
        src_path = SRC

    if hashes.get(SRC) == hashes.get(src_path) and all(
            hashes.get(c) == hashes.get(SRC) for c in copies):
        print("\n✅ 已一致, 无需同步")
        return 0

    print(f"\n从 [{os.path.basename(src_path)}] 分发到 {len(copies)} 份副本:")
    for c in copies:
        d = os.path.dirname(c)
        if not os.path.isdir(d):
            print(f"  ⚠ 跳过(目录不存在): {c}")
            continue
        shutil.copyfile(src_path, c)
        print(f"  ✓ {c}")

    after = {p: md5(p) for p in paths}
    ok = len(set(v for v in after.values() if v)) == 1 and all(after.values())
    print(f"\n{'✅ 同步完成, 指纹一致' if ok else '⚠ 同步后仍不一致, 请检查'}")
    for p in paths:
        print(f"  {after[p]}  {os.path.basename(p)}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
