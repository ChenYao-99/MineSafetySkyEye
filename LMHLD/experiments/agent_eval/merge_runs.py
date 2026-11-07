# -*- coding: utf-8 -*-
"""
merge_runs.py — 合并两次评测结果（断点续测后拼接）
====================================================
场景: 首轮 19 个场景中因运行环境故障(实例崩溃)失败若干, 用 --only 补跑后
      把补跑结果回填到首轮, 得到完整结果集, 再统一重算汇总。

用法: python merge_runs.py <首轮 results.json> <补跑 results.json> [输出路径]
"""
import os
import sys
import json
from datetime import datetime

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, r"D:\LMHLD")
from eval_agent_trajectory import summarize   # noqa: E402


def main():
    a_path, b_path = sys.argv[1], sys.argv[2]
    out_path = sys.argv[3] if len(sys.argv) > 3 else os.path.join(
        os.path.dirname(a_path), "results_merged.json")
    a = json.load(open(a_path, encoding="utf-8"))
    b = json.load(open(b_path, encoding="utf-8"))

    retry = {r["id"]: r for r in b["scenarios"] if "error" not in r}
    merged, replaced = [], []
    for r in a["scenarios"]:
        if "error" in r and r["id"] in retry:
            merged.append(retry[r["id"]])
            replaced.append(r["id"])
        else:
            merged.append(r)

    agg = summarize(merged, "合并后 (首轮 + 补跑)")
    out = {"when": datetime.now().isoformat(),
           "sources": [os.path.basename(a_path), os.path.basename(b_path)],
           "replaced_by_retry": replaced,
           "scenarios": merged, "summary": agg,
           "history": a.get("history", [])}
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\n补跑回填 {len(replaced)} 个: {replaced}")
    print(f"合并结果: {out_path}")

    still = [r["id"] for r in merged if "error" in r]
    if still:
        print(f"⚠ 仍失败 {len(still)} 个: {still}")
    else:
        print(f"✅ 全部 {len(merged)} 个场景成功")


if __name__ == "__main__":
    main()
