# -*- coding: utf-8 -*-
"""
eval_agent_trajectory.py — 多智能体推理轨迹双层评估
======================================================
对标 Earth-Agent (ICLR 2026) 的双层评估协议, 适配到矿山监测多智能体系统:
  · 轨迹级 (step-by-step): Any-Order / In-Order / Step-Wise(严格前缀) / Exact-Match
    + 禁忌智能体违规 (预处理链/匹配类不应出现)
  · 结果级 (end-to-end): 报告要点命中率 / 风险等级一致性 / 效率(步数比) / 耗时

用法:
  python eval_agent_trajectory.py            # 跑全部 20 场景 (真实 API, ~20-30 分钟)
  python eval_agent_trajectory.py --history  # 仅评估历史 trace (results/ 目录) 作为对照
输出: experiments/agent_eval/ 下 results.json + 报告.md
"""
import os
import re
import sys
import json
import time
import base64
import argparse
import urllib.request
from datetime import datetime

import numpy as np
import cv2

sys.stdout.reconfigure(encoding="utf-8")
# 端口可用环境变量覆盖 —— 便于在隔离实例上做 A/B 回归而不打扰 live 服务,
# 也便于用不同 KAT_OUT 保留每一轮结果(不覆盖既有基线)。
API = os.environ.get("KAT_API", "http://localhost:5100/api")
OUT = os.environ.get("KAT_OUT", r"D:\LMHLD\experiments\agent_eval")
HISTORY = r"D:\GeoMMAgent\results"          # 历史 trace 目录
P_IMG = r"D:\竞赛材料\真实场景验证\影像"

# ── 参考标注 (对标 Earth-Bench 的 reference steps) ──
EXPECTED_SEQ = ["SegAgent", "DetAgent", "ClsAgent", "ReasoningAgent"]   # 检测评估型规范序列
REQUIRED_AGENTS = {"ReasoningAgent"}          # 必须包含
FORBIDDEN_AGENTS = {"FormatConversionAgent", "ImageFilterAgent", "ScaleAgent",
                    "SuperResolutionAgent", "MatchingAgent"}            # 禁忌(无关步骤)

# 报告要点检查
RE_REPORT = {
    "风险术语": re.compile(r"预警|风险等级|蓝|黄|橙|红"),
    "定量数字": re.compile(r"\d+(\.\d+)?\s*%|\d+\s*像素|\d+\s*处"),
    "处置建议": re.compile(r"建议|措施|处置|撤离|监测"),
    "字母残留": re.compile(r"^\s*[A-E]\s*$", re.M),                 # 禁止: 单字母残留
    "错误残留": re.compile(r"Agent error|Error code|Traceback"),     # 禁止: 错误文本
}
DRONE = ["无人机", "监测"]


def post(path, payload, timeout=300):
    req = urllib.request.Request(API + path,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def b64_of(name):
    path = os.path.join(P_IMG, name if name.endswith(".png") else name + ".png")
    img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    return base64.b64encode(cv2.imencode(".png", img)[1].tobytes()).decode()


# ── 轨迹指标 (协议复刻自 Earth-Agent/evaluate/step_by_step.py) ──
def seq_any_order(actual, expected):
    if not expected:
        return 1.0
    return len(set(expected) & set(actual)) / len(set(expected))


def seq_in_order(actual, expected):
    """贪心顺序匹配(允许中间夹杂其他)"""
    if not expected:
        return 1.0
    matched, it = 0, iter(actual)
    for e in expected:
        for a in it:
            if a == e:
                matched += 1
                break
        else:
            break
    return matched / len(expected)


def seq_step_wise(actual, expected):
    """严格前缀匹配(首个不匹配即停)"""
    if not expected:
        return 0.0
    correct = 0
    for e, a in zip(expected, actual):
        if e == a:
            correct += 1
        else:
            break
    return correct / len(expected)


def seq_exact_match(actual, expected):
    return 1.0 if list(actual) == list(expected) else 0.0


def report_score(answer, det_risk_name=""):
    """报告要点命中率 + 等级一致性"""
    if not answer:
        return {"要点命中率": 0.0, "等级一致": 0, "违规": ["空报告"]}
    got, viol = {}, []
    checks = ["风险术语", "定量数字", "处置建议"]
    for k in checks:
        got[k] = bool(RE_REPORT[k].search(answer))
    if RE_REPORT["字母残留"].search(answer.strip()):
        viol.append("字母选项残留")
    if RE_REPORT["错误残留"].search(answer):
        viol.append("错误文本残留")
    level_ok = int(any(t in answer for t in re.findall(r"[蓝黄橙红]色", det_risk_name))
                   if det_risk_name else 0)
    return {"要点命中率": round(sum(got.values()) / len(checks), 3),
            "等级一致": level_ok, "违规": viol, "明细": got}


# ── 场景定义 ──
def build_scenarios():
    sc = []
    # 数据集模式: 6 区域 × 2 样本
    for region, tag in [("Emilia-Romagna_Italy_32", "意大利"), ("Gorkha_Nepal_32", "尼泊尔"),
                        ("Hokkaido_Japan_128", "日本"), ("Jiuzhaigou_China_128", "九寨沟"),
                        ("Wenchuan_China_224", "汶川"), ("Taiwan_China_224", "台湾")]:
        for idx in (0, 2):
            sc.append({"id": f"ds_{tag}_{idx}", "kind": "dataset", "type": "检测评估",
                       "payload": {"region": region, "sample_index": idx, "num": 4},
                       "endpoint": "/pipeline"})
    # 上传模式
    for scene, model, tag in [("baige_landslide", "base", "白格-基座"),
                              ("baige_landslide", "baige", "白格-适配"),
                              ("baige_landslide", "unified", "白格-统一"),
                              ("fushun_west_mine", "base", "抚顺")]:
        sc.append({"id": f"up_{tag}", "kind": "upload", "type": "检测评估",
                   "payload": {"image": b64_of(scene), "model_tag": model},
                   "endpoint": "/pipeline_upload"})
    # 变化对比
    sc.append({"id": "cmp_baige", "kind": "compare", "type": "变化对比",
               "payload": {"image_t1": b64_of("baige_landslide"),
                           "image_t2": b64_of("baige_landslide")},
               "endpoint": "/pipeline_compare"})
    # 边界场景
    sc.append({"id": "edge_tiny", "kind": "upload", "type": "检测评估",
               "payload": {"image": b64_of("fushun_pit_z16"), "model_tag": "base"},
               "endpoint": "/pipeline_upload"})
    sc.append({"id": "edge_cmp_same", "kind": "compare", "type": "变化对比",
               "payload": {"image_t1": b64_of("fushun_west_mine"),
                           "image_t2": b64_of("fushun_west_mine")},
               "endpoint": "/pipeline_compare"})
    return sc


# ── 运行与评估 ──
def run_scenario(s):
    t0 = time.time()
    try:
        r = post(s["endpoint"], s["payload"])
    except Exception as e:
        return {"id": s["id"], "error": str(e), "elapsed": round(time.time() - t0, 1)}
    elapsed = round(time.time() - t0, 1)
    agents = [a["name"] for a in r.get("agents", [])]
    det = r.get("detection", {})
    risk = det.get("risk_name", "")
    ans = r.get("final_answer", "")
    quality = r.get("quality") or {}
    m = {
        "id": s["id"], "type": s["type"], "elapsed": elapsed,
        "guard_pass": quality.get("guard_pass"),
        "repaired": quality.get("repaired"),
        "degraded": quality.get("degraded"),
        "self_eval_pass": (quality.get("self_eval") or {}).get("overall_pass"),
        "is_mock": r.get("is_mock"), "agents": agents, "risk": risk,
        "any_order": round(seq_any_order(agents, EXPECTED_SEQ), 3),
        "in_order": round(seq_in_order(agents, EXPECTED_SEQ), 3),
        "step_wise": round(seq_step_wise(agents, EXPECTED_SEQ), 3),
        "exact_match": seq_exact_match(agents, EXPECTED_SEQ),
        "forbidden": sorted(set(agents) & FORBIDDEN_AGENTS),
        "n_steps": len(agents),
        "report": report_score(ans, risk),
    }
    return m


def eval_history():
    """评估历史 trace (results/ 下 run.json) 作为改造前对照"""
    rows = []
    if not os.path.isdir(HISTORY):
        return rows
    for d in sorted(os.listdir(HISTORY)):
        f = os.path.join(HISTORY, d, "trace_full.json")
        if not os.path.exists(f):
            continue
        try:
            r = json.load(open(f, encoding="utf-8"))
        except Exception:
            continue
        agents = [s.get("agent_name", "") for s in r.get("agent_steps", [])]
        agents = [a for a in agents if a]
        if not agents:
            continue
        rows.append({
            "id": d, "agents": agents, "n_steps": len(agents),
            "any_order": round(seq_any_order(agents, EXPECTED_SEQ), 3),
            "in_order": round(seq_in_order(agents, EXPECTED_SEQ), 3),
            "step_wise": round(seq_step_wise(agents, EXPECTED_SEQ), 3),
            "exact_match": seq_exact_match(agents, EXPECTED_SEQ),
            "forbidden": sorted(set(agents) & FORBIDDEN_AGENTS),
            "report": report_score(r.get("final_answer", "")),
        })
    return rows


def summarize(rows, title):
    ok = [r for r in rows if "error" not in r]
    n = len(ok) or 1
    agg = {
        "场景数": len(rows), "成功": len(ok),
        "Any-Order": round(sum(r["any_order"] for r in ok) / n, 3),
        "In-Order": round(sum(r["in_order"] for r in ok) / n, 3),
        "Step-Wise": round(sum(r["step_wise"] for r in ok) / n, 3),
        "Exact-Match": round(sum(r["exact_match"] for r in ok) / n, 3),
        "禁忌违规率": round(sum(1 for r in ok if r["forbidden"]) / n, 3),
        "报告要点命中": round(sum(r["report"]["要点命中率"] for r in ok) / n, 3),
        "平均步数": round(sum(r["n_steps"] for r in ok) / n, 2),
        "守卫通过率": round(sum(1 for r in ok if r.get("guard_pass")) / n, 3),
        "降级率": round(sum(1 for r in ok if r.get("degraded")) / n, 3),
        "自评估通过率": round(sum(1 for r in ok if r.get("self_eval_pass")) / n, 3),
    }
    if ok and "elapsed" in ok[0]:
        agg["平均耗时s"] = round(sum(r["elapsed"] for r in ok) / n, 1)
    print(f"\n═══ {title} ═══")
    for k, v in agg.items():
        print(f"  {k}: {v}")
    return agg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--history", action="store_true", help="仅评估历史 trace")
    ap.add_argument("--only", nargs="*", default=None,
                    help="只跑指定场景 id (断点续测用, 如 --only ds_日本_2 edge_cmp_same)")
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)

    hist = eval_history()
    if hist:
        summarize(hist, "历史轨迹 (改造前基线)")

    if not args.history:
        sc = build_scenarios()
        if args.only:
            sc = [s for s in sc if s["id"] in set(args.only)]
            print(f"仅运行指定 {len(sc)} 个场景: {[s['id'] for s in sc]}")
        print(f"运行 {len(sc)} 个场景 (真实多智能体调用)...")
        rows = []
        for i, s in enumerate(sc):
            m = run_scenario(s)
            rows.append(m)
            st = m.get("agents", m.get("error", "?"))
            print(f"  [{i+1}/{len(sc)}] {m['id']:<18} {m.get('elapsed','-')}s "
                  f"agents={st if isinstance(st, str) else len(st)}")
        agg = summarize(rows, "改造后 (白名单+定量工具)")
        out = {"when": datetime.now().isoformat(), "scenarios": rows,
               "summary": agg, "history": hist}
        with open(os.path.join(OUT, "results.json"), "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
        print(f"\n结果: {os.path.join(OUT, 'results.json')}")
    else:
        with open(os.path.join(OUT, "history_eval.json"), "w", encoding="utf-8") as f:
            json.dump(hist, f, ensure_ascii=False, indent=2)
        print(f"\n历史评估: {os.path.join(OUT, 'history_eval.json')}")


if __name__ == "__main__":
    main()
