# -*- coding: utf-8 -*-
"""
check_services.py — 演示前的服务体检（只诊断，不启动任何东西）
================================================================
演示日出问题时，先跑这个，一眼看出卡在哪一层。
**只读**：不启动、不重启、不改配置，随时可以安全运行。

用法：python check_services.py

覆盖的检查层（从下往上，任一层失败都能定位）：
  ① 端口      5100 后端 / 5101 检测服务 是否在监听
  ② 健康      两个 /api/health 是否返回 200 且字段正常
  ③ 依赖      检测侧权重文件是否齐（UPLOAD_MODELS / WEIGHTS 里每一条）
  ④ 模型      后端报告的 detector_loaded / geomm_loaded
  ⑤ 配置      各上传通道的阈值
  ⑥ 硬件      GPU 显存（检测服务常驻约 4GB；剩太少会 OOM）
  ⑦ 网络      Qwen API 是否配置（报告生成依赖，仅提示不判失败）
"""
import json
import os
import shutil
import subprocess
import sys
import urllib.error
import urllib.request

sys.stdout.reconfigure(encoding="utf-8")

BACKEND = "http://localhost:5100"
DETECTOR = "http://localhost:5101"
ROOT = r"D:\LMHLD"
AGENT = r"D:\GeoMMAgent"

OK, BAD, WARN = "[OK]  ", "[FAIL]", "[WARN]"
problems = []


def hit(url, timeout=6):
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return r.status, json.loads(r.read().decode())
    except urllib.error.HTTPError as e:
        return e.code, None
    except Exception as e:
        return None, str(e)


def head(title):
    print(f"\n{'=' * 68}\n{title}\n{'=' * 68}")


def main():
    print("矿安天眼 · 演示前服务体检  （只诊断，不启动任何东西）")

    # ① 端口
    head("① 端口监听")
    listening = set()
    try:
        out = subprocess.run(["netstat", "-ano"], capture_output=True, text=True,
                             timeout=20).stdout
        for line in out.splitlines():
            if "LISTENING" in line:
                for p in ("5100", "5101"):
                    if f":{p} " in line or line.rstrip().endswith(p):
                        listening.add(p)
    except Exception as e:
        print(f"  {WARN} netstat 调用失败: {e}")
    for p, name in (("5100", "后端(智能体)"), ("5101", "检测服务(GPU)")):
        if p in listening:
            print(f"  {OK} {p} {name} 在监听")
        else:
            print(f"  {BAD} {p} {name} **未监听**")
            problems.append(f"{p} ({name}) 未启动")

    # ② 健康
    head("② 健康检查")
    for url, name in ((f"{BACKEND}/api/health", "后端 5100"),
                      (f"{DETECTOR}/api/health", "检测 5101")):
        st, body = hit(url)
        if st == 200 and isinstance(body, dict):
            extra = ""
            if name.startswith("后端"):
                extra = (f"  detector_loaded={body.get('detector_loaded')} "
                         f"geomm_loaded={body.get('geomm_loaded')}")
            else:
                extra = f"  detector_loaded={body.get('detector_loaded')}"
            print(f"  {OK} {name}: 200{extra}")
            if name.startswith("后端") and not body.get("detector_loaded"):
                problems.append("后端报告 detector_loaded=False（检测服务没连上）")
        else:
            print(f"  {BAD} {name}: {st if st else body}")
            problems.append(f"{name} 健康检查失败")

    # ③ 权重文件
    head("③ 权重文件")
    try:
        sys.path.insert(0, ROOT)
        from detector_service import UPLOAD_MODELS, UPLOAD_THRESHOLDS
        for k, v in UPLOAD_MODELS.items():
            p = v[0]
            if os.path.exists(p):
                sz = os.path.getsize(p) / 1e6
                print(f"  {OK} 上传通道 {k:<9} 阈值={UPLOAD_THRESHOLDS.get(k, 0.5):<5} "
                      f"{sz:.0f}MB")
            else:
                print(f"  {BAD} 上传通道 {k:<9} 权重缺失: {p}")
                problems.append(f"通道 {k} 权重缺失")
        from detector_service import WEIGHTS
        for k, v in WEIGHTS.items():
            p = v[0]
            print(f"  {OK if os.path.exists(p) else BAD} 区域模型 {k:<22} "
                  f"{'就绪' if os.path.exists(p) else '缺失: ' + p}")
            if not os.path.exists(p):
                problems.append(f"区域模型 {k} 权重缺失")
    except Exception as e:
        print(f"  {WARN} 无法读取服务配置（需在能 import detector_service 的环境跑）: {e}")

    # ④ 硬件
    head("④ GPU 显存")
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.total,memory.used,memory.free",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=15).stdout.strip()
        tot, used, free = [int(x) for x in out.split(",")]
        pct = free / tot * 100
        mark = OK if free > 2500 else (WARN if free > 1200 else BAD)
        print(f"  {mark} 总 {tot}MB / 已用 {used}MB / **空闲 {free}MB**（{pct:.0f}%）")
        if free <= 2500:
            print(f"       检测服务常驻约 4GB；大图（2304²）推理需 ≥2.5GB 空闲。")
            print(f"       若即将演示，先关掉占显存的其它程序（训练/浏览器硬件加速等）。")
        if free <= 1200:
            problems.append(f"GPU 空闲显存仅 {free}MB，大图推理会 OOM")
    except Exception as e:
        print(f"  {WARN} 读不到 GPU 信息: {e}")

    # ④b 谁在占显存 —— 本机是 8GB 笔记本显卡，后台桌面程序的硬件加速会吃掉好几个 GB
    try:
        out = subprocess.run(["nvidia-smi"], capture_output=True, text=True,
                             timeout=20).stdout
        block = out.split("Processes")[-1] if "Processes" in out else ""
        users = []
        for line in block.splitlines():
            parts = line.split()
            if "C+G" not in parts and "C" not in parts:
                continue
            # 行格式: | 0 N/A N/A <pid> C+G <路径\进程名> N/A |
            if len(parts) < 3:
                continue
            raw = parts[-3]                       # 倒数第 3 项是进程路径
            exe = raw.split("\\")[-1].strip()     # 只留进程名
            if not exe.lower().endswith(".exe"):
                continue
            users.append(("python" in exe.lower(), exe))
        gfx = [t for is_py, t in users if not is_py]
        gfx = sorted(set(gfx))
        if gfx:
            print(f"  {WARN} 另有 {len(gfx)} 个后台程序在用显卡的图形加速（各占一两百 MB）:")
            for t in gfx[:14]:
                print(f"       · {t}")
            if len(gfx) > 14:
                print(f"       · ...（还有 {len(gfx) - 14} 个）")
            print("       **演示前建议关掉**：微信 / 浏览器 / VS Code / Zotero 等")
            print("       （它们的硬件加速会与检测服务争显存；关掉即可，不影响本项目）")
    except Exception:
        pass

    # ⑤ 网络 / LLM
    head("⑤ 报告生成依赖（Qwen API）")
    env = os.path.join(AGENT, ".env")
    if os.path.exists(env):
        txt = open(env, encoding="utf-8", errors="ignore").read()
        has = "DASHSCOPE_API_KEY" in txt or "QWEN" in txt.upper()
        print(f"  {OK if has else WARN} {env} 存在"
              f"{'，且含密钥字段' if has else '，但未见密钥字段'}")
    else:
        print(f"  {WARN} 未找到 {env} —— 检测与报告模板仍可用，"
              f"但**多智能体报告生成会失败**")

    # 结论
    head("结论")
    if not problems:
        print(f"  {OK} 一切就绪，可以演示。")
        print("       提示：检测服务是单线程（threaded=False），连续点击会排队，属正常。")
        return 0
    print(f"  {BAD} 发现 {len(problems)} 个问题：")
    for p in problems:
        print(f"      · {p}")
    print("\n  处置办法（按顺序试）：")
    print("   1. 双击  D:\\GeoMMAgent\\start_backend_5100.bat")
    print("      （必须**双击**。从 git bash / PowerShell 脚本里调用会因编码与引号问题失败：")
    print("       bat 的注释会被 cmd 按 GBK 误当命令执行，报 'xxx 不是内部或外部命令'。）")
    print("   2. 等 30~60 秒（后端要加载 RAG 知识库与检测模型），再跑一次本脚本。")
    print("   3. 仍不行：看桌面弹出的 'KAT-Backend' 控制台窗口里的报错。")
    print("   4. 检测服务 5101 未起来时，后端启动日志里会有 '滑坡检测服务' 相关提示；")
    print("      也可直接手动跑： D:\\Miniconda3\\envs\\lmhld\\python.exe D:\\LMHLD\\detector_service.py 5101")
    return 1


if __name__ == "__main__":
    sys.exit(main())
