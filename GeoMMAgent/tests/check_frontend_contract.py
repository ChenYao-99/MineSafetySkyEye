# -*- coding: utf-8 -*-
"""
check_frontend_contract.py — 前端演示页静态契约检查
=====================================================
重排 HTML 时最容易踩的坑不是语法错, 而是**引用错位**: 改了 id 却漏改某处
getElementById、删了函数却留着 onclick —— 这类错误浏览器只在运行时静默失败或
在控制台抛一个没人看的 null, 演示时才炸。

本脚本把这份契约静态查一遍。**只读**, 不修改任何文件。

检查项:
  1. getElementById("x") 的每个字面量 x  →  必须有 id="x"        [FAIL]
  2. 内联 on*="f(...)" 引用的每个函数 f  →  必须有定义            [FAIL]
     (含 <script> 内 innerHTML 模板串里的 onclick, 如 runUploadPipeline())
  3. querySelector/All("#x ...") 的 #x   →  必须有 id="x"        [FAIL]
  4. 同一 id 出现多次                                              [FAIL]
  5. 定义了但全文件无人调用的函数                                   [WARN]
  6. 有 id 但全文件无人引用                                        [WARN]

用法:
  python check_frontend_contract.py                    # 查规范源
  python check_frontend_contract.py <path.html>        # 查指定文件
"""
import os
import re
import sys
from collections import Counter

sys.stdout.reconfigure(encoding="utf-8")

DEFAULT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "矿安天眼_前端演示(后端版).html")

# JS 里不算"函数定义"的关键字, 避免把 if(/for(/while( 误判成调用
KEYWORDS = {
    "if", "for", "while", "switch", "catch", "return", "typeof", "new", "delete",
    "void", "in", "of", "do", "else", "function", "await", "yield", "throw", "case",
}

# 内联 handler 里可能直接调用的浏览器内置函数 —— 不属于"未定义"
BUILTINS = {
    "alert", "confirm", "prompt", "print", "open", "close", "fetch", "parseInt",
    "parseFloat", "isNaN", "encodeURIComponent", "decodeURIComponent", "atob", "btoa",
    "setTimeout", "setInterval", "clearTimeout", "clearInterval", "URL",
}


def extract(src):
    """拆出 <style> / <script> / 其余 HTML 三段"""
    style = "\n".join(m.group(1) for m in re.finditer(r"<style[^>]*>(.*?)</style>", src, re.S))
    script = "\n".join(m.group(1) for m in re.finditer(r"<script(?![^>]*\bsrc=)[^>]*>(.*?)</script>", src, re.S))
    html = re.sub(r"<style[^>]*>.*?</style>", "", src, flags=re.S)
    html = re.sub(r"<script(?![^>]*\bsrc=)[^>]*>.*?</script>", "", html, flags=re.S)
    return style, script, html


def find_ids(text):
    """所有 id="..." / id='...'（静态 HTML 与 JS 模板串一并扫）"""
    return re.findall(r"""\bid\s*=\s*["']([^"']+)["']""", text)


def find_getelementbyid(script):
    return re.findall(r"""getElementById\(\s*["']([^"']+)["']\s*\)""", script)


def find_hash_selectors(script):
    """querySelector("#a .b") → ['a']"""
    out = []
    for m in re.finditer(r"""querySelector(?:All)?\(\s*["']([^"']+)["']""", script):
        out.extend(re.findall(r"#([A-Za-z_][\w-]*)", m.group(1)))
    return out


def find_inline_handlers(*texts):
    """HTML 属性与 JS 模板串里的 on*="f(...)" → 函数名列表"""
    names = []
    pat = re.compile(r"""\bon(?:click|change|input|submit|load|error|mouseover)\s*=\s*["']([^"']*)["']""")
    for text in texts:
        for m in pat.finditer(text):
            for fn in re.findall(r"([A-Za-z_$][\w$]*)\s*\(", m.group(1)):
                names.append(fn)
    return names


def find_defs(script):
    """函数定义名 → 定义形式"""
    defs = {}
    for m in re.finditer(r"\bfunction\s+([A-Za-z_$][\w$]*)\s*\(", script):
        defs[m.group(1)] = "function"
    # const f = () => / const f = async () => / const f = function
    for m in re.finditer(
            r"\b(?:const|let|var)\s+([A-Za-z_$][\w$]*)\s*=\s*(?:async\s*)?(?:function\b|\([^)]*\)\s*=>|[A-Za-z_$][\w$]*\s*=>)",
            script):
        defs.setdefault(m.group(1), "arrow")
    return defs


def check_report_slots(script, id_set):
    """动态 id 约定检查 —— 本页报告块的 id 是 rp(slot, part) 拼出来的模板串,
    上面那套字面量检查**看不到它们**, 而这里恰恰是改动风险最集中的地方。

    约定:
      rp(slot, part)  →  id="rp-{slot}-{part}"
      rp(slot, "")    →  id="rp-{slot}"        （报告块本体）

    slot 取值只可能来自字面量调用 rp("upload", ...)（变量形式 rp(slot,...) 的 slot
    由调用方传入, 无法静态枚举）; part 取值则从两种形式里都能收。
    """
    fails = []
    lit = re.findall(r"""\brp\(\s*["']([^"']*)["']\s*,\s*["']([^"']*)["']\s*\)""", script)
    any_part = re.findall(r"""\brp\(\s*[^,)]+,\s*["']([^"']+)["']\s*\)""", script)
    if not lit:
        return fails

    slots = sorted({s for s, _ in lit})
    parts = sorted(set(any_part))

    for s in slots:
        if f"rp-{s}" not in id_set:
            fails.append(f'rp("{s}", "") —— 报告块需要 id="rp-{s}"，但 HTML 里没有')
        for p in parts:
            if f"rp-{s}-{p}" not in id_set:
                fails.append(f'rp("{s}", "{p}") —— 需要 id="rp-{s}-{p}"，但 HTML 里没有')

    covered = {f"rp-{s}" for s in slots} | {f"rp-{s}-{p}" for s in slots for p in parts}
    for i in sorted(x for x in id_set if x.startswith("rp-") and x not in covered):
        fails.append(f'id="{i}" 没有任何 rp(...) 调用会用到它（孤儿报告槽）')
    return fails


def check_slot_names(script):
    """slot 名写错时, rp() 会拼出一个不存在的 id → getElementById 返回 null →
    要么静默不渲染, 要么在 .classList 上抛异常。字面量检查完全看不到这类错误
    （它只查 id 与函数的对应关系, 不查字符串参数的值）。

    权威取值来自 REPORT_ACTIONS 的键 —— 那是个字面量对象, 静态可枚举。
    """
    fails = []
    m = re.search(r"const\s+REPORT_ACTIONS\s*=\s*\{(.*?)\n\};", script, re.S)
    if not m:
        return fails
    valid = set(re.findall(r"^\s*(\w+)\s*:", m.group(1), re.M))
    if not valid:
        return fails

    checks = [
        (r"""renderPipelineResult\(\s*[^,)]+,\s*["']([^"']+)["']\s*\)""", "renderPipelineResult(data, slot)"),
        (r"""\bresetReport\(\s*["']([^"']+)["']\s*\)""", "resetReport(slot)"),
        (r"""\bexportReport\(\s*["']([^"']+)["']\s*\)""", "exportReport(slot)"),
        (r"""\bexportVector\(\s*[^,)]+,\s*["']([^"']+)["']\s*\)""", "exportVector(format, slot)"),
    ]
    for pat, desc in checks:
        for got in sorted(set(re.findall(pat, script))):
            if got not in valid:
                fails.append(f'{desc} 传了未知 slot "{got}"'
                             f'（REPORT_ACTIONS 里只有 {sorted(valid)}）')
    return fails


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT
    if not os.path.exists(path):
        print(f"✗ 找不到文件: {path}")
        return 2
    src = open(path, encoding="utf-8").read()
    style, script, html = extract(src)

    all_text = style + "\n" + script + "\n" + html
    ids = find_ids(all_text)
    id_count = Counter(ids)
    id_set = set(ids)

    used = find_getelementbyid(script)
    hash_sel = find_hash_selectors(script)
    handlers = find_inline_handlers(html, script)
    defs = find_defs(script)

    fails, warns = [], []

    # 1. getElementById 目标
    missing = sorted({u for u in used if u not in id_set})
    for u in missing:
        fails.append(f'getElementById("{u}") —— HTML 里没有 id="{u}"')

    # 2. 内联 handler 的函数
    undef = sorted({h for h in handlers if h not in defs and h not in BUILTINS})
    for h in undef:
        fails.append(f'内联 on*="{h}(...)" —— <script> 里没有 {h} 的定义')

    # 3. querySelector 的 #id
    for h in sorted({h for h in hash_sel if h not in id_set}):
        fails.append(f'querySelector("#{h} ...") —— HTML 里没有 id="{h}"')

    # 4. 重复 id
    for i, n in sorted(id_count.items()):
        if n > 1:
            fails.append(f'id="{i}" 出现了 {n} 次（id 必须唯一）')

    # 4b. 动态拼出的报告槽 id（字面量检查看不见，单独按约定核）
    fails.extend(check_report_slots(script, id_set))
    fails.extend(check_slot_names(script))
    rp_covered = {x for x in id_set if x.startswith("rp-")}

    # 5. 定义了但无人引用。
    #    判据是"出现次数 > 定义处的 1 次", 而不是"是否在标识符集合里" ——
    #    后者恒为真（定义本身就算一次出现），会让这条检查彻底失效。
    #    两类合法例外要排除:
    #      · 只从 HTML 内联 handler 调用（调用点在 HTML 里, 脚本内自然只出现 1 次）
    #      · 立即执行函数表达式 (function foo(){...})()  —— 定义即调用, 本就没有调用点
    for name in sorted(defs):
        if name in handlers:
            continue
        if re.search(rf"[(!]\s*function\s+{re.escape(name)}\b", script):
            continue
        if len(re.findall(rf"\b{re.escape(name)}\b", script)) <= 1:
            warns.append(f"{name}() 已定义但全文件无人引用")

    # 6. 有 id 但无人引用。按**带引号的整串**匹配，避免子串误判
    #    （id="cmpT1" 会被 id="cmpT1Prev" 的出现掩盖）。
    # 经 data-target="x" 间接引用的（场景导航：getElementById(p.dataset.target)）
    data_targets = set(re.findall(r"""\bdata-target\s*=\s*["']([^"']+)["']""", html + script))
    for i in sorted(id_set):
        if i in used or i in hash_sel or i in rp_covered or i in data_targets:
            continue
        if f'"{i}"' in script or f"'{i}'" in script:
            continue
        warns.append(f'id="{i}" 未被任何 JS 引用')

    # ── 报告 ──
    print(f"契约检查: {os.path.basename(path)}")
    print(f"  id 共 {len(id_set)} 个 · getElementById 引用 {len(set(used))} 个 · "
          f"querySelector(#) {len(set(hash_sel))} 个")
    print(f"  内联 handler 引用 {len(set(handlers))} 个函数 · 脚本内定义 {len(defs)} 个函数")

    if fails:
        print(f"\n[FAIL] {len(fails)} 项契约断裂:")
        for f in fails:
            print(f"   ✗ {f}")
    if warns:
        print(f"\n[WARN] {len(warns)} 项提示（不一定是问题）:")
        for w in warns:
            print(f"   · {w}")
    if not fails:
        print("\n[OK] 契约完整")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
