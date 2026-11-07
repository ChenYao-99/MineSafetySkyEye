# -*- coding: utf-8 -*-
"""
test_geometry_guard.py — 报告几何数字一致性校验的回归测试
============================================================
被测对象: backend.py 的 check_geometry_grounding / validate_report(几何部分)。

与 test_summary_fault_fix.py 同样的做法: **从 backend.py 源码提取真实函数文本**
执行, 而不是复制一份实现 —— 保证测的就是线上跑的那段代码。

背景: 已存档的报告曾凭空写出"长度分别约为 180 像素、120 像素和 95 像素",
背后没有任何几何计算。几何工具注入后, 报告里的几何数字必须来自工具输出;
越界的数字只记软告警(不进 missing), 以防正则误判触发修复死循环。

用法: python tests/test_geometry_guard.py
"""
import ast
import os
import re as _re
import sys

sys.stdout.reconfigure(encoding="utf-8")
HERE = os.path.dirname(os.path.abspath(__file__))
BACKEND = os.path.join(os.path.dirname(HERE), "backend.py")
OK = True

WANTED = {"_quality_stats", "REPAIR_HINTS", "_bump",
          "_geom_number_in_whitelist", "check_geometry_grounding", "validate_report",
          "_GEOM_PX_RE", "_geom_ctx_pattern"}


def chk(name, cond, extra=""):
    global OK
    print(f"  {'✓' if cond else '✗'} {name} {extra}")
    if not cond:
        OK = False


def load_guard():
    """提取守卫相关的模块级对象与函数, 在隔离命名空间执行。"""
    with open(BACKEND, encoding="utf-8") as f:
        src = f.read()
    tree = ast.parse(src)
    ns = {}
    exec("import re as _re", ns)
    found = set()
    for node in tree.body:
        nm = None
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
            nm = node.targets[0].id
        elif isinstance(node, ast.FunctionDef):
            nm = node.name
        if nm in WANTED:
            exec(compile(ast.get_source_segment(src, node), BACKEND, "exec"), ns)
            found.add(nm)
    missing = WANTED - found
    if missing:
        raise RuntimeError(f"backend.py 中未找到: {sorted(missing)}")
    return ns


def main():
    ns = load_guard()
    check = ns["check_geometry_grounding"]
    validate = ns["validate_report"]

    print("① 无白名单时: 不新增检查项（保持向后兼容）")
    a = ("检测到橙色-高风险区域，面积占比 12.5%，共 3 像素级区域。" * 12)
    miss = validate(a, "橙色-高风险")
    chk("不出现『几何数字引用』", "几何数字引用" not in miss, f"missing={miss}")

    print("② 有白名单: 引用真实几何 → 通过")
    wl = {"937", "179.3", "54.94", "12", "3", "220"}
    a2 = ("该滑坡体最大实例面积 937 像素，周长 179.3 像素，长轴约 54.94 像素，"
          "共 3 处区域，面积占比 12%。建议加强监测并制定撤离方案。" * 8)
    miss2 = validate(a2, "橙色-高风险", wl)
    chk("不缺失『几何数字引用』", "几何数字引用" not in miss2, f"missing={miss2}")
    cited, bad = check(a2, wl)
    chk("判定为已引用", cited is True)
    chk("无越界告警", bad == [], f"bad={bad}")

    print("③ 有白名单但完全没引用几何 → 判缺失")
    a3 = ("检测到风险区域，建议加强监测与巡查，制定应急预案，确保人员安全。" * 10)
    miss3 = validate(a3, "橙色-高风险", wl)
    chk("缺失『几何数字引用』", "几何数字引用" in miss3, f"missing={miss3}")

    print("④ 明显编造尺寸 → 软告警（不进 missing）")
    a4 = ("该滑坡体最大长度约为 500 像素，宽度约 95 像素，面积 937 像素。"
          "建议加强监测并制定撤离方案。" * 8)
    miss4 = validate(a4, "橙色-高风险", wl)
    cited4, bad4 = check(a4, wl)
    chk("引用判定为真（含白名单数字 937）", cited4 is True)
    chk("越界片段被抓出（500/95，真实长轴仅 54.94）", len(bad4) >= 2, f"bad={bad4}")
    chk("越界不进入 missing(防修复死循环)", "几何数字越界" not in miss4, f"missing={miss4}")

    print("④b 能力边界（如实记录，非缺陷）: 贴着真值的杜撰无法识别")
    # 真实周长 179.3px, 模型写"约 180 像素" —— 落在容差内, 判为合法四舍五入。
    # 这正是已存档报告中那句"长度约为 180 像素"的形态: 单项判据抓不到它,
    # 所以几何检查只能作软告警, 真正的防线是"把工具输出注入提示词"本身。
    _c, bad_near = check("最大长度约为 180 像素。", wl)
    chk("180 与真实 179.3 相差 0.4%, 按容差视为合法", bad_near == [], f"bad={bad_near}")

    print("④c 实测误报回归: 比率与倍数表述不得判为几何越界")
    _c, bad_pct = check("该区域面积占比为 3.1%，约为最大区域面积的7倍。", wl)
    chk("『面积占比』『面积的7倍』均不误报", bad_pct == [], f"bad={bad_pct}")
    _c2, bad_m = check("建议撤离 500 米范围内的作业人员，24 小时内完成排查。", wl)
    chk("米制/时间表述不误报(仅像素单位受检)", bad_m == [], f"bad={bad_m}")

    print("⑤ ±5% 容差: 四舍五入的引用不算越界")
    _, bad5 = check("最大长度约 55 像素，周长 179 像素。", wl)
    chk("54.94→55 容差内", bad5 == [], f"bad={bad5}")
    _, bad6 = check("最大长度约 70 像素。", wl)
    chk("70 明显越界被抓出", len(bad6) == 1, f"bad={bad6}")

    print("⑥ 数字边界: 白名单含 12 时不应把 123 判为命中")
    wl7 = {"12"}
    _, bad7 = check("外接框长度 123 像素。", wl7)
    chk("123 不被 12 命中", len(bad7) == 1, f"bad={bad7}")
    cited7, bad8 = check("共 12 处区域。", wl7)
    chk("独立的 12 正常命中", cited7 is True and bad8 == [], f"bad={bad8}")

    print("⑦ 退化输入不抛异常")
    for nm, args in [("空答案", ("", wl)), ("None答案", (None, wl)),
                     ("空白名单", ("长度 999 像素", set())),
                     ("None白名单", ("长度 999 像素", None))]:
        try:
            check(*args)
            validate(args[0] or "", "橙色-高风险", args[1])
            chk(f"不抛异常[{nm}]", True)
        except Exception as e:
            chk(f"不抛异常[{nm}]", False, repr(e))

    print("⑧ 原有 6 项检查未被削弱")
    base = ("检测到橙色-高风险，面积占比 12.5%，建议立即撤离并加强监测。" * 10)
    miss8 = validate(base, "橙色-高风险")
    chk("合格的纯文本报告仍通过", miss8 == [], f"missing={miss8}")
    chk("空报告仍被拦", validate("", "橙色-高风险") == ["空报告或过短"])
    chk("字母残留仍被拦", "无字母残留" in validate(
        ("A" + chr(10) + base), "橙色-高风险"))
    chk("错误残留仍被拦", "无错误残留" in validate(
        ("Agent error: boom" + base), "橙色-高风险"))
    chk("等级不一致仍被拦", "风险等级一致" in validate(base, "蓝色-低风险"))

    print(f"\n{'✅ 全部通过' if OK else '❌ 存在失败项'}")
    return 0 if OK else 1


if __name__ == "__main__":
    sys.exit(main())
