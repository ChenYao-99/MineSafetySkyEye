# -*- coding: utf-8 -*-
"""
test_summary_fault_fix.py — CAMEL 摘要故障级联修复的回归测试
==============================================================
被测对象: backend.py 中的 `_install_summary_fault_fix`。

本测试**从 backend.py 源码中提取该函数的原始文本**再执行(ast 解析), 而不是
复制一份实现 —— 保证测的就是真实上线的那段代码, 不会出现"测试过了但线上是
另一个版本"的失真。

故障回顾 (证据见 backend_startup.log):
  ① 摘要触发 → camel 对多模态 list 型 content 做 str() → base64 图像进提示词 → 400
  ② 摘要返回空串 → 上游仍调用 _update_memory_with_summary
  ③ 该方法 isinstance(content, str) 漏判多模态 → 记忆被清空且无 user 角色 → 调度 400

用法: python tests/test_summary_fault_fix.py
"""
import ast
import os
import sys
import types

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

BACKEND = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "backend.py")

OK = True


def chk(name, cond, extra=""):
    global OK
    print(f"  {'✓' if cond else '✗'} {name} {extra}")
    if not cond:
        OK = False


def load_patch_fn():
    """从 backend.py 源码提取 _install_summary_fault_fix 并执行(不 import 整个后端)。"""
    with open(BACKEND, encoding="utf-8") as f:
        src = f.read()
    tree = ast.parse(src)
    node = next((n for n in tree.body
                 if isinstance(n, ast.FunctionDef) and n.name == "_install_summary_fault_fix"),
                None)
    if node is None:
        raise RuntimeError("backend.py 中未找到 _install_summary_fault_fix")
    seg = ast.get_source_segment(src, node)
    ns = {}
    exec(compile(seg, BACKEND, "exec"), ns)
    return ns["_install_summary_fault_fix"]


# ── 伪造一个足够用的 ChatAgent 替身 ─────────────────────────
class FakeMemory:
    def __init__(self, messages):
        self._m = list(messages)
        self.clear_count = 0

    def get_context(self):
        return list(self._m), None


class FakeAgent:
    def __init__(self, messages):
        self.memory = FakeMemory(messages)
        self._summary_token_count = 0
        self.model_backend = types.SimpleNamespace(
            token_counter=types.SimpleNamespace(
                count_tokens_from_messages=lambda msgs: 7))
        self.restored = []

    def clear_memory(self, reset_summary_state=True):
        self.memory._m = [m for m in self.memory._m if m.get("role") == "system"]
        self.memory.clear_count += 1

    def update_memory(self, msg, role):
        from camel.types import OpenAIBackendRole
        r = "assistant" if role == OpenAIBackendRole.ASSISTANT else "user"
        self.memory._m.append({"role": r, "content": getattr(msg, "content", "")})
        if r == "user":
            self.restored.append(getattr(msg, "content", ""))


def main():
    install = load_patch_fn()
    install()

    from camel.agents import ChatAgent
    from camel.messages import BaseMessage

    BIG_B64 = "A" * 700_000                      # 模拟 513KB JPEG 的 base64 量级
    mm_content = [
        {"type": "text", "text": "请分析该矿区滑坡风险并给出处置建议。"},
        {"type": "image_url",
         "image_url": {"url": f"data:image/jpeg;base64,{BIG_B64}"}},
    ]

    print("① 摘要提示词不得含 base64")
    msgs = [{"role": "system", "content": "sys"},
            {"role": "user", "content": mm_content}]
    text, user_msgs = ChatAgent._build_conversation_text_from_messages(None, msgs)
    chk("输出不含 base64 字样", "base64" not in text)
    chk("输出长度合理(未被图像撑爆)", len(text) < 500, f"实={len(text)} 字符")
    chk("保留了文字部分", "滑坡风险" in text)
    chk("图像以占位符表示", "[图像]" in text)
    chk("user_messages 提取到文字", any("滑坡风险" in u for u in user_msgs))

    print("② 字符串里混入 data URI 也要截断")
    dirty = "前文 data:image/png;base64," + BIG_B64 + " 后文"
    t2, _ = ChatAgent._build_conversation_text_from_messages(
        None, [{"role": "user", "content": dirty}])
    chk("data URI 被截断", "base64" not in t2 and "[图像]" in t2, f"实={len(t2)} 字符")

    print("③ 空摘要不得清空记忆（安全网）")
    update_mem = ChatAgent._update_memory_with_summary   # 补丁打在类上, 显式传 self
    agent = FakeAgent([{"role": "system", "content": "sys"},
                       {"role": "user", "content": mm_content}])
    before = len(agent.memory._m)
    update_mem(agent, "", include_summaries=False)
    chk("未调用 clear_memory", agent.memory.clear_count == 0)
    chk("记忆条数不变", len(agent.memory._m) == before, f"{before} -> {len(agent.memory._m)}")
    update_mem(agent, None)
    chk("summary=None 也不清记忆", agent.memory.clear_count == 0)
    update_mem(agent, "   ")
    chk("空白摘要也不清记忆", agent.memory.clear_count == 0)

    print("④ 摘要成功但最后一条是 user 消息时, 记忆里必须仍有 user 角色（补偿恢复）")
    # 纯文本 user 消息: 上游自己会恢复, 补偿逻辑不应重复添加
    a2 = FakeAgent([{"role": "system", "content": "sys"},
                    {"role": "user", "content": "纯文本问题"}])
    update_mem(a2, "这是摘要内容", include_summaries=False)
    users = [m for m in a2.memory._m if m.get("role") == "user"]
    chk("纯文本场景: 有 user 角色", len(users) >= 1, f"实={len(users)}")
    chk("纯文本场景: 未重复补偿", len(a2.restored) <= 1, f"补偿次数={len(a2.restored)}")

    # 多模态最后一条 user 消息: 这正是上游 isinstance 判断漏掉的场景
    a3 = FakeAgent([{"role": "system", "content": "sys"},
                    {"role": "user", "content": mm_content}])
    update_mem(a3, "这是摘要内容", include_summaries=False)
    users3 = [m for m in a3.memory._m if m.get("role") == "user"]
    chk("多模态场景: 补偿后有 user 角色", len(users3) >= 1, f"实={len(users3)}")
    chk("补偿内容为纯文字", all("base64" not in str(m.get("content", "")) for m in users3))
    chk("记忆未含 base64", all("base64" not in str(m.get("content", ""))
                              for m in a3.memory._m))

    print("⑤ 无 user 消息的消息列表不应抛异常")
    try:
        ChatAgent._build_conversation_text_from_messages(None, [{"role": "system", "content": "s"}])
        chk("空会话不抛异常", True)
    except Exception as e:
        chk("空会话不抛异常", False, repr(e))
    try:
        ChatAgent._build_conversation_text_from_messages(None, [])
        chk("空列表不抛异常", True)
    except Exception as e:
        chk("空列表不抛异常", False, repr(e))

    print(f"\n{'✅ 全部通过' if OK else '❌ 存在失败项'}")
    return 0 if OK else 1


if __name__ == "__main__":
    sys.exit(main())
