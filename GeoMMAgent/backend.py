"""
GeoMMAgent Flask 后端服务
在本地运行 GeoMMAgent 并提供 HTTP API
用法：python backend.py
然后从启动后的本地 HTTP 网址使用前端
"""

import os
import io
import sys
import json
import time
import base64
import uuid
import hashlib
import pathlib
import subprocess
import urllib.request
import urllib.error
import numpy as np
import copy
import threading
from collections import OrderedDict
from report_grounding import build_grounding, check_structured_geometry, fallback_report
from review_handoff import build_review_record
from evidence_reporting import objective_facts, workflow_status
from comparison_reporting import export_comparison
from http_contract import install_json_errors, validate_analysis_fields
from case_archive import CaseArchive, ArchiveError
from runtime_identity import runtime_fingerprint
import sqlite3
from datetime import datetime
from project_runtime import configure_runtime_paths
configure_runtime_paths()

from flask import Flask, request, jsonify, send_file
from local_access import install_local_access

app = Flask(__name__)
install_local_access(app)
app.config["MAX_CONTENT_LENGTH"] = 48 * 1024 * 1024
install_json_errors(app)
BACKEND_CODE_SHA256 = hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest()


@app.route('/')
def frontend_page():
    return send_file(pathlib.Path(__file__).resolve().parent/'矿安天眼_前端演示(后端版).html')

# ── 路径设置 ──────────────────────────────────────────
BASE_DIR = pathlib.Path(__file__).parent.resolve()


@app.errorhandler(ArchiveError)
def archive_error(error):
    return jsonify({'error': str(error), 'status_code': error.status, **error.details}), error.status


@app.errorhandler(sqlite3.Error)
def archive_database_error(error):
    app.logger.error('案例归档数据库操作失败: %s', type(error).__name__)
    return jsonify({'error': '案例归档暂不可用，请检查D盘空间与数据库访问状态', 'status_code': 503}), 503

# GeoMMAgent 源码目录（D:\GeoMMAgent\）
# 注意：你的目录结构是 D:\GeoMMAgent\configs/ coordinator/ toolkit/ 等
# 不再有额外的 GeoMMAgent 子目录层级
GEOMM_DIR = BASE_DIR

# 加载 .env（D:\GeoMMAgent\.env）
from dotenv import load_dotenv
env_path = BASE_DIR / ".env"
if env_path.exists():
    load_dotenv(dotenv_path=str(env_path))
    print(f".env loaded from {env_path}")
else:
    print(f"WARNING: .env not found at {env_path}")

# Bound provider requests as well as the outer task; respect an explicit user setting.
os.environ.setdefault("MODEL_TIMEOUT", "60")
LOCAL_FACTS_ONLY = os.environ.get('KAT_LOCAL_FACTS_ONLY', '0').lower() in ('1', 'true')
CASE_ARCHIVE_PATH = pathlib.Path(os.environ.get('KAT_CASE_ARCHIVE_PATH', str(BASE_DIR / 'data/case_archive/cases.sqlite3'))).resolve()
if CASE_ARCHIVE_PATH.drive.upper() == 'C:':
    raise RuntimeError('案例数据库必须保存在D盘项目目录')
case_archive = CaseArchive(CASE_ARCHIVE_PATH)

# 添加 GeoMMAgent 源码目录到 Python 路径
sys.path.insert(0, str(GEOMM_DIR))


# ── 离线容错: tiktoken 词表下载受限时改用本地 Qwen 词表 ──
# camel 初始化会调用 tiktoken 下载 o200k_base 词表, 本机网络无法访问
# 该服务时会永久挂起(实测); 这里用本地 Qwen 词表(零依赖解析)替代。
def _install_offline_tokenizer():
    try:
        import tiktoken
        import base64 as _b64
        _qv = BASE_DIR / 'resources/qwen.tiktoken'
        if not _qv.exists():
            _qv = pathlib.Path(r"D:\Miniconda3\Lib\site-packages\dashscope\resources\qwen.tiktoken")
        _mergeable = {}
        with open(_qv, "rb") as f:
            for line in f:
                parts = line.split()
                if len(parts) == 2:
                    _mergeable[_b64.b64decode(parts[0])] = int(parts[1])
        _pat = (r"'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}"
                r"| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+")
        _enc = tiktoken.Encoding(
            name="qwen_local", pat_str=_pat, mergeable_ranks=_mergeable,
            special_tokens={"<|endoftext|>": len(_mergeable),
                            "<|im_start|>": len(_mergeable) + 1,
                            "<|im_end|>": len(_mergeable) + 2})
        import camel.utils.token_counting as _tc
        _tc.get_model_encoding = lambda v: _enc
        print(f"[tokenizer] 离线模式: 本地 Qwen 词表({len(_mergeable)} 条目)")
    except Exception as e:
        try:
            import camel.utils.token_counting as _tc

            class _Est:
                def encode(self, text):
                    return [0] * max(1, len(text) // 4)

            _tc.get_model_encoding = lambda v: _Est()
            print(f"[tokenizer] 估算模式(兜底): {e}")
        except Exception:
            pass


# ── 稳定性: CAMEL 摘要故障级联修复 (2026-09-22) ─────────────
# 实测故障链 (证据见 backend_startup.log):
#   ① 触发摘要时 camel 对多模态消息的 list 型 content 做 str(), 把
#      data:...;base64 图像整块拼进摘要提示词 —— 上传场景单图实测 513KB,
#      base64 约 68 万字符, 远超 qwen-vl-max 的 129024 token 上限 → 400;
#   ② 摘要失败返回空串, 但上游不检查成败即调用 _update_memory_with_summary;
#   ③ 该方法用 isinstance(content, str) 取"最后一条用户消息", 而多模态
#      content 是 list → 判断失败 → last_user_message=None → clear_memory()
#      清空后未恢复 user 消息 → 下次调度发出 [system, assistant("")] →
#      无 user 角色 → 400 → 表现为 "Dispatch parsing failed"。
# 因此先前的"调度解析失败 = JSON 格式问题"归因是错的, 根因是记忆被清空。
# 两处补丁: ① 摘要文本只取文字部分; ② 空摘要不清记忆 + 补偿恢复 user 消息。
def _install_summary_fault_fix():
    try:
        from camel.agents import ChatAgent
        from camel.messages import BaseMessage
        from camel.types import OpenAIBackendRole
    except Exception as e:                                  # 守卫: 不得阻断启动
        print(f"[summary-fix] 跳过(导入失败): {e}")
        return

    def _strip_data_uri(s):
        """兜底: 字符串中若混入 data URI, 截断为占位符。"""
        i = s.find("data:")
        if i < 0 or ";base64," not in s:
            return s
        return s[:i] + "[图像]"

    def _text_only(content):
        """list 型多模态 content → 纯文字; 图像以占位符代替, 绝不内联 base64。"""
        if isinstance(content, str):
            return _strip_data_uri(content)
        if isinstance(content, (list, tuple)):
            parts = []
            for p in content:
                if isinstance(p, dict):
                    if "text" in p and p.get("type") in (None, "text"):
                        parts.append(str(p["text"]))
                    else:
                        parts.append("[图像]")
                else:
                    parts.append(_strip_data_uri(str(p)))
            return " ".join(x for x in parts if x)
        return str(content)

    # ① 摘要提示词不得含 base64
    _orig_build = ChatAgent._build_conversation_text_from_messages

    def _safe_build(self, messages, include_summaries=False):
        safe = []
        for m in messages:
            try:
                d = dict(m)
            except Exception:
                try:
                    d = {k: getattr(m, k) for k in
                         ("role", "content", "name", "tool_calls") if hasattr(m, k)}
                except Exception:
                    safe.append({"role": "user", "content": "[消息不可解析]"})
                    continue
            if "content" in d:
                d["content"] = _text_only(d["content"])
            safe.append(d)
        return _orig_build(self, safe, include_summaries)

    ChatAgent._build_conversation_text_from_messages = _safe_build

    # ② 空摘要不得清空记忆; 多模态的最后一条 user 消息要能恢复
    _orig_update = ChatAgent._update_memory_with_summary

    def _safe_update(self, summary, include_summaries=False):
        if summary is None or not str(summary).strip():
            print("[summary-fix] 摘要为空(上游生成失败), 跳过记忆更新以免清空上下文")
            return
        last_user_text = None
        try:
            msgs, _ = self.memory.get_context()
            for m in msgs:
                role = m.get("role") if isinstance(m, dict) else getattr(m, "role", "")
                cont = m.get("content") if isinstance(m, dict) else getattr(m, "content", "")
                if role == "user":
                    t = _text_only(cont).strip()
                    if t:
                        last_user_text = t
        except Exception:
            pass

        _orig_update(self, summary, include_summaries)

        # 补偿: 上游的 isinstance(str) 判断会漏掉多模态消息, 导致记忆里没有 user 角色
        if last_user_text:
            try:
                msgs, _ = self.memory.get_context()
                has_user = any(
                    (m.get("role") if isinstance(m, dict) else getattr(m, "role", "")) == "user"
                    for m in msgs)
                if not has_user:
                    self.update_memory(
                        BaseMessage.make_user_message(
                            role_name="user", content=last_user_text),
                        OpenAIBackendRole.USER)
                    print("[summary-fix] 已补偿恢复最后一条用户消息(上游判断漏掉多模态)")
            except Exception as e:
                print(f"[summary-fix] 补偿恢复失败: {e}")

    ChatAgent._update_memory_with_summary = _safe_update
    print("[summary-fix] 已安装: 摘要不含base64 + 空摘要不清记忆 + user消息补偿恢复")


# ── 初始化 GeoMMAgent ────────────────────────────────
_cfg = {}
coordinator = None
try:
    from configs import load_config, create_model_from_config
    from configs.loader import DEFAULT_MODEL

    config = load_config() or {}
    if not isinstance(config, dict):
        raise ValueError("GeoMMAgent 配置须为映射")
    _cfg = config

    def create_model(model_type, temperature=0, max_tokens=1600):
        return create_model_from_config(model_type, temperature, max_tokens)

    if not LOCAL_FACTS_ONLY:
        _install_offline_tokenizer()
        _install_summary_fault_fix()
        from coordinator import AgentCoordinator

        coordinator_settings = _cfg.get("coordinator", {})
        coordinator_model = create_model(
            coordinator_settings.get("model", DEFAULT_MODEL),
            coordinator_settings.get("temperature", 0),
            max_tokens=coordinator_settings.get("max_tokens", 768),
        )
        # Validate the online coordinator at startup. Each request owns its workers.
        coordinator = AgentCoordinator(model=coordinator_model)
        print("GeoMMAgent 在线协调器初始化成功")
except Exception as e:
    print(f"GeoMMAgent 初始化失败: {e}")
    print("自动分析不可用；检测链路仅输出带来源的事实汇总")

# ── 检测侧仓库 (LMHLD) 的模块依赖 ─────────────────────
# 本后端会引用检测侧仓库(LMHLD)的三个模块:
#   knowledge_base (RAG 知识库) / geo_tools (定量分析) / geo_geometry (多粒度几何)
# 这是**跨仓库依赖**, 且检测侧运行在独立的 lmhld 环境(本进程只读其纯 numpy 模块)。
#
# 设计原则: 跨仓依赖必须显式声明 + 可配置 + 缺失时降级而非崩溃。
#   · 路径可用环境变量 KAT_LMHLD_DIR 覆盖(默认 D:\LMHLD)
#   · 目录不存在时给出明确告警, 服务仍能启动, 仅相关功能降级
# 这样两个仓库各自可独立克隆运行(功能降级), 边界清晰。
LMHLD_DIR = os.environ.get("KAT_LMHLD_DIR", r"D:\LMHLD")
if os.path.isdir(LMHLD_DIR):
    sys.path.insert(0, LMHLD_DIR)
    print(f"检测侧仓库已挂载: {LMHLD_DIR}")
else:
    print(f"WARNING: 检测侧仓库目录不存在: {LMHLD_DIR}")
    print("         定量分析/几何量测/RAG 知识库将降级不可用;")
    print("         如目录在别处, 请设置环境变量 KAT_LMHLD_DIR 指向它")

# ── 露天矿安全知识库 (RAG 检索) ──────────────────────
kb = None
try:
    from knowledge_base import KnowledgeBase
    kb = KnowledgeBase()
    print(f"露天矿安全知识库已加载: {len(kb.docs)} 条知识 / {len(kb.sources)} 个来源")
except Exception as e:
    print(f"WARNING: 知识库加载失败: {e}")

# 定量分析工具集 (对标 Earth-Agent 的 Index/Analysis 工具包)
try:
    from geo_tools import format_quant_block
    print("定量分析工具集已加载 (NDVI/区域统计/趋势/对比)")
except Exception as e:
    format_quant_block = None
    print(f"WARNING: 定量工具加载失败: {e}")


# ── 滑坡检测服务 (detector_service.py, lmhld 环境) ────
DETECTOR_PY = os.environ.get("KAT_DETECTOR_PY", r"D:\Miniconda3\envs\lmhld\python.exe")
DETECTOR_SERVICE = str(pathlib.Path(LMHLD_DIR) / "detector_service.py")
DETECTOR_PORT = int(os.environ.get("DETECTOR_PORT", 5101))
DETECTOR_URL = f"http://localhost:{DETECTOR_PORT}"
_detector_proc = None
AUTO_RECOVER_DETECTOR=os.environ.get('KAT_AUTO_RECOVER_DETECTOR','1').lower() in ('1','true')
_detector_supervisor=None
BACKEND_RUNTIME_SHA256=runtime_fingerprint(BASE_DIR,'backend')
EXPECTED_DETECTOR_RUNTIME_SHA256=runtime_fingerprint(LMHLD_DIR,'detector')


def detector_identity_matches(identity):
    return bool(identity and identity.get('service')=='kat-detector'
                and identity.get('runtime_sha256')==EXPECTED_DETECTOR_RUNTIME_SHA256
                and os.path.normcase(os.path.abspath(identity.get('runtime_root','')))==os.path.normcase(os.path.abspath(LMHLD_DIR)))


def detector_port_active():
    import socket
    for host in ('127.0.0.1','::1'):
        try:
            with socket.create_connection((host,DETECTOR_PORT),timeout=.2):return True
        except OSError:pass
    return False


def start_detector_service():
    """启动滑坡检测服务子进程 (独立 GPU 环境)"""
    global _detector_proc
    if not (os.path.exists(DETECTOR_PY) and os.path.exists(DETECTOR_SERVICE)):
        print("WARNING: 检测服务脚本不存在, 检测功能不可用")
        return
    # 先探测是否已有实例在跑 —— 原先无条件拉起, 在"重启后端但检测服务仍在"的
    # 常见场景下会再起一个: Windows 允许 SO_REUSEADDR 同端口双绑定, 结果两个
    # 检测服务同时占同一端口与显存(实测 2026-09-22), 请求落点还不确定。
    try:
        with urllib.request.urlopen(DETECTOR_URL + "/api/health", timeout=2) as r:
            if r.status == 200:
                identity=json.loads(r.read().decode())
                if not detector_identity_matches(identity):
                    print('WARNING: 已有检测服务的代码或知识版本不一致，请停止旧服务后重新启动')
                    return
                print("滑坡检测服务已在运行, 直接复用:", DETECTOR_URL)
                return
    except Exception:
        pass
    if detector_port_active():
        print('WARNING: 检测端口已有监听但未通过健康检查，未额外启动检测进程')
        return
    try:
        _detector_proc = subprocess.Popen(
            [DETECTOR_PY, DETECTOR_SERVICE, str(DETECTOR_PORT)],
            creationflags=subprocess.CREATE_NO_WINDOW,
        )
        # 等待服务就绪 (最多 30s)
        for _ in range(30):
            try:
                with urllib.request.urlopen(DETECTOR_URL + "/api/health", timeout=1) as r:
                    if r.status == 200:
                        if not detector_identity_matches(json.loads(r.read().decode())):
                            print('WARNING: 检测服务来源不匹配，未报告启动成功')
                            return
                        print("滑坡检测服务已就绪:", DETECTOR_URL)
                        return
            except Exception:
                time.sleep(1)
        print("WARNING: 检测服务启动超时, 检测功能暂不可用")
    except Exception as e:
        print(f"WARNING: 检测服务启动失败: {e}")


def detector_request(path, payload=None, method="POST"):
    """Preserve detector validation errors and status codes through the HTTP proxy."""
    if path != '/api/health':
        identity=detector_request('/api/health',method='GET')
        if identity is None:
            return {'error':'检测服务不可用，请启动检测服务','status_code':503}
        if not detector_identity_matches(identity):
            return {'error':'检测服务代码或知识版本不一致，请停止旧服务后重新启动；未使用旧版本生成结果',
                    'status_code':503,'inference_status':'runtime_version_mismatch'}
    try:
        if method == "GET":
            req = DETECTOR_URL + path
        else:
            req = urllib.request.Request(DETECTOR_URL + path, data=json.dumps(payload).encode(),
                                         headers={"Content-Type": "application/json"})
        timeout=3 if path=='/api/health' else int(os.environ.get("KAT_PROXY_TIMEOUT", 600))
        with urllib.request.urlopen(req, timeout=timeout) as response:
            return json.loads(response.read().decode())
    except urllib.error.HTTPError as exc:
        try:
            error = json.loads(exc.read().decode())
        except Exception:
            error = {"error": f"检测服务返回HTTP {exc.code}"}
        error["status_code"] = exc.code
        return error
    except Exception as exc:
        print(f"[proxy] 检测服务请求失败 {path}: {exc}")
        return None


# ── 调度约束协调器 (对标 Earth-Agent: 抑制"无关步骤") ──
# 矿山监测任务不需要图像预处理链与选择题匹配, 直接排除
NEVER_WORKERS = ("formatconversion", "imagefilter", "scale", "superresolution",
                 "matching", "super_resolution")
# 场景白名单: 只允许与任务相关的智能体参与调度
SCENARIO_WORKERS = {
    "detect": ("segagent", "detagent", "clsagent", "searchagent", "reasoningagent"),
    "compare": ("segagent", "detagent", "clsagent", "searchagent", "reasoningagent"),
    "qa": ("clsagent", "detagent", "segagent", "searchagent", "retrievalagent",
           "reasoningagent"),
}

def get_open_coordinator(scenario="detect"):
    """Each request owns fresh agent memories, including the dispatcher."""
    from configs import build_agents_from_config
    from coordinator import AgentCoordinator
    allowed = SCENARIO_WORKERS.get(scenario)
    def keep(agent_name):
        name = agent_name.lower()
        return not any(n in name for n in NEVER_WORKERS) and (
            not allowed or any(n in name for n in allowed))

    scenario_agents = {
        name: settings
        for name, settings in _cfg.get("agents", {}).items()
        if keep(name)
    }
    workers = build_agents_from_config({**_cfg, "agents": scenario_agents}, open_ended=True)
    coord = AgentCoordinator(model=create_model(
        _cfg.get("coordinator", {}).get("model", DEFAULT_MODEL), max_tokens=768))
    coord.register_workers(workers)
    return coord


# ── 多智能体超时保护 ──────────────────────────────────
from concurrent.futures import ThreadPoolExecutor, TimeoutError as _FutTimeout

_llm_check_cache = {"t": 0.0, "ok": None}
_llm_slots = threading.BoundedSemaphore(2)
_llm_check_lock = threading.Lock()
_report_lock = threading.Lock()
_report_cache = OrderedDict()

def llm_fast_check(ttl=300):
    if LOCAL_FACTS_ONLY:
        return False
    now = time.time()
    with _llm_check_lock:
        if _llm_check_cache["ok"] is not None and now - _llm_check_cache["t"] < ttl:
            return _llm_check_cache["ok"]
    try:
        from camel.agents import ChatAgent
        agent = ChatAgent(system_message="You are a helpful assistant.",
                          model=create_model(_cfg.get('coordinator', {}).get('model', DEFAULT_MODEL), max_tokens=32),
                          retry_attempts=1)
        result = run_with_timeout(lambda: agent.step("回复: 好"), timeout=10)
        from evidence_integrity import completion_status
        ok = result is not None and bool(result.msgs) and completion_status(result)['complete']
    except Exception:
        ok = False
    with _llm_check_lock:
        _llm_check_cache.update(t=time.time(), ok=ok)
    return ok

def run_with_timeout(fn, timeout=120):
    """Bound outstanding work, including threads which are still running after timeout."""
    slots = _llm_slots
    if not slots.acquire(blocking=False):
        print("[pipeline] 模型任务仍占用执行槽，返回明确降级状态")
        return None
    def owned_call():
        try:
            return fn()
        finally:
            slots.release()
    ex = ThreadPoolExecutor(max_workers=1)
    try:
        fut = ex.submit(owned_call)
    except Exception:
        slots.release()
        ex.shutdown(wait=False)
        raise
    try:
        return fut.result(timeout=timeout)
    except _FutTimeout:
        print("[pipeline] 模型任务超时；后台任务结束前仍占用执行槽")
        return None
    except Exception as exc:
        print(f"[pipeline] 模型任务失败: {exc}")
        return None
    finally:
        ex.shutdown(wait=False)


# ── 工具函数 ──────────────────────────────────────────
def _safe_filename(s):
    import re
    return re.sub(r"[^\w\-]+", "_", s).strip("_") or "agent"


VLM_MAX_DIM = 1280   # 多模态模型输入上限保护 (过大会被上游拒绝)


def _prepare_vlm_image(image_bytes):
    """大图压缩: 长边 > VLM_MAX_DIM 时等比缩放, 统一转 JPEG 以减少体积"""
    from PIL import Image as _Image
    try:
        im = _Image.open(io.BytesIO(image_bytes)).convert("RGB")
        if max(im.size) > VLM_MAX_DIM:
            r = VLM_MAX_DIM / max(im.size)
            im = im.resize((max(1, int(im.width * r)), max(1, int(im.height * r))), _Image.LANCZOS)
        buf = io.BytesIO()
        im.save(buf, "JPEG", quality=85)
        return buf.getvalue()
    except Exception as e:
        print(f"[vlm] 图像压缩失败, 使用原图: {e}")
        return image_bytes


def write_trace_json(trace, out_dir, sample_id=None, ground_truth=None):
    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    steps_dir = out_dir / "steps"
    steps_dir.mkdir(exist_ok=True)

    payload = {
        "sample_id": sample_id,
        "ground_truth": ground_truth,
        "image_path": trace.image_path,
        "original_task": trace.original_task,
        "dispatch_plan": trace.dispatch_plan,
        "final_answer": trace.final_answer,
        "evidence_images": trace.evidence_images,
    }
    with (out_dir / "run.json").open("w", encoding="utf-8") as f:
        import json
        json.dump(payload, f, ensure_ascii=False, indent=2)
    with (out_dir / "trace_full.json").open("w", encoding="utf-8") as f:
        import json
        json.dump(trace.to_dict(), f, ensure_ascii=False, indent=2)
    for s in trace.agent_steps:
        import json, re
        fname = f"{s.step:02d}_{_safe_filename(s.agent_name)}.json"
        with (steps_dir / fname).open("w", encoding="utf-8") as f:
            json.dump({
                "step": s.step,
                "agent_index": s.agent_index,
                "agent_name": s.agent_name,
                "subtask": s.subtask,
                "input_text": s.input_text,
                "output": s.output,
            }, f, ensure_ascii=False, indent=2)


# ── API 路由 ──────────────────────────────────────────

@app.before_request
def validate_json_request():
    if request.method == "POST":
        data = request.get_json(silent=True)
        if not isinstance(data, dict):
            return jsonify({"error": "请求必须为 JSON 对象"}), 400
        error = validate_analysis_fields(data)
        if error:
            return jsonify({'error': error}), 400
        if data.get('case_id') and request.endpoint in ('pipeline', 'pipeline_upload', 'pipeline_compare'):
            case_archive.get_case(data['case_id'])

@app.route("/api/health", methods=["GET"])
def health():
    det = detector_request("/api/health", method="GET")
    return jsonify({
        "status": "ok",
        'service': 'kat-backend', 'backend_code_sha256': BACKEND_CODE_SHA256,
        'backend_runtime_sha256':BACKEND_RUNTIME_SHA256,
        'detector_runtime_compatible':detector_identity_matches(det),
        'runtime_pid':os.getpid(),'runtime_root':str(BASE_DIR),
        'archive_database_path':str(case_archive.path),
        'detector_recovery_enabled':bool(AUTO_RECOVER_DETECTOR and _detector_supervisor and _detector_proc is not None),
        'detector_recovery':_detector_supervisor.snapshot() if _detector_supervisor else {'state':'not_started'},
        'runtime_python_prefix':sys.prefix, 'http_server':request.environ.get('SERVER_SOFTWARE', 'unknown'),
        'http_runtime':app.config.get('KAT_HTTP_RUNTIME','in_process_test'),
        "geomm_loaded": coordinator is not None and not LOCAL_FACTS_ONLY,
        'reporting_mode': 'local_facts_only' if LOCAL_FACTS_ONLY else 'cloud_optional',
        'report_workflow': 'evidence_report' if _cfg.get('report_workflow',{}).get('enabled',False) else 'general_dispatch',
        "detector_online": bool(det and det.get("status") == "ok"),
        "detector_loaded": bool(det and det.get("detector_loaded")),
        "knowledge_docs": len(kb.docs) if kb else 0,
        "knowledge_sources": len(kb.sources) if kb else 0,
    })


@app.route("/api/analyze", methods=["POST"])
def analyze():
    """
    接收: { question: str, image: base64字符串 }
    返回: { agents: [...], final_answer: str, dispatch_plan: [...] }
    """
    try:
        data = request.get_json()
        question = data.get("question", "").strip()
        image_b64 = data.get("image", "").strip()

        if not question:
            return jsonify({"error": "question is required"}), 400
        if not image_b64:
            return jsonify({"error": "image is required"}), 400

        # 保存临时图片 (大图自动压缩, 防上游多模态体积超限)
        img_id = str(uuid.uuid4())[:8]
        img_dir = BASE_DIR / "uploads"
        img_dir.mkdir(exist_ok=True)
        img_path = img_dir / f"{img_id}.jpg"
        img_bytes = _prepare_vlm_image(base64.b64decode(image_b64))
        with open(img_path, "wb") as f:
            f.write(img_bytes)

        run_id = f"run_{img_id}_{datetime.now().strftime('%H%M%S')}"
        run_dir = BASE_DIR / "results" / run_id

        if coordinator is not None and llm_fast_check():
            qa_coord = get_open_coordinator("qa")
            from evidence_integrity import evidence_context
            qa_coord.set_evidence_context(evidence_context(request_id=run_id))
            trace = run_with_timeout(lambda: qa_coord.run_with_trace(question, image_path=str(img_path)), 150)
            if trace is None:
                return jsonify({"error": "自动问答调用失败或超时", "is_mock": False}), 503
            write_trace_json(trace, run_dir)
            if not trace.final_answer or any(getattr(s, 'status', 'complete') != 'complete' for s in trace.agent_steps):
                return jsonify({'error': '智能体输出不完整，未作为有效问答结果使用',
                                'workflow':workflow_status(trace),
                                'agents':[{'step':s.step,'name':s.agent_name,'status':s.status} for s in trace.agent_steps],
                                'analysis_status': 'incomplete', 'is_mock': False, 'run_id': run_id}), 503
            agents = [
                {
                    "step": s.step,
                    "name": s.agent_name,
                    "subtask": s.subtask,
                    "output": s.output,
                }
                for s in trace.agent_steps
            ]
            return jsonify({
                "run_id": run_id,
                "trace_dir": str(run_dir),
                "agents": agents,
                "dispatch_plan": trace.dispatch_plan,
                "workflow": workflow_status(trace),
                "final_answer": trace.final_answer,
                "image_path": str(img_path),
                "is_mock": False,
            })
        else:
            # Model availability is reported explicitly; no invented answer.
            return jsonify({"error": "模型服务不可用，未生成问答结果", "is_mock": False}), 503

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/api/detect", methods=["POST"])
def detect_proxy():
    """滑坡检测 (代理到 GPU 检测服务)"""
    data = request.get_json(silent=True) or {}
    result = detector_request("/api/detect", data)
    if result is None:
        return jsonify({"error": "检测服务不可用, 请检查 detector_service 是否启动"}), 503
    return jsonify(result), result.get("status_code", 503) if "error" in result else 200


RISK_CN = {"blue": "蓝色-未检出明显异常", "yellow": "黄色-少量疑似异常",
           "orange": "橙色-较多疑似异常", "red": "红色-大量疑似异常",
           "unknown":"待复核-证据冲突"}


@app.route("/api/geometry_export", methods=["POST"])
def geometry_export_proxy():
    """实例几何矢量导出 (代理到检测服务)。只做格式化, 不重新检测。"""
    data = request.get_json(silent=True) or {}
    result = detector_request("/api/geometry_export", data)
    if result is None:
        return jsonify({"error": "检测服务不可用"}), 503
    return jsonify(result), result.get("status_code", 503) if "error" in result else 200



def knowledge_metadata(chunks):
    fields = ("id", "category", "title", "source_title", "source_url", "publisher",
              "published_at", "checked_at", "locator", "evidence_type", "scope", "verification",
              "full_text_url", "related_sources")
    return [{key: chunk.get(key) for key in fields} for chunk in chunks]


def cache_report(result, case_id=None):
    if "error" in result:
        return result
    report_id = uuid.uuid4().hex
    result["report_id"] = report_id
    result['generated_at'] = datetime.now().astimezone().isoformat(timespec='seconds')
    result['objective_facts'] = objective_facts(result['detection'])
    result['objective_facts_source'] = 'same_detector_result'
    result['review_handoff'] = build_review_record(result)
    result = case_archive.save_report(result, case_id)
    with _report_lock:
        now = time.time()
        expired = [key for key, (created, _) in _report_cache.items() if now - created > 1800]
        for key in expired:
            del _report_cache[key]
        _report_cache[report_id] = (now, copy.deepcopy(result))
        while len(_report_cache) > 8:
            _report_cache.popitem(last=False)
    return result


def report_for_export(data, generate):
    report_id = data.get("report_id")
    if report_id:
        try:
            return case_archive.get_report(str(report_id))
        except ArchiveError as error:
            if error.status != 404: raise
            return {'error': '报告标识不存在，请从案例历史选择已归档的分析结果', 'status_code': 410}
    return generate()


def report_facts(det):
    return {"stats": det.get("stats", {}), "risk_name": det.get("risk_name", "待复核"),
            "roi": det.get("roi"), "band_meta": det.get("band_meta"), "scale_meta":det.get('scale_meta'),
            "model_tag": det.get("model_tag"), "threshold": det.get("threshold"),
            "model_info": det.get("model_info"),
            'request_id':det.get('request_id'),'input_quality':det.get('input_quality'),
            'work_context':det.get('work_context'),
            "risk_basis": det.get("risk_basis")}


def build_rag_question(stats, risk_cn, quant=None):
    quant_text = ""
    if quant and format_quant_block is not None:
        try:
            quant_text = format_quant_block(**quant) + "\n\n"
        except Exception as exc:
            print(f"[pipeline] 定量块无法渲染: {exc}")
    scope = "监测范围内" if stats.get("scope") == "roi" else "本次影像范围内"
    if stats.get('unverified_pixels',0):
        scope='有效非冲突像素范围内'
    coverage = (f"掩膜面积占比{stats['area_ratio'] * 100:.1f}%，" if stats.get('area_ratio_available',True)
                else '当前没有可判定像素，面积占比不可计算，')
    question = (quant_text + f"在{scope}，模型输出{stats['num_regions']}处疑似异常区域，"
                + coverage +
                f"最大区域面积{stats['max_region_area']}像素，原型筛查分级为{risk_cn}。"
                "该分级来自预测区域面积比例，不能等同于法定预警或失稳概率。"
                "请输出中文报告：检测事实与范围、识别局限、人工复核与监测建议。"
                "几何量来自预测掩膜计算，不是现场量测真值；不得改变数值的指标、实例或单位。"
                "未提供真实NIR时不得推断NDVI；未提供坡度、位移与裂缝实测时不得编造这些数据。"
                "不得凭一幅影像确认滑坡、预测事故时间或断言人员安全。"
                "只引用提供资料的条目编号，区分法规摘要、技术资料与项目说明。"
                "直接输出正文，不输出A/B/C/D选项。")
    if stats.get('unverified_pixels',0):
        question += f"另有{stats['unverified_pixels']}像素证据冲突，尚未判定，不能写成正常或安全区域。必须保持待复核状态。"
    chunks = kb.retrieve("光学筛查 面积占比 监测范围 人工复核 分区稳定性 " + risk_cn, k=4) if kb else []
    if chunks:
        question = "经核对的知识资料：\n" + kb.format_context(chunks) + "\n\n" + question
    return question, knowledge_metadata(chunks)



# ── 报告质量守卫 + 自评估 (对标 ThinkGeo/Earth-Agent 的质量校验) ──
import re as _re

_quality_stats = {"checks": {}, "repair_attempts": 0, "repair_success": 0,
                  "degraded": 0, "selfeval_total": 0, "selfeval_pass": 0,
                  "runs": 0}

REPAIR_HINTS = {
    '观测依据与案例一致':'严格核对给定案例编号与各时相采集日期；缺少真实近红外时不声称测得NDVI，不补造现场位移或失稳时间，冲突像素保持未知',
    "检测数值一致": "区域数量和面积占比必须与提供的检测统计和空间范围一致",
    '筛查结论不越界':'影像掩膜不能证明边坡安全稳定、排除失稳或预测不会发生滑坡；冲突像素必须保持待复核',
    "引用来源一致": "用所提供资料的条目编号引用，不能编造或使用未提供的编号",
    "风险等级一致": "必须明确写出与检测结果一致的筛查分级(如'橙色-较多疑似异常')，待复核不能写成安全",
    "定量数字": "必须引用具体数值(面积占比%、像素数或区域处数)",
    "处置建议": "必须给出监测建议或应急处置措施",
    "无错误残留": "不得包含任何错误文本",
    "无字母残留": "不得输出 A/B/C/D/E 选项字母",
    "长度充足": "报告须完整充分(不少于200字)",
    "空报告或过短": "必须输出完整的中文评估报告",
    "几何数字引用": "必须引用几何形态工具输出中的具体数值(如实例面积/周长/长轴)",
    "几何数字越界": "报告中的几何数值必须取自几何形态工具输出, 不得自行估计",
    "几何数值一致": "按同一实例、同一指标和单位引用可靠事实；不能把周长写成长轴",
}

# 越界判据: **带像素单位的数字** —— 这是精度最高的锚点。
# 为什么不用"语境词 + 数字": 实测连续误报 —— 「面积占比为 3.1%」「…面积的7倍」
# 都会被"面积"这个语境词抓进来, 而自然语言里的几何语境词是枚举不完的。
# 为什么像素单位可靠: 提示词里只有几何工具输出带像素量, 知识库/通识建议
# 只会出现米/公里/百分比 —— 所以"带 px 的数字"几乎只可能来自检测结果,
# 不在白名单里就说明是模型自己编的(或四舍五入到偏离真值)。
_GEOM_PX_RE = _re.compile(r"(\d+(?:\.\d+)?)\s*(?:像素|px)", _re.I)


def _geom_ctx_pattern():
    """保留给文档/调试: 语言层面的几何语境(不再用于判定, 见上方说明)。"""
    return _re.compile(
        r"(长度|宽度|周长|长轴|短轴|直径|外接框|面积(?!占比|比例|率)|跨度|尺寸)"
        r"[^\d]{0,6}(\d+(?:\.\d+)?)(?!\s*%)")


def _geom_number_in_whitelist(val, whitelist, tol=0.03):
    """数值按 ±3% 容差比对白名单 —— 报告常有"约 132 像素"这类四舍五入。

    ⚠ 能力边界(如实记录): 本判据只能抓**明显编造**(如真实长轴 54.94px 却写
    "约 500 像素"), 无法识别"贴着真值的杜撰" —— 若模型写"长度约 180px"而真实
    周长为 179.3px, 会被判为合法四舍五入。故它只作软告警, 绝不作为质量门禁。
    """
    try:
        v = float(val)
    except (TypeError, ValueError):
        return True
    for w in whitelist:
        try:
            wv = float(w)
        except (TypeError, ValueError):
            continue
        if wv == 0:
            if abs(v) < 1e-9:
                return True
        elif abs(v - wv) / max(abs(wv), 1e-9) <= tol:
            return True
    return False


def check_geometry_grounding(answer, geom_whitelist):
    """检查报告对几何数字的引用情况。返回 (是否引用了白名单数字, 越界片段列表)。

    越界 = 出现「长度/周长/长轴/面积…」等几何语境词后跟了一个不在白名单里的数字,
    意味着模型很可能在编造尺寸(已存档的报告曾凭空写出"长度约为 180 像素")。

    ⚠ 只作**软告警**(记入 quality.warnings), 不进入 missing、不触发修复循环 ——
    正则对自然语言的误判若触发修复, 修复后的文本仍可能被判越界, 形成死循环。
    """
    if isinstance(geom_whitelist, dict):
        from report_grounding import check_structured_geometry
        return check_structured_geometry(answer, geom_whitelist)
    if not geom_whitelist:
        return None, []
    nums = sorted({str(w) for w in geom_whitelist if str(w).strip()},
                  key=len, reverse=True)
    rx = _re.compile(r"(?<![\d.])(" + "|".join(_re.escape(n) for n in nums) + r")(?![\d.])") \
        if nums else None
    cited = bool(rx and rx.search(answer or ""))
    bad = []
    for m in _GEOM_PX_RE.finditer(answer or ""):
        if not _geom_number_in_whitelist(m.group(1), geom_whitelist):
            bad.append(m.group(0))
    return cited, bad[:6]


def validate_report(answer, risk_name="", geom_whitelist=None, facts=None):
    """确定性质量校验: 返回缺失项列表 (空 = 通过)

    geom_whitelist 非空时额外要求「几何数字引用」—— 报告须真正引用检测出的几何量,
    否则说明多智能体忽略了注入的几何工具输出(与"定量数字"检查同源, 但更具体)。
    """
    if not answer or len(answer.strip()) < 50:
        _bump("空报告或过短")
        return ["空报告或过短"]
    lvl = _re.findall(r"[蓝黄橙红]色", risk_name or "")
    checks = {
        "风险等级一致": (not lvl) or any(t in answer for t in lvl),
        "定量数字": bool(_re.search(r"\d+(\.\d+)?\s*%|\d+\s*像素|\d+\s*处", answer)),
        "处置建议": bool(_re.search(r"建议|措施|处置|撤离|监测", answer)),
        "无错误残留": not bool(_re.search(r"Agent error|Error code|Traceback", answer)),
        "无字母残留": not bool(_re.search(r"^\s*[A-E]\s*$", answer, _re.M)),
        "长度充足": len(answer) >= 200,
    }
    if geom_whitelist:
        cited, bad = check_geometry_grounding(answer, geom_whitelist)
        checks["几何数字引用"] = bool(cited)
        if isinstance(geom_whitelist, dict):
            checks["几何数值一致"] = not bad
    if facts:
        from report_grounding import check_detection_numbers, check_unsupported_safety_claims, check_observation_claims,check_known_input_claims
        checks["检测数值一致"] = not check_detection_numbers(answer, facts)
        checks['筛查结论不越界'] = not check_unsupported_safety_claims(answer,facts)
        checks['观测依据与案例一致'] = not (check_observation_claims(answer,facts)+check_known_input_claims(answer,facts))
        if facts.get('stats',{}).get('unverified_pixels',0):
            checks['风险等级一致'] = '待复核' in answer
    missing = [k for k, ok in checks.items() if not ok]
    for k in missing:
        _bump(k)
    return missing


def _bump(key):
    _quality_stats["checks"][key] = _quality_stats["checks"].get(key, 0) + 1


def repair_report(answer, missing, facts_context=""):
    """按缺失项用文本大模型修复一次 (比重新调度多智能体快得多, ~10-15s)"""
    if not (coordinator is not None and llm_fast_check()):
        return None
    hints = "；".join(REPAIR_HINTS.get(m, m) for m in missing)
    NL = chr(10)
    prompt = ("以下是一份滑坡预警报告草稿, 请补全并改进后输出完整报告。" + NL
              + f"必须满足: {hints}" + NL + NL + "可靠事实与资料（优先于草稿；不能新增未提供的数值）:" + NL + facts_context + NL + NL
              + "报告草稿:" + NL + answer + NL + NL
              + "请直接输出改进后的完整中文报告(无需说明修改过程)。")
    def _ask():
        model = create_model("qwen-max")
        from camel.agents import ChatAgent
        agent = ChatAgent(system_message="你是露天矿安全报告编辑专家。", model=model, retry_attempts=1)
        resp = agent.step(prompt)
        from evidence_integrity import completion_status
        if not completion_status(resp)['complete']:
            return None
        return resp.msgs[0].content if resp.msgs else None
    return run_with_timeout(_ask, timeout=90)


def run_self_eval(question, agents, answer, image_path):
    """自评估智能体: 直接调用其四维评分工具 (逻辑/空间/领域/准确性 → overall_pass)"""
    if not _cfg.get('agents',{}).get('SelfEvaluationAgent',{}).get('enabled',False):
        return None
    try:
        from toolkit.evaluation_metrics import run_self_evaluation_metrics
        trace_text = chr(10).join(
            f"[{a.get('step')}] {a.get('name')}: {str(a.get('output', ''))[:500]}"
            for a in agents)
        out = run_with_timeout(lambda: run_self_evaluation_metrics(
            question=question, execution_trace=trace_text,
            candidate_answer=answer, image_path=image_path or ""), timeout=90)
        if out:
            d = json.loads(out)
            _quality_stats["selfeval_total"] += 1
            if d.get("overall_pass"):
                _quality_stats["selfeval_pass"] += 1
            return d
    except Exception as e:
        print(f"[selfeval] 失败: {e}")
    return None


def enforce_report_quality(question, agents, answer, risk_name, image_path,
                           geom_whitelist=None, facts=None):
    """质量守卫总流程: 校验 → 按缺失项修复一次 → 仍不合格标记降级; 附自评估
    返回 (final_answer, quality_dict)

    geom_whitelist 非空时额外做几何数字一致性软校验(只告警不拦截)。
    """
    _quality_stats["runs"] += 1
    from report_grounding import check_report_citations, check_source_detail_claims
    def check(candidate):
        missing = validate_report(candidate, risk_name, geom_whitelist, facts)
        if not check_report_citations(candidate, question):
            missing.append("引用来源一致")
            _bump("引用来源一致")
        if check_source_detail_claims(candidate,question):
            missing.append('引用细节有依据')
            _bump('引用细节有依据')
        return missing
    from report_grounding import check_observation_claims
    original_observation_errors = check_observation_claims(answer,facts)
    missing = check(answer)
    repaired = False
    if missing:
        _quality_stats["repair_attempts"] += 1
        fixed = repair_report(answer, missing, facts_context=question)
        if fixed and fixed.strip():
            answer = fixed.strip()
            repaired = True
            still = check(answer)
            if not still:
                _quality_stats["repair_success"] += 1
            missing = still
    degraded = bool(missing)
    if degraded:
        _quality_stats["degraded"] += 1
    warnings = []
    if geom_whitelist:
        try:
            _cited, bad = check_geometry_grounding(answer, geom_whitelist)
            if bad:
                warnings.append("几何数字越界: " + " | ".join(bad))
                _bump("几何数字越界告警")
        except Exception as e:
            print(f"[quality] 几何一致性检查失败: {e}")
    se = run_self_eval(question, agents, answer, image_path)
    quality = {
        "guard_pass": not degraded,
        "missing": missing,
        "repaired": repaired,
        "degraded": degraded,
        "warnings": warnings,
        "original_observation_errors": original_observation_errors,
        "self_eval": ({
            "overall_pass": se.get("overall_pass"),
            "summary": (se.get("summary") or "")[:200],
        } if se else None),
    }
    return answer, quality


def analyze_with_agents(question, overlay_b64, scenario="detect", risk_name="",
                        geom_whitelist=None, facts=None, case_id=None):
    overlay_path, agents, answer = None, [], None
    facts = copy.deepcopy(facts or {})
    facts['case_id'] = str(case_id) if case_id is not None else None
    from report_grounding import report_input_context
    question = question + '\n\n' + report_input_context(facts)
    from evidence_integrity import evidence_context
    fact_meta = (facts or {}).get('band_meta') or {}
    times = [fact_meta[k].get('acquisition_datetime') for k in ('t1','t2') if isinstance(fact_meta.get(k),dict)]
    if not times:
        times = [fact_meta.get('acquisition_datetime')]
    sources = [{'evidence_id': key, 'scope': 'global_reference', 'role': 'reference_not_current_observation'}
               for key in sorted(set(_re.findall(r'\[([A-Z]+-\d{3})\]',question)))]
    scope = evidence_context(case_id=case_id,
        request_id=(facts or {}).get('request_id') or uuid.uuid4().hex, capture_times=times, records=sources)
    reason = "已选择本地事实汇总模式" if LOCAL_FACTS_ONLY else "模型服务不可用"
    if coordinator is not None and llm_fast_check():
        try:
            raw = _prepare_vlm_image(base64.b64decode(overlay_b64))
            tmp_dir = BASE_DIR / "temp_images"
            tmp_dir.mkdir(exist_ok=True)
            overlay_path = tmp_dir / f"landslide_{uuid.uuid4().hex[:12]}.jpg"
            overlay_path.write_bytes(raw)
            if _cfg.get('report_workflow',{}).get('enabled',False):
                from report_workflow import ReportCoordinator
                coord = ReportCoordinator(_cfg,facts)
            else:
                coord = get_open_coordinator(scenario)
            coord.set_evidence_context(scope)
            trace = run_with_timeout(lambda: coord.run_with_trace(question, image_path=str(overlay_path)), 150)
            if trace is not None:
                agents = [{"step": s.step, "name": s.agent_name, "subtask": s.subtask, "output": s.output,
                           'status': getattr(s,'status','complete'), 'finish_reasons':getattr(s,'finish_reasons',[]),
                           'validation_errors':getattr(s,'validation_errors',[])}
                          for s in trace.agent_steps]
                answer = trace.final_answer
                if isinstance(answer, str) and answer.strip() in "ABCDE" and len(answer.strip()) == 1:
                    answer = next((s.output for s in reversed(trace.agent_steps)
                                   if "reason" in s.agent_name.lower()), None)
                if not answer or any(getattr(s,'status','complete') != 'complete' or "Agent error" in str(s.output) or "Error code:" in str(s.output)
                                     for s in trace.agent_steps):
                    answer, reason = None, "智能体链路未完整执行"
            else:
                reason = "模型任务失败、超时或执行槽已占用"
        except Exception as exc:
            print(f"[pipeline] 分析失败: {exc}")
            reason = "智能体调用失败"
    if answer is None:
        _quality_stats["runs"] += 1
        _quality_stats["degraded"] += 1
        answer = fallback_report(facts, reason)
        quality = {"guard_pass": False, "missing": ["智能体分析未完成"], "degraded": True,
                   "repaired": False, "warnings": [], "self_eval": None,
                   "generation": {"status": "fallback", "source": "detection_summary", "reason": reason}}
    else:
        answer, quality = enforce_report_quality(question, agents, answer, risk_name,
                                                str(overlay_path or ""), geom_whitelist, facts)
        if quality["degraded"]:
            answer = fallback_report(facts, "分析草稿未通过事实与质量校验")
        quality["generation"] = {"status": "fallback" if quality["degraded"] else "live",
                                 "source": "detection_summary" if quality["degraded"] else "multi_agent"}
    quality['evidence_context'] = scope
    quality['human_review_required'] = True
    quality['automated_checks_scope'] = '工程完整性、显式数值与来源约束；不认证全部地质语义'
    trace_value=locals().get('trace')
    quality['workflow'] = coord.status(trace_value) if trace_value is not None and hasattr(locals().get('coord'),'status') else workflow_status(trace_value)
    if quality['generation']['status']=='live' and len(agents)==1:
        quality['generation']['source']='single_agent'
    return agents, answer, str(overlay_path) if overlay_path else None, quality


def build_geometry_quant(stats, meters_per_pixel=None, topk=3):
    regions = (stats or {}).get("regions") or []
    if not regions:
        return None, None
    n_total = stats.get("num_regions_total", stats.get("num_regions"))
    geo = {"regions": regions[:topk], "num_regions_total": n_total}
    if meters_per_pixel:
        geo["meters_per_pixel"] = float(meters_per_pixel)
    return geo, build_grounding(regions[:topk], n_total, meters_per_pixel)


def run_pipeline(region, sample_index=None, num=8, fuse=False, ds_reject=False, case_id=None):
    """
    全链路: 滑坡检测 → 自动生成问题 → RAG注入 → 多智能体风险评估 → 预警报告
    供 /api/pipeline 与 /api/report 复用
    返回 dict (含 error 键时表示失败)
    """
    # 1. 检测
    options = {"region": region, "num": num, "fuse": fuse, "ds_reject": ds_reject}
    if sample_index is not None:
        if isinstance(sample_index, bool) or not isinstance(sample_index, int) or sample_index < 0:
            return {'error': 'sample_index 必须是非负整数', 'status_code': 400}
        options.update(start=sample_index, num=1)
    det = detector_request("/api/detect", options)
    if det is None:
        return {"error": "检测服务不可用"}
    if "error" in det:
        return det
    if not det.get("samples"):
        return {"error": "检测无结果"}

    # 2. 选定样本: 指定索引, 否则默认最高风险
    samples = det["samples"]
    target = None
    if sample_index is not None:
        for s in samples:
            if s["index"] == int(sample_index):
                target = s
                break
    if target is None and sample_index is not None:
        return {'error': '检测服务未返回所选样本，请重新选择并检测；报告未使用其他样本替代', 'status_code': 422}
    if target is None:
        order = {"blue": 0, "yellow": 1, "orange": 2, "red": 3,'unknown':4}
        target = max(samples, key=lambda s: order.get(s["risk_level"], 0))

    # 3. 问题构造 + 定量分析 + RAG 注入
    quant = {}
    if target.get("quant", {}).get("ndvi",{}).get('available',False):
        quant["index_stats"] = target["quant"]["ndvi"]
    quant["region_stats"] = {"num_regions": target["stats"]["num_regions"],
                             "num_regions_total": target["stats"].get(
                                 "num_regions_total", target["stats"]["num_regions"]),
                             "area_ratio": target["stats"]["area_ratio"],
                             "max_region_area": target["stats"]["max_region_area"],
                             "risk_name": RISK_CN[target["risk_level"]]}
    # 多粒度几何注入: 报告可引用真实的实例面积/周长/长轴 —— 替代 VLM 凭空编造尺寸
    geom, geom_wl = build_geometry_quant(target["stats"])
    if geom:
        quant["geometry"] = geom
    question, knowledge_meta = build_rag_question(target["stats"], RISK_CN[target["risk_level"]], quant)

    # 4. 多智能体分析 (带几何数字白名单 → 报告须引用真实几何, 编造的尺寸会被软告警)
    agents, final_answer, overlay_path, quality = analyze_with_agents(
        question, target["overlay_b64"], "detect", RISK_CN[target["risk_level"]],
        geom_whitelist=geom_wl, facts=report_facts(target), case_id=case_id)

    return cache_report({
        "region": region,
        "target_index": target["index"],
        "detection": target,
        "question": question,
        "agents": agents,
        "final_answer": final_answer,
        "knowledge": knowledge_meta,
        "quality": quality,
        "image_path": overlay_path,
        "is_mock": False,
        "generation": quality["generation"],
    }, case_id)


def run_pipeline_upload(image_b64, model_tag="mine", meters_per_pixel=None,
                        roi=None, channel_order="auto", nir_synth=False, instance_tau=0, case_id=None, analysis_grid='native'):
    payload = {"image": image_b64, "model_tag": model_tag, "channel_order": channel_order,
               "nir_synth": nir_synth, "instance_tau": instance_tau,'analysis_grid':analysis_grid}
    if meters_per_pixel not in (None, ""):
        payload["meters_per_pixel"] = meters_per_pixel
    if roi is not None:
        payload["roi"] = roi
    det = detector_request("/api/detect_upload", payload)
    if det is None:
        return {"error": "检测服务不可用", "status_code": 503}
    if "error" in det:
        return det
    stats = det["stats"]
    risk_cn = det["risk_name"]
    mpp = (det.get("scale_meta") or {}).get("meters_per_pixel")
    quant = {"region_stats": det.get("quant", {}).get("region")}
    ndvi = det.get("quant", {}).get("ndvi")
    if ndvi and ndvi.get("available", True):
        quant["index_stats"] = ndvi
    geom, whitelist = build_geometry_quant(stats, mpp)
    if geom:
        quant["geometry"] = geom
    question, knowledge = build_rag_question(stats, risk_cn, quant)
    if det.get("band_meta"):
        question += "\n输入波段说明：" + det["band_meta"].get("note", "")
    context=det.get('work_context') or {}
    if context.get('status')=='matched_public_expected_work_reference':
        context_text=(f"[WORKCTX-001] 公开年度作业参考：{context.get('source_title','')}；"
                      f"时间范围：{context.get('source_period','')}；"
                      f"本次已报告掩膜中有{context.get('predicted_pixels_overlapping_work_reference',0)}像素与参考重叠。"
                      "原始掩膜和筛查分级保持原样。该资料支持作业原因的复核线索，"
                      "不确认具体日级作业时间、事故发生、事故排除或边坡安全。"
                      f"来源：{context.get('source_url','')}")
        question+='\n'+context_text
        knowledge=list(knowledge)+[{'id':'WORKCTX-001','title':'公开年度采矿清表参考','source_title':context.get('source_title'),
                     'source_url':context.get('source_url'),'category':'作业原因复核','publisher':'Queensland Government',
                     'evidence_type':'年度遥感地类与现期外观参考；非现场安全鉴定',
                     'scope':context.get('source_period'),'locator':'SLATS original mine-clearing feature layer255',
                     'content':context_text}]
    facts = report_facts(det)
    agents, answer, image_path, quality = analyze_with_agents(
        question, det["overlay_b64"], "detect", risk_cn, geom_whitelist=whitelist, facts=facts, case_id=case_id)
    return cache_report({"region": "upload", "target_index": None, "detection": det,
                         "question": question, "agents": agents, "final_answer": answer,
                         "knowledge": knowledge, "quality": quality, "image_path": image_path,
                         "generation": quality["generation"], "is_mock": False}, case_id)


@app.route("/api/detect_upload", methods=["POST"])
def detect_upload_proxy():
    """上传影像检测 (代理到 GPU 检测服务)"""
    data = request.get_json(silent=True) or {}
    result = detector_request("/api/detect_upload", data)
    if result is None:
        return jsonify({"error": "检测服务不可用"}), 503
    return jsonify(result), result.get("status_code", 503) if "error" in result else 200


@app.route('/api/inspect_upload',methods=['POST'])
def inspect_upload_proxy():
    data=request.get_json(silent=True) or {}
    result=detector_request('/api/inspect_upload',data)
    if result is None: return jsonify({'error':'影像读取服务不可用'}),503
    return jsonify(result),result.get('status_code',503) if 'error' in result else 200


@app.route("/api/compare", methods=["POST"])
def compare_proxy():
    """多期影像变化对比 (代理到 GPU 检测服务)"""
    data = request.get_json(silent=True) or {}
    result = detector_request("/api/compare", data)
    if result is None:
        return jsonify({"error": "检测服务不可用"}), 503
    return jsonify(result), result.get("status_code", 503) if "error" in result else 200


def run_pipeline_compare(b64_t1, b64_t2, model_tag="mine", registration="auto", channel_order="auto", case_id=None):
    """
    多期影像变化趋势全链路: 变化检测 → 问题构造+RAG → 多智能体评估 → 趋势报告
    返回结构与 run_pipeline 一致 (detection 为变化统计, target_index=None)
    """
    det = detector_request("/api/compare", {"image_t1": b64_t1, "image_t2": b64_t2,
        "model_tag": model_tag, "registration": registration, "channel_order": channel_order})
    if det is None:
        return {"error": "检测服务不可用"}
    if "error" in det:
        return det
    st = det["stats"]

    # 变化趋势专用问题模板
    trend_parts = []
    if st["n_added_regions"] > 0:
        trend_parts.append(f"新增 {st['n_added_regions']} 处模型掩膜差分连通区域")
    if st["n_expanded_regions"] > 0:
        trend_parts.append(f"{st['n_expanded_regions']} 处掩膜扩大差分连通区域")
    if st["n_removed_regions"] > 0:
        trend_parts.append(f"{st['n_removed_regions']} 处掩膜消退差分连通区域")
    trend_desc = "、".join(trend_parts) if trend_parts else "本次差分连通区域计数未显示新增、扩大或消退"
    risk_level = (RISK_CN['red'] if st["t2_ratio"] >= 0.15 else
                  RISK_CN['orange'] if st["t2_ratio"] >= 0.05 else
                  RISK_CN['yellow'] if st["t2_ratio"] >= 0.01 else RISK_CN['blue'])

    question = (
        f"这是一处露天矿边坡在前后两个时期(T1→T2)的卫星遥感影像变化对比结果: "
        f"{trend_desc}。模型掩膜像素占比由 T1 的 {st['t1_ratio'] * 100:.2f}% 变化为 T2 的 "
        f"{st['t2_ratio'] * 100:.2f}%（净变化 {st['net_ratio'] * 100:+.2f} 个百分点），"
        f"T2按掩膜面积比例的原型筛查分级为{risk_level}，不能等同于法定预警。"
        f"这些量在共同有效像素范围内计算；差分连通区域不等于事故数量或匹配实例数量。"
        f"掩膜变化不能直接证明现场事故、边坡稳定或地面位移，变化原因应保持待复核。"
        f"请基于变化特征输出中文结构化报告: 1) 掩膜变化及量测依据; "
        f"2) 开采、堆排、复垦、观测条件与灾害等候选原因及核查要求; "
        f"3) 对应的监测与人工复核建议。"
        f"注意: 这是开放式评估任务, 直接输出报告内容, 不要输出 A/B/C/D/E 等字母选项。"
    )
    knowledge_meta = []
    if kb is not None:
        chunks = kb.retrieve("露天矿边坡 滑坡 变化 监测 预警 处置措施", k=2)
        if chunks:
            question = (f"请结合以下露天矿安全知识库资料进行分析:\n"
                        f"{kb.format_context(chunks)}\n\n{question}")
            knowledge_meta = knowledge_metadata(chunks)

    # 注入两期对比定量块
    quant_cmp = {}
    q = det.get("quant", {})
    if q.get("ndvi_t1", {}).get("available", False) and q.get("ndvi_t2", {}).get("available", False):
        # geo_tools 属检测侧仓库, 需容错: 缺失时跳过定量块而非让整个接口报错
        # (本调用点此前无保护, 是四处跨仓导入中唯一裸露的一处)
        try:
            from geo_tools import compare_periods as _cmp
            quant_cmp["compare"] = _cmp(
                {"area_ratio": st["t1_ratio"], "num_regions": 0, "risk_name": "T1"},
                {"area_ratio": st["t2_ratio"], "num_regions": st["n_added_regions"],
                 "risk_name": "T2"})
            quant_cmp["index_stats"] = q["ndvi_t2"]
        except Exception as e:
            print(f"[compare] 两期对比定量块跳过(geo_tools 不可用): {e}")
    if quant_cmp and format_quant_block is not None:
        try:
            blk = format_quant_block(**quant_cmp)
            if blk:
                question = blk + chr(10) + chr(10) + question
        except Exception:
            pass
    agents, final_answer, overlay_path, quality = analyze_with_agents(
        question + "\n配准与共同范围：" + json.dumps(det.get("registration", {}), ensure_ascii=False),
        det["change_overlay_b64"], "compare", risk_level,
        facts={"kind": "compare", "stats": st, "risk_name": risk_level,
               "model_tag": det.get("model_tag"), "model_info": det.get("model_info"),
               'request_id':det.get('request_id'),'input_quality':det.get('input_quality'),
               "band_meta": det.get("band_meta"), "registration": det.get("registration"),
               'scale_meta':det.get('scale_meta')}, case_id=case_id)

    return cache_report({
        "region": "compare",
        "target_index": None,
        "detection": {
            "overlay_b64": det["change_overlay_b64"],
            "t1_overlay_b64": det["t1_overlay_b64"],
            "t2_overlay_b64": det["t2_overlay_b64"],
            "t2_rgb_b64": det.get("t2_rgb_b64"),
            "stats": st,
            "change_instances": det.get("change_instances"),
            "size": det.get("size"),
            "scale_meta": det.get("scale_meta"),
            "registration": det.get("registration"),
            'request_id':det.get('request_id'),'input_quality':det.get('input_quality'),
            'band_meta':det.get('band_meta'),'threshold':det.get('threshold'),
            "model_tag": det.get("model_tag"),
            "model_info": det.get("model_info"),
            "risk_level": risk_level,
            "risk_name": risk_level,
        },
        "question": question,
        "agents": agents,
        "final_answer": final_answer,
        "knowledge": knowledge_meta,
        "quality": quality,
        "image_path": overlay_path,
        "is_mock": False,
        "generation": quality["generation"],
    }, case_id)


@app.route("/api/pipeline_compare", methods=["POST"])
def pipeline_compare():
    """多期影像变化趋势报告: 变化检测 → RAG → 多智能体评估"""
    data = request.get_json(silent=True) or {}
    t1, t2 = data.get("image_t1", ""), data.get("image_t2", "")
    if not t1 or not t2:
        return jsonify({"error": "image_t1 and image_t2 are required"}), 400
    result = run_pipeline_compare(t1, t2, model_tag=data.get("model_tag", "mine"),
                                  registration=data.get("registration", "auto"),
                                  channel_order=data.get("channel_order", "auto"), case_id=data.get('case_id'))
    status = result.get("status_code", 503) if "error" in result else 200
    return jsonify(result), status


@app.route("/api/pipeline", methods=["POST"])
def pipeline():
    """全链路: 滑坡检测 → RAG → 多智能体评估 → 预警报告"""
    data = request.get_json(silent=True) or {}
    result = run_pipeline(
        region=data.get("region", "Emilia-Romagna_Italy_32"),
        sample_index=data.get("sample_index"),
        num=int(data.get("num", 8)),
        fuse=bool(data.get("fuse", False)),
        ds_reject=bool(data.get("ds_reject", False)),
        case_id=data.get('case_id'),
    )
    status = result.get("status_code", 503) if "error" in result else 200
    return jsonify(result), status


KB_ANSWER_SYSTEM = (
    "你是露天矿安全领域的专家助理。请根据用户提供的知识库资料回答问题。\n"
    "要求:\n"
    "1. 只能基于提供的知识库资料回答, 不得编造资料中没有的内容;\n"
    "2. 引用资料时在句末标注来源编号, 格式如 [WARN-001];\n"
    "3. 回答结构清晰、条理分明, 直接给出答案, 无需客套;\n"
    "4. 如果资料不足以回答, 明确说明'现有知识库资料无法完整回答', 并给出资料中能找到的相关信息。"
)


def generate_kb_answer(question, chunks):
    """检索增强生成: 知识块 + 问题 → Qwen 合成带引用的回答
    失败返回 None (调用方降级为纯检索展示)"""
    if coordinator is None or not llm_fast_check():
        return None
    try:
        context = kb.format_context(chunks)
        model = create_model("qwen-max")
        from camel.agents import ChatAgent
        agent = ChatAgent(system_message=KB_ANSWER_SYSTEM, model=model, retry_attempts=1)

        def _ask():
            resp = agent.step(
                f"知识库资料:\n{context}\n\n"
                f"问题: {question}\n请给出回答。"
            )
            from evidence_integrity import completion_status
            if not completion_status(resp)['complete']:
                return None
            answer = resp.msgs[0].content if resp.msgs else None
            allowed_ids = {chunk["id"] for chunk in chunks}
            cited_ids = set(_re.findall(r"\[([A-Z]+-\d{3})\]", answer or ""))
            return answer if cited_ids and cited_ids <= allowed_ids else None

        return run_with_timeout(_ask, timeout=60)
    except Exception as e:
        print(f"[knowledge] 回答生成失败: {e}")
        return None


@app.route("/api/report", methods=["POST"])
def report():
    data = request.get_json(silent=True) or {}
    result = report_for_export(data, lambda: run_pipeline(
        region=data.get("region", "Emilia-Romagna_Italy_32"), sample_index=data.get("sample_index"),
        num=int(data.get("num", 8)), fuse=bool(data.get("fuse", False)), ds_reject=bool(data.get("ds_reject", False))))
    if "error" in result:
        return jsonify(result), result.get("status_code", 503)
    try:
        return jsonify({"report_b64": render_report_png(result), "target_index": result["target_index"],
                        "report_id": result.get("report_id")})
    except Exception as exc:
        return jsonify({"error": f"报告渲染失败: {exc}"}), 500


def render_report_png(result):
    """matplotlib 渲染 A4 风格预警报告 PNG → base64 (匿名安全: 无校名/人名)"""
    import base64 as _b64
    import io as _io
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image
    plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei"]
    plt.rcParams["axes.unicode_minus"] = False

    det = result["detection"]
    stats = det["stats"]
    answer = (result.get("final_answer") or "")
    decisions=(result.get('review_handoff') or {}).get('accepted_human_review_decisions') or []
    if decisions:
        answer += '\n\n人工复核意见（不改写原始量测；身份和依据尚未独立核验）\n'
        for decision in decisions[:5]:
            answer += (f"{decision.get('target_key')}：{decision.get('status_label')}；"
                       f"说明：{str(decision.get('note',''))[:400]}；依据：{str(decision.get('evidence',''))[:400]}\n")
        if len(decisions)>5:answer += '其余人工意见及全部修改历史见复核记录JSON。\n'
    # 清理 Markdown 纯样式符号: ** __ ` ` 与行首 #
    import re as _re
    answer = _re.sub(r"\*\*|__", "", answer)
    answer = _re.sub(r"`([^`]*)`", r"\1", answer)
    answer = _re.sub(r"(?m)^#{1,6}\s*", "", answer)
    from report_grounding import wrap_report_text
    answer = wrap_report_text(answer, 100)
    n_est = len(answer.splitlines())
    fig_h = max(11.7, 6.6 + (n_est + 4) * 0.18)

    fig = plt.figure(figsize=(8.27, fig_h), dpi=110)
    fig.patch.set_facecolor("white")

    # 标题栏
    fig.text(0.5, 1 - .45 / fig_h, "矿安天眼 · 露天矿滑坡辅助筛查报告", ha="center",
             fontsize=15, fontweight="bold", color="#003466")
    idx = result.get("target_index")
    src_label = f"样本 #{idx}" if idx is not None else "上传影像"
    region_label = result.get("region", "unknown")
    output_label = "多智能体分析" if result.get("generation", {}).get("status") == "live" else "检测事实汇总"
    fig.text(0.5, 1 - .75 / fig_h, f"来源: {region_label} | {src_label} | {output_label}",
             ha="center", fontsize=9, color="#666666")

    # 左: 检测标注图
    overlay = np.array(Image.open(_io.BytesIO(_b64.b64decode(det["overlay_b64"]))))
    ax_img = fig.add_axes([0.05, 1 - 5.4 / fig_h, 0.52, 4.2 / fig_h])
    ax_img.imshow(overlay)
    ax_img.axis("off")
    ax_img.set_title("疑似异常标注图 (红色 = 预测区域" +
                     ("，蓝框 = 监测范围)" if det.get("roi") else ")"), fontsize=9)

    # 右: 风险统计 + 几何形态 + 智能体摘要 + 知识依据
    ax_txt = fig.add_axes([0.60, 1 - 5.6 / fig_h, 0.35, 4.4 / fig_h])
    ax_txt.axis("off")
    n_total = stats.get("num_regions_total", stats["num_regions"])
    n_txt = f"{stats['num_regions']} 处" + (
        f"(共 {n_total})" if n_total != stats["num_regions"] else "")
    lines = [f"◆ 筛查分级: {det['risk_name']}",
             f"◆ 风险区域: {n_txt}",
             f"◆ 像素占比: {stats['area_ratio'] * 100:.1f}%",
             f"◆ 最大区域: {stats['max_region_area']} px"]
    if stats.get("iou") is not None:
        lines.append(f"◆ 数据集IoU: {stats['iou']}")
    lines.append("◆ 统计范围: " + ("监测范围内" if stats.get("scope") == "roi" else "整幅影像"))
    # 几何形态: 最多加 2 行 —— 右侧文本框每行仅约 23 个汉字宽, 行数/字数过多会溢出。
    # 面积一律由实例多边形给出(实测外接框只能还原 48.9% 的滑坡面积)。
    regs = stats.get("regions") or []
    if regs:
        r0 = regs[0]
        if r0.get("area_m2"):
            lines.append(f"◆ 最大实例: {r0['area_m2']:.0f} 平方米")
            lines.append(f"◆ 长轴/周长: {r0.get('major_axis_m', '-')} / "
                         f"{r0.get('perimeter_m', '-')} m")
        else:
            lines.append(f"◆ 最大实例: 周长 {r0.get('perimeter_px', '-')} px")
            lines.append(f"◆ 长轴/紧致度: {r0.get('major_axis_px', '-')} px / "
                         f"{r0.get('circularity', '-')}")
    generation = result.get("generation", {})
    lines.append("◆ 输出来源: " + ("多智能体分析" if generation.get("status") == "live" else "检测事实汇总"))
    lines += ["", "智能体协作:"]
    for a in result.get("agents", []):
        lines.append(f"  {a['step']}. {a['name']}")
    if result.get("knowledge"):
        lines.append("")
        lines.append("知识依据:")
        for k in result["knowledge"]:
            title = k["title"]
            if len(title) > 14:
                title = title[:14] + "…"
            lines.append(f"  · [{k['id']}] {title}")
    panel = wrap_report_text("\n".join(lines), 44)
    panel_font = min(8.8, max(6.5, 4.4 * 72 / (max(len(panel.splitlines()), 1) * 1.45)))
    ax_txt.text(0, 1, panel, va="top", ha="left", fontsize=panel_font,
                linespacing=1.45, wrap=False)

    # 底部: 报告全文
    ax_ans = fig.add_axes([0.05, .4 / fig_h, 0.90, (fig_h - 6.4) / fig_h])
    ax_ans.axis("off")
    ax_ans.text(0, 1, f"【{output_label}】\n{answer}", va="top", ha="left",
                fontsize=9, linespacing=1.45, wrap=False)

    buf = _io.BytesIO()
    fig.savefig(buf, format="png", facecolor="white")
    plt.close(fig)
    return _b64.b64encode(buf.getvalue()).decode()


@app.route("/api/pipeline_upload", methods=["POST"])
def pipeline_upload():
    data = request.get_json(silent=True) or {}
    if not data.get("image"):
        return jsonify({"error": "image is required"}), 400
    result = run_pipeline_upload(data["image"], **{
        key: data[key] for key in ("model_tag", "meters_per_pixel", "roi", "channel_order", "nir_synth", "instance_tau", 'case_id','analysis_grid')
        if key in data})
    return jsonify(result), result.get("status_code", 503) if "error" in result else 200


@app.route("/api/report_upload", methods=["POST"])
def report_upload():
    data = request.get_json(silent=True) or {}
    if not data.get("report_id") and not data.get("image"):
        return jsonify({"error": "image or report_id is required"}), 400
    result = report_for_export(data, lambda: run_pipeline_upload(data["image"], **{
        key: data[key] for key in ("model_tag", "meters_per_pixel", "roi", "channel_order", "nir_synth", "instance_tau",'analysis_grid')
        if key in data}))
    if "error" in result:
        return jsonify(result), result.get("status_code", 503)
    try:
        return jsonify({"report_b64": render_report_png(result), "region": "upload",
                        "report_id": result.get("report_id")})
    except Exception as exc:
        return jsonify({"error": f"报告渲染失败: {exc}"}), 500


@app.route('/api/report_compare', methods=['POST'])
def report_compare():
    data = request.get_json(silent=True)
    if not isinstance(data, dict) or not isinstance(data.get('report_id'), str) or not data['report_id']:
        return jsonify({'error': '请提供本次双期报告的 report_id'}), 400
    if data.get('format', 'html') not in ('html', 'json'):
        return jsonify({'error': 'format 仅支持 html 或 json'}), 400
    result = report_for_export(data, lambda: {'error': '导出必须来自同次缓存结果', 'status_code': 400})
    if 'error' in result:
        return jsonify(result), result.get('status_code', 503)
    if result.get('region') != 'compare':
        return jsonify({'error': '请使用双期变化报告的 report_id'}), 400
    try:
        return jsonify(export_comparison(result, data.get('format', 'html')))
    except ValueError as exc:
        return jsonify({'error': str(exc)}), 422


@app.route('/api/review_record', methods=['POST'])
def review_record():
    data = request.get_json(silent=True)
    if not isinstance(data, dict) or not data.get('report_id'):
        return jsonify({'error': '请提供本次报告的 report_id'}), 400
    result = report_for_export(data, lambda: {'error': '复核记录必须来自已生成的同次结果', 'status_code': 400})
    if 'error' in result:
        return jsonify(result), result.get('status_code', 503)
    record = result.get('review_handoff') or build_review_record(result)
    if not record['trace']['trace_complete']:
        return jsonify({'error': '本次结果缺少完整请求与权重标识，请重新生成后导出'}), 422
    fmt=data.get('format','json')
    if fmt not in ('json','html'):return jsonify({'error':'复核交接格式只支持json或html'}),400
    if fmt=='html':
        from review_packet import review_html
        evidence=None
        evidence_path=pathlib.Path(LMHLD_DIR)/'experiments/frozen_default_independent_audit_20261007/results.json'
        if evidence_path.is_file():
            audit=json.loads(evidence_path.read_text(encoding='utf-8'))
            metrics=audit['primary_reported_production_mask']
            evidence={'weights_sha256':audit['model_artifact']['weights_sha256'],
                'metrics':{'recall':metrics['recall_on_scored_published_reference'],'precision':metrics['precision_on_scored_published_reference'],'IoU':metrics['IoU_on_scored_published_reference']}}
        content=review_html(result,evidence)
        return jsonify({'report_id':result['report_id'],'filename':'矿安天眼_离线复核_'+result['report_id']+'.html',
            'mime_type':'text/html;charset=utf-8','content_b64':base64.b64encode(content.encode('utf-8')).decode('ascii')})
    content = json.dumps(record, ensure_ascii=False, indent=2, allow_nan=False) + '\n'
    return jsonify({'record': record, 'report_id': result['report_id'],
                    'filename': '矿安天眼_复核记录_' + result['report_id'] + '.json',
                    'content_b64': base64.b64encode(content.encode('utf-8')).decode('ascii')})


@app.route('/api/cases', methods=['GET','POST'])
def cases_api():
    if request.method == 'POST':
        data=request.get_json(silent=True) or {}
        return jsonify(case_archive.create_case(data.get('mine_name'),data.get('title'),data.get('notes',''))),201
    try:limit=int(request.args.get('limit',50));offset=int(request.args.get('offset',0))
    except ValueError:raise ArchiveError('分页参数须为整数')
    return jsonify(case_archive.list_cases(request.args.get('q',''),limit,offset))


@app.route('/api/cases/<case_id>', methods=['GET','PATCH'])
def case_detail_api(case_id):
    if request.method == 'PATCH':
        data=request.get_json(silent=True)
        if not isinstance(data,dict):raise ArchiveError('请求必须为JSON对象')
        return jsonify(case_archive.edit_case(case_id,data))
    try:limit=int(request.args.get('limit',50));offset=int(request.args.get('offset',0))
    except ValueError:raise ArchiveError('分页参数须为整数')
    return jsonify(case_archive.list_reports(case_id,limit,offset))


@app.route('/api/archive/reports/<report_id>', methods=['GET'])
def archived_report_api(report_id):
    return jsonify(case_archive.get_report(report_id))


@app.route('/api/archive/reports/<report_id>/reviews', methods=['POST'])
def submit_review_api(report_id):
    return jsonify(case_archive.submit_review(report_id,request.get_json(silent=True)))


@app.route('/api/archive/reports/<report_id>/manual_annotations', methods=['POST'])
def submit_manual_annotation_api(report_id):
    return jsonify(case_archive.submit_manual_annotation(report_id,request.get_json(silent=True)))


@app.route('/api/case_archive/backup', methods=['POST'])
def case_archive_backup_api():
    path=case_archive.backup()
    response=send_file(path,mimetype='application/vnd.sqlite3',as_attachment=True,download_name=path.name)
    # Wrap the stream so WSGI close runs our callback on completion/disconnect.
    response.direct_passthrough=False
    response.call_on_close(lambda:path.unlink(missing_ok=True))
    return response


@app.route("/api/stats", methods=["GET"])
def stats():
    """运行统计: 质量守卫/修复/降级/自评估通过率"""
    runs = max(_quality_stats["runs"], 1)
    ds = {}
    if not LOCAL_FACTS_ONLY:
        try:
            from coordinator.coordinator import get_dispatch_stats
            ds = get_dispatch_stats()
        except Exception:
            pass
    return jsonify({
        "runs": _quality_stats["runs"],
        "dispatch": ds,
        "guard_fail_checks": _quality_stats["checks"],
        "repair_attempts": _quality_stats["repair_attempts"],
        "repair_success": _quality_stats["repair_success"],
        "degraded": _quality_stats["degraded"],
        "degraded_rate": round(_quality_stats["degraded"] / runs, 3),
        "selfeval_total": _quality_stats["selfeval_total"],
        "selfeval_pass": _quality_stats["selfeval_pass"],
        "selfeval_pass_rate": round(_quality_stats["selfeval_pass"] / max(_quality_stats["selfeval_total"], 1), 3),
    })


@app.route("/api/knowledge", methods=["POST"])
def knowledge_query():
    """知识库问答 (RAG 检索增强生成)"""
    data = request.get_json(silent=True) or {}
    query = data.get("query", "").strip()
    if not query:
        return jsonify({"error": "query is required"}), 400
    if kb is None:
        return jsonify({"error": "知识库未加载"}), 503
    docs = kb.retrieve(query, k=3)
    if not docs:
        return jsonify({"query": query, "answer": "", "docs": [],
                        "answer_note": "未检索到相关知识"}), 200
    answer = generate_kb_answer(query, docs)
    resp = {"query": query, "docs": docs}
    if answer:
        resp["answer"] = answer
    else:
        resp["answer"] = ""
        resp["answer_note"] = "自动回答未完成或引用无法核验，已显示检索资料与原文来源。"
    return jsonify(resp)




if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5100))
    # 调试/热重载默认关闭 —— 2026-09-22 事故后改为显式开关:
    # 原先硬编码 debug=True, werkzeug 的 reloader 会监视被 import 的模块
    # (实测连 D:\LMHLD\eval_agent_trajectory.py 都在监视范围内)。任何一次
    # 文件改动都会重启服务并掐断正在处理的请求; 更严重的是 Windows 上 reloader
    # 的子进程存在套接字继承缺陷(WinError 10038), 子进程起不来时服务会直接停摆
    # —— 实测已发生一次(19:47 编辑评测脚本 → 后端再未恢复)。
    # 需要热重载时: set KAT_DEBUG=1
    debug = os.environ.get("KAT_DEBUG", "0").strip() in ("1", "true", "True")
    start_detector_service()  # 拉起 GPU 滑坡检测服务
    if AUTO_RECOVER_DETECTOR and not debug:
        from detector_supervision import DetectorSupervisor
        _detector_supervisor=DetectorSupervisor(lambda:_detector_proc,start_detector_service,detector_port_active)
        _detector_supervisor.start()
    print(f"\n后端启动中: http://localhost:{port}"
          f"  (热重载: {'开' if debug else '关'})\n")
    print(f"从 http://127.0.0.1:{port}/ 使用分析、归档与复核功能\n")
    from server_runtime import serve_app
    serve_app(app, port, debug=debug)
