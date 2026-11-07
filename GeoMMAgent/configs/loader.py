# configs/loader.py
# Load YAML config and build per-agent models + AgentCoordinator.

import os
import pathlib
from typing import Any, Dict, List, Optional

import yaml

AGENT_REGISTRY = {}


def _ensure_registry():
    if AGENT_REGISTRY:
        return
    from exec_agents import (
        FormatConversionAgent,
        ImageFilterAgent,
        ScaleAgent,
        SuperResolutionAgent,
        ClsAgent,
        DetAgent,
        SegAgent,
        SearchAgent,
        RetrievalAgent,
        ReasoningAgent,
        MatchingAgent,
        SelfEvaluationAgent,
    )
    AGENT_REGISTRY.update({
        "FormatConversionAgent": FormatConversionAgent,
        "ImageFilterAgent": ImageFilterAgent,
        "ScaleAgent": ScaleAgent,
        "SuperResolutionAgent": SuperResolutionAgent,
        "ClsAgent": ClsAgent,
        "DetAgent": DetAgent,
        "SegAgent": SegAgent,
        "SearchAgent": SearchAgent,
        "RetrievalAgent": RetrievalAgent,
        "ReasoningAgent": ReasoningAgent,
        "MatchingAgent": MatchingAgent,
        "SelfEvaluationAgent": SelfEvaluationAgent,
    })


def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    if config_path is None:
        config_path = str(
            pathlib.Path(__file__).parent / "GeoMMBench.yaml"
        )
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def apply_seg_env_from_config(cfg: Dict[str, Any]) -> None:
    """SegAgent / perception：DeepLab 权重路径写入环境变量（setdefault）。"""
    seg = cfg.get("agents", {}).get("SegAgent") or {}
    dws = seg.get("deeplab_weights")
    if not dws:
        return
    w0 = dws[0] if isinstance(dws, list) else None
    if not isinstance(w0, dict):
        return
    project_root = pathlib.Path(__file__).resolve().parents[1]

    p = w0.get("path")
    if p:
        if not pathlib.Path(p).is_absolute():
            p = str((project_root / p).resolve())
        os.environ.setdefault("GEOMM_DEEPLAB_WEIGHTS", p)
    if w0.get("num_classes") is not None:
        os.environ.setdefault(
            "GEOMM_DEEPLAB_NUM_CLASSES",
            str(int(w0["num_classes"])),
        )
    if w0.get("output_stride") is not None:
        os.environ.setdefault(
            "GEOMM_DEEPLAB_OUTPUT_STRIDE",
            str(int(w0["output_stride"])),
        )


def apply_gme_env_from_config(cfg: Dict[str, Any]) -> None:
    """
    将 ``gme:`` 段写入 ``GEOMM_*`` 环境变量（仅 ``setdefault``，不覆盖已有值）。
    便于与 SearchAgent / gme_filter 共用；``.env`` 中已设的项优先生效。
    """
    gme = cfg.get("gme") or {}
    if not gme:
        return

    project_root = pathlib.Path(__file__).resolve().parents[1]

    def _abs(p: str) -> str:
        p = str(p).strip()
        if not p:
            return p
        pp = pathlib.Path(p)
        if not pp.is_absolute():
            return str((project_root / pp).resolve())
        return str(pp.resolve())

    mp = gme.get("model_path")
    if mp:
        os.environ.setdefault("GEOMM_GME_MODEL_PATH", _abs(mp))

    if gme.get("top_k") is not None:
        os.environ.setdefault("GEOMM_GME_TOP_K", str(int(gme["top_k"])))

    if gme.get("img_batch") is not None:
        os.environ.setdefault("GEOMM_GME_IMG_BATCH", str(int(gme["img_batch"])))

    if gme.get("max_evidence_images") is not None:
        os.environ.setdefault(
            "GEOMM_MAX_EVIDENCE_IMAGES",
            str(int(gme["max_evidence_images"])),
        )

    instr = gme.get("instruction")
    if instr:
        os.environ.setdefault("GEOMM_GME_INSTRUCTION", str(instr).strip())

    if gme.get("disable") is True:
        os.environ.setdefault("GEOMM_GME_DISABLE", "1")


STREAM_ONLY_MODELS = {"qvq-max", "qvq-plus"}
DEFAULT_MODEL = 'qwen3-vl-plus'


def create_model_from_config(
    model_type: str, temperature: float = 0, max_tokens: int = 1600,
):
    from camel.models import ModelFactory
    from camel.types import ModelPlatformType

    config = {"temperature": temperature, 'max_tokens': max_tokens}
    if model_type.startswith(('qwen3-', 'qwen3.')):
        config['extra_body'] = {'enable_thinking': False}
    if model_type in STREAM_ONLY_MODELS:
        config["stream"] = True
    return ModelFactory.create(
        model_platform=ModelPlatformType.QWEN,
        model_type=model_type,
        model_config_dict=config,
        url=os.environ.get('QWEN_BASE_URL') or os.environ.get('QWEN_API_BASE_URL') or 'https://dashscope.aliyuncs.com/compatible-mode/v1',
        timeout=float(os.environ.get('MODEL_TIMEOUT', '45')),
        max_retries=0,
    )


def build_agents_from_config(
    cfg: Dict[str, Any],
    open_ended: bool = False,
) -> List[Dict[str, Any]]:
    """Build worker dicts for all enabled agents according to config."""
    apply_gme_env_from_config(cfg)
    apply_seg_env_from_config(cfg)
    _ensure_registry()
    agents_cfg = cfg.get("agents", {})
    worker_dicts = []

    for agent_name, agent_cfg in agents_cfg.items():
        if not agent_cfg.get("enabled", True):
            continue

        cls = AGENT_REGISTRY.get(agent_name)
        if cls is None:
            continue

        model = create_model_from_config(
            model_type=agent_cfg.get("model", DEFAULT_MODEL),
            temperature=agent_cfg.get("temperature", 0),
            max_tokens=agent_cfg.get('max_tokens', 1600),
        )

        kwargs = {"model": model}
        if open_ended and agent_name == 'ReasoningAgent':
            kwargs['system_prompt_override'] = (
                '你是露天矿影像监测的推理角色。根据当前任务、工具事实和前序角色输出生成中文分析报告。'
                '先核对数值、单位、时相和资料所属案例；发现冲突时保持待复核，不把其他矿区的量测并入本次结果。'
                '按检测事实、解释及局限、人工复核和监测建议展开。引用仅限给定来源编号。'
                '未知观测不得补造，不能由掩膜确认事故、安全或失稳时刻。'
                '这是开放式任务，输出正文，不输出选项或答案字母。')

        project_root = pathlib.Path(__file__).resolve().parents[1]

        if agent_name == "SegAgent" and agent_cfg.get("deeplab_weights"):
            dws = []
            for w in agent_cfg["deeplab_weights"]:
                w = dict(w)
                p = w.get("path")
                if p and not pathlib.Path(p).is_absolute():
                    w["path"] = str(project_root / p)
                dws.append(w)
            kwargs["deeplab_weights"] = dws
        elif "yolo_weights" in agent_cfg:
            kwargs["yolo_weights"] = agent_cfg["yolo_weights"]
        elif "yolo_model_path" in agent_cfg:
            yolo_path = agent_cfg["yolo_model_path"]
            if not pathlib.Path(yolo_path).is_absolute():
                yolo_path = str(project_root / yolo_path)
            kwargs["yolo_model_path"] = yolo_path

        # 超分权重 / 第三方源码路径：供 toolkit.general.super_resolve 读取
        if agent_name == "SuperResolutionAgent":
            rp = (
                agent_cfg.get("sr_weights")
                or agent_cfg.get("realesrgan_model_path")
                or agent_cfg.get("realesrgan_weights")
            )
            if rp:
                if not pathlib.Path(rp).is_absolute():
                    rp = str(project_root / rp)
                os.environ["GEOMM_SR_WEIGHTS"] = rp
            rr = agent_cfg.get("sr_repo") or agent_cfg.get("realesrgan_repo")
            if rr:
                if not pathlib.Path(rr).is_absolute():
                    rr = str(project_root / rr)
                os.environ["GEOMM_SR_REPO"] = rr
            tile = agent_cfg.get("sr_tile", agent_cfg.get("realesrgan_tile"))
            if tile is not None:
                os.environ["GEOMM_SR_TILE"] = str(int(tile))

        agent_instance = cls(**kwargs)
        worker = agent_instance.as_worker_dict()
        if open_ended and agent_name == 'ReasoningAgent':
            worker['description'] = 'ReasoningAgent: 核对当前案例、工具事实与前序证据，输出完整中文监测解释与复核建议。'
        worker_dicts.append(worker)

    return worker_dicts
