# -*- coding: utf-8 -*-
"""
detector_service.py — 矿安天眼 · 滑坡检测独立服务 (GPU)
=========================================================
运行环境: lmhld conda 环境 (torch 2.5.1 CUDA)
由 D:\\GeoMMAgent\\backend.py 启动时自动拉起 (端口 5001)

API:
  GET  /api/health              → {status, detector_loaded}
  GET  /api/regions             → 可用数据集区域列表
  POST /api/detect              → {region, num} 数据集模式检测
  POST /api/detect_upload       → {image} base64 上传影像检测 (滑窗)

用法: python detector_service.py [port]
"""
import os
import sys
import json
import base64
import io
import csv
import gc
import threading
import uuid
from functools import wraps
import numpy as np
import cv2
import torch
from flask import Flask, request, jsonify, g
from local_access import install_local_access

LMHLD_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, LMHLD_DIR)
from detector import LandslideDetector, RISK_NAMES, ModelNumericalError, ModelComponentError
from spatial_input_contract import SpatialInputContract
from model_registry import ModelRegistry, ModelArtifactError
from mine_work_context import annotate_known_work
from input_quality import InputQualityError, inspect_quality, require_usable_input, prepare_inference_context
from protected_inputs import input_split
from geo_tools import compute_spectral_index, compute_region_stats, compare_periods
from upload_utils import decode_upload, detection_options, align_periods,period_chronology
from raster_upload import public_band_meta,observed_mask,resolve_mpp,calibrated_spectral_image
from geo_geometry import (mask_to_regions, region_record, attach_scale,
                          regions_to_geojson, regions_to_csv, instance_overlap,
                          instance_prob_filter,
                          label_instances, MAX_REGIONS_LITE, LITE_MAX_VERTICES,
                          DEFAULT_SIMPLIFY_EPS)

from data_paths import resolve_data_root
from runtime_identity import runtime_fingerprint
DETECTOR_RUNTIME_SHA256=runtime_fingerprint(LMHLD_DIR)
DATA_ROOT = resolve_data_root("native")
MODELS_ROOT = os.environ.get("KAT_MODELS_ROOT", os.path.join(LMHLD_DIR, "models_prior"))
model_registry = ModelRegistry(os.environ.get('KAT_MODEL_REGISTRY', os.path.join(LMHLD_DIR, 'model_registry.json')), MODELS_ROOT)
# 区域 → (权重路径, patch_size)  使用先验加权训练模型
# 九寨沟用 λ=0.5 校准模型(λ 旋钮校准成果); 其余用 λ=1.0
WEIGHTS = {
    "Emilia-Romagna_Italy_32": (os.path.join(MODELS_ROOT, "Emilia-Romagna_Italy_32_prior1.0_seed42_pre_dice", "best_model.pth"), 32),  # 精度冲刺: 预训练+Dice (+1.66pp)
    "Gorkha_Nepal_32": (os.path.join(MODELS_ROOT, "Gorkha_Nepal_32_prior1.0_seed42", "best_model.pth"), 32),
    "Hokkaido_Japan_128": (os.path.join(MODELS_ROOT, "Hokkaido_Japan_128_prior1.0_seed42", "best_model.pth"), 128),
    "Jiuzhaigou_China_128": (os.path.join(MODELS_ROOT, "Jiuzhaigou_China_128_prior0.5_seed42_pre_dice", "best_model.pth"), 128),  # 精度冲刺: (+2.42pp, 小区召回+25pp)
    "Wenchuan_China_224": (os.path.join(MODELS_ROOT, "Wenchuan_China_224_prior1.0_seed42_pre_dice", "best_model.pth"), 224),  # 精度冲刺: (+0.59pp)
    "Taiwan_China_224": (os.path.join(MODELS_ROOT, "Taiwan_China_224_prior1.0_seed42", "best_model.pth"), 224),
}
# 多尺度融合: Hokkaido 区域可用 32px 细粒度模型融合
FUSION_REGION = "Hokkaido_Japan_128"
FUSION32_WEIGHTS = os.path.join(MODELS_ROOT, "Hokkaido_Japan_32_prior1.0_seed42", "best_model.pth")
# 上传影像域适配模型 (0-255 归一化输入, 微调自 Emilia λ=1.0)
UPLOAD_WEIGHTS = os.path.join(MODELS_ROOT, "Emilia-Romagna_Italy_32_upload_seed42", "best_model.pth")
# 可选的场景适配模型 (逐矿区小样本适配 / 跨域统一模型)
# 每项: (权重路径, patch_size, 归一化协议, 输入缩放)
UPLOAD_MODELS = {
    "base": (UPLOAD_WEIGHTS, 32, "minmax", 1.0),
    # 白格场景适配模型: 120 样本适配, 目标域响应 0.32→0.57, 森林 0.24
    "baige": (os.path.join(MODELS_ROOT, "baige_adapted_ps32", "best_model.pth"), 32, "minmax", 1.0),
    # 跨域统一模型(6 区域联合训练, 原始DN域): 真实影像 ×10 缩放输入,
    # 白格场景选择性 +0.29(免适配), 优于基座(+0.11), 接近专用适配模型(+0.33)
    "unified": (os.path.join(MODELS_ROOT, "unified_pre_seed42", "best_model.pth"), 32, None, 10.0),
    # 真实影像域适配模型(第十三组成果): 基座 + 真实域负样本(城区/农田/港口/林地/裸地)
    # + LMHLD 混合数据(防遗忘) + NIR dropout(训练时 50% 概率把 NIR 换成 R)。
    # 实测: 真实负样本图块误报率 88.97%→6.33%(**留出集口径**); 未见场景(抚顺)整幅
    # 风险占比 30%→1.4%; 白格滑坡分离度 +0.079→+0.243; 源域 F1 保留 89%(0.737→0.657)。
    # ⚠ 原记 88.8%→5.8% 为污染口径(评估 5501 个, 其中 2500 训练见过 → 45% 重叠);
    #   第十六组复核后修正, 见 experiments/nir_train/leak_quantify.json。
    # 协议与 base 相同(逐图块 min-max)。
    "realfit": (os.path.join(MODELS_ROOT, "real_ft_v1_seed42", "best_model.pth"), 32, "minmax", 1.0),
    # 真实**矿区**域适配模型（第十九/二十组）: realfit 配方 + 矿区负样本
    # （准东干旱戈壁 600 + 工业设施 1400 —— **只用不可能变形的安全层**，
    #  采场台阶/排土场**明确排除**）。
    # 第二十组 T1b 实测: 同阈值 0.40 下对比第十九组的 MIN8(800) ——
    #   矿区误报 5.13/7.45/7.63% vs 10.74/12.41/11.60%（**降约一半**）;
    #   白格滑坡区风险 38.7% vs 40.6%（持平）、背景 11.5% vs 12.9%（更低）、
    #   分离度 +0.2076 vs +0.1980; 源域 F1 保留 88.6%; 相对分离度 0.895 vs 0.781。
    # 第二十组同时证明「边界纪律」成立: 把采坑也作负样本的那组**误报更低但白格分离度
    #   掉 40%**（0.125 vs 0.208）—— 只看误报会做出完全错误的选择。
    "mine": (os.path.join(MODELS_ROOT, "real_ft_T1b_ours_seed42", "best_model.pth"),
             32, "minmax", 1.0),
    # Optional, internally calibrated probability blend; no field-safety claim.
    "rgb_guarded": (os.path.join(MODELS_ROOT, "rgb_guarded_20261001", "adapter.json"),
                    32, "minmax", 1.0),
}

# 各上传通道的判定阈值（与 UPLOAD_MODELS 分离, 避免改动其元组结构 —— 有 7 处消费者
# 按 (路径, patch, 协议, 缩放) 四元组解包）。默认 0.5。
UPLOAD_THRESHOLDS = {
    "base": 0.5, "baige": 0.5, "unified": 0.5, "realfit": 0.5,
    # 第二十组: 矿区负样本使模型整体更保守, 配套下调以保持召回（实测阈值曲线见
    # experiments/mining_neg/t1b_threshold.json）
    "mine": 0.40,
    "rgb_guarded": 0.50,
}
DEFAULT_REGION = "Emilia-Romagna_Italy_32"

_detector32 = None       # 融合用小尺度模型(懒加载)
_upload_detector = None  # 上传检测用域适配模型(懒加载)
_upload_detectors = {}   # 多场景适配模型缓存 {tag: detector}

app = Flask(__name__)
install_local_access(app)
app.config["MAX_CONTENT_LENGTH"] = 48 * 1024 * 1024

@app.errorhandler(413)
def oversized_request(error):
    return jsonify({'error': '请求超过服务大小限制，请裁剪影像或缩小共同分析范围后重试',
                    'status_code': 413, 'max_request_bytes': app.config['MAX_CONTENT_LENGTH']}), 413

_inference_lock = threading.Lock()


def exclusive_inference(function):
    """Bound both GPU concurrency and model-switch races without an unbounded queue."""
    @wraps(function)
    def guarded(*args, **kwargs):
        if not getattr(g,'inference_slot',False) and not _inference_lock.acquire(blocking=False):
            response = jsonify({'error': '检测正在处理其他请求，请稍后重试',
                                'inference_status': 'busy', 'retry_after_seconds': 2})
            response.headers['Retry-After'] = '2'
            return response, 503
        g.inference_slot=True
        try:
            return function(*args, **kwargs)
        except torch.cuda.OutOfMemoryError:
            _release_models()  # Clear only while the failing request still owns capacity.
            raise
        finally:
            g.inference_slot=False
            _inference_lock.release()
    return guarded


@app.after_request
def trace_response(response):
    if getattr(g,'inference_slot',False):
        g.inference_slot=False
        _inference_lock.release()
    request_id = getattr(g, 'request_id', uuid.uuid4().hex)
    response.headers['X-KAT-Request-ID'] = request_id
    if response.is_json:
        body = response.get_json()
        if isinstance(body, dict):
            body['request_id'] = request_id
            response.set_data(app.json.dumps(body))
    return response


@app.before_request
def validate_json_request():
    g.request_id = uuid.uuid4().hex
    if request.method=='POST' and request.endpoint in ('detect','detect_upload','compare','inspect_upload'):
        if not _inference_lock.acquire(blocking=False):
            response=jsonify({'error':'影像处理正在进行，请稍后重试','inference_status':'busy','retry_after_seconds':2})
            response.headers['Retry-After']='2'
            return response,503
        g.inference_slot=True
    if request.method == "POST" and not isinstance(request.get_json(silent=True), dict):
        return jsonify({"error": "请求必须为 JSON 对象"}), 400

detector = None
_region_cache = {}   # region → test images/labels npy 缓存


def _get_region_npy(region):
    """加载区域 test 集 (缓存)"""
    if region not in _region_cache:
        region_dir = os.path.join(DATA_ROOT, region)
        _region_cache.clear()
        imgs = np.load(os.path.join(region_dir, "test_images.npy"), mmap_mode='r')
        lbls = np.load(os.path.join(region_dir, "test_labels.npy"), mmap_mode='r')
        _region_cache[region] = (imgs, lbls)
    return _region_cache[region]


def _load_detector():
    global detector
    if detector is not None:
        return
    for region, (wpath, ps) in WEIGHTS.items():
        if os.path.exists(wpath):
            _load_detector_region(region)
            return
    print("[detector_service] 未找到权重文件!")


def _encode_img(arr):
    """numpy RGB 数组 → base64 PNG
    注意: cv2.imencode 按 BGR 处理, 传入的是 RGB 数组, 须先转换, 否则红蓝互换"""
    bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR) if arr.ndim == 3 else arr
    ok, buf = cv2.imencode(".png", bgr)
    return base64.b64encode(buf.tobytes()).decode()


def _prepare_upload(image_b64, channel_order="AUTO", return_meta=False):
    """Legacy array API; endpoints request band metadata and report validation errors."""
    try:
        image, meta = decode_upload(image_b64, channel_order)
    except ValueError:
        if return_meta:
            raise
        return None
    return (image, meta) if return_meta else image


def _raster_rgb(image,meta):
    visible=image[:,:,:3].astype(np.float32).copy(); valid=observed_mask(meta,image.shape[:2])
    for c in range(3):
        lo,hi=np.percentile(visible[:,:,c][valid],[2,98])
        visible[:,:,c]=np.clip((visible[:,:,c]-lo)/max(hi-lo,1e-6),0,1)*255
    visible[~valid]=0
    return np.ascontiguousarray(visible[:,:,::-1].astype(np.uint8))


def _upload_spectral_index(image,meta,index):
    if meta.get('nir_source')=='measured' and not meta.get('spectral_indices_available',False):
        return {'index':index.upper(),'available':False,'nir_source':'measured',
                'reason':'真实NIR存在，但辐射定标未知，不能输出物理光谱指数'}
    result=compute_spectral_index(image,index,nir_source=meta['nir_source'])
    result['radiometry_status']=meta.get('radiometry_status','unknown')
    return result


@app.route('/api/inspect_upload',methods=['POST'])
@exclusive_inference
def inspect_upload():
    data=request.get_json(silent=True) or {}
    try:
        opts=detection_options(data)
        image,meta=_prepare_upload(data.get('image',''),opts['channel_order'],True)
        mpp,source=resolve_mpp(opts['meters_per_pixel'],meta)
    except ValueError as exc:
        return jsonify({'error':str(exc)}),400
    rgb=_raster_rgb(image,meta)
    if max(rgb.shape[:2])>2048:
        ratio=2048/max(rgb.shape[:2]); rgb=cv2.resize(rgb,None,fx=ratio,fy=ratio,interpolation=cv2.INTER_AREA)
    return jsonify({'preview_b64':_encode_img(rgb),'image_shape':list(image.shape[:2]),
                    'band_meta':public_band_meta(meta),'scale_meta':{'meters_per_pixel':mpp,'source':source} if mpp else None,
                    'preview_is_display_only':True, 'input_quality':inspect_quality(image,meta),
                    'data_split':input_split(meta)})


def _parse_roi(raw, shape):
    """解析「监测范围」→ 像素框 (x0,y0,x1,y1)；非法返回 None。

    接受两种写法（比例写法不受前端缩放影响，推荐）:
      · [x0, y0, x1, y1]                                      —— 像素
      · {"x0":..,"y0":..,"x1":..,"y1":..,"unit":"fraction"}   —— 0~1 比例
    越界自动裁剪；面积 < 图幅 1% 判为非法（统计样本太小没有意义）。
    一律不抛异常 —— 全链路不得有未捕获异常（一次 500 会打挂整个请求）。
    """
    H, W = shape
    try:
        if isinstance(raw, dict):
            vals = [float(raw[k]) for k in ("x0", "y0", "x1", "y1")]
            if str(raw.get("unit", "fraction")) == "fraction":
                vals = [vals[0] * W, vals[1] * H, vals[2] * W, vals[3] * H]
        elif isinstance(raw, (list, tuple)) and len(raw) == 4:
            vals = [float(v) for v in raw]
            if max(vals) <= 1.0:            # 四个数都在 0~1 → 视为比例
                vals = [vals[0] * W, vals[1] * H, vals[2] * W, vals[3] * H]
        else:
            return None
        x0, x1 = sorted((max(0.0, min(vals[0], W)), max(0.0, min(vals[2], W))))
        y0, y1 = sorted((max(0.0, min(vals[1], H)), max(0.0, min(vals[3], H))))
        if (x1 - x0) * (y1 - y0) < 0.01 * H * W:
            return None
        return int(x0), int(y0), int(x1), int(y1)
    except (TypeError, ValueError, KeyError):
        return None


@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "detector_loaded": detector is not None,
                    'service':'kat-detector','runtime_pid':os.getpid(),
                    'runtime_sha256':DETECTOR_RUNTIME_SHA256,
                    'runtime_root':LMHLD_DIR,
                    'runtime_python_prefix':sys.prefix, 'http_server':request.environ.get('SERVER_SOFTWARE', 'unknown'),
                    'http_runtime':app.config.get('KAT_HTTP_RUNTIME','in_process_test'),
                    'cuda_memory': {'allocated':torch.cuda.memory_allocated(),'reserved':torch.cuda.memory_reserved(),
                                    'peak_allocated':torch.cuda.max_memory_allocated(),'peak_reserved':torch.cuda.max_memory_reserved()} if torch.cuda.is_available() else None,
                    'inference_busy':_inference_lock.locked(), 'maximum_parallel_inference':1,
                    'upload_models_loaded':list(_upload_detectors),
                    "regions": list(WEIGHTS.keys())})


@app.route("/api/regions", methods=["GET"])
def regions():
    return jsonify({"regions": list(WEIGHTS.keys())})


@app.route("/api/detect", methods=["POST"])
@exclusive_inference
def detect():
    """
    数据集模式: 从指定区域 test 集检测 N 个样本
    请求: {region, num(默认8), start(起始索引, 默认0)}
    返回: {samples: [{index, risk_level, risk_name, stats, rgb_b64, overlay_b64}]}
    """
    data = request.get_json(silent=True) or {}
    region = data.get("region", DEFAULT_REGION)
    if not isinstance(region,str) or region not in WEIGHTS:
        return jsonify({"error": f"未知区域 {region}"}), 400
    num, start = data.get('num',8), data.get('start',0)
    if isinstance(num,bool) or not isinstance(num,int) or not 1 <= num <= 64 or isinstance(start,bool) or not isinstance(start,int) or start < 0:
        return jsonify({'error':'num须为1到64的整数，start须为非负整数'}),400
    fuse, ds_reject = data.get('fuse',False), data.get('ds_reject',False)
    if not isinstance(fuse,bool) or not isinstance(ds_reject,bool):
        return jsonify({'error':'fuse和ds_reject须为布尔值'}),400
    if (fuse and region!=FUSION_REGION) or (ds_reject and not fuse):
        return jsonify({'error':'融合仅支持Hokkaido，D-S复核须同时开启融合'}),400
    current = _load_detector_region(region)
    if current is None:
        return jsonify({'error':f'区域 {region} 的权重不可用，未使用其他区域模型'}),503
    imgs, lbls = _get_region_npy(region)
    n = len(imgs)
    if start >= n:
        return jsonify({'error':'start超出该区域测试集范围'}),400
    end = min(start + num, n)
    idxs = list(range(start, end))

    probs = current.predict_batch(imgs[start:end])
    # 多尺度融合模式 (仅 Hokkaido 区域支持: 128px 大尺度 + 32px 细粒度)
    fused = False
    ds_rejected_px = 0
    rejected = np.zeros(probs.shape,bool)
    if fuse and region == FUSION_REGION:
        det32 = _load_detector32()
        if det32 is None:
            return jsonify({'error':'所选融合模型不可用，未执行单模型替代'}),503
        if det32 is not None:
            probs32 = np.stack([
                det32.detect_image(imgs[i].transpose(1, 2, 0), stride=32)
                for i in range(start, end)
            ])
            primary_probs = probs
            probs = _fuse_probs(primary_probs, probs32)
            fused = True
            # D-S 冲突拒绝: 两尺度证据冲突 K>θ → 判定不确定, 不触发预警 (抑制误报)
            if ds_reject:
                k_map = _ds_conflict_map(primary_probs, probs32)
                rejected = k_map > DS_REJECT_THETA
                ds_rejected_px = int(rejected.sum())
    masks = (probs >= detector.threshold).astype(np.uint8)
    masks[rejected] = 0
    gt = (lbls[start:end, 0] > 0).astype(np.uint8)

    samples = []
    for k, i in enumerate(idxs):
        r = detector.analyze_mask(masks[k])
        accepted = ~rejected[k]
        inter = (np.logical_and(masks[k], gt[k])&accepted).sum()
        union = (np.logical_or(masks[k], gt[k])&accepted).sum()
        r["iou"] = round(float(inter) / (union + 1e-6), 4)
        r['unverified_pixels'] = int(rejected[k].sum())
        r['valid_pixels'] = int(accepted.sum())
        r['area_ratio'] = round(r['landslide_pixels']/max(r['valid_pixels'],1),4)
        r['area_ratio_available'] = bool(r['valid_pixels'])
        r['iou_available'] = bool(r['valid_pixels'] and union)
        if not r['iou_available']:
            r['iou']=None
        r['iou_scope'] = 'nonconflicting_pixels'
        r['screening_status'] = 'needs_review' if r['unverified_pixels'] else 'screening_complete'
        if r['unverified_pixels']:
            r['risk_level']='unknown'
        r["label_ratio"] = round(float(gt[k].sum() / gt[k].size), 4)

        rgb = detector.to_rgb(imgs[i].transpose(1, 2, 0))
        overlay = detector.overlay(rgb, masks[k])
        # 定量分析工具: 光谱指数(基于原始 4 通道数据)
        try:
            quant = {"ndvi": {'index':'NDVI','available':False,'nir_source':'dataset_BGRN',
                              'reason':'演示图块未附原始辐射定标，不能作为物理NDVI测量'}}
        except Exception:
            quant = {}
        samples.append({
            "index": int(i),
            "quant": quant,
            "risk_level": r["risk_level"],
            "risk_name": RISK_NAMES[r["risk_level"]][0],
            "risk_color": RISK_NAMES[r["risk_level"]][1],
            "stats": {k2: v for k2, v in r.items() if k2 != "risk_level"},
            "rgb_b64": _encode_img(rgb),
            "overlay_b64": _encode_img(overlay),
        })

    return jsonify({"region": region, "total": n, "fused": fused,
                    'model_info':getattr(current,'artifact_info',None),
                    "ds_rejected_px": ds_rejected_px, "samples": samples})


@app.route("/api/detect_upload", methods=["POST"])
@exclusive_inference
def detect_upload():
    data = request.get_json(silent=True) or {}
    try:
        opts = detection_options(data)
        tag = opts["model_tag"]
        if tag not in UPLOAD_MODELS:
            raise ValueError(f"未知检测模型: {tag}")
        if tag == "rgb_guarded" and opts["nir_synth"]:
            raise ValueError("RGB联合适配验证版未校准合成NIR输入")
        image, band_meta = _prepare_upload(data.get("image", ""), opts["channel_order"], True)
        mpp,mpp_source=resolve_mpp(opts['meters_per_pixel'],band_meta)
        roi_raw = data.get("roi")
        box = _parse_roi(roi_raw, image.shape[:2]) if roi_raw not in (None, "", [], {}) else None
        if roi_raw not in (None, "", [], {}) and box is None:
            raise ValueError("roi 非法：范围须位于图幅内且面积不少于1%")
        valid=observed_mask(band_meta,image.shape[:2]); scope_valid=valid.copy()
        if box:
            x0,y0,x1,y1=box; scope_valid[:y0]=False; scope_valid[y1:]=False
            scope_valid[:,:x0]=False; scope_valid[:,x1:]=False
        if not scope_valid.any(): raise ValueError('监测范围内没有有效观测像素，不能给出低风险结论')
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    requested_scope = np.ones(valid.shape,bool)
    if box:
        requested_scope[:]=False
        requested_scope[y0:y1,x0:x1]=True
    input_quality = require_usable_input(image,band_meta,requested_scope)
    split=input_split(band_meta)
    if split['reserved']:
        return jsonify({'error':'该影像涉及隔离的保留测试矿范围，日常上传不能执行预测；最终评估须使用冻结的一次测试流程',
                        'inference_status':'reserved_test_protected','case_id':split['case_id']}),409
    try:
        spatial=SpatialInputContract(image,valid,mpp,opts['analysis_grid'])
    except ValueError as error:return jsonify({'error':str(error)}),400
    source_valid=valid
    valid=spatial.native_support;scope_valid=valid&requested_scope
    if not scope_valid.any():return jsonify({'error':'参考尺度下监测范围没有完整有效观测单元'}),422
    if spatial.record['resampled']:
        # Recheck the actual model input: averaging can remove all source detail.
        analysis_scope=cv2.resize(requested_scope.astype(np.float32),
            (spatial.image.shape[1],spatial.image.shape[0]),interpolation=cv2.INTER_AREA)>0
        try:
            analysis_quality=require_usable_input(spatial.image,
                {**band_meta,'_valid_mask':spatial.valid},analysis_scope)
        except InputQualityError as exc:
            exc.quality['analysis_scope']=spatial.mode
            raise
        analysis_quality['analysis_scope']=spatial.mode
        input_quality['analysis_input']=analysis_quality
    band_meta['spatial_input_contract']=spatial.record
    if opts['analysis_grid']!='native':band_meta['note']+='；'+spatial.record['note']
    loaded = _load_upload_detector(tag)
    if loaded is None:
        return jsonify({"error": f"检测模型 {tag} 的权重不可用", "model_tag": tag}), 503
    upload_det, patch_norm, input_scale = loaded
    synth = _load_nir_synth(upload_det.device) if opts["nir_synth"] else None
    if opts["nir_synth"] and synth is None:
        return jsonify({"error": "所选NIR合成器不可用，未执行替代推理"}), 503
    if synth is not None and band_meta["nir_source"] == "measured":
        return jsonify({"error": "影像已有真实NIR，不能用合成波段覆盖"}), 400
    inference_image,context=prepare_inference_context(spatial.image,spatial.valid)
    input_quality['inference_context']=context
    prob = upload_det.detect_image(inference_image * input_scale, patch_norm=patch_norm,
                                  nir_synth_model=synth, max_batch=opts["max_batch"])
    prob=spatial.restore_probability(prob)
    if not np.isfinite(prob).all():
        return jsonify({'error':'模型概率含非有限数，未生成筛查等级'}),503
    mask = (prob >= upload_det.threshold).astype(np.uint8)
    if opts["instance_tau"] > 0:
        mask = instance_prob_filter(mask, prob, tau=opts["instance_tau"])
    mask*=valid.astype(np.uint8)
    whole = upload_det.analyze_mask(mask,min_region_area=spatial.min_region_area)
    n_valid=int(valid.sum()); whole_ratio=float(whole['landslide_pixels'])/n_valid
    whole.update(total_pixels=n_valid,valid_pixels=n_valid,nodata_pixels=int(mask.size-n_valid),
                 area_ratio=round(whole_ratio,4),risk_level=upload_det._risk_level(whole_ratio))
    stats = dict(whole)
    stats["pred_ratio"] = round(whole_ratio,4)
    stats["instance_tau"] = opts["instance_tau"]
    stats["scope"] = "whole_image"
    roi_meta = None
    if box:
        x0, y0, x1, y1 = box
        scoped = np.zeros_like(mask)
        scoped[y0:y1, x0:x1] = mask[y0:y1, x0:x1]
        stats = upload_det.analyze_mask(scoped,min_region_area=spatial.min_region_area)
        area = (x1 - x0) * (y1 - y0)
        n_inside=int(scope_valid.sum()); n_outside=n_valid-n_inside
        inside = stats["landslide_pixels"] / n_inside
        # ROI clipping may drop a small inside fragment. Subtracting independently
        # filtered inside statistics from whole-image statistics assigns that
        # dropped fragment to the outside. Count the whole-image accepted mask
        # on the actual outside support instead.
        whole_labels,_=label_instances(mask,min_area=spatial.min_region_area)
        outside_pixels=int(((whole_labels>0)&valid&~requested_scope).sum())
        outside=outside_pixels/n_outside if n_outside else None
        stats.update(area_ratio=round(inside, 4), risk_level=upload_det._risk_level(inside),
                     pred_ratio=round(inside,4),total_pixels=n_inside,valid_pixels=n_inside,nodata_pixels=area-n_inside,
                     area_ratio_whole=whole["area_ratio"],
                     pred_ratio_whole=round(whole_ratio,4),
                     instance_tau=opts["instance_tau"], scope="roi")
        roi_meta = {"box_px": list(box), "area_px": area,
                    'valid_area_px':n_inside,'nodata_area_px':area-n_inside,
                    "area_ratio": round(area / mask.size, 4),
                    "risk_inside": stats["area_ratio"],
                    "risk_outside": round(outside, 4) if outside is not None else None,
                    "outside_available": bool(n_outside),
                    "outside_valid_pixels": n_outside,
                    "outside_reported_pixels": outside_pixels,
                    "outside_count_basis": "whole-image accepted components intersected with observed outside pixels",
                    "mean_prob_inside": round(float(prob[scope_valid].mean()), 4),
                    "level_inside": stats["risk_level"],
                    "level_inside_name": RISK_NAMES[stats["risk_level"]][0],
                    "level_outside": upload_det._risk_level(outside) if outside is not None else None,
                    "note": "实例、面积占比、报告均按监测范围内统计；范围外仅作参考"}
    source_observed_in_scope=int((source_valid&requested_scope).sum())
    stats.update(source_nodata_pixels=int(requested_scope.sum())-source_observed_in_scope,
        inference_excluded_pixels=source_observed_in_scope-int(scope_valid.sum()),
        nodata_pixels_definition='analysis-unavailable pixels, including source NoData and reference-cell exclusions')
    if mpp is not None:
        attach_scale(stats["regions"], mpp)
        stats['landslide_area_m2']=round(stats['landslide_pixels']*mpp*mpp,3)
        stats['observed_area_m2']=round(int(scope_valid.sum())*mpp*mpp,3)
    scale_meta = {"meters_per_pixel": mpp, "source": mpp_source} if mpp else None
    spectral_image = calibrated_spectral_image(image,band_meta)[scope_valid][:,None,:]
    quant = {name: _upload_spectral_index(spectral_image, band_meta, name)
             for name in ("ndvi", "ndwi")}
    quant["region"] = {"num_regions": stats["num_regions"],
                        "num_regions_total": stats["num_regions_total"],
                        "area_ratio": stats["area_ratio"],
                        "max_region_area": stats["max_region_area"],
                        "risk_name": RISK_NAMES[stats["risk_level"]][0]}
    band_meta["model_nir_source"] = "synthetic" if synth is not None else band_meta["nir_source"]
    rgb = _raster_rgb(image,band_meta) if band_meta.get('format') in ('GeoTIFF','TIFF') else upload_det.to_rgb(image)
    display_source=scoped if box else mask
    display_labels,_=label_instances(display_source,min_area=spatial.min_region_area)
    display_mask=(display_labels>0).astype(np.uint8)
    display_contract={'basis':'same connected-component filter as reported area and exported geometry',
        'scope':stats['scope'],'min_region_area_native_pixels':spatial.min_region_area,
        'raw_threshold_pixels':int(display_source.sum()),'reported_region_pixels':int(display_mask.sum()),
        'small_component_excluded_pixels':int(display_source.sum()-display_mask.sum()),
        'classification_threshold_changed':False}
    overlay = upload_det.overlay(rgb, display_mask)
    if box:
        cv2.rectangle(overlay, (x0, y0), (x1 - 1, y1 - 1), (30, 111, 217), 2)
    work_context=annotate_known_work(scoped if box else mask,band_meta,scope_valid,stats)
    return jsonify({"risk_level": stats["risk_level"],
                    "risk_name": RISK_NAMES[stats["risk_level"]][0],
                    "risk_color": RISK_NAMES[stats["risk_level"]][1],
                    "stats": {k: v for k, v in stats.items() if k != "risk_level"},
                    "quant": quant, "band_meta": public_band_meta(band_meta), "scale_meta": scale_meta,
                    'input_quality':input_quality,
                    'work_context':work_context,
                    'mask_display_contract':display_contract,
                    "roi": roi_meta, "model_tag": tag,
                    "model_info": _upload_model_info(tag, upload_det, patch_norm),
                    "risk_basis": "预测掩膜面积占比的原型筛查分级，不代表法定预警等级或失稳概率",
                    "threshold": float(upload_det.threshold), "nir_synth": synth is not None,
                    "rgb_b64": _encode_img(rgb), "overlay_b64": _encode_img(overlay),"size":list(image.shape[:2])})


def _load_detector_region(region):
    """按区域加载对应权重(含 patch_size)"""
    global detector
    entry = WEIGHTS.get(region)
    if not entry or not os.path.isfile(entry[0]):
        return None
    artifact = model_registry.verify('region:'+region, entry[0], {'patch_size':entry[1], 'threshold':.5, 'patch_norm':None, 'input_scale':1.0})
    if detector is not None and detector.region == region and getattr(detector,'artifact_info',None)==artifact:
        return detector
    _release_models()
    detector = LandslideDetector(entry[0], patch_size=entry[1])
    detector.region = region
    detector.artifact_info = artifact
    return detector


def _release_models():
    global detector, _detector32, _upload_detector, _nir_synth
    detector = _detector32 = _upload_detector = _nir_synth = None
    _upload_detectors.clear()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _load_detector32():
    """懒加载多尺度融合用 32px 模型"""
    global _detector32
    if _detector32 is None and os.path.exists(FUSION32_WEIGHTS):
        artifact = model_registry.verify('component:fusion32',FUSION32_WEIGHTS,{'patch_size':32,'threshold':.5,'patch_norm':None,'input_scale':1.0})
        _detector32 = LandslideDetector(FUSION32_WEIGHTS, patch_size=32)
        _detector32.artifact_info = artifact
    return _detector32


def _load_upload_detector(tag="base"):
    """Load precisely the requested model. Missing weights never fall back silently."""
    global _upload_detector
    entry = UPLOAD_MODELS.get(tag)
    if entry is None or not os.path.isfile(entry[0]):
        return None
    artifact = model_registry.verify('upload:'+tag,entry[0],{'patch_size':entry[1],'threshold':UPLOAD_THRESHOLDS.get(tag,.5),'patch_norm':entry[2],'input_scale':entry[3]})
    if tag in _upload_detectors and getattr(_upload_detectors[tag][0],'artifact_info',None)==artifact:
        return _upload_detectors[tag]
    _release_models()
    if tag == "rgb_guarded":
        from rgb_adapter import load_guarded_adapter
        try:
            det = load_guarded_adapter(entry[0], MODELS_ROOT, patch_size=entry[1],
                                       threshold=UPLOAD_THRESHOLDS.get(tag, 0.5))
        except (ValueError, FileNotFoundError) as exc:
            print(f"[detector_service] RGB适配配置不可用: {exc}")
            return None
    else:
        det = LandslideDetector(entry[0], patch_size=entry[1],
                                threshold=UPLOAD_THRESHOLDS.get(tag, 0.5))
    det.region = f"upload_{tag}"
    det.artifact_info = artifact
    pair = (det, entry[2], entry[3])
    _upload_detectors[tag] = pair
    if tag == "base":
        _upload_detector = det
    return pair


def _upload_model_info(tag, detector, patch_norm):
    info = {"tag": tag, "weights": os.path.basename(os.path.dirname(UPLOAD_MODELS[tag][0])),
            "threshold": float(detector.threshold), "patch_norm": patch_norm,
            'artifact':getattr(detector,'artifact_info',None),
            'inference_implementation':'streaming_windows_v3_observed_context'}
    if tag=='mine':
        info.update(validation_status='development_recall_insufficient',
                    validation_note='真实矿区开发验证中召回仍不足，疑似区域需结合原始影像复核')
    if tag == "rgb_guarded":
        info.update(adapter_weight=detector.adapter_weight,
                    validation_status="public_reference_regression_only",
                    validation_note=detector.validation_note)
    return info


DS_REJECT_THETA = 0.7   # D-S 冲突拒绝阈值 (实验验证 θ=0.7: F1 最优且误报下降)

# ── RGB→NIR 合成器 (第十五组, 可选开关) ──────────────────
# Esri 等 RGB 影像没有近红外波段, 而实测其缺失代价 −32.7pp(第十三组)。
# 该合成器用 LMHLD 同块配准的 RGB/NIR 训练（实际 22,780 对）, 在逐图块 min-max 域重建 NIR。
# **默认关闭**: 实测真实滑坡场景明显改善(分离度 +0.02~0.06), 但矿山场景对已适配
# 模型会略增误报(抚顺 realfit 1.34%→2.09%) —— 故不改变既有默认行为。
NIR_SYNTH_WEIGHTS = os.path.join(MODELS_ROOT, "nir_synth_seed42", "best_model.pth")
_nir_synth = None


def _load_nir_synth(device):
    """懒加载 NIR 合成器 (Unet-resnet18 3ch→1ch)"""
    global _nir_synth
    if _nir_synth is None and os.path.exists(NIR_SYNTH_WEIGHTS):
        model_registry.verify('component:nir_synth',NIR_SYNTH_WEIGHTS,{'input':'BGR/minmax/255','output':'synthetic_NIR'})
        try:
            import torch
            import segmentation_models_pytorch as smp
            m = smp.Unet(encoder_name="resnet18", encoder_weights=None,
                         in_channels=3, classes=1)
            m.load_state_dict(torch.load(NIR_SYNTH_WEIGHTS, weights_only=True,
                                         map_location="cpu"))
            m.eval()
            _nir_synth = m.to(device)
            print("[detector_service] NIR 合成器已加载 (RGB→NIR, 第十五组)")
        except Exception as e:
            print(f"[detector_service] NIR 合成器加载失败: {e}")
            _nir_synth = None
    return _nir_synth


def _fuse_probs(probs128, probs32):
    """多尺度融合: 逐像素最大(验证中小区召回最优)"""
    return np.maximum(probs128, probs32)


def _ds_conflict_map(probs128, probs32):
    """D-S 冲突因子: K = p128*(1-p32) + (1-p128)*p32 (两证据互斥程度)"""
    return probs128 * (1 - probs32) + (1 - probs128) * probs32


# ── 变化分析颜色 (BGR) ──
C_ADDED = (0, 0, 255)        # 红: 新增
C_EXPANDED = (0, 140, 255)   # 橙: 扩大
C_REMOVED = (60, 180, 60)    # 绿: 消退
C_STABLE = (80, 80, 220)     # 淡红: 持续风险
EXPAND_RATIO = 1.2           # 面积增幅超 1.2 倍视为"扩大"


def _classify_changes(mask1, mask2, valid_mask=None):
    """两期掩膜变化分类(连通域级)
    返回: added/expanded/stable/removed 掩膜 + 统计"""
    mask1 = (mask1 > 0).astype(np.uint8)
    mask2 = (mask2 > 0).astype(np.uint8)
    if mask1.ndim!=2 or mask1.shape!=mask2.shape:
        raise ValueError('两期掩膜须为相同范围的二维数组')
    observed=np.ones(mask2.shape,bool) if valid_mask is None else np.asarray(valid_mask,bool)
    if observed.shape!=mask2.shape or not observed.any():
        raise ValueError('变化统计需要相同范围的有效观测')
    mask1*=observed;mask2*=observed
    H, W = mask2.shape
    n2, lab2, st2, _ = cv2.connectedComponentsWithStats(mask2, 8)
    overlap=np.bincount(lab2[mask1>0],minlength=n2)
    present=np.arange(n2)>0
    is_added=present&(overlap==0)
    is_expanded=present&~is_added&(st2[:,cv2.CC_STAT_AREA]>overlap*EXPAND_RATIO)
    is_stable=present&~is_added&~is_expanded
    added=is_added[lab2].astype(np.uint8)
    expanded=is_expanded[lab2].astype(np.uint8)
    stable=is_stable[lab2].astype(np.uint8)
    n_added=int(is_added.sum());n_expanded=int(is_expanded.sum())

    # 消退: T1 中有、T2 中无 (像素级, 去零星)
    removed_px = (mask1 > 0) & (mask2 == 0)
    removed = np.zeros((H, W), np.uint8)
    n_removed = 0
    if removed_px.any():
        n3, lab3, st3, _ = cv2.connectedComponentsWithStats(removed_px.astype(np.uint8), 8)
        keep=(st3[:,cv2.CC_STAT_AREA]>=10)&(np.arange(n3)>0)
        removed=keep[lab3].astype(np.uint8)
        n_removed=int(keep.sum())

    total = int(observed.sum())
    stats = {
        "n_added_regions": n_added,
        "n_expanded_regions": n_expanded,
        "n_removed_regions": n_removed,
        "added_ratio": round(float(added.sum()) / total, 4),
        "t2_ratio": round(float(mask2.sum()) / total, 4),
        "t1_ratio": round(float(mask1.sum()) / total, 4),
        "net_ratio": round((float(mask2.sum()) - float(mask1.sum())) / total, 4),
        "valid_pixels": total,
        "ratio_basis": "common_observed_pixels",
    }
    return {"added": added, "expanded": expanded, "stable": stable,
            "removed": removed, "stats": stats}


def _overlay_classified(rgb, change):
    """分类结果 → 彩色标注图"""
    out = rgb.astype(np.float32).copy()
    for key, color, alpha in (("added", C_ADDED, 0.7), ("expanded", C_EXPANDED, 0.65),
                              ("removed", C_REMOVED, 0.55), ("stable", C_STABLE, 0.35)):
        m = change[key] > 0
        if m.any():
            out[m] = out[m] * (1 - alpha) + np.array(color, dtype=np.float32) * alpha
    return np.clip(out, 0, 255).astype(np.uint8)


def _change_instances(mask1, mask2, max_regions=MAX_REGIONS_LITE):
    """模型掩膜连通实例的变化几何，与像素级 _classify_changes 互补。

    匹配判据用**包含度** |A∩B| / min(|A|,|B|) 而非 IoU:
    同一处滑坡扩大时 IoU 会因面积差必然下降, 用 IoU 匹配会把「扩大」误判成
    「删除 + 新增」; 包含度对单向扩张稳定(小者仍几乎完全落在扩大后的区域内)。

    返回 {"instances": [...], "removed_instances": [...], "summary": {...}}
    其中 expansion_ratio 即"扩大倍数" —— 这是替代 VLM 凭空编造尺寸的直接依据。
    """
    MATCH_CONT = 0.5
    lab1, areas1 = label_instances(mask1)
    lab2, areas2 = label_instances(mask2)
    n1, n2 = len(areas1), len(areas2)
    cont = instance_overlap(lab1, lab2, n1, n2)["containment"]

    matched_t1, recs = set(), []
    n_added = n_expanded = n_stable = 0
    for j in range(n2):
        rec = region_record(lab2, j + 1, mask2.shape,
                            DEFAULT_SIMPLIFY_EPS, LITE_MAX_VERTICES)
        if rec is None:
            continue
        if n1 == 0:
            kind, ratio, mc, mi = "added", None, 0.0, None
        else:
            i = int(cont[:, j].argmax())
            mc = float(cont[i, j])
            if mc >= MATCH_CONT:
                ratio = float(areas2[j]) / max(float(areas1[i]), 1e-9)
                kind = "expanded" if ratio > EXPAND_RATIO else "stable"
                mi = i
                matched_t1.add(i)
            else:
                kind, ratio, mi = "added", None, None
        rec.update({"change": kind,
                    "expansion_ratio": (round(ratio, 3) if ratio else None),
                    "matched_containment": round(mc, 3),
                    "matched_t1_index": mi})
        recs.append(rec)
        n_added += (kind == "added")
        n_expanded += (kind == "expanded")
        n_stable += (kind == "stable")

    removed = []
    for i in range(n1):
        if i not in matched_t1:
            r = region_record(lab1, i + 1, mask1.shape,
                              DEFAULT_SIMPLIFY_EPS, LITE_MAX_VERTICES)
            if r is not None:
                r["change"] = "removed"
                removed.append(r)

    recs.sort(key=lambda r: -r["area_px"])
    removed.sort(key=lambda r: -r["area_px"])
    exps = sorted([r["expansion_ratio"] for r in recs
                   if r["change"] == "expanded" and r["expansion_ratio"]],
                  reverse=True)
    return {
        "instances": recs[:max_regions],
        "removed_instances": removed[:max_regions],
        "summary": {
            "n_added": int(n_added), "n_expanded": int(n_expanded),
            "n_stable": int(n_stable), "n_removed": len(removed),
            "max_expansion_ratio": (exps[0] if exps else None),
            "expansion_ratios_top": exps[:5],
            "match_containment_threshold": MATCH_CONT,
            "expand_ratio_threshold": EXPAND_RATIO,
            "n_t1_instances": n1, "n_t2_instances": n2,
            "truncated_t2": bool(len(recs) > max_regions),
            "truncated_removed": bool(len(removed) > max_regions),
            "truncated": bool(len(recs) > max_regions or len(removed) > max_regions),
        },
    }


@app.route("/api/compare", methods=["POST"])
@exclusive_inference
def compare():
    data = request.get_json(silent=True) or {}
    try:
        opts = detection_options(data)
        if opts["model_tag"] not in UPLOAD_MODELS:
            raise ValueError("未知检测模型")
        if opts['analysis_grid']!='native':raise ValueError('参考尺度模式目前用于单期分析；两期影像请先统一到相同的已核验格网')
        if opts["model_tag"] == "rgb_guarded" and opts["nir_synth"]:
            raise ValueError("RGB联合适配验证版未校准合成NIR输入")
        img1, meta1 = _prepare_upload(data.get("image_t1", ""), opts["channel_order"], True)
        img2, meta2 = _prepare_upload(data.get("image_t2", ""), opts["channel_order"], True)
        quality1 = require_usable_input(img1,meta1)
        quality2 = require_usable_input(img2,meta2)
        chronology=period_chronology(meta1,meta2,np.array_equal(img1,img2))
        g1,g2=meta1.get('georeference'),meta2.get('georeference')
        if g1 and g2:
            if g1['crs']!=g2['crs'] or not np.allclose(g1['transform'],g2['transform'],rtol=0,atol=1e-6):
                raise ValueError('两期GeoTIFF坐标参考或格网不一致，请先统一投影、分辨率和范围')
        valid1=observed_mask(meta1,img1.shape[:2]); valid2=observed_mask(meta2,img2.shape[:2])
        registration_mode=data.get('registration','auto')
        same_raster_grid=bool(g1 and g2)
        if same_raster_grid and registration_mode=='auto':
            registration_mode='prealigned'
        img1, valid, registration = align_periods(img1, img2, registration_mode)
        if same_raster_grid:
            registration['basis']='same_uploaded_raster_grid'
            registration['independent_ground_accuracy_verified']=False
            registration['note']='按两期文件相同投影与格网对比，保留原始观测；地理标签不代表地面定位精度已独立核验'
        if registration.get('transform'):
            matrix=np.array(registration['transform'],np.float32)
            valid1=cv2.warpAffine(valid1.astype(np.float32),matrix,(img2.shape[1],img2.shape[0]),
                                  flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT)>.999999
        valid=(valid.astype(bool)&valid1&valid2).astype(np.uint8)
        if float(valid.mean())<.6: raise ValueError('两期实际共同有效观测不足60%，无法可靠比较')
        common_quality={}
        for epoch,image,meta in (('t1',img1,meta1),('t2',img2,meta2)):
            # Source texture hidden by the other epoch is not comparable evidence.
            try:
                quality=require_usable_input(image,{**meta,'_valid_mask':valid},valid)
            except InputQualityError as exc:
                exc.quality.update(analysis_scope='common_valid_pixels',epoch=epoch)
                raise
            quality.update(analysis_scope='common_valid_pixels',epoch=epoch)
            common_quality[epoch]=quality
        registration['valid_ratio']=round(float(valid.mean()),4)
        registration['nodata_excluded']=bool(meta1.get('nodata_pixels',0) or meta2.get('nodata_pixels',0))
        registration['raster_grids_checked']=bool(g1 and g2)
        registration['chronology']=chronology
    except InputQualityError:
        raise
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    for meta in (meta1,meta2):
        split=input_split(meta)
        if split['reserved']:
            return jsonify({'error':'保留测试影像禁止用于日常两期检测','inference_status':'reserved_test_protected','case_id':split['case_id']}),409
    loaded = _load_upload_detector(opts["model_tag"])
    if loaded is None:
        return jsonify({"error": "所选检测模型的权重不可用"}), 503
    upload_det, patch_norm, input_scale = loaded
    synth = _load_nir_synth(upload_det.device) if opts["nir_synth"] else None
    if opts["nir_synth"] and synth is None:
        return jsonify({"error": "所选NIR合成器不可用"}), 503
    if synth is not None and (meta1["nir_source"] == "measured" or meta2["nir_source"] == "measured"):
        return jsonify({"error": "不能覆盖真实NIR波段"}), 400
    masks = []
    for epoch,image in (('t1',img1),('t2',img2)):
        # Restrict BOTH epochs' context to the SAME observations. Date-specific
        # missing pixels must not perturb normalization on comparable pixels.
        inference_image,context=prepare_inference_context(image,valid)
        common_quality[epoch]['inference_context']=context
        prob = upload_det.detect_image(inference_image * input_scale, patch_norm=patch_norm,
                                      nir_synth_model=synth, max_batch=opts["max_batch"])
        if not np.isfinite(prob).all():
            return jsonify({'error':'模型概率含非有限数，未生成变化结论'}),503
        mask = (prob >= upload_det.threshold).astype(np.uint8)
        if opts["instance_tau"]:
            mask = instance_prob_filter(mask, prob, tau=opts["instance_tau"])
        masks.append(mask * valid)
    mask1, mask2 = masks
    change = _classify_changes(mask1, mask2, valid)
    n_valid = int(valid.sum())
    st = change["stats"]
    rgb1=upload_det.to_rgb(img1)
    rgb2=_raster_rgb(img2,meta2) if meta2.get('format') in ('GeoTIFF','TIFF') else upload_det.to_rgb(img2)
    quant = {"ndvi_t1": _upload_spectral_index(calibrated_spectral_image(img1,meta1)[valid > 0][:, None, :], meta1, "ndvi"),
             "ndvi_t2": _upload_spectral_index(calibrated_spectral_image(img2,meta2)[valid > 0][:, None, :], meta2, "ndvi")}
    return jsonify({"quant": quant, "band_meta": {"t1": public_band_meta(meta1), "t2": public_band_meta(meta2)},
                    'input_quality':{'t1':quality1,'t2':quality2,'common_valid':common_quality},
                    "registration": registration, "model_tag": opts["model_tag"],
                    "model_info": _upload_model_info(opts["model_tag"], upload_det, patch_norm),
                    "threshold": float(upload_det.threshold),
                    "t1_overlay_b64": _encode_img(upload_det.overlay(rgb1, mask1)),
                    "t2_overlay_b64": _encode_img(upload_det.overlay(rgb2, mask2)),
                    "t2_rgb_b64": _encode_img(rgb2),
                    "change_overlay_b64": _encode_img(_overlay_classified(rgb2, change)),
                    "stats": st, "change_instances": _change_instances(mask1, mask2),
                    "size": list(img2.shape[:2])})


@app.errorhandler(ModelNumericalError)
def _numerical_inference_failure(exc):
    return jsonify({'error':str(exc),'inference_status':'failed_numerics'}),503


@app.errorhandler(InputQualityError)
def _unusable_observation(exc):
    return jsonify({'error':str(exc),'inference_status':'unusable_observation','input_quality':exc.quality}),422


@app.errorhandler(ModelArtifactError)
@app.errorhandler(ModelComponentError)
def _unavailable_model(exc):
    return jsonify({'error':str(exc),'inference_status':'failed_component'}),503


@app.errorhandler(torch.cuda.OutOfMemoryError)
def _exhausted_gpu(exc):
    return jsonify({'error':'本次推理显存不足，请缩小影像或批量后重试','inference_status':'resource_exhausted'}),503


@app.errorhandler(Exception)
def _log_unhandled_exception(e):
    """把未捕获异常写入日志并把错误类型回传 —— 否则服务以 CREATE_NO_WINDOW
    方式启动, 异常既无控制台也无线索, 只能看到一句 500。"""
    from werkzeug.exceptions import HTTPException
    if isinstance(e, HTTPException):
        return jsonify({"error": e.description}), e.code
    import traceback
    from datetime import datetime as _dt
    tb = traceback.format_exc()
    try:
        with open(os.path.join(LMHLD_DIR, "detector_errors.log"), "a",
                  encoding="utf-8") as f:
            f.write(f"\n=== {_dt.now():%Y-%m-%d %H:%M:%S}  {request.path} ===\n{tb}")
    except Exception:
        pass
    return jsonify({"error": str(e), "type": type(e).__name__}), 500


@app.route("/api/geometry_export", methods=["POST"])
def geometry_export():
    """把已检测出的实例几何导出为 GeoJSON / CSV。

    请求: {regions: [...],            # 由 /api/detect 或 /api/detect_upload 返回
           format: "geojson"|"csv",
           meters_per_pixel?: float,  # 有则输出经纬度与米制量
           origin_latlon?: [lat, lon],# 影像左上角(取自 fetch_mine_imagery 的 meta.json)
           image_shape?: [H, W],
           min_area?: int,            # 过滤过小实例
           name?: str}
    返回: {filename, content_b64, n_features, note}

    设计说明: 本端点只做"格式化", 不做重新检测 —— 几何在检测接口已一次算好,
    原样传回即可。这样既不需要再占 GPU, 也避免前后端各写一套几何逻辑导致漂移。
    注意: regions 受检测响应的实例数上限约束(默认 32), 如需全量需调大该上限。
    """
    data = request.get_json(silent=True) or {}
    regions = data.get("regions") or []
    if not isinstance(regions, list) or not regions:
        return jsonify({"error": "regions 为空"}), 400
    fmt = str(data.get("format", "geojson")).lower()
    mpp = data.get("meters_per_pixel")
    origin = data.get("origin_latlon")
    georef=data.get('georeference')
    try:
        mpp = float(mpp) if mpp not in (None, "", 0, "0") else None
        if mpp is not None and (not np.isfinite(mpp) or mpp<=0): raise ValueError('分辨率必须为有限正数')
        if georef:
            if not isinstance(georef,dict) or not isinstance(georef.get('crs'),str): raise ValueError('栅格地理参考格式无效')
            affine=georef.get('transform')
            if not isinstance(affine,list) or len(affine)!=6 or not np.isfinite(np.array(affine,dtype=float)).all():
                raise ValueError('栅格仿射变换须为6个有限数')
            mpp,_=resolve_mpp(mpp,{'georeference':georef})
    except (TypeError, ValueError) as exc:
        return jsonify({'error':str(exc)}),400
    min_area = int(data.get("min_area") or 0)
    name = str(data.get("name") or "landslide")

    keep = [r for r in regions if int(r.get("area_px", 0)) >= min_area]
    spatial=data.get('spatial_input_contract')
    protocol=None
    if isinstance(spatial,dict) and spatial.get('mode')=='reference_10m':
        protocol={key:spatial.get(key) for key in ['mode','source_meters_per_pixel','reference_meters_per_pixel','effective_pixel_size_m','analysis_shape','display_grid','note']}
        protocol.update(protocol_source='client_returned_detection_metadata',ground_boundary_precision_verified=False,accuracy_on_real_high_resolution_sensors_verified=False)
    if fmt == "csv":
        content = regions_to_csv(keep, meters_per_pixel=mpp)
        if protocol:
            # Each detached CSV row retains the input scale and its precision limits.
            rows=list(csv.reader(io.StringIO(content)))
            stream=io.StringIO(newline='');writer=csv.writer(stream,lineterminator='\n')
            writer.writerow(rows[0]+list(protocol))
            values=[json.dumps(value,ensure_ascii=False) if isinstance(value,(list,dict)) else value for value in protocol.values()]
            writer.writerows(row+values for row in rows[1:])
            content=stream.getvalue()
        fname = f"{name}.csv"
        n_features=len(keep)
    else:
        try:
            geo = regions_to_geojson(keep, meters_per_pixel=mpp,
                                     origin_latlon=origin,image_shape=data.get('image_shape'),
                                     name=name,georeference=georef)
        except Exception:
            return jsonify({'error':'几何地理参考无效，请检查坐标系及仿射变换'}),400
        if protocol:geo['spatial_input_contract']=protocol
        content = json.dumps(geo, ensure_ascii=False)
        fname = f"{name}.geojson"
        n_features=len(geo['features'])
    note = []
    if not mpp:
        note.append("未提供 meters_per_pixel, 坐标为像素坐标且无米制量")
    if georef:
        note=['坐标按上传文件仿射/投影转换；文件地理标签未经独立地面核验']
    elif not origin:
        note.append("未提供 origin_latlon, 未做地理配准")
    if any(r.get('hole_contours_truncated') for r in keep): note.append('部分复杂实例孔洞轮廓已截断，不能用于完整边界面积核算')
    if protocol:note.append('按约10米参考尺度分析；投影回原图不增加边界细节，真实高分辨率精度待验收')
    return jsonify({
        "filename": fname,
        "content_b64": base64.b64encode(content.encode("utf-8")).decode(),
        "n_features": n_features,
        "note": "; ".join(note) or "已含经纬度与米制量(局部等距近似)",
    })


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else int(os.environ.get("DETECTOR_PORT", 5101))
    print(f"\n[detector_service] 启动: http://localhost:{port}")
    from server_runtime import serve_app
    serve_app(app, port)
