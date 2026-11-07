"""Upload validation and spatial registration; no model or Flask dependency."""
import base64
import math
import os
import hashlib
import io
from datetime import datetime,timezone

import cv2
import numpy as np
from PIL import Image
from raster_upload import is_tiff,decode_tiff


def positive_number(value, name, optional=False):
    if optional and value in (None, ""):
        return None
    if isinstance(value, bool):
        raise ValueError(f"{name} 必须是有限正数")
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{name} 必须是有限正数") from None
    if not math.isfinite(number) or number <= 0:
        raise ValueError(f"{name} 必须是有限正数")
    return number


def detection_options(data, default_model="mine"):
    if not isinstance(data, dict):
        raise ValueError("请求必须为 JSON 对象")
    tag = data.get("model_tag", default_model)
    if not isinstance(tag, str):
        raise ValueError("model_tag 必须是字符串")
    order = str(data.get("channel_order", "auto")).upper()
    if order not in ("AUTO", "BGRN", "RGBN"):
        raise ValueError("channel_order 只能为 auto、BGRN 或 RGBN")
    try:
        tau = float(data.get("instance_tau") or 0)
    except (TypeError, ValueError):
        raise ValueError("instance_tau 必须为 0 到 1 之间的数值") from None
    if not math.isfinite(tau) or not 0 <= tau <= 1:
        raise ValueError("instance_tau 必须为 0 到 1 之间的数值")
    synth = data.get("nir_synth", False)
    if not isinstance(synth, bool):
        raise ValueError("nir_synth 必须是布尔值")
    grid=data.get('analysis_grid','native')
    if grid not in ('native','reference_10m'):raise ValueError('analysis_grid只支持native或reference_10m')
    if grid=='reference_10m' and tag!='mine':raise ValueError('10米参考尺度模式仅核验了mine模型的输入链路')
    batch = int(os.environ.get("KAT_MAX_BATCH", "256"))
    if not 1 <= batch <= 512:
        raise ValueError("KAT_MAX_BATCH 必须为1到512之间的整数")
    return {"model_tag": tag, "channel_order": order, "instance_tau": tau,
            "nir_synth": synth, "max_batch": batch,'analysis_grid':grid,
            "meters_per_pixel": positive_number(data.get("meters_per_pixel"),
                                                "meters_per_pixel", optional=True)}


def decode_upload(image_b64, channel_order="AUTO"):
    if not isinstance(image_b64, str) or not image_b64:
        raise ValueError("image 必须是非空 base64 字符串")
    encoded = image_b64.split(",", 1)[-1] if image_b64.startswith("data:") else image_b64
    max_bytes = int(os.environ.get("KAT_MAX_UPLOAD_BYTES", str(32 * 1024 * 1024)))
    if len(encoded) > (max_bytes * 4 // 3 + 8):
        raise ValueError("影像文件超过上传大小限制")
    try:
        raw = base64.b64decode(encoded, validate=True)
    except Exception:
        raise ValueError("图片解码失败") from None
    if len(raw)>max_bytes:
        raise ValueError("影像文件超过上传大小限制")
    max_pixels = int(os.environ.get("KAT_MAX_IMAGE_PIXELS", "16777216"))
    if is_tiff(raw):
        return decode_tiff(raw,channel_order.upper(),max_pixels)
    # Compressed headers must be checked before OpenCV allocates a full raster.
    try:
        with Image.open(io.BytesIO(raw), formats=['PNG', 'JPEG', 'BMP', 'WEBP']) as header:
            if header.width * header.height > max_pixels or min(header.size) < 2:
                raise ValueError('影像尺寸不在允许范围内')
    except (OSError, Image.DecompressionBombError):
        raise ValueError('图片格式或尺寸不在允许范围内') from None
    try:
        image = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_UNCHANGED)
    except cv2.error:
        raise ValueError("图片解码失败") from None
    if image is None:
        raise ValueError("图片解码失败")
    if image.shape[0] * image.shape[1] > max_pixels or min(image.shape[:2]) < 2:
        raise ValueError("影像尺寸不在允许范围内")
    order = channel_order.upper()
    measured = order in ("BGRN", "RGBN")
    if measured:
        if image.ndim != 3 or image.shape[2] != 4:
            raise ValueError("真实近红外影像须包含 4 个波段，并声明 BGRN/RGBN 顺序")
        if order == "RGBN":
            image = image[..., [2, 1, 0, 3]]
        nir_source = "measured"
        note = "第4波段由用户声明为真实近红外；光谱指数仍需正确辐射定标"
    else:
        # An ordinary 4-channel image is BGRA. Alpha is never a spectral band.
        if image.ndim == 2:
            image = np.stack([image] * 3, axis=-1)
        if image.shape[2] not in (3, 4):
            raise ValueError("普通影像仅支持灰度、RGB 或 RGBA")
        had_alpha = image.shape[2] == 4
        valid = image[...,3] > 0 if had_alpha else np.ones(image.shape[:2],bool)
        image = image[..., :3]
        image = np.concatenate([image, image[..., 2:3]], axis=-1)
        nir_source = "red_proxy"
        note = "缺少真实近红外；复制红波段仅供检测器输入，不能用于 NDVI/NDWI"
        if had_alpha:
            note += "；透明度仅用于排除完全透明像素"
    image = image.astype(np.float32)
    if not np.isfinite(image).all():
        raise ValueError("影像包含非有限数值")
    meta = {"channel_order": "BGRN", "input_channel_order": order,
                   'input_sha256':hashlib.sha256(raw).hexdigest(),
                   "nir_source": nir_source, "spectral_indices_available": False,'radiometry_status':'unknown',
                   'nir_verified_by':'user_declaration' if measured else 'no_measured_NIR',
                   "note": note}
    if not measured:
        image[~valid] = 0
        if not valid.any():
            raise ValueError('影像全部透明，没有可用观测像素')
        meta.update(_valid_mask=valid,observed_pixels=int(valid.sum()),nodata_pixels=int((~valid).sum()))
    return image, meta


def _gray(image):
    rgb = image[..., :3]
    lo, hi = float(rgb.min()), float(rgb.max())
    rgb8 = np.clip((rgb - lo) / max(hi - lo, 1e-6) * 255, 0, 255).astype(np.uint8)
    return cv2.cvtColor(rgb8, cv2.COLOR_BGR2GRAY)


def period_chronology(meta1,meta2,identical=False):
    first,second=meta1.get('acquisition_datetime'),meta2.get('acquisition_datetime')
    info={'status':'not_verified','date_source':'provided_raster_headers','independent_capture_time_verified':False}
    if not first or not second:return info
    try:
        a=datetime.fromisoformat(str(first).replace('Z','+00:00'))
        b=datetime.fromisoformat(str(second).replace('Z','+00:00'))
    except ValueError:
        raise ValueError('影像采集时间标签无效，不能进行时序变化解释') from None
    if (a.tzinfo is None)!=(b.tzinfo is None):raise ValueError('两期采集时间的时区声明不一致，不能猜测顺序')
    if b<a or (b==a and not identical):raise ValueError('T2采集时间必须晚于T1；同一次观测不能解释为新增或消退')
    info.update(status='same_observation' if b==a else 'provided_order_checked',t1=str(first),t2=str(second))
    return info


def align_periods(image_t1, image_t2, mode="auto"):
    """Align T1 to T2; failed/unverifiable registration stops change analysis."""
    shape = image_t2.shape[:2]
    if mode == "prealigned":
        if image_t1.shape[:2] != shape:
            raise ValueError("已配准影像须具有相同尺寸、范围与分辨率")
        return image_t1, np.ones(shape, np.uint8), {
            "status": "declared_prealigned", "reference": "T2",
            "valid_ratio": 1.0, "note": "用户声明两期已配准；系统未验证地理参考"}
    if mode != "auto":
        raise ValueError("registration 只能为 auto 或 prealigned")
    if image_t1.shape == image_t2.shape and np.array_equal(image_t1, image_t2):
        return image_t1, np.ones(shape, np.uint8), {
            "status": "identical", "reference": "T2", "valid_ratio": 1.0}
    orb = cv2.ORB_create(nfeatures=5000)
    k1, d1 = orb.detectAndCompute(_gray(image_t1), None)
    k2, d2 = orb.detectAndCompute(_gray(image_t2), None)
    error = "两期影像无法可靠配准；请使用相同范围的正射影像，或确认已配准后选择对应模式"
    if d1 is None or d2 is None:
        raise ValueError(error)
    pairs = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(d1, d2, k=2)
    good = [p[0] for p in pairs if len(p) == 2 and p[0].distance < .7 * p[1].distance]
    if len(good) < 20:
        raise ValueError(error)
    a = np.float32([k1[m.queryIdx].pt for m in good])
    b = np.float32([k2[m.trainIdx].pt for m in good])
    matrix, inliers = cv2.estimateAffinePartial2D(a, b, method=cv2.RANSAC,
                                               ransacReprojThreshold=2.0)
    if matrix is None or inliers is None:
        raise ValueError(error)
    keep = inliers.ravel().astype(bool)
    scale = float(np.hypot(matrix[0, 0], matrix[1, 0]))
    angle = abs(float(np.degrees(np.arctan2(matrix[1, 0], matrix[0, 0]))))
    residual = np.linalg.norm(a @ matrix[:, :2].T + matrix[:, 2] - b, axis=1)
    if keep.sum() < 16 or keep.mean() < .4 or not .8 <= scale <= 1.25 or angle > 15:
        raise ValueError(error)
    if float(np.median(residual[keep])) > 2:
        raise ValueError(error)
    aligned = cv2.warpAffine(image_t1, matrix, (shape[1], shape[0]),
                             flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    valid = cv2.warpAffine(np.ones(image_t1.shape[:2], np.uint8), matrix,
                          (shape[1], shape[0]), flags=cv2.INTER_NEAREST,
                          borderMode=cv2.BORDER_CONSTANT)
    if float(valid.mean()) < .6:
        raise ValueError("两期影像共同覆盖范围不足60%，无法可靠比较")
    return aligned, valid, {"status": "registered", "reference": "T2",
                            "transform": matrix.tolist(), "matches": len(good),
                            "inliers": int(keep.sum()),
                            "median_error_px": round(float(np.median(residual[keep])), 3),
                            "valid_ratio": round(float(valid.mean()), 4),
                            "note": "变化仅在共同有效范围内统计；配准残差不等于地理定位精度"}
