"""Observable input defects; weather/stability cannot be certified from RGB alone."""
import numpy as np
from raster_upload import observed_mask


class InputQualityError(ValueError):
    def __init__(self, message, quality):
        super().__init__(message)
        self.quality = quality


def inspect_quality(image, meta, scope=None):
    valid = observed_mask(meta, image.shape[:2])
    requested = np.ones(valid.shape, bool) if scope is None else np.asarray(scope, bool)
    if requested.shape != valid.shape:
        raise ValueError('质量检查范围与影像不一致')
    inside = valid & requested
    total = int(requested.sum())
    # Information outside the requested/observed area cannot validate this task.
    visible = np.asarray(image[:, :, :3], np.float32)[inside]
    ranges = np.ptp(visible, axis=0) if visible.size else np.zeros(3)
    scale = float(np.max(np.abs(visible))) if visible.size else 0.0
    informative = bool(np.any(ranges > max(scale * 1e-6, 1e-6)))
    return {'schema_version': 1, 'scope_pixels': total,
            'information_scope': 'requested_observed_pixels',
            'observed_pixels': int(inside.sum()),
            'observed_ratio': float(inside.sum()) / max(total, 1),
            'visible_has_spatial_information': informative,
            'weather_assessment': 'provided_SCL_screening' if meta.get('observation_quality',{}).get('source')=='provided_Sentinel2_SCL' else 'not_verified',
            'observation_quality':meta.get('observation_quality'),
            'ground_stability_assessment': 'not_verified',
            'status': 'usable_observation' if informative and inside.any() else 'unusable_observation'}


def require_usable_input(image, meta, scope=None):
    quality = inspect_quality(image, meta, scope)
    if quality['status'] != 'usable_observation':
        raise InputQualityError('影像无有效观测或可见光空间信息，不能生成低风险结论', quality)
    return quality


def prepare_inference_context(image, valid, method='nearest_observed'):
    """Prepare image-only context; never change support or quantitative inputs.

    All channels use one nearest observed pixel, preserving its spectral tuple.
    Filled pixels remain unobserved and must still be excluded from statistics.
    """
    a=np.asarray(image);observed=np.asarray(valid,bool)
    if a.ndim!=3 or observed.shape!=a.shape[:2] or not observed.any():
        raise ValueError('推理上下文需要对齐的有效观测范围')
    if method not in ('nearest_observed','zero_common'):
        raise ValueError('未知的推理上下文协议')
    if not np.isfinite(a[observed]).all():
        raise ValueError('有效观测内含非有限数')
    missing=~observed;n=int(missing.sum())
    record={'method':method if n else 'unchanged_fully_observed',
            'context_only_pixels':n,'observed_pixels':int(observed.sum()),
            'filled_pixels_are_observations':False,'spectral_indices_use_original_observations':True}
    if not n:return a,record
    result=a.copy()
    if method=='nearest_observed':
        from scipy.ndimage import distance_transform_edt
        nearest=distance_transform_edt(missing,return_distances=False,return_indices=True)
        result[missing]=a[nearest[0][missing],nearest[1][missing]]
    else:result[missing]=0
    return result,record
