"""Small source-trained spatial scale gate; offline research, no default hook.

Inputs are actual four-band imagery, global/local probabilities, and supplied
nominal GSD. Label arrays, region IDs and coordinates are never inference inputs.
"""
from __future__ import annotations
import cv2
import numpy as np

FEATURES = (
    'global_probability', 'local_probability', 'local_minus_global', 'absolute_disagreement',
    'global_entropy', 'local_entropy', 'global_gradient', 'local_gradient',
    'normalized_blue', 'normalized_green', 'normalized_red', 'normalized_nir',
    'normalized_nir_red_contrast', 'visible_brightness', 'visible_gradient', 'nir_gradient',
    'log_nominal_gsd', 'log_parent_ground_extent',
)
METADATA_FEATURES = (16, 17)


def gradient(a):
    dx = cv2.Sobel(a, cv2.CV_32F, 1, 0, ksize=3) / 8
    dy = cv2.Sobel(a, cv2.CV_32F, 0, 1, ksize=3) / 8
    return np.sqrt(dx*dx + dy*dy)


def feature_map(image_chw, global_probability, local_probability, nominal_gsd):
    raw = np.asarray(image_chw, np.float32)
    p, q = [np.asarray(v, np.float32) for v in (global_probability, local_probability)]
    if raw.ndim != 3 or raw.shape[0] != 4 or raw.shape[1:] != p.shape or q.shape != p.shape:
        raise ValueError('Actual BGRN image and both probabilities must share the native grid')
    if not np.isfinite(raw).all() or not np.isfinite(p).all() or not np.isfinite(q).all():
        raise ValueError('Non-finite scale-gate input')
    if np.any(p < 0) or np.any(p > 1) or np.any(q < 0) or np.any(q > 1):
        raise ValueError('Invalid probability range')
    if not np.isfinite(nominal_gsd) or nominal_gsd <= 0:
        raise ValueError('Positive nominal GSD required')
    lo = raw.min((1, 2), keepdims=True)
    span = raw.max((1, 2), keepdims=True) - lo
    x = np.divide(raw-lo, span, out=np.zeros_like(raw), where=span > 0)
    def entropy(v):
        v = np.clip(v, 1e-6, 1-1e-6)
        return -(v*np.log(v)+(1-v)*np.log1p(-v))
    brightness = x[:3].mean(0)
    # This contrast uses individually normalized channels. It is NOT physical NDVI.
    contrast = (x[3]-x[2])/(x[3]+x[2]+.01)
    return np.stack([p,q,q-p,np.abs(q-p),entropy(p),entropy(q),gradient(p),gradient(q),
                     *x,contrast,brightness,gradient(brightness),gradient(x[3]),
                     np.full_like(p,np.log(nominal_gsd)),
                     np.full_like(p,np.log(p.shape[-1]*nominal_gsd))], axis=-1)


def fuse(feature_array, global_probability, local_probability, coefficients):
    if tuple(coefficients['features']) != FEATURES:
        raise ValueError('Feature schema mismatch')
    a = np.asarray(feature_array, np.float32)
    if a.shape[-1] != len(FEATURES) or not np.isfinite(a).all():
        raise ValueError('Invalid gate features')
    mean = np.asarray(coefficients['mean'], np.float32)
    std = np.asarray(coefficients['std'], np.float32)
    weights = np.asarray(coefficients['weights'], np.float32)
    if (any(v.shape != (len(FEATURES),) for v in (mean, std, weights)) or np.any(std <= 0)
            or not all(np.isfinite(v).all() for v in (mean,std,weights))
            or not np.isfinite(coefficients['bias'])):
        raise ValueError('Invalid fitted gate normalization')
    z = np.clip((a-mean)/std, -8, 8)
    logit = np.clip(z@weights+coefficients['bias'], -40, 40)
    alpha = 1/(1+np.exp(-logit))
    p, q = np.asarray(global_probability), np.asarray(local_probability)
    if p.shape != a.shape[:-1] or q.shape != p.shape:
        raise ValueError('Fusion grids differ')
    return ((1-alpha)*p+alpha*q).astype(np.float32), alpha.astype(np.float32)
