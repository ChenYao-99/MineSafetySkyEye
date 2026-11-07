"""Offline spectral-ablation and native-tile inference helpers.

No truth arrays, case identities or calibration are accepted by these helpers.
RGB means BGR plus copied red, never measured or reconstructed near infrared.
"""
from __future__ import annotations

import cv2
import numpy as np
import component_filter as filt


def make_input_views(image, mode):
    from validation.cross_region_scale import make_views
    image = np.asarray(image)
    if image.ndim != 3 or image.shape[0] not in (3, 4):
        raise ValueError('CHW BGR or BGRN input required')
    if mode == 'rgb_proxy':
        # Discard NIR BEFORE validation, normalization or interpolation.
        bgr = np.asarray(image[:3], np.float32)
        image = np.concatenate([bgr, bgr[2:3]], axis=0)
    elif mode != 'actual_bgrn' or image.shape[0] != 4:
        raise ValueError('Explicit rgb_proxy or actual_bgrn mode required')
    return make_views(image)[0]


def infer_grids(model, views, batch=96):
    import torch
    views = np.asarray(views)
    if views.ndim != 5 or views.shape[1:] != (5, 4, 64, 64) or not len(views):
        raise ValueError('Expected N x five x four x64x64 views')
    if not np.isfinite(views).all() or views.min() < 0 or views.max() > 1:
        raise ValueError('Research inputs must be normalized to 0..1')
    if isinstance(batch, bool) or not isinstance(batch, int) or batch < 1:
        raise ValueError('Positive integer batch required')
    device = next(model.parameters()).device

    def forward(images, kind=0):
        result = []
        with torch.inference_mode():
            for i in range(0, len(images), batch):
                x = torch.as_tensor(np.ascontiguousarray(images[i:i+batch]), device=device).float()
                if kind in (1, 2, 3): x = x.rot90(kind, (-2, -1))
                elif kind == 4: x = x.flip(-1)
                p = model(x).sigmoid()[:, 0]
                if kind in (1, 2, 3): p = p.rot90(-kind, (-2, -1))
                elif kind == 4: p = p.flip(-1)
                if not torch.isfinite(p).all(): raise ValueError('Nonfinite model output')
                result.append(p.cpu().numpy())
        return np.concatenate(result)

    whole = forward(views[:, 0]); tta = whole.copy()
    for k in (1, 2, 3, 4): tta += forward(views[:, 0], k)
    local = forward(views[:, 1:].reshape(-1, 4, 64, 64)).reshape(-1, 4, 64, 64)
    return {'global_grid': whole, 'local_grids': local, 'tta_grid': tta / 5}


def native_maps(grids, side):
    if not isinstance(side, int) or side < 2 or side % 2:
        raise ValueError('Even native parent side required')
    out = {'global': [], 'local': [], 'tta5': []}; half = side // 2
    for p, q, t in zip(grids['global_grid'], grids['local_grids'], grids['tta_grid']):
        local = np.empty((side, side), np.float32)
        for i in range(4):
            y, x = i // 2 * half, i % 2 * half
            local[y:y+half, x:x+half] = cv2.resize(q[i], (half, half), interpolation=cv2.INTER_LINEAR)
        out['global'].append(cv2.resize(p, (side, side), interpolation=cv2.INTER_LINEAR))
        out['local'].append(local)
        out['tta5'].append(cv2.resize(t, (side, side), interpolation=cv2.INTER_LINEAR))
    return out


def masks_from_policy(p, q, tta, gsd, policy):
    pred = filt.extract_components(p, q, policy['pixel_threshold'], policy['anchor_threshold'], gsd)
    a = policy['area_policy']
    scores = filt.component_scores(pred, None, 'area_guard')
    keep = filt.keep_components(pred, scores, a['cutoff'], a['protect_anchor'])
    return {'tta5': np.asarray(tta, np.float64) >= policy['tta_threshold'],
            'multiscale_raw': pred.labels > 0,
            'area_guard': filt.component_mask(pred, keep)}, pred, keep


def tile_positions(height, width, side=32):
    if min(height, width) < 2 or side < 2 or side % 2:
        raise ValueError('Invalid scene or native tile dimensions')
    return [(y, x) for y in range(0, height, side) for x in range(0, width, side)]


def fill_invalid(image, valid):
    """Image-only nearest-valid fill; scoring still excludes original NoData."""
    from scipy.ndimage import distance_transform_edt
    image = np.asarray(image, np.float32); valid = np.asarray(valid, bool)
    if image.ndim != 3 or image.shape[1:] != valid.shape or not valid.any():
        raise ValueError('Valid aligned image mask required')
    if not np.isfinite(image[:, valid]).all(): raise ValueError('Nonfinite valid pixels')
    if valid.all(): return image.copy()
    nearest = distance_transform_edt(~valid, return_distances=False, return_indices=True)
    return image[:, nearest[0], nearest[1]]


def scene_grids(model, image, mode, side=32, parent_batch=64):
    """Stream nonoverlapping native parents; do not resize an entire scene to64."""
    image = np.asarray(image); h, w = image.shape[1:]
    positions = tile_positions(h, w, side)
    pad = np.pad(image, ((0, 0), (0, (-h) % side), (0, (-w) % side)), mode='reflect')
    packed = []
    for start in range(0, len(positions), parent_batch):
        views = np.stack([make_input_views(pad[:, y:y+side, x:x+side], mode)
                          for y, x in positions[start:start+parent_batch]])
        packed.append(infer_grids(model, views))
    return {k: np.concatenate([g[k] for g in packed]) for k in packed[0]}, positions


def scene_masks(grids, shape, gsd, policy, side=32):
    h, w = shape; positions = tile_positions(h, w, side)
    if len(grids['global_grid']) != len(positions): raise ValueError('Scene tile count changed')
    native = native_maps(grids, side)
    masks = {k: np.zeros((h, w), bool) for k in ('tta5', 'multiscale_raw', 'area_guard')}
    for i, (y, x) in enumerate(positions):
        selected, _, _ = masks_from_policy(native['global'][i], native['local'][i], native['tta5'][i], gsd, policy)
        hh, ww = min(side, h-y), min(side, w-x)
        for k in masks: masks[k][y:y+hh, x:x+ww] = selected[k][:hh, :ww]
    return masks


def partial_reference_counts(mask, truth, normal, valid):
    mask, truth, normal, valid = map(np.asarray, (mask, truth, normal, valid))
    if not (mask.shape == truth.shape == normal.shape == valid.shape):
        raise ValueError('Partial references must align exactly')
    if not np.isin(truth, [0, 1, 255]).all() or not np.isin(normal, [0, 1]).all():
        raise ValueError('Unsupported reference values')
    valid = valid.astype(bool); mask = mask.astype(bool)
    pos = (truth == 1) & valid; bg = (truth == 0) & valid
    normal = (normal == 1) & valid
    # A later independent normal review may annotate pixels left unknown by an
    # earlier partial label. It need not be a subset of that older background.
    # A positive/normal conflict must still fail rather than be silently scored.
    if np.any(normal & pos): raise ValueError('Reviewed normal conflicts with positive reference')
    result = {}
    for key, scope in [('positive_core', pos), ('weak_background', bg), ('reviewed_normal', normal)]:
        n = int(scope.sum()); hits = int((mask & scope).sum())
        result[key] = {'pixels': n, 'activated': hits, 'rate': hits/n if n else None}
    result['unknown_pixels_excluded'] = int(((truth == 255) & valid).sum())
    result['reviewed_normal_from_partial_unknown_pixels'] = int((normal & (truth == 255)).sum())
    result['invalid_pixels_excluded'] = int((~valid).sum())
    return result
