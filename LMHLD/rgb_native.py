"""Three-band research inference. No fourth channel, NIR synthesis or truth input."""
from __future__ import annotations
import numpy as np
import rgb_scale


def make_views(image, label=None):
    from validation.cross_region_scale import make_views as original_views
    a = np.asarray(image)
    if a.ndim != 3 or a.shape[0] not in (3, 4):
        raise ValueError('Native BGR CHW required')
    bgr = np.asarray(a[:3], np.float32)
    # Reuse the frozen interpolation/parent normalization, then remove the
    # redundant channel. Actual NIR is discarded before every preprocessing.
    v, y = original_views(np.concatenate([bgr, bgr[2:3]], axis=0), label)
    return np.ascontiguousarray(v[:, :3]), y


def infer_grids(model, views, batch=64):
    import torch
    a = np.asarray(views)
    if a.ndim != 5 or a.shape[1:] != (5, 3, 64, 64) or not len(a):
        raise ValueError('Expected N x five x three x64x64')
    if not np.isfinite(a).all() or a.min() < 0 or a.max() > 1:
        raise ValueError('Finite normalized native BGR views required')
    if isinstance(batch, bool) or not isinstance(batch, int) or batch < 1:
        raise ValueError('Positive batch required')
    device = next(model.parameters()).device

    def forward(x, kind=0):
        parts = []
        with torch.inference_mode():
            for start in range(0, len(x), batch):
                z = torch.as_tensor(np.ascontiguousarray(x[start:start+batch]), device=device).float()
                if kind in (1, 2, 3): z = z.rot90(kind, (-2, -1))
                elif kind == 4: z = z.flip(-1)
                p = model(z).sigmoid()[:, 0]
                if kind in (1, 2, 3): p = p.rot90(-kind, (-2, -1))
                elif kind == 4: p = p.flip(-1)
                if p.shape[1:] != (64, 64) or not torch.isfinite(p).all():
                    raise ValueError('Invalid native RGB output')
                parts.append(p.cpu().numpy())
        return np.concatenate(parts)

    p = forward(a[:, 0]); t = p.copy()
    for kind in (1, 2, 3, 4): t += forward(a[:, 0], kind)
    q = forward(a[:, 1:].reshape(-1, 3, 64, 64)).reshape(-1, 4, 64, 64)
    return {'global_grid': p, 'local_grids': q, 'tta_grid': t/5}


def scene_grids(model, image, parent_batch=48):
    image = np.asarray(image)
    if image.ndim != 3 or image.shape[0] != 3:
        raise ValueError('Three BGR bands required; four-band input not accepted')
    h, w = image.shape[1:]; side = 32
    positions = rgb_scale.tile_positions(h, w, side)
    pad = np.pad(image, ((0, 0), (0, (-h) % side), (0, (-w) % side)), mode='reflect')
    blocks = []
    for start in range(0, len(positions), parent_batch):
        v = np.stack([make_views(pad[:, y:y+side, x:x+side])[0]
                      for y, x in positions[start:start+parent_batch]])
        # Native32 probability maps preserve all later mask decisions while
        # avoiding retention of six much larger64x64 view probability grids.
        maps = rgb_scale.native_maps(infer_grids(model, v), 32)
        blocks.append({k: np.stack(z) for k, z in maps.items()})
    return {k: np.concatenate([b[k] for b in blocks]) for k in blocks[0]}


def scene_masks(native, shape, gsd, policy):
    h, w = shape; positions = rgb_scale.tile_positions(h, w, 32)
    if any(np.asarray(native[k]).shape != (len(positions), 32, 32) for k in ('global', 'local', 'tta5')):
        raise ValueError('Invalid native scene probability cache')
    masks = {k: np.zeros((h, w), bool) for k in ('tta5', 'multiscale_raw', 'area_guard')}
    for i, (y, x) in enumerate(positions):
        selected, _, _ = rgb_scale.masks_from_policy(native['global'][i], native['local'][i], native['tta5'][i], gsd, policy)
        hh, ww = min(32, h-y), min(32, w-x)
        for k in masks: masks[k][y:y+hh, x:x+ww] = selected[k][:hh, :ww]
    return masks
