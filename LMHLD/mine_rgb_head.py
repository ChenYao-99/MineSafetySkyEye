"""Frozen RGB decoder features and source-mine-only weak-reference learning.

The classifier never accepts mine IDs, coordinates, dates or truth at inference.
Only explicit conservative cores and reviewed-normal pixels provide supervision.
"""
from __future__ import annotations
from runtime_paths import configure_runtime_paths
configure_runtime_paths()
import cv2
import numpy as np
from scipy.special import expit


def inverse(a, kind):
    if kind in (1, 2, 3): return a.rot90(-kind, (-2, -1))
    return a.flip(-1) if kind == 4 else a


def forward_features(model, parents, check_forward=False):
    import torch
    a = np.asarray(parents)
    if a.ndim != 4 or a.shape[1:] != (3, 64, 64) or not np.isfinite(a).all() or a.min() < 0 or a.max() > 1:
        raise ValueError('Finite normalized BGR3x64x64 parents required')
    if model.training: raise ValueError('Frozen model must be in eval mode')
    device = next(model.parameters()).device
    with torch.inference_mode():
        x = torch.as_tensor(np.ascontiguousarray(a), device=device).float()
        total = None; feature_sum = None; single = None
        for k in range(5):
            z = x.rot90(k, (-2, -1)) if k in (1, 2, 3) else x.flip(-1) if k == 4 else x
            decoded = model.decoder(model.encoder(z)); logits = model.segmentation_head(decoded)
            if decoded.shape[1:] != (16, 64, 64) or not torch.isfinite(decoded).all() or not torch.isfinite(logits).all():
                raise ValueError('Invalid frozen decoder output')
            if check_forward and k == 0 and not torch.equal(logits, model(z)):
                raise ValueError('Manual feature path differs from original CNN forward')
            p = inverse(logits.sigmoid()[:, 0], k); f = inverse(decoded, k)
            if k == 0: single = p.clone()
            total = p if total is None else total+p
            feature_sum = f if feature_sum is None else feature_sum+f
        return (total/5).cpu().numpy(), single.cpu().numpy(), (feature_sum/5).cpu().numpy()


def scene_features(model, bgr_chw, parent_batch=64, native_side=32):
    import rgb_native
    a = np.asarray(bgr_chw)
    if a.ndim != 3 or a.shape[0] != 3 or min(a.shape[1:]) < 2 or not np.isfinite(a).all():
        raise ValueError('Finite three-band scene required')
    if native_side not in (32, 64): raise ValueError('Predeclared native contexts32 or64 only')
    h, w = a.shape[1:]; side = native_side
    padded = np.pad(a, ((0, 0), (0, (-h) % side), (0, (-w) % side)), mode='reflect')
    positions = [(y, x) for y in range(0, h, side) for x in range(0, w, side)]
    out = {'tta': np.empty((h, w), np.float32), 'single': np.empty((h, w), np.float32),
           'embedding': np.empty((h, w, 16), np.float16)}
    for start in range(0, len(positions), parent_batch):
        pos = positions[start:start+parent_batch]
        views = np.stack([rgb_native.make_views(padded[:, y:y+side, x:x+side])[0][0] for y, x in pos])
        p, s, f = forward_features(model, views, check_forward=start == 0)
        for i, (y, x) in enumerate(pos):
            hh, ww = min(side, h-y), min(side, w-x)
            out['tta'][y:y+hh, x:x+ww] = cv2.resize(p[i], (side, side), interpolation=cv2.INTER_LINEAR)[:hh, :ww]
            out['single'][y:y+hh, x:x+ww] = cv2.resize(s[i], (side, side), interpolation=cv2.INTER_LINEAR)[:hh, :ww]
            features = cv2.resize(f[i].transpose(1, 2, 0), (side, side), interpolation=cv2.INTER_LINEAR)
            out['embedding'][y:y+hh, x:x+ww] = features[:hh, :ww].astype(np.float16)
    if not all(np.isfinite(v).all() for v in out.values()): raise ValueError('Nonfinite feature cache')
    return out


def supervision(truth, normal, valid):
    truth, normal, valid = np.asarray(truth), np.asarray(normal), np.asarray(valid, bool)
    if not (truth.shape == normal.shape == valid.shape) or not np.isin(truth, [0, 1, 255]).all() or not np.isin(normal, [0, 1]).all():
        raise ValueError('Explicit aligned partial and reviewed-normal masks required')
    positive = (truth == 1) & valid; negative = (normal == 1) & valid
    if np.any(positive & negative): raise ValueError('Positive/normal reference conflict')
    return positive, negative


def matrix(features):
    p = np.asarray(features['tta'], np.float32); f = np.asarray(features['embedding'], np.float32)
    if f.shape != (*p.shape, 16) or not np.isfinite(f).all() or not np.isfinite(p).all() or p.min() < 0 or p.max() > 1:
        raise ValueError('Finite aligned frozen features required')
    logit = np.log(np.clip(p, 1e-6, 1-1e-6)/(1-np.clip(p, 1e-6, 1-1e-6)))
    return np.concatenate([f, logit[..., None]], axis=-1).reshape(-1, 17)


def fit_head(features, references, source_ids, forbidden_id, cap=2048):
    from sklearn.linear_model import LogisticRegression
    if forbidden_id in source_ids or set(features) != set(source_ids) or set(references) != set(source_ids) or len(source_ids) < 2:
        raise ValueError('Explicit source mines only; held mine forbidden')
    xx = []; yy = []; ww = []; records = []
    for i, case in enumerate(sorted(source_ids)):
        f = matrix(features[case]); positive, negative = supervision(*references[case])
        for label, mask, trust in [(1, positive, .6), (0, negative, .25)]:
            indices = np.flatnonzero(mask.ravel())
            if not len(indices): raise ValueError('Source mine lacks positive/normal support')
            rng = np.random.default_rng(20261020+131*i+17*label)
            indices = np.sort(rng.choice(indices, min(cap, len(indices)), replace=False))
            xx.append(f[indices]); yy.append(np.full(len(indices), label, np.uint8))
            ww.append(np.full(len(indices), trust/len(indices), np.float64))
            records.append({'case': case, 'label': label, 'indices': indices.tolist(), 'weak_loss_weight': trust})
    x, y = np.concatenate(xx).astype(np.float64), np.concatenate(yy)
    # Equal mine/class mass before the recorded .6/.25 weak loss multipliers.
    w = np.concatenate(ww)*len(x)/(2*len(source_ids))
    mean = np.average(x, axis=0, weights=w); scale = np.sqrt(np.average((x-mean)**2, axis=0, weights=w)); scale = np.maximum(scale, 1e-6)
    z = np.clip((x-mean)/scale, -8, 8)
    model = LogisticRegression(C=1., solver='lbfgs', max_iter=1000, tol=1e-7).fit(z, y, sample_weight=w)
    if int(model.n_iter_[0]) >= 1000: raise ValueError('Source linear head did not converge')
    return {'kind': 'source_only_L2_logistic', 'mean': mean.tolist(), 'scale': scale.tolist(),
            'coef': model.coef_[0].tolist(), 'intercept': float(model.intercept_[0]),
            'source_ids': sorted(source_ids), 'forbidden_id': forbidden_id, 'iterations': int(model.n_iter_[0]), 'samples': records}


def predict_head(head, features):
    x = matrix(features); result = np.empty(len(x), np.float32)
    mean, scale, coef = [np.asarray(head[k], np.float64) for k in ('mean', 'scale', 'coef')]
    if mean.shape != (17,) or scale.shape != (17,) or coef.shape != (17,) or np.any(scale <= 0) or not all(np.isfinite(a).all() for a in (mean, scale, coef)) or not np.isfinite(head['intercept']):
        raise ValueError('Invalid source head')
    for start in range(0, len(x), 65536):
        z = np.clip((x[start:start+65536].astype(np.float64)-mean)/scale, -8, 8)
        result[start:start+len(z)] = expit(z@coef+head['intercept']).astype(np.float32)
    return result.reshape(features['tta'].shape)


def calibrate(scores, references, source_ids, forbidden_id, production_rates, base_rates):
    if forbidden_id in source_ids or set(scores) != set(source_ids) or set(references) != set(source_ids) or len(source_ids) != 3:
        raise ValueError('Three source calibration mines required, held mine forbidden')
    budget = min(float(np.mean([production_rates[c] for c in source_ids]))+.01,
                 .8*float(np.mean([base_rates[c] for c in source_ids])))
    tails = {}; caps = {}
    for c in source_ids:
        _, normal = supervision(*references[c]); p = np.asarray(scores[c], np.float32)
        if p.shape != normal.shape or not np.isfinite(p).all() or p.min() < 0 or p.max() > 1: raise ValueError('Invalid source score')
        values = p[normal].astype(np.float64)
        if not len(values): raise ValueError('Explicit normal support required')
        h = np.bincount(np.minimum((values*1000).astype(np.int32), 1000), minlength=1001)
        tails[c] = np.r_[np.cumsum(h[::-1])[::-1]/h.sum(), 0.]
        caps[c] = min(production_rates[c]+.03, base_rates[c]+.01)
    macro = np.mean(list(tails.values()), axis=0); ok = macro <= budget
    for c in source_ids: ok &= tails[c] <= caps[c]
    i = int(np.flatnonzero(ok)[0])
    return {'threshold': i/1000, 'source_budget': budget, 'source_mean_normal_activation': float(macro[i]),
            'source_normal_caps': caps, 'source_normal_activation': {c: float(v[i]) for c, v in tails.items()}, 'source_ids': list(source_ids), 'held_used': False}


def core_rates(scores, references, threshold):
    out = {}
    for c, p in scores.items():
        pos, _ = supervision(*references[c]); out[c] = float((np.asarray(p, np.float64)[pos] >= threshold).mean())
    return out
