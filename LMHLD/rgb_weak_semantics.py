"""Pointwise weak semantic heads, with explicit whole-mine isolation.

Region IDs are SOURCE supervision weights only, never inference features.
Unknown pixels are absent from support, not assigned negative labels.
"""
import numpy as np
from scipy.special import expit


def predict(model, features):
    x = np.asarray(features, np.float64)
    mean, scale, coef = [np.asarray(model[k], np.float64) for k in ('mean', 'scale', 'coef')]
    if x.ndim != 2 or x.shape[1] != 17 or mean.shape != (17,) or scale.shape != (17,) or coef.shape != (17,) or np.any(scale <= 0) or not all(np.isfinite(a).all() for a in (x, mean, scale, coef)):
        raise ValueError('Finite frozen17-dimensional features/head required')
    return expit(np.clip((x-mean)/scale, -8, 8)@coef+model['intercept']).astype(np.float32)


def fit(support, sources, forbidden, arm='pixel', cap=2048):
    from sklearn.linear_model import LogisticRegression
    if arm not in ('pixel', 'region_balanced') or set(support) != set(sources) or forbidden in sources or len(sources) < 2:
        raise ValueError('Explicit source-only supervision and registered head required')
    xx = []; yy = []; ww = []; samples = []; masses = []
    for i, case in enumerate(sorted(sources)):
        s = support[case]; index = np.asarray(s['index']); labels = np.asarray(s['label']); region = np.asarray(s['region'])
        if index.ndim != 1 or np.any(np.diff(index) <= 0) or not np.isin(labels, [0, 1]).all() or region.shape != labels.shape or s['x'].shape != (len(index), 17): raise ValueError('Sparse support must be unique, ordered, binary and aligned')
        for label, trust in [(1, .6), (0, .25)]:
            available = index[labels == label]
            if not len(available): raise ValueError('Each source needs positive and explicit normal support')
            rng = np.random.default_rng(20261020+131*i+17*label)
            chosen = np.sort(rng.choice(available, min(cap, len(available)), replace=False)); pos = np.searchsorted(index, chosen)
            if arm == 'region_balanced' and label == 0:
                all_regions = np.unique(region[labels == 0]); chosen_regions = region[pos]
                if 0 in all_regions or not np.array_equal(np.unique(chosen_regions), all_regions): raise ValueError('Normal-region support missing from identical sampled pixels')
                weight = np.empty(len(pos), np.float64)
                for r in all_regions:
                    sel = chosen_regions == r; weight[sel] = trust/(len(all_regions)*int(sel.sum()))
                    masses.append({'case': case, 'region': int(r), 'samples': int(sel.sum()), 'unscaled_mass': float(weight[sel].sum())})
            else: weight = np.full(len(pos), trust/len(pos), np.float64)
            xx.append(s['x'][pos]); yy.append(np.full(len(pos), label, np.uint8)); ww.append(weight)
            samples.append({'case': case, 'label': label, 'indices': chosen.tolist(), 'weak_loss_weight': trust})
    x, y = np.concatenate(xx).astype(np.float64), np.concatenate(yy); w = np.concatenate(ww)*len(x)/(2*len(sources))
    mean = np.average(x, axis=0, weights=w); scale = np.maximum(np.sqrt(np.average((x-mean)**2, axis=0, weights=w)), 1e-6)
    z = np.clip((x-mean)/scale, -8, 8); model = LogisticRegression(C=1., solver='lbfgs', max_iter=1000, tol=1e-7).fit(z, y, sample_weight=w)
    if int(model.n_iter_[0]) >= 1000: raise ValueError('Weak semantic head did not converge')
    return {'kind': arm, 'mean': mean.tolist(), 'scale': scale.tolist(), 'coef': model.coef_[0].tolist(), 'intercept': float(model.intercept_[0]),
            'sources': sorted(sources), 'forbidden': forbidden, 'iterations': int(model.n_iter_[0]), 'samples': samples, 'normal_region_masses': masses}


def calibration(scores, support, sources, forbidden, production_rates, baseline_rates):
    if set(scores) != set(sources) or set(support) != set(sources) or forbidden in sources or len(sources) != 3: raise ValueError('Three source mines only')
    budget = min(float(np.mean([production_rates[c] for c in sources]))+.01, .8*float(np.mean([baseline_rates[c] for c in sources])))
    caps = {}; tails = {}
    for c in sources:
        p = np.asarray(scores[c], np.float32)
        if p.shape != support[c]['label'].shape or not np.isfinite(p).all() or p.min() < 0 or p.max() > 1: raise ValueError('Invalid sparse score')
        neg = p[support[c]['label'] == 0].astype(np.float64)
        if not len(neg): raise ValueError('Explicit reviewed normal support missing')
        h = np.bincount(np.minimum((neg*1000).astype(np.int32), 1000), minlength=1001)
        tails[c] = np.r_[np.cumsum(h[::-1])[::-1]/h.sum(), 0.]
        caps[c] = min(production_rates[c]+.03, baseline_rates[c]+.01)
    macro = np.mean(list(tails.values()), axis=0); eligible = macro <= budget
    for c in sources: eligible &= tails[c] <= caps[c]
    i = int(np.flatnonzero(eligible)[0])
    return {'threshold': i/1000, 'source_budget': budget, 'source_normal_caps': caps,
            'source_normal_activation': {c: float(t[i]) for c, t in tails.items()}, 'source_mean_normal_activation': float(macro[i]), 'sources': list(sources), 'held_used': False}


def rates(scores, support, cutoff):
    core = {}; normal = {}
    for c, p in scores.items():
        labels = support[c]['label']; p = np.asarray(p, np.float64)
        core[c] = float((p[labels == 1] >= cutoff).mean()); normal[c] = float((p[labels == 0] >= cutoff).mean())
    return {'core': core, 'normal': normal, 'mean_core': float(np.mean(list(core.values()))), 'mean_normal': float(np.mean(list(normal.values())))}
