"""Source-first check: spend the old five-view budget on spatial detail instead."""
from pathlib import Path
import sys
import time
import numpy as np
sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parent)]
from runtime_paths import configure_runtime_paths
configure_runtime_paths()
import rgb_transfer_regions as regions
import rgb_resolution as resolution
exp = regions.exp; base = exp.base
PRIOR = exp.ROOT/'experiments/rgb_transfer_regions_20261020'


def register(out):
    out = Path(out).resolve()
    if out.drive.upper() != 'D:' or not out.is_relative_to(exp.ROOT/'experiments'): raise ValueError('D experiments only')
    out.mkdir(parents=True, exist_ok=True)
    if (out/'protocol.json').exists():
        p = base.read(out/'protocol.json'); exp.old_rgb.check_assets(p); return p
    old = base.read(PRIOR/'protocol.json'); frozen = dict(old['frozen_assets'])
    paths = [Path(__file__), exp.ROOT/'rgb_resolution.py', exp.ROOT/'tests/test_rgb_resolution.py',
             exp.ROOT/'validation/rgb_transfer_regions.py', PRIOR/'protocol.json', PRIOR/'data_manifest.json', PRIOR/'results.json', PRIOR/'verification.json']
    for g in old['held_indices']:
        for seed in old['seeds']:
            d = exp.folder(PRIOR, regions.ARM, g, seed)
            paths += [d/'training.json', d/'source_selected.pth', d/'rgb_policy.json', d/'result.json']
    frozen.update({str(a): base.sha(a) for a in paths})
    p = {'registered_at': base.now(), 'frozen_assets': frozen, 'held_indices': old['held_indices'], 'seeds': old['seeds'], 'pilot_held_index': 2,
         'question': 'Can preserved spatial detail at approximately matched convolutional pixel budget replace five low-resolution transformations without losing small slides or raising background errors?',
         'rule': 'Image-only original parent side>=128: one whole-parent128 forward. Smaller parents: unchanged64 TTA5. No GSD, region identifier, label or target statistic in inference rule.',
         'normalization': 'Frozen per-parent BGR minmax01 BEFORE resizing; identical area/linear interpolation and float16 cache->float32 arithmetic. Same normalized pretrained weights, no fitting/BN update.',
         'compute': 'Large parents:128^2 vs5*64^2 input pixels (.8 ratio). This is an approximate spatial-FLOPs comparison, not an exact equal-latency assertion. Small parents unchanged; real source forward latency measured separately.',
         'source_selection_gate': {'min_macro_small_cover_gain': .02, 'min_macro_small_iou_gain': .02, 'max_macro_f1_loss': .005, 'max_region_f1_loss': .02, 'max_macro_fpr_increase': .002, 'max_region_fpr_increase': .01},
         'calibration': 'Other source validation halves independently calibrate both methods to mean2% background FPR; source selection halves screen the registered rule.',
         'automatic_order': 'First the two held2 SOURCE-only policies. Only if BOTH pass, compute all six fold source policies. Each admitted source policy may then evaluate its own held region; failed policies have no new target forward. No refit or threshold search on targets.',
         'target_gate': 'Require each seed mean small coverage and strictIoU gains>=.02, F1 loss<=.005, FPR increase<=.002, macroFPR<=.03 and any-regionFPR<=.05; no automatic production change.',
         'scope': 'Previously exposed regional development folds. No official test, mine or reserved-case prediction and no new blind acceptance claim.',
         'retention': 'Packed source/target masks, source background histograms and tiny policies/results; discard temporary float probabilities. No copied weights or raw source data.',
         'production_changed': False, 'training_updates': 0, 'mine_predictions': 0, 'official_test_predictions': 0, 'reserved_predictions': 0}
    exp.old_rgb.check_assets(p); base.write(out/'protocol.json', p); return p


def folder(out, held, seed):
    d = Path(out)/f'fold{held}'/f'seed{seed}'; d.mkdir(parents=True, exist_ok=True); return d


def metrics(probs, labels, components, q):
    rows = [base.parent_counts(p, y, c, q) for p, y, c in zip(probs, labels, components)]
    return base.aggregate_counts(rows)


def macro(rows):
    keys = ('f1', 'small_cover50', 'small_iou50', 'background_fpr', 'false_components_per_mpix')
    return {k: float(np.mean([v[k] for v in rows.values() if v[k] is not None])) for k in keys}


def source_gate(old, new, limits):
    a, b = macro(old), macro(new)
    checks = {'small_coverage_gain': b['small_cover50'] >= a['small_cover50']+limits['min_macro_small_cover_gain'],
              'small_iou_gain': b['small_iou50'] >= a['small_iou50']+limits['min_macro_small_iou_gain'],
              'f1_retained': b['f1'] >= a['f1']-limits['max_macro_f1_loss'],
              'each_region_f1': max(old[g]['f1']-new[g]['f1'] for g in old) <= limits['max_region_f1_loss'],
              'fpr_retained': b['background_fpr'] <= a['background_fpr']+limits['max_macro_fpr_increase'],
              'each_region_fpr': max(new[g]['background_fpr']-old[g]['background_fpr'] for g in old) <= limits['max_region_fpr_increase']}
    return {'old': a, 'new': b, 'delta': {k: b[k]-a[k] for k in a}, 'checks': checks, 'passed': all(checks.values())}


def group_probabilities(model, data, region, g):
    import torch
    import cv2
    x = data[g]['views'][:, 0]
    regions.transfer.runtime.predict_tta(model, x[:2])
    torch.cuda.synchronize(); started = time.perf_counter()
    old = exp.parent.short.native_probabilities(regions.transfer.runtime.predict_tta(model, x), data[g]['labels'])
    torch.cuda.synchronize(); old_time = time.perf_counter()-started
    native_side = len(data[g]['labels'][0])
    if native_side < 128: return old, old, {'old_seconds': old_time, 'new_seconds': old_time, 'n': len(old), 'native_side': native_side, 'unchanged': True}
    split = region['splits']['val']; raw = np.load(split['image_path'], mmap_mode='r')
    views = np.stack([resolution.whole_view(raw[i, :3]) for i in split['indices']])
    # Preprocessing excluded from BOTH forward timing measurements.
    resolution.predict(model, views[:2]); torch.cuda.synchronize(); started = time.perf_counter()
    pred = resolution.predict(model, views); torch.cuda.synchronize(); new_time = time.perf_counter()-started
    new = [cv2.resize(a, (native_side, native_side), interpolation=cv2.INTER_LINEAR) for a in pred]
    return old, new, {'old_seconds': old_time, 'new_seconds': new_time, 'n': len(old), 'native_side': native_side, 'unchanged': False}


def source(out, held, seed, data, manifest, p):
    import torch
    d = folder(out, held, seed)
    if (d/'source.json').exists(): return base.read(d/'source.json')
    model = regions.load_model(exp.folder(PRIOR, regions.ARM, held, seed)); hist = {m: {} for m in ('tta64', 'detail128')}; saved = {}; timings = {}
    temp = d/'temporary'; temp.mkdir(exist_ok=True)
    for g in range(6):
        if g == held: continue
        a, b, timings[str(g)] = group_probabilities(model, data, manifest['regions'][g], g)
        pos = exp.parent.gate.meta_split(len(data[g]['labels']), g)[1]; labels = [data[g]['labels'][i] for i in pos]
        for method, probs in [('tta64', a), ('detail128', b)]:
            hist[method][g] = exp.parent.source_histogram([probs[i] for i in pos], labels)
            np.savez_compressed(temp/f'{method}_g{g}.npz', probability=np.stack(probs))
    thresholds = {}; calibrations = {}
    for m in hist: thresholds[m], calibrations[m] = exp.parent.threshold_from_histograms(hist[m], held, .02)
    if thresholds['tta64'] != base.read(exp.folder(PRIOR, regions.ARM, held, seed)/'rgb_policy.json')['tta_threshold']:
        raise ValueError('Frozen TTA64 calibration was not reproduced')
    np.savez_compressed(d/'source_histograms.npz', **{f'{m}_g{g}': h for m, a in hist.items() for g, h in a.items()})
    results = {m: {'selection': {}, 'calibration': {}, 'threshold': thresholds[m], 'source_calibration': calibrations[m]} for m in hist}
    for g in range(6):
        if g == held: continue
        fit, cal = exp.parent.gate.meta_split(len(data[g]['labels']), g)
        for m in hist:
            path = temp/f'{m}_g{g}.npz'
            with np.load(path) as z: probs = z['probability']
            for i, a in enumerate(probs): saved[f'{m}_g{g}_p{i}'] = np.packbits((a.astype(np.float64) >= thresholds[m]).ravel())
            for split, ids in [('selection', fit), ('calibration', cal)]:
                results[m][split][str(g)] = metrics([probs[i] for i in ids], [data[g]['labels'][i] for i in ids], [data[g]['components'][i] for i in ids], thresholds[m])
            path.unlink()
    temp.rmdir(); np.savez_compressed(d/'source_masks.npz', **saved)
    gate = source_gate(results['tta64']['selection'], results['detail128']['selection'], p['source_selection_gate'])
    record = {'frozen_at': base.now(), 'held_index': held, 'seed': seed, 'methods': results, 'source_gate': gate, 'forward_timings': timings,
              'source_masks_sha256': base.sha(d/'source_masks.npz'), 'source_histograms_sha256': base.sha(d/'source_histograms.npz'), 'no_target_forward': True}
    base.write(d/'source.json', record); del model; torch.cuda.empty_cache()
    print('RESOLUTION_SOURCE_GATE', held, seed, gate, flush=True); return record


def verify_source(out, held, seed, data):
    d = folder(out, held, seed); r = base.read(d/'source.json')
    if base.sha(d/'source_masks.npz') != r['source_masks_sha256'] or base.sha(d/'source_histograms.npz') != r['source_histograms_sha256']: raise ValueError('Resolution source caches changed')
    with np.load(d/'source_histograms.npz') as h, np.load(d/'source_masks.npz') as z:
        for method, result in r['methods'].items():
            q, cal = exp.parent.threshold_from_histograms({g: h[f'{method}_g{g}'] for g in range(6) if g != held}, held, .02)
            if q != result['threshold'] or cal != result['source_calibration']: raise ValueError('Resolution source threshold replay failed')
            for g in range(6):
                if g == held: continue
                fit, calibration = exp.parent.gate.meta_split(len(data[g]['labels']), g)
                for split, ids in [('selection', fit), ('calibration', calibration)]:
                    masks = [regions.transfer.unpack_mask(z[f'{method}_g{g}_p{i}'], data[g]['labels'][i].shape) for i in ids]
                    actual = metrics(masks, [data[g]['labels'][i] for i in ids], [data[g]['components'][i] for i in ids], .5)
                    if actual != result[split][str(g)]: raise ValueError('Resolution source score replay failed')
    return True


def target(out, held, seed, data, manifest):
    import torch
    d = folder(out, held, seed); policy = base.read(d/'source.json')
    if not policy['source_gate']['passed']: raise ValueError('Rejected source policy must not access target')
    model = regions.load_model(exp.folder(PRIOR, regions.ARM, held, seed)); old, new, timing = group_probabilities(model, data, manifest['regions'][held], held)
    scores = {}; masks = {}
    for method, probs in [('tta64', old), ('detail128', new)]:
        q = policy['methods'][method]['threshold']; scores[method] = metrics(probs, data[held]['labels'], data[held]['components'], q)
        for i, a in enumerate(probs): masks[f'{method}_p{i}'] = np.packbits((a.astype(np.float64) >= q).ravel())
    np.savez_compressed(d/'target_masks.npz', **masks)
    r = {'completed_at': base.now(), 'held_index': held, 'seed': seed, 'methods': scores, 'timing': timing,
         'policy_sha256': base.sha(d/'source.json'), 'target_masks_sha256': base.sha(d/'target_masks.npz')}
    base.write(d/'target.json', r); del model; torch.cuda.empty_cache(); return r


def verify_target(out, held, seed, data):
    d = folder(out, held, seed); r = base.read(d/'target.json'); policy = base.read(d/'source.json')
    if not policy['source_gate']['passed'] or r['policy_sha256'] != base.sha(d/'source.json') or r['target_masks_sha256'] != base.sha(d/'target_masks.npz'):
        raise ValueError('Resolution target policy/cache identity changed')
    with np.load(d/'target_masks.npz') as z:
        for method in ('tta64', 'detail128'):
            masks = [regions.transfer.unpack_mask(z[f'{method}_p{i}'], y.shape) for i, y in enumerate(data[held]['labels'])]
            if metrics(masks, data[held]['labels'], data[held]['components'], .5) != r['methods'][method]:
                raise ValueError('Resolution target score replay failed')
    return True


def summarize_targets(targets, p):
    rows = []
    for seed in p['seeds']:
        sub = [r for r in targets if r['seed'] == seed]
        if len(sub) != len(p['held_indices']):
            rows.append({'seed': seed, 'complete_source_admission': False, 'passed': False}); continue
        a = macro({r['held_index']: r['methods']['tta64'] for r in sub}); b = macro({r['held_index']: r['methods']['detail128'] for r in sub})
        checks = {'coverage_gain': b['small_cover50'] >= a['small_cover50']+.02, 'iou_gain': b['small_iou50'] >= a['small_iou50']+.02,
                  'f1_retained': b['f1'] >= a['f1']-.005, 'fpr_retained': b['background_fpr'] <= a['background_fpr']+.002,
                  'macro_absolute_fpr': b['background_fpr'] <= .03, 'each_region_absolute_fpr': max(r['methods']['detail128']['background_fpr'] for r in sub) <= .05}
        rows.append({'seed': seed, 'complete_source_admission': True, 'baseline': a, 'candidate': b, 'checks': checks, 'passed': all(checks.values())})
    return rows


def run(out):
    out = Path(out); p = register(out); exp.parent.setup_runtime(); manifest = base.read(PRIOR/'data_manifest.json'); data = exp.load_validation(manifest)
    pilots = [source(out, p['pilot_held_index'], seed, data, manifest, p) for seed in p['seeds']]
    passed = all(r['source_gate']['passed'] for r in pilots)
    base.write(out/'pilot_decision.json', {'decided_at': base.now(), 'source_only': True, 'passed_both_seeds': passed})
    targets = []; done = [(p['pilot_held_index'], seed) for seed in p['seeds']]
    if passed:
        for held in p['held_indices']:
            for seed in p['seeds']:
                r = source(out, held, seed, data, manifest, p)
                if (held, seed) not in done: done.append((held, seed))
                if r['source_gate']['passed']: targets.append(target(out, held, seed, data, manifest))
    for held, seed in done: verify_source(out, held, seed, data)
    for r in targets: verify_target(out, r['held_index'], r['seed'], data)
    exp.old_rgb.check_assets(p)
    target_summary = summarize_targets(targets, p)
    base.write(out/'results.json', {'completed_at': base.now(), 'source_pilot_passed_both_seeds': passed, 'source_folds_checked': len(done),
        'target_folds_predicted': len(targets), 'targets': targets, 'target_summary': target_summary, 'admitted_both_seeds': all(r['passed'] for r in target_summary), 'production_changed': False, 'training_updates': 0})
    base.write(out/'verification.json', {'verified_at': base.now(), 'status': 'passed', 'source_policies_rebuilt': len(done), 'target_scores_rebuilt': len(targets), 'protected_assets_unchanged': True})
    exp.state(out, status='completed', stage='source_first_detail_check_verified', source_pilot_passed=passed, target_predictions=len(targets))


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser(); p.add_argument('--output', required=True); args = p.parse_args()
    try: run(Path(args.output))
    except BaseException as e: exp.state(Path(args.output), status='interrupted_or_failed', error=str(e)); raise
