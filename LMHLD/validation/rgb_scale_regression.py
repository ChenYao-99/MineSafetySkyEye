"""Fixed-weight RGB ablation followed by four seen-mine pipeline regressions.

Source-only policies are persisted before any target predictions in each fit.
All old experiment assets and default registries remain read only.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import sys
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'validation')]
from runtime_paths import configure_runtime_paths
configure_runtime_paths()
import rgb_scale as rgb
import component_filter as filt
import component_screen as screen
import component_screen_acceptance as acceptance
from mine_data_protocol import assert_development_cases

parent = screen.parent
base = screen.base
PRIOR = ROOT/'experiments/component_screen_20261020'
FULL = screen.PRIOR
ACCEPT = ROOT/'experiments/component_acceptance_20261020'
OLD_RGB = ROOT/'experiments/task_rgb_adapter_20261015'
METHODS = ('tta5', 'multiscale_raw', 'area_guard', 'naive_bgrn_policy')
METRICS = ('f1', 'small_cover50', 'small_iou50', 'background_fpr', 'false_components_per_mpix')
DEFAULT = ROOT/'models_prior/real_ft_T1b_ours_seed42/best_model.pth'


def folder(root, held, seed):
    return Path(root)/f'fold_{held}_{base.REGIONS[held][0]}'/f'seed{seed}'


def check_assets(protocol):
    for path, digest in protocol['frozen_assets'].items():
        if base.sha(path) != digest: raise ValueError(f'Frozen asset changed: {path}')


def register(out):
    out = Path(out).resolve()
    if out.drive.upper() != 'D:' or not out.is_relative_to(ROOT/'experiments'):
        raise ValueError('Use D:/LMHLD/experiments')
    out.mkdir(parents=True, exist_ok=True)
    if (out/'protocol.json').exists():
        p = base.read(out/'protocol.json'); check_assets(p); return p
    old = base.read(ACCEPT/'protocol.json')
    frozen = dict(old['frozen_assets'])
    paths = [Path(__file__), ROOT/'rgb_scale.py', ROOT/'tests/test_rgb_scale.py',
             ACCEPT/'protocol.json', ACCEPT/'results.json', ACCEPT/'verification.json',
             PRIOR/'results.json', PRIOR/'verification.json', OLD_RGB/'mine_evaluation_plan.json',
             ROOT/'validation/mine_data_protocol.py', ROOT/'detector.py', DEFAULT]
    plan = base.read(OLD_RGB/'mine_evaluation_plan.json')
    ids = [r['mine'] for r in plan['cases']]
    assert_development_cases(ids)
    for r in plan['cases']:
        for key in ('image', 'label', 'normal'):
            if base.sha(r[key+'_path']) != r[key+'_sha256']: raise ValueError('Mine input changed')
            paths.append(Path(r[key+'_path']))
    for g in old['held_indices']:
        for seed in old['seeds']:
            for name in ('filter_policy.json', 'result.json'):
                paths.append(folder(PRIOR, g, seed)/name)
    for path in paths: frozen[str(path)] = base.sha(path)
    p = {'registered_at': base.now(), 'frozen_assets': frozen,
         'question': 'Does the source-calibrated area guard survive removal of measured NIR, and does this fixed research pipeline transfer to four already-seen mine scenes?',
         'held_indices': old['held_indices'], 'seeds': old['seeds'],
         'methods': list(METHODS), 'primary_method': 'area_guard',
         'architecture': 'Existing six full-source Unet++/ResNet34 weights, frozen, no training.',
         'rgb_input': 'Original BGR channels only; fourth channel copies red BEFORE independent parent/band minmax0..1 and crop/resize64. Discard true NIR before all preprocessing. Not actual camera/sensor RGB validation and not synthetic measured NIR.',
         'source_calibration': dict(screen.CAL),
         'pixel_calibration': 'Original source calibration halves only, equal-region mean background FPR<=2%, lowest .001-grid threshold separately for global, TTA5 and 50:50 global/local mean.',
         'area_calibration': 'Same source calibration halves and constraints as predecessor. No classifier or new trainable parameters. No formal statistical risk guarantee.',
         'naive_control': 'Apply the old actual-BGRN pixel/anchor/area policy unchanged to RGB probabilities. Diagnostic only.',
         'target_cohort': 'Original256 val parents per held region, three regions x two seeds. Previously exposed development only. Previous682 acceptance parents are NOT reopened here. No new blind claim.',
         'limits': old['limits'], 'extra_limits': old['extra_limits'],
         'absolute_rgb_fpr_limits': {'macro_each_seed': .03, 'region_each_seed': .05},
         'mine_cases': plan['cases'], 'mine_primary_fold': 2,
         'mine_model_selection': 'Use first protocol fold Hokkaido-held, both seeds fixed before predictions; do not select folds/models on mine results.',
         'mine_inputs': ['actual_bgrn', 'rgb_proxy'],
         'mine_grid': '32 native pixels, nonoverlapping stride32, reflect bottom/right padding. Each parent normalized0..1 once, whole/quadrants64. Screen within each parent before stitching/cropping, exactly as source calibration; edge splitting remains a limitation. No scene resizing to64.',
         'mine_nodata': 'All-band raster valid intersection. Nearest valid image-only fill for inference in BOTH pipelines; exclude original NoData from all scored scopes. Verify exact CRS/affine/shape alignment.',
         'production_control': 'Current registered mine default, native32 stride32, per-band minmax0..255, thresholds .4 primary and .5 secondary; probability maps recomputed. Different architecture/preprocessing pipeline comparison, not a causal scale-only ablation.',
         'mine_scoring': 'Positive-core pixel recovery and weak/reviewed-normal activation only. Unknown255 excluded, not assumed safe. No complete incident IoU, full-scene precision or field false-alarm claim.',
         'mine_development_gate': {'macro_core_gain_each_seed': .05, 'macro_reviewed_normal_increase_max_each_seed': .01, 'any_mine_core_loss_max': .03},
         'mine_exposure_ledger': 'Write this round in exposure_ledger.json; existing registry files are frozen historical snapshots, remain unchanged. Four already-seen development mines only.',
         'new_cnn_training_updates': 0, 'external_calls': 0, 'official_test_predictions': 0,
         'reserved_mine_predictions': 0, 'production_changed': False}
    check_assets(p); base.write(out/'protocol.json', p); return p


def load_group(manifest, g, mode):
    v = manifest['regions'][g]['splits']['val']
    x = np.load(v['image_path'], mmap_mode='r'); y = np.load(v['label_path'], mmap_mode='r')
    views = []; labels = []
    for i, hx, hy in zip(v['indices'], v['raw_parent_sha256'], v['label_parent_sha256']):
        if base.array_sha(x[i]) != hx or base.array_sha(y[i]) != hy: raise ValueError('Selected validation row changed')
        views.append(rgb.make_input_views(x[i], mode)); labels.append(y[i, 0].astype(np.uint8))
    return {'views': np.stack(views), 'labels': labels, 'indices': v['indices'],
            'components': [base.gt_components(t, manifest['regions'][g]['gsd_m']) for t in labels]}


def load_model(policy):
    import torch
    import segmentation_models_pytorch as smp
    if base.sha(policy['base_weight']) != policy['base_weight_sha256']: raise ValueError('CNN changed')
    model = smp.UnetPlusPlus(encoder_name='resnet34', encoder_weights=None, in_channels=4, classes=1).cuda()
    model.load_state_dict(torch.load(policy['base_weight'], map_location='cpu', weights_only=True)); model.eval()
    if parent.short.state_hash(model) != policy['base_state_sha256']: raise ValueError('CNN state identity mismatch')
    return model


def original_policy(row, held, seed):
    result = base.read(folder(FULL, held, seed)/'scale_result.json')
    return {'pixel_threshold': row['pixel_threshold'], 'anchor_threshold': row['anchor_threshold'],
            'tta_threshold': result['methods']['tta5']['threshold'], 'area_policy': row['policies']['area_guard']}


def fit_rgb_policy(out, manifest, held, seed, protocol, model, old):
    d = folder(out, held, seed); d.mkdir(parents=True, exist_ok=True)
    data = {}; maps = {}; histograms = {k: {} for k in ('global', 'mean', 'tta')}
    for g in range(6):
        if g == held: continue
        data[g] = load_group(manifest, g, 'rgb_proxy')
        grids = rgb.infer_grids(model, data[g]['views'])
        native = rgb.native_maps(grids, len(data[g]['labels'][0]))
        p, q = native['global'], native['local']; maps[g] = (p, q)
        pos = parent.gate.meta_split(len(p), g)[1]
        labels = [data[g]['labels'][i] for i in pos]
        for key, prob in [('global', p), ('mean', [.5*(a+b) for a, b in zip(p, q)]), ('tta', native['tta5'])]:
            histograms[key][g] = parent.source_histogram([prob[i] for i in pos], labels)
        del data[g]['views']
        print('RGB_SOURCE', held, seed, g, flush=True)
    thresholds = {}; calibrations = {}
    for key, h in histograms.items():
        thresholds[key], calibrations[key] = parent.threshold_from_histograms(h, held, .02)
    fit, cal, packed, records = screen.build_source_cases(data, maps, manifest, held, thresholds['mean'], thresholds['global'])
    pool = filt.CalibrationPool(cal, held)
    baseline = screen.full_calibration_metrics(pool, np.ones(len(pool.area), bool))
    scores = screen.score_cases(cal, None, 'area_guard')
    area = screen.calibrate(pool, scores, True, baseline, protocol['source_calibration'])
    records = [r for r in records if r['split'] == 'calibration']
    wanted = {r['key']+suffix for r in records for suffix in ('_labels', '_features', '_anchor')}
    np.savez_compressed(d/'source_components.npz', **{k: v for k, v in packed.items() if k in wanted})
    np.savez_compressed(d/'source_histograms.npz', **{f'{k}_g{g}': v for k, rows in histograms.items() for g, v in rows.items()})
    policy = {'frozen_at': base.now(), 'held_index': held, 'seed': seed, 'source_groups': sorted(data),
              'pixel_threshold': thresholds['mean'], 'anchor_threshold': thresholds['global'], 'tta_threshold': thresholds['tta'],
              'pixel_calibrations': calibrations, 'area_policy': area, 'source_records': records,
              'source_components_sha256': base.sha(d/'source_components.npz'),
              'source_histograms_sha256': base.sha(d/'source_histograms.npz'),
              'base_weight': old['base_weight'], 'base_weight_sha256': old['base_weight_sha256'],
              'base_state_sha256': old['base_state_sha256'], 'target_used_for_calibration': False,
              'protocol_sha256': base.sha(Path(out)/'protocol.json')}
    base.write(d/'rgb_policy.json', policy); return policy


def score_target(grids, data, gsd, policy, naive):
    native = rgb.native_maps(grids, len(data['labels'][0])); counts = {k: [] for k in METHODS}
    for i, truth in enumerate(data['labels']):
        p, q, t = (native[k][i] for k in ('global', 'local', 'tta5'))
        masks, _, _ = rgb.masks_from_policy(p, q, t, gsd, policy)
        masks['naive_bgrn_policy'] = rgb.masks_from_policy(p, q, t, gsd, naive)[0]['area_guard']
        for key, mask in masks.items():
            counts[key].append(base.parent_counts(mask, truth, data['components'][i], .5))
    return {k: {'held_out': base.aggregate_counts(v), 'per_parent_counts': v} for k, v in counts.items()}


def run_rgb(out, protocol, manifest):
    import torch
    for held in protocol['held_indices']:
        for seed in protocol['seeds']:
            d = folder(out, held, seed); path = d/'result.json'
            if path.exists(): continue
            old = base.read(folder(PRIOR, held, seed)/'filter_policy.json'); model = load_model(old)
            started = time.perf_counter(); policy = fit_rgb_policy(out, manifest, held, seed, protocol, model, old)
            opened = base.now(); target = load_group(manifest, held, 'rgb_proxy')
            grids = rgb.infer_grids(model, target['views'])
            np.savez_compressed(d/'target_grids.npz', **grids, original_target_indices=np.array(target['indices']))
            methods = score_target(grids, target, manifest['regions'][held]['gsd_m'], policy, original_policy(old, held, seed))
            base.write(path, {'held_index': held, 'seed': seed, 'target_opened_at': opened, 'policy_frozen_at': policy['frozen_at'],
                             'policy_sha256': base.sha(d/'rgb_policy.json'), 'target_grids_sha256': base.sha(d/'target_grids.npz'),
                             'target_rows': len(target['labels']), 'methods': methods, 'seconds': time.perf_counter()-started,
                             'full_bgrn_reference': base.read(folder(PRIOR, held, seed)/'result.json')['methods']['area_guard']['held_out']})
            print('RGB_FIT_COMPLETE', held, seed, {k: v['held_out']['f1'] for k, v in methods.items()}, flush=True)
            del model, target, grids; torch.cuda.empty_cache()


def load_mine(row):
    import rasterio
    assert_development_cases([row['mine']])
    arrays = {}; signatures = {}; valid = None
    for key in ('image', 'label', 'normal'):
        if base.sha(row[key+'_path']) != row[key+'_sha256']: raise ValueError('Mine input hash changed')
        with rasterio.open(row[key+'_path']) as src:
            arrays[key] = src.read(); signatures[key] = (src.shape, src.transform, src.crs)
            if key == 'image':
                valid = np.all(src.read_masks() > 0, axis=0)
                gsd = abs(src.transform.a)
                if src.count != 4 or src.crs is None or not src.crs.is_projected or src.crs.linear_units != 'metre' or src.transform.b or src.transform.d or abs(src.transform.e) != gsd:
                    raise ValueError('Explicit square projected-metre BGRN input required')
    if signatures['image'] != signatures['label'] or signatures['image'] != signatures['normal']:
        raise ValueError('Reference grids do not align; no automatic resampling')
    x = rgb.fill_invalid(arrays['image'], valid)
    return x, arrays['label'][0], arrays['normal'][0], valid, gsd


def run_mines(out, protocol):
    import torch
    from detector import LandslideDetector
    out = Path(out); d = out/'mines'; d.mkdir(exist_ok=True)
    if (d/'results.json').exists(): return
    # Persist explicit roles and planned exposures BEFORE opening images for inference.
    ledger = {'registered_at': base.now(), 'protocol_sha256': base.sha(out/'protocol.json'),
              'purpose': 'Repeated development regression, NOT independent mine acceptance.',
              'case_ids': [r['mine'] for r in protocol['mine_cases']],
              'reserved_cases_opened_or_predicted': [], 'registry_files_changed': False,
              'baselines': [], 'research_predictions': []}
    ledger['initial_aborted_exposures'] = protocol.get('implementation_amendment', {}).get('initial_mine_exposures', [])
    base.write(out/'exposure_ledger.json', ledger)
    baseline = LandslideDetector(DEFAULT, device='cuda', patch_size=32, threshold=.4)
    control = {}
    for row in protocol['mine_cases']:
        x, truth, normal, valid, gsd = load_mine(row); case = row['mine']; control[case] = {}
        for mode in protocol['mine_inputs']:
            a = x.copy()
            if mode == 'rgb_proxy': a[3] = a[2]
            prob = baseline.detect_image(a.transpose(1, 2, 0), stride=32, patch_norm='minmax', max_batch=128)
            path = d/f'{case}_production_{mode}.npz'; np.savez_compressed(path, probability=prob)
            control[case][mode] = {str(t): rgb.partial_reference_counts(prob.astype(np.float64) >= t, truth, normal, valid) for t in (.4, .5)}
            ledger['baselines'].append({'case': case, 'input_mode': mode, 'completed_at': base.now(), 'cache': str(path), 'cache_sha256': base.sha(path)})
            base.write(out/'exposure_ledger.json', ledger)
    del baseline; torch.cuda.empty_cache()
    rows = []
    for seed in protocol['seeds']:
        held = protocol['mine_primary_fold']; old = base.read(folder(PRIOR, held, seed)/'filter_policy.json')
        rgb_policy = base.read(folder(out, held, seed)/'rgb_policy.json'); model = load_model(old)
        for mode in protocol['mine_inputs']:
            policy = rgb_policy if mode == 'rgb_proxy' else original_policy(old, held, seed)
            for row in protocol['mine_cases']:
                x, truth, normal, valid, gsd = load_mine(row); started = time.perf_counter()
                grids, positions = rgb.scene_grids(model, x, mode)
                path = d/f"{row['mine']}_seed{seed}_{mode}.npz"; np.savez_compressed(path, **grids)
                masks = rgb.scene_masks(grids, truth.shape, gsd, policy)
                result = {'case': row['mine'], 'seed': seed, 'held_index': held, 'input_mode': mode,
                          'tile_count': len(positions), 'shape': list(truth.shape), 'gsd_m': gsd,
                          'methods': {k: rgb.partial_reference_counts(v, truth, normal, valid) for k, v in masks.items()},
                          'cache_path': str(path), 'cache_sha256': base.sha(path), 'seconds': time.perf_counter()-started}
                rows.append(result)
                ledger['research_predictions'].append({k: result[k] for k in ('case', 'seed', 'held_index', 'input_mode', 'tile_count', 'cache_path', 'cache_sha256')})
                base.write(out/'exposure_ledger.json', ledger)
                print('MINE_COMPLETE', row['mine'], seed, mode, result['methods']['area_guard'], flush=True)
        del model; torch.cuda.empty_cache()
    base.write(d/'results.json', {'completed_at': base.now(), 'production_control': control, 'research': rows,
                                'scope': protocol['mine_scoring'], 'reserved_mine_predictions': 0})


def summarize(out, protocol):
    out = Path(out); rows = [base.read(folder(out, g, s)/'result.json') for g in protocol['held_indices'] for s in protocol['seeds']]
    macro = {k: {m: float(np.mean([r['methods'][k]['held_out'][m] for r in rows])) for m in METRICS} for k in METHODS}
    full = {m: float(np.mean([r['full_bgrn_reference'][m] for r in rows])) for m in METRICS}
    by_seed = []
    for seed in protocol['seeds']:
        sub = [r for r in rows if r['seed'] == seed]
        mm = {k: {m: float(np.mean([r['methods'][k]['held_out'][m] for r in sub])) for m in METRICS} for k in METHODS}
        worst = max(r['methods']['multiscale_raw']['held_out']['small_cover50']-r['methods']['area_guard']['held_out']['small_cover50'] for r in sub)
        reduction, checks = acceptance.assess(mm, worst, protocol['limits'], protocol['extra_limits'])
        absolute = {'macro_fpr': mm['area_guard']['background_fpr'] <= protocol['absolute_rgb_fpr_limits']['macro_each_seed'],
                    'all_region_fpr': max(r['methods']['area_guard']['held_out']['background_fpr'] for r in sub) <= protocol['absolute_rgb_fpr_limits']['region_each_seed']}
        by_seed.append({'seed': seed, 'means': mm, 'false_component_reduction': reduction, 'filter_checks': checks, 'absolute_fpr_checks': absolute})
    mine = base.read(out/'mines/results.json'); mine_summary = []
    limits = protocol['mine_development_gate']
    for mode in protocol['mine_inputs']:
        for seed in protocol['seeds']:
            sub = [r for r in mine['research'] if r['input_mode'] == mode and r['seed'] == seed]
            core = []; norm = []; losses = []
            for r in sub:
                a = r['methods']['area_guard']; b = mine['production_control'][r['case']][mode]['0.4']
                core.append(a['positive_core']['rate']-b['positive_core']['rate'])
                norm.append(a['reviewed_normal']['rate']-b['reviewed_normal']['rate'])
                losses.append(b['positive_core']['rate']-a['positive_core']['rate'])
            checks = {'macro_core_gain': float(np.mean(core)) >= limits['macro_core_gain_each_seed'],
                      'normal_activation_control': float(np.mean(norm)) <= limits['macro_reviewed_normal_increase_max_each_seed'],
                      'no_large_case_recall_regression': max(losses) <= limits['any_mine_core_loss_max']}
            mine_summary.append({'input_mode': mode, 'seed': seed, 'macro_core_gain': float(np.mean(core)),
                                 'macro_reviewed_normal_change': float(np.mean(norm)), 'worst_case_core_loss': max(losses), 'checks': checks})
    check_assets(protocol)
    result = {'completed_at': base.now(), 'status': 'completed', 'rgb_target_macro': macro, 'actual_bgrn_reference_macro': full,
              'rgb_by_seed': by_seed, 'rgb_filter_gate_passed': all(v for r in by_seed for v in r['filter_checks'].values()),
              'rgb_absolute_fpr_gate_passed': all(v for r in by_seed for v in r['absolute_fpr_checks'].values()),
              'mine_by_mode_seed': mine_summary, 'mine_development_gate_passed': all(v for r in mine_summary for v in r['checks'].values()),
              'target_parent_seed_predictions': sum(r['target_rows'] for r in rows), 'different_target_parents': 768,
              'different_development_mines': 4, 'research_mine_scene_predictions': len(mine['research']),
              'new_cnn_training_updates': 0, 'official_test_predictions': 0, 'reserved_mine_predictions': 0,
              'external_calls': 0, 'production_changed': False,
              'scope': 'Missing-band ablation on exposed validation; four partial-reference seen mines. No independent/complete-boundary/field acceptance.'}
    base.write(out/'results.json', result)
    base.write(out/'state.json', {k: v for k, v in result.items() if k not in ('rgb_target_macro', 'rgb_by_seed', 'actual_bgrn_reference_macro', 'mine_by_mode_seed')})
    print('RGB_MINE_ROUND_COMPLETE', {k: result[k] for k in ('rgb_filter_gate_passed', 'rgb_absolute_fpr_gate_passed', 'mine_development_gate_passed')}, flush=True)
    return result


def verify(out):
    out = Path(out); protocol = base.read(out/'protocol.json'); check_assets(protocol)
    manifest = base.read(FULL/'data_manifest.json'); rebuilt = 0; replayed = 0
    for g in protocol['held_indices']:
        for seed in protocol['seeds']:
            d = folder(out, g, seed); result = base.read(d/'result.json'); policy = base.read(d/'rgb_policy.json')
            if base.sha(d/'rgb_policy.json') != result['policy_sha256']: raise ValueError('RGB policy changed')
            accepted_protocols = {base.sha(out/'protocol.json')}
            if 'implementation_amendment' in protocol:
                accepted_protocols.add(protocol['implementation_amendment']['initial_protocol_sha256'])
            if policy['protocol_sha256'] not in accepted_protocols: raise ValueError('Policy protocol provenance changed')
            for name, key in [('source_components.npz', 'source_components_sha256'), ('source_histograms.npz', 'source_histograms_sha256')]:
                if base.sha(d/name) != policy[key]: raise ValueError('Source cache changed')
            hist = np.load(d/'source_histograms.npz')
            for key, pk in [('global', 'anchor_threshold'), ('mean', 'pixel_threshold'), ('tta', 'tta_threshold')]:
                h = {i: hist[f'{key}_g{i}'] for i in policy['source_groups']}
                threshold, cal = parent.threshold_from_histograms(h, g, .02)
                if threshold != policy[pk] or cal != policy['pixel_calibrations'][key]: raise ValueError('RGB pixel calibration differs')
            hist.close(); packed = np.load(d/'source_components.npz'); cases = {i: [] for i in policy['source_groups']}; labels = {}
            for rec in policy['source_records']:
                i = rec['group']; v = manifest['regions'][i]['splits']['val']
                if i not in labels: labels[i] = np.load(v['label_path'], mmap_mode='r')
                raw = labels[i][rec['original_index']]
                if base.array_sha(raw) != rec['label_sha256']: raise ValueError('Source truth changed')
                truth = raw[0].astype(np.uint8); k = rec['key']; lab = packed[k+'_labels']
                areas = np.bincount(lab.ravel().astype(np.int64))[1:]
                pred = filt.PredictionComponents(lab, packed[k+'_features'], packed[k+'_anchor'], areas)
                cases[i].append(filt.observed(pred, truth, base.gt_components(truth, manifest['regions'][i]['gsd_m']), i, rec['position']))
            pool = filt.CalibrationPool(cases, g); baseline = screen.full_calibration_metrics(pool, np.ones(len(pool.area), bool))
            area = screen.calibrate(pool, screen.score_cases(cases, None, 'area_guard'), True, baseline, protocol['source_calibration'])
            if area != policy['area_policy']: raise ValueError('RGB area calibration differs')
            packed.close()
            if base.sha(d/'target_grids.npz') != result['target_grids_sha256']: raise ValueError('RGB target cache changed')
            grids = np.load(d/'target_grids.npz'); data = load_group(manifest, g, 'rgb_proxy')
            if not np.array_equal(grids['original_target_indices'], data['indices']): raise ValueError('Target rows changed')
            old = base.read(folder(PRIOR, g, seed)/'filter_policy.json')
            methods = score_target(grids, data, manifest['regions'][g]['gsd_m'], policy, original_policy(old, g, seed))
            if methods != result['methods']: raise ValueError('RGB metrics differ')
            model = load_model(old); fresh = rgb.infer_grids(model, data['views'][:3])
            for k, a in fresh.items():
                if not np.allclose(a, grids[k][:3], atol=2e-6, rtol=0): raise ValueError('RGB forward replay differs')
            del model; grids.close(); rebuilt += 1
            print('RGB_VERIFIED', g, seed, flush=True)
    mine = base.read(out/'mines/results.json')
    by_id = {r['mine']: r for r in protocol['mine_cases']}
    for row in mine['research']:
        _, truth, normal, valid, gsd = load_mine(by_id[row['case']])
        path = Path(row['cache_path'])
        if base.sha(path) != row['cache_sha256']: raise ValueError('Mine grids changed')
        g = protocol['mine_primary_fold']; seed = row['seed']
        old = base.read(folder(PRIOR, g, seed)/'filter_policy.json')
        policy = base.read(folder(out, g, seed)/'rgb_policy.json') if row['input_mode'] == 'rgb_proxy' else original_policy(old, g, seed)
        with np.load(path) as grids:
            masks = rgb.scene_masks(grids, truth.shape, gsd, policy)
        scores = {k: rgb.partial_reference_counts(v, truth, normal, valid) for k, v in masks.items()}
        if scores != row['methods']: raise ValueError('Mine metrics differ')
        replayed += 1
    ledger = base.read(out/'exposure_ledger.json')
    for row in ledger['baselines']:
        _, truth, normal, valid, _ = load_mine(by_id[row['case']]); path = Path(row['cache'])
        if base.sha(path) != row['cache_sha256']: raise ValueError('Production probability changed')
        with np.load(path) as packed: p = packed['probability']
        scores = {str(t): rgb.partial_reference_counts(p.astype(np.float64) >= t, truth, normal, valid) for t in (.4, .5)}
        if scores != mine['production_control'][row['case']][row['input_mode']]: raise ValueError('Control metrics differ')
    check_assets(protocol)
    value = {'verified_at': base.now(), 'status': 'passed', 'source_policies_and_target_metrics_rebuilt': rebuilt,
             'forward_replays': rebuilt*3, 'mine_scene_metrics_rebuilt': replayed, 'production_controls_rebuilt': len(ledger['baselines']),
             'protected_assets_unchanged': True, 'reserved_cases_opened_or_predicted': [], 'production_changed': False}
    base.write(out/'verification.json', value); print('VERIFIED', value, flush=True)


def main():
    p = argparse.ArgumentParser(); p.add_argument('--output', required=True)
    p.add_argument('--stage', choices=('register', 'run', 'verify'), default='run'); a = p.parse_args()
    out = Path(a.output).resolve()
    if a.stage == 'verify': parent.setup_runtime(); verify(out); return
    protocol = register(out)
    if a.stage == 'register': return
    parent.setup_runtime(); manifest = base.read(FULL/'data_manifest.json')
    base.write(out/'state.json', {'status': 'running', 'started_at': base.now(), 'production_changed': False})
    run_rgb(out, protocol, manifest); run_mines(out, protocol); summarize(out, protocol)


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8'); main()
