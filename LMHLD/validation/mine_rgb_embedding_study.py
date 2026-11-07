"""Two-seed leave-one-mine-out weak background/embedding/context validation.

Only source normal masks choose cutoffs; source OOF core checks admit a route.
All original masks, previous studies, weights, registries and defaults are read only.
"""
from pathlib import Path
import sys
import time
import numpy as np
sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parent)]
from runtime_paths import configure_runtime_paths
configure_runtime_paths()
import mine_rgb_head as head
import rgb_final_training as final
import rgb_native_detector as runtime

exp = final.exp; base = exp.base
PRIOR = exp.ROOT/'experiments/rgb_final_all_regions_20261020'
CASES = ('Amyntaio', 'LasCruces', 'Hpakant', 'CN_XJ2023')
CONTEXTS = (32, 64)


def register(out):
    from mine_data_protocol import assert_development_cases, registry
    out = Path(out).resolve()
    if out.drive.upper() != 'D:' or not out.is_relative_to(exp.ROOT/'experiments'): raise ValueError('Use D experiments')
    out.mkdir(parents=True, exist_ok=True)
    path = out/'protocol.json'
    if path.exists():
        p = base.read(path); exp.old_rgb.check_assets(p); return p
    prior = base.read(PRIOR/'protocol.json'); assert_development_cases(CASES)
    reg = registry(); groups = [reg['cases'][c].get('mine_group') or reg['cases'][c].get('mine_id') or c for c in CASES]
    if len(set(groups)) != 4: raise ValueError('Whole-mine groups must be distinct')
    paths = [Path(__file__), exp.ROOT/'mine_rgb_head.py', exp.ROOT/'tests/test_mine_rgb_embedding.py',
             PRIOR/'protocol.json', PRIOR/'results.json', PRIOR/'verification.json', PRIOR/'data_manifest.json',
             exp.ROOT/'model_registry.json', exp.ROOT/'datasets/input_reserve_registry.json',
             exp.ROOT/'datasets/mine_benchmark_20261002/mine_case_registry.json',
             exp.ROOT/'datasets/mine_benchmark_20261002/review_v2_20261003/review_manifest.json']
    for seed in (42, 43):
        for name in ('candidate.json', 'source_selected.pth', 'mine_results.json', 'source_calibration.npz'):
            paths.append(PRIOR/f'seed{seed}'/name)
    for c in prior['mine_cases']:
        for key in ('image', 'label', 'normal'):
            path0 = Path(c[key+'_path'])
            if base.sha(path0) != c[key+'_sha256']: raise ValueError('Case input changed')
            paths.append(path0)
    p = {'registered_at': base.now(), 'frozen_assets': {str(p): base.sha(p) for p in paths},
         'question': 'Can frozen native-RGB decoder features learn normal-operation distinctions on source mines and transfer more safely than changing a probability threshold? Does a native64 context help?',
         'cases': list(CASES), 'seeds': [42, 43], 'contexts': list(CONTEXTS), 'mine_cases': prior['mine_cases'],
         'primary': 'Source-admitted decoder-feature head; matched probability-only controls at same source normal budget. Context chosen only on source OOF performance.',
         'base': 'Frozen seven-source3-band RGB Unet++/ResNet34; none of the four mine scenes used for backbone gradients. No new CNN gradients.',
         'features': '16 final decoder channels averaged after inverse transforms of same5 CNN views, native-grid resized and storedFP16; plus logit of the frozen TTA probability. No coordinates, mineID, dates, labels or raw colour statistics at inference.',
         'parity_audit_overhead': 'First identity batch recomputes original full-model logits once per scene/context to verify decoder extraction. Extra diagnostic preprocessing forwards are not counted as candidate runtime savings.',
         'context': 'Native32 or64 square parents, parent BGR minmax01 then64 input. Context64 needs about1/4 the CNN windows per scene; compare pipeline effects, no equal-per-scene-FLOP claim.',
         'weak_reference': 'Published-figure conservative positive interiors and explicit project-reviewed constructed normals; .6 positive/.25 normal loss weights (algorithmic weak-reference convention, not calibrated confidence). No expert GIS, complete boundaries, geotechnical safety or field acceptance claim.',
         'reference_dates': {'Amyntaio': 'published2017-06-20 vs input2017-06-28; conservative buffered core', 'LasCruces': '2019-01-25 vs2019-01-27', 'Hpakant': 'same2020-08-03 calendar day, different sensor/time', 'CN_XJ2023': '2023-02-24 reference vs2023-02-23 input'},
         'normal_role_reuse': 'Old review manifest evaluation-only statements belong to preceding frozen studies. This new DEVELOPMENT experiment reuses only three SOURCE mine masks for learning/calibration; outer held normal/positive masks stay excluded from its policy. Prior results remain frozen, no new blind claim.',
         'classifier': 'Fixed L2 logistic C1, weighted source-only mean/std and clip+-8. At most2048 unique pixels/mine/class, fixed deterministic sampling, equal mine/class mass before weak loss weights. All explicitly reviewed normals eligible, not unknown background.',
         'nested_split': 'Outer train3/held1 whole mine. Inner train2/held1 across the three source mines creates OOF scores. Final head fit on all3 sources only after policy cutoff/gate definitions fixed.',
         'negative_only_cutoff': 'Lowest .001-grid threshold with equal-source normal activation <=min(mean production normal+.01, .8*mean current RGB32 normal); each source <=min(production+.03, RGB32+.01). Positive labels only validate conflicts and assess retention AFTER cutoff choice.',
         'source_head_gate': {'max_macro_core_loss_vs_rgb32': .01, 'max_any_source_core_loss': .03, 'min_core_gain_vs_same_context_probability_control': .02},
         'source_context_only_gate': {'max_macro_core_loss_vs_rgb32': .01, 'max_any_source_core_loss': .03, 'min_core_gain_vs_calibrated_context32_probability': .02},
         'selection': 'Admit eligible embedding32/64 or context-only64. Maximize source OOF mean core coverage, tie embedding first then smaller context. Otherwise explicit unchanged-RGB32 fallback. Never choose on outer target.',
         'diagnostic_gates': {'min_normal_reduction_vs_rgb32_each_seed': .20, 'max_macro_core_loss_vs_rgb32_each_seed': .01, 'max_any_mine_core_loss': .03, 'max_normal_increase_vs_production_each_seed': .01},
         'old_joint_promotion_gates': prior['mine_limits'],
         'old_joint_gate_limitation': 'A rejection-only filter cannot raise a seed whose baseline recall gain was below5pp; this learned classifier may add or remove pixels. Original joint gates remain reported, never relaxed or called contest standards.',
         'single_view': 'Separately source-calibrate identity forward on seven LMHLD source val calibration halves to2% equal-region FPR; admit only if selection-half sourceF1 loss<=.005 macro/.02 each region and selection FPR increase<=.002 macro/.005 each region vs frozen TTA. About80% fewer forwards, not a preclaimed wall-clock speedup; no mine-specific tuning.',
         'scope': 'Four historically exposed DEVELOPMENT mines, repeated weak-reference transfer, two seeds. No independent mine/complete accident boundary/real-user trial claim.',
         'production_changed': False, 'reserved_predictions': 0, 'official_test_predictions': 0, 'external_calls': 0, 'new_CNN_training_updates': 0}
    exp.old_rgb.check_assets(p); base.write(out/'protocol.json', p); return p


def image_only(row):
    import rasterio
    from mine_data_protocol import assert_development_cases
    assert_development_cases([row['mine']])
    if base.sha(row['image_path']) != row['image_sha256']: raise ValueError('Image changed')
    with rasterio.open(row['image_path']) as f:
        image = f.read(); valid = np.all(f.read_masks() > 0, axis=0)
        if image.shape[0] != 4 or f.res != (10., 10.) or f.crs is None: raise ValueError('Expected registered native10m BGRN scene')
        grid = {'shape': list(f.shape), 'crs': str(f.crs), 'transform': list(f.transform)}
    return exp.rgb.fill_invalid(image[:3], valid), valid, grid


def references(row, feature_meta, allowed_ids):
    import rasterio
    if row['mine'] not in allowed_ids: raise ValueError('Held reference cannot enter source learning/calibration')
    values = []
    for key in ('label', 'normal'):
        if base.sha(row[key+'_path']) != row[key+'_sha256']: raise ValueError('Reference changed')
        with rasterio.open(row[key+'_path']) as f:
            if {'shape': list(f.shape), 'crs': str(f.crs), 'transform': list(f.transform)} != feature_meta['grid']: raise ValueError('Reference/image grids differ')
            values.append(f.read(1))
    valid = np.load(feature_meta['valid_path'])
    ref = (*values, valid); head.supervision(*ref); return ref


def extract(out, seed, p):
    import torch
    d = Path(out)/f'seed{seed}'; d.mkdir(exist_ok=True)
    candidate = runtime.load_candidate(PRIOR/f'seed{seed}/candidate.json', device='cuda')
    rows = {}
    for row in p['mine_cases']:
        case = row['mine']; meta_path = d/f'{case}_features.json'
        if meta_path.exists(): rows[case] = base.read(meta_path); continue
        image, valid, grid = image_only(row); valid_path = d/f'{case}_valid.npy'; np.save(valid_path, valid)
        meta = {'case': case, 'grid': grid, 'valid_path': str(valid_path), 'valid_sha256': base.sha(valid_path), 'contexts': {},
                'backbone_config_sha256': base.sha(PRIOR/f'seed{seed}/candidate.json'), 'threshold': candidate.threshold,
                'image_only_no_reference_read': True, 'started_at': base.now()}
        for context in CONTEXTS:
            started = time.perf_counter(); features = head.scene_features(candidate.model, image, native_side=context)
            if context == 32:
                with np.load(PRIOR/f'seed{seed}'/(case+'_probability.npz')) as z:
                    err = float(np.max(np.abs(features['tta']-z['probability'])))
                    if err > 2e-6: raise ValueError('Frozen RGB32 probability replay differs')
                meta['base_probability_max_absolute_error'] = err
            path = d/f'{case}_context{context}.npz'; np.savez_compressed(path, **features)
            meta['contexts'][str(context)] = {'path': str(path), 'sha256': base.sha(path), 'seconds': time.perf_counter()-started}
        meta['completed_at'] = base.now(); base.write(meta_path, meta); rows[case] = meta
        print('MINE_FEATURES', seed, case, meta['base_probability_max_absolute_error'], flush=True)
    del candidate; torch.cuda.empty_cache(); return rows


def load_features(meta, context):
    r = meta['contexts'][str(context)]
    if base.sha(r['path']) != r['sha256']: raise ValueError('Frozen features changed')
    with np.load(r['path']) as z: return {k: z[k] for k in z.files}


def rate_summary(scores, refs, cutoff):
    core = head.core_rates(scores, refs, cutoff); normal = {}
    for c, prob in scores.items():
        _, n = head.supervision(*refs[c]); normal[c] = float((np.asarray(prob, np.float64)[n] >= cutoff).mean())
    return {'core': core, 'normal': normal, 'mean_core': float(np.mean(list(core.values()))), 'mean_normal': float(np.mean(list(normal.values())))}


def source_policy(out, seed, held, metas, p, prod, persist=True):
    sources = [c for c in CASES if c != held]; rows = {r['mine']: r for r in p['mine_cases']}
    refs = {c: references(rows[c], metas[c], sources) for c in sources}
    config = base.read(PRIOR/f'seed{seed}/candidate.json'); t = config['threshold']
    source32 = {c: load_features(metas[c], 32) for c in sources}
    base_score = rate_summary({c: f['tta'] for c, f in source32.items()}, refs, t)
    prod_rates = {c: prod[c]['0.4']['reviewed_normal']['rate'] for c in sources}; base_rates = base_score['normal']
    results = {}; sparse = {}; fitted = {}
    for context in CONTEXTS:
        features = source32 if context == 32 else {c: load_features(metas[c], context) for c in sources}
        probability = {c: f['tta'] for c, f in features.items()}
        pc = head.calibrate(probability, refs, sources, held, prod_rates, base_rates)
        pr = rate_summary(probability, refs, pc['threshold'])
        oof = {}; inner = {}
        for calibration_case in sources:
            fit = [c for c in sources if c != calibration_case]
            model = head.fit_head({c: features[c] for c in fit}, {c: refs[c] for c in fit}, fit, calibration_case)
            if held in model['source_ids'] or calibration_case in model['source_ids']: raise ValueError('Mine isolation failed')
            inner[calibration_case] = model; oof[calibration_case] = head.predict_head(model, features[calibration_case])
        hc = head.calibrate(oof, refs, sources, held, prod_rates, base_rates); hr = rate_summary(oof, refs, hc['threshold'])
        worst = max(base_score['core'][c]-hr['core'][c] for c in sources)
        gate = {'core_retained': hr['mean_core'] >= base_score['mean_core']-.01,
                'all_source_core_retained': worst <= .03, 'beats_probability_control': hr['mean_core'] >= pr['mean_core']+.02}
        results[f'embedding_{context}'] = {'context': context, 'kind': 'embedding', 'calibration': hc, 'source_scores': hr,
            'checks': gate, 'eligible': all(gate.values()), 'inner_heads': inner, 'probability_control': {'calibration': pc, 'source_scores': pr}}
        fitted[f'embedding_{context}'] = head.fit_head(features, refs, sources, held)
        results[f'probability_{context}'] = {'context': context, 'kind': 'probability', 'calibration': pc, 'source_scores': pr}
        for c in sources:
            pos, normal = head.supervision(*refs[c]); prefix = f'{context}_{c}'
            sparse[prefix+'_pos'] = oof[c][pos]; sparse[prefix+'_normal'] = oof[c][normal]
        del features, oof
    context_result = results['probability_64']; control32 = results['probability_32']
    s = context_result['source_scores']; gate = {'core_retained': s['mean_core'] >= base_score['mean_core']-.01,
        'all_source_core_retained': max(base_score['core'][c]-s['core'][c] for c in sources) <= .03,
        'beats_context32_probability': s['mean_core'] >= control32['source_scores']['mean_core']+.02}
    context_result.update(checks=gate, eligible=all(gate.values()))
    eligible = [(name, r) for name, r in results.items() if r.get('eligible')]
    chosen = max(eligible, key=lambda x: (x[1]['source_scores']['mean_core'], x[1]['kind'] == 'embedding', -x[1]['context']))[0] if eligible else 'unchanged_base'
    directory = Path(out)/f'seed{seed}'/f'held_{held}'; directory.mkdir(exist_ok=True)
    if persist:
        np.savez_compressed(directory/'source_oof_support.npz', **sparse)
    else:
        with np.load(directory/'source_oof_support.npz') as cached:
            if set(cached.files) != set(sparse) or any(not np.array_equal(cached[k], a) for k, a in sparse.items()):
                raise ValueError('Source OOF scores did not reproduce')
    record = {'frozen_at': base.now(), 'seed': seed, 'held_mine': held, 'source_mines': sources,
              'source_baseline': base_score, 'methods': results, 'final_heads': fitted, 'chosen': chosen,
              'source_oof_support_sha256': base.sha(directory/'source_oof_support.npz'),
              'source_only_policy': True, 'target_reference_used': False, 'protocol_sha256': base.sha(Path(out)/'protocol.json')}
    if persist:
        base.write(directory/'policy.json', record); print('SOURCE_MINE_POLICY', seed, held, chosen, flush=True)
    return record


def evaluate(out, seed, held, policy, metas, p):
    row = next(r for r in p['mine_cases'] if r['mine'] == held)
    ref = references(row, metas[held], [held]); base_features = load_features(metas[held], 32)
    t = base.read(PRIOR/f'seed{seed}/candidate.json')['threshold']; scores = {'unchanged_base': (base_features['tta'], t)}
    selected = policy['chosen']; prediction_path = None
    if selected != 'unchanged_base':
        method = policy['methods'][selected]; f = base_features if method['context'] == 32 else load_features(metas[held], 64)
        q = head.predict_head(policy['final_heads'][selected], f) if method['kind'] == 'embedding' else f['tta']
        scores['selected'] = (q, method['calibration']['threshold'])
        control = policy['methods'][f"probability_{method['context']}"]
        scores['probability_control'] = (f['tta'], control['calibration']['threshold'])
        scores['probability32_control'] = (base_features['tta'], policy['methods']['probability_32']['calibration']['threshold'])
        prediction_path = Path(out)/f'seed{seed}'/f'held_{held}'/'selected_probability.npz'
        np.savez_compressed(prediction_path, probability=q)
    methods = {k: exp.rgb.partial_reference_counts(np.asarray(prob, np.float64) >= threshold, *ref) for k, (prob, threshold) in scores.items()}
    if selected == 'unchanged_base': methods['selected'] = methods['unchanged_base']
    result = {'completed_at': base.now(), 'seed': seed, 'held_mine': held, 'chosen': selected, 'methods': methods,
              'policy_sha256': base.sha(Path(out)/f'seed{seed}'/f'held_{held}'/'policy.json'),
              'new_head_target_prediction': selected.startswith('embedding'),
              'selected_cache': str(prediction_path) if prediction_path else None,
              'selected_cache_sha256': base.sha(prediction_path) if prediction_path else None}
    base.write(Path(out)/f'seed{seed}'/f'held_{held}'/'result.json', result); return result


def run(out):
    out = Path(out); p = register(out); exp.parent.setup_runtime()
    prior = base.read(PRIOR/'protocol.json'); exp.old_rgb.check_assets(prior)
    prod = base.read(exp.ROOT/'experiments/rgb_native_20261020/mines/results.json')['production_control']
    all_results = []
    ledger_path = out/'exposure_ledger.json'
    ledger = base.read(ledger_path) if ledger_path.exists() else {'registered_at': base.now(), 'purpose': p['scope'], 'feature_extractions': [], 'held_evaluations': [], 'reserved_cases_opened_or_predicted': []}
    for seed in p['seeds']:
        exp.state(out, status='running', stage='frozen_rgb_features', seed=seed); metas = extract(out, seed, p)
        ledger['feature_extractions'].extend({'seed': seed, 'case': c, 'contexts': list(CONTEXTS), 'label_free': True} for c in CASES
            if not any(r['seed'] == seed and r['case'] == c for r in ledger['feature_extractions']))
        base.write(out/'exposure_ledger.json', ledger)
        policies = {}
        for held in CASES:
            exp.state(out, status='running', stage='nested_source_mine_learning', seed=seed, held_mine=held)
            path = out/f'seed{seed}'/f'held_{held}'/'policy.json'
            policies[held] = base.read(path) if path.exists() else source_policy(out, seed, held, metas, p, prod)
        # All four SOURCE policies are persisted before outer metric evaluation.
        for held in CASES:
            path = out/f'seed{seed}'/f'held_{held}'/'result.json'
            r = base.read(path) if path.exists() else evaluate(out, seed, held, policies[held], metas, p)
            all_results.append(r)
            if not any(a['seed'] == seed and a['held_mine'] == held for a in ledger['held_evaluations']):
                ledger['held_evaluations'].append({k: r[k] for k in ('seed', 'held_mine', 'chosen', 'new_head_target_prediction')})
            base.write(out/'exposure_ledger.json', ledger)
            print('HELD_MINE_RESULT', seed, held, r['chosen'], r['methods']['selected'], flush=True)
    by_seed = []
    for seed in p['seeds']:
        rows = [r for r in all_results if r['seed'] == seed]
        old_core, new_core, old_norm, new_norm, prod_core, prod_norm = native_means(rows, prod)
        worst = max(r['methods']['unchanged_base']['positive_core']['rate']-r['methods']['selected']['positive_core']['rate'] for r in rows)
        gates = {'normal_reduced20pct': new_norm <= .8*old_norm, 'core_retained_macro': new_core >= old_core-.01,
                 'core_retained_each_mine': worst <= .03, 'near_production_normal': new_norm <= prod_norm+.01}
        joint = {'production_core_gain5pp': new_core >= prod_core+.05, 'normal_control': new_norm <= prod_norm+.01,
                 'no_large_case_core_loss': max(prod[r['held_mine']]['0.4']['positive_core']['rate']-r['methods']['selected']['positive_core']['rate'] for r in rows) <= .03}
        by_seed.append({'seed': seed, 'baseline_macro_core': float(old_core), 'selected_macro_core': float(new_core),
                        'baseline_macro_normal': float(old_norm), 'selected_macro_normal': float(new_norm),
                        'normal_reduction': float(1-new_norm/max(old_norm, 1e-12)), 'checks': gates, 'original_joint_checks': joint,
                        'admitted_heads': sum(r['chosen'].startswith('embedding') for r in rows), 'admitted_context_only': sum(r['chosen'] == 'probability_64' for r in rows)})
    exp.old_rgb.check_assets(p)
    result = {'completed_at': base.now(), 'by_seed': by_seed, 'diagnostic_passed_both_seeds': all(v for r in by_seed for v in r['checks'].values()),
              'old_joint_passed_both_seeds': all(v for r in by_seed for v in r['original_joint_checks'].values()), 'models': all_results,
              'scope': p['scope'], 'production_changed': False, 'new_CNN_training_updates': 0, 'reserved_predictions': 0}
    base.write(out/'results.json', result); exp.state(out, status='completed', stage='verification_pending', diagnostic_passed=result['diagnostic_passed_both_seeds'])
    print('MINE_RGB_EMBEDDING_COMPLETE', by_seed, flush=True)
    verify(out)
    single_view(out)


def native_means(rows, prod):
    """Keep summaries JSON-native, including booleans produced by comparisons."""
    return tuple(float(np.mean(v)) for v in (
        [r['methods']['unchanged_base']['positive_core']['rate'] for r in rows],
        [r['methods']['selected']['positive_core']['rate'] for r in rows],
        [r['methods']['unchanged_base']['reviewed_normal']['rate'] for r in rows],
        [r['methods']['selected']['reviewed_normal']['rate'] for r in rows],
        [prod[r['held_mine']]['0.4']['positive_core']['rate'] for r in rows],
        [prod[r['held_mine']]['0.4']['reviewed_normal']['rate'] for r in rows]))


def verify(out):
    out = Path(out); p = base.read(out/'protocol.json'); exp.old_rgb.check_assets(p)
    prod = base.read(exp.ROOT/'experiments/rgb_native_20261020/mines/results.json')['production_control']
    checked = 0; targets = 0
    for seed in p['seeds']:
        metas = {c: base.read(out/f'seed{seed}'/f'{c}_features.json') for c in CASES}
        for c, meta in metas.items():
            if base.sha(meta['valid_path']) != meta['valid_sha256']: raise ValueError('Valid mask changed')
            if meta['backbone_config_sha256'] != base.sha(PRIOR/f'seed{seed}/candidate.json'): raise ValueError('Feature backbone changed')
            for context in CONTEXTS: load_features(meta, context)
        for held in CASES:
            d = out/f'seed{seed}'/f'held_{held}'; old = base.read(d/'policy.json')
            accepted = {base.sha(out/'protocol.json')}
            if 'implementation_amendment' in p: accepted.add(p['implementation_amendment']['initial_protocol_sha256'])
            if old['protocol_sha256'] not in accepted: raise ValueError('Unknown policy protocol version')
            if base.sha(d/'source_oof_support.npz') != old['source_oof_support_sha256']: raise ValueError('OOF support changed')
            replay = source_policy(out, seed, held, metas, p, prod, persist=False)
            for key in replay:
                if key not in ('frozen_at', 'protocol_sha256') and replay[key] != old[key]: raise ValueError(f'Nested source policy changed: {key}')
            r = base.read(d/'result.json')
            if base.sha(d/'policy.json') != r['policy_sha256']: raise ValueError('Target policy changed')
            if r['selected_cache']:
                if base.sha(r['selected_cache']) != r['selected_cache_sha256']: raise ValueError('Selected scores changed')
                chosen = old['chosen']; spec = old['methods'][chosen]; f = load_features(metas[held], spec['context'])
                score = head.predict_head(old['final_heads'][chosen], f) if spec['kind'] == 'embedding' else f['tta']
                with np.load(r['selected_cache']) as z:
                    if not np.array_equal(score, z['probability']): raise ValueError('Held score replay differs')
                row = next(a for a in p['mine_cases'] if a['mine'] == held); ref = references(row, metas[held], [held])
                metric = exp.rgb.partial_reference_counts(score.astype(np.float64) >= spec['calibration']['threshold'], *ref)
                if metric != r['methods']['selected']: raise ValueError('Held metrics differ')
                targets += 1
            checked += 1
            print('MINE_POLICY_VERIFIED', seed, held, flush=True)
    exp.old_rgb.check_assets(p)
    base.write(out/'verification.json', {'verified_at': base.now(), 'status': 'passed', 'nested_source_policies_refitted': checked,
                                        'source_oof_arrays_rebuilt': checked, 'admitted_target_scores_rebuilt': targets,
                                        'protected_assets_unchanged': True, 'production_changed': False})
    exp.state(out, status='completed', stage='embedding_verified_single_view_pending', verified_policies=checked)


def single_view(out):
    import torch
    out = Path(out); p = base.read(out/'protocol.json'); manifest = base.read(PRIOR/'data_manifest.json')
    directory = out/'single_view'; directory.mkdir(exist_ok=True); summaries = []
    for seed in p['seeds']:
        config = base.read(PRIOR/f'seed{seed}/candidate.json'); detector = runtime.load_candidate(PRIOR/f'seed{seed}/candidate.json', device='cuda')
        cache = {}; hist = {}; data = {}; single_seconds = 0.; tta_seconds = 0.; single_n = 0; tta_n = 0
        for g, region in enumerate(manifest['regions']):
            v = region['splits']['val']; x = np.load(v['image_path'], mmap_mode='r'); y = np.load(v['label_path'], mmap_mode='r')
            views = np.stack([exp.native.make_views(x[i, :3])[0][0] for i in v['indices']]); truth = [y[i, 0].astype(np.uint8) for i in v['indices']]
            torch.cuda.synchronize(); started = time.perf_counter()
            prob = exp.parent.short.forward(detector.model, views, batch=64)
            torch.cuda.synchronize(); single_seconds += time.perf_counter()-started; single_n += len(views)
            a, b = exp.parent.gate.meta_split(len(truth), g)
            native = exp.parent.short.native_probabilities(prob, truth); hist[g] = exp.parent.source_histogram([native[i] for i in b], [truth[i] for i in b])
            torch.cuda.synchronize(); started = time.perf_counter()
            tta = runtime.predict_tta(detector.model, views[a])
            torch.cuda.synchronize(); tta_seconds += time.perf_counter()-started; tta_n += len(a)
            data[g] = (truth, native, a, tta)
            cache[f'g{g}_single'] = prob; cache[f'g{g}_selection_tta'] = tta; cache[f'g{g}_histogram'] = hist[g]
        threshold, calibration = final.threshold(hist); before = {}; after = {}
        for g, (truth, native, pos, tta) in data.items():
            labels = [truth[i] for i in pos]
            before[str(g)] = exp.parent.short.pixel_metrics(exp.parent.short.native_probabilities(tta, labels), labels, config['threshold'])
            after[str(g)] = exp.parent.short.pixel_metrics([native[i] for i in pos], labels, threshold)
        delta = float(np.mean([after[g]['f1']-before[g]['f1'] for g in before])); worst = max(before[g]['f1']-after[g]['f1'] for g in before)
        fpr_delta = float(np.mean([after[g]['background_fpr']-before[g]['background_fpr'] for g in before]))
        worst_fpr = max(after[g]['background_fpr']-before[g]['background_fpr'] for g in before)
        eligible = delta >= -.005 and worst <= .02 and fpr_delta <= .002 and worst_fpr <= .005
        cache_path = directory/f'seed{seed}_source.npz'; np.savez_compressed(cache_path, **cache)
        policy = {'frozen_at': base.now(), 'seed': seed, 'single_threshold': threshold, 'source_calibration': calibration,
                  'source_selection_before': before, 'source_selection_after': after, 'source_macro_f1_change': delta,
                  'worst_source_f1_loss': worst, 'source_gate_passed': eligible, 'source_cache_sha256': base.sha(cache_path),
                  'source_selection_fpr_change': fpr_delta, 'worst_source_fpr_increase': worst_fpr,
                  'forward_timings': {'single_seconds': single_seconds, 'single_parents': single_n, 'tta_seconds': tta_seconds, 'tta_parents': tta_n,
                                      'per_parent_forward_speed_ratio': (tta_seconds/tta_n)/(single_seconds/single_n), 'scope': 'source-batch forward timing, not end-to-end scene/task/user timing'},
                  'source_only': True, 'expected_forward_count_change': 'five to one per native32 parent'}
        base.write(directory/f'seed{seed}_policy.json', policy)
        mine_rows = []
        if eligible:
            for row in p['mine_cases']:
                c = row['mine']; meta = base.read(out/f'seed{seed}'/f'{c}_features.json'); features = load_features(meta, 32); ref = references(row, meta, CASES)
                mine_rows.append({'case': c, 'single': exp.rgb.partial_reference_counts(features['single'].astype(np.float64) >= threshold, *ref),
                                  'tta5': exp.rgb.partial_reference_counts(features['tta'].astype(np.float64) >= config['threshold'], *ref)})
        summary = {'policy': policy, 'mine_rows': mine_rows, 'no_new_mine_CNN_forwards': True}
        summaries.append(summary); base.write(directory/f'seed{seed}_result.json', summary)
        print('SINGLE_VIEW_ADMISSION', seed, eligible, delta, worst, flush=True)
        del detector, data, cache; torch.cuda.empty_cache()
    exp.old_rgb.check_assets(p)
    base.write(directory/'results.json', {'completed_at': base.now(), 'seeds': summaries,
                                         'source_gate_passed_both_seeds': all(r['policy']['source_gate_passed'] for r in summaries),
                                         'production_changed': False})
    exp.state(out, status='completed', stage='all_planned_stages_completed', verified_policies=8,
              embedding_diagnostic_passed=base.read(out/'results.json')['diagnostic_passed_both_seeds'],
              single_view_source_passed_both=all(r['policy']['source_gate_passed'] for r in summaries))


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(); parser.add_argument('--output', required=True); a = parser.parse_args()
    try: run(Path(a.output))
    except BaseException as error: exp.state(Path(a.output), status='interrupted_or_failed', error=str(error)); raise
