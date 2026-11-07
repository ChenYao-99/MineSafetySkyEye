"""Bounded semantic supervision probe on the new pretrained RGB representation."""
from pathlib import Path
import sys
import numpy as np
sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parent)]
from runtime_paths import configure_runtime_paths
configure_runtime_paths()
import rgb_weak_semantics as semantic
import mine_rgb_head as features
import rgb_semantic_audit as audit
import mine_rgb_embedding_study as old
import rgb_transfer_model as architecture
exp = audit.exp; base = audit.base
PRIOR = audit.PRIOR
AUDIT = exp.ROOT/'experiments/rgb_semantic_audit_20261021'
CASES = old.CASES
ARMS = ('pixel', 'region_balanced')


def register(out):
    from mine_data_protocol import assert_development_cases
    out = Path(out).resolve()
    if out.drive.upper() != 'D:' or not out.is_relative_to(exp.ROOT/'experiments'): raise ValueError('D experiments only')
    out.mkdir(parents=True, exist_ok=True)
    if (out/'protocol.json').exists():
        p = base.read(out/'protocol.json'); exp.old_rgb.check_assets(p); return p
    assert_development_cases(CASES); parent = base.read(PRIOR/'protocol.json'); frozen = dict(base.read(AUDIT/'protocol.json')['frozen_assets'])
    paths = [Path(__file__), exp.ROOT/'rgb_weak_semantics.py', exp.ROOT/'tests/test_rgb_weak_semantics.py', exp.ROOT/'mine_rgb_head.py',
             exp.ROOT/'rgb_transfer_model.py', exp.ROOT/'validation/mine_rgb_embedding_study.py', AUDIT/'results.json', AUDIT/'verification.json',
             exp.ROOT/'experiments/mine_rgb_embedding_20261020/protocol.json', exp.ROOT/'experiments/mine_rgb_embedding_20261020/results.json',
             exp.ROOT/'experiments/mine_rgb_embedding_20261020/verification.json', exp.ROOT/'experiments/rgb_native_20261020/mines/results.json']
    for seed in (42, 43): paths.append(PRIOR/'imagenet_normalized'/f'seed{seed}'/'source_selected.pth')
    frozen.update({str(a): base.sha(a) for a in paths})
    p = {'registered_at': base.now(), 'frozen_assets': frozen, 'cases': list(CASES), 'mine_cases': parent['mine_cases'], 'seeds': [42, 43],
         'question': 'Does the materially stronger pretrained RGB representation make existing weak mine semantics transferable, and does equal weighting of reviewed normal regions help its bench/plant error concentration?',
         'changed_information_condition': 'Backbone is the newly verified ImageNet-initialized normalized7-source RGB model, not the scratch RGB backbone used by the prior rejected feature-head study. No new strong labels claimed.',
         'inference': 'Frozen native32 parent ->64, TTA5, sixteen inverse-TTA-averaged decoder channels plus logit(probability). Same17 features and classifier as previous semantic diagnostic. No polygon/region/mine/date/coordinates/labels in prediction inputs.',
         'supervision': 'Existing conservative positive cores and seventeen explicit project-reviewed normal regions only. Weak weights .6/.25 are algorithmic conventions, not measured confidence. Unknown and annual work context excluded. Outer held mine never enters any learning/calibration statistic.',
         'data_roles': 'Three SOURCE development mines per outer fold may be reused for weak supervised learning. Prior evaluation-only roles belong to old frozen studies; this new DEVELOPMENT protocol explicitly changes SOURCE use only. All four previously exposed; not new blind data.',
         'arms': list(ARMS), 'classifier': 'L2 logistic C1, lbfgs1000 tol1e-7; weighted source-only mean/std floor1e-6; clip+-8. No hyperparameter search.',
         'matched_samples': 'Both head arms use exactly the old deterministic at-most2048 original pixel indices/mine/class. Same total weak class mass and C. region_balanced distributes negative mass equally over existing source-normal polygons, not by polygon pixel count; source region IDs are weights only.',
         'nested_split': 'Outer train3/held1. Inner train2 and predict the third source, repeat3 for source OOF. Cutoff calibrated on source OOF NORMAL ONLY. Fit final source3 head after eligible strategy is determined. Persist all8 source policies before new outer-head scores.',
         'normal_budget': 'Same former diagnostic rule: <=min(mean production normal+.01,.8*pretrained baseline normal) and eachsource<=min(production+.03,baseline+.01); lowest.001 threshold. Positive cores assess retention after cutoff, not cutoff selection.',
         'admission': {'max_macro_core_loss': .01, 'max_any_source_core_loss': .03, 'min_core_gain_vs_calibrated_probability': .02, 'min_balanced_gain_vs_pixel': .01},
         'selection': 'Admit pixel if source core-retention and probability-control gates pass. Balanced must additionally beat the same-sample pixel head by1pp source mean core. Choose eligible max source core, ties simpler pixel. Otherwise frozen pretrained baseline, no new target head scoring.',
         'diagnostic': 'Per seed require mean reviewed-normal20% reduction, mean core loss<=1pp, anyminecoreloss<=3pp and mean normal<=production+.01. Also report original production+5ppcore/+1ppnormal/no3pploss gate; these are engineering criteria, not contest rules.',
         'retention': 'Retain only sparse reference-support feature vectors (pointwise inference equivalent), indices/weak labels/region tags, source OOF arrays and coefficients. Delete full-scene temporary features in memory; no full weight copies.',
         'scope': 'Four historically observed DEVELOPMENT mines. Support-only accuracy, not dense full-scene precision or complete accident IoU. No claim about hidden instability, safe active mining, professional trial or independent acceptance.',
         'production_changed': False, 'new_CNN_training_updates': 0, 'reserved_predictions': 0, 'official_test_predictions': 0, 'external_calls': 0}
    exp.old_rgb.check_assets(p); base.write(out/'protocol.json', p); return p


def support(out, seed, p):
    import torch
    d = Path(out)/f'seed{seed}'; d.mkdir(exist_ok=True); config = PRIOR/'imagenet_normalized'/f'seed{seed}'/'candidate.json'
    model = architecture.load_candidate(config, device='cuda').model
    review = base.read(audit.REVIEW); xj = base.read(audit.XJ); packs = {}; meta = {}
    for row in p['mine_cases']:
        c = row['mine']; path = d/(c+'_support.npz'); metadata = d/(c+'_support.json')
        if path.exists() and metadata.exists():
            r = base.read(metadata)
            if base.sha(path) != r['sha256'] or base.sha(config) != r['backbone_config_sha256']: raise ValueError('Semantic support provenance changed')
            with np.load(path) as z: packs[c] = {k: z[k] for k in z.files}
            meta[c] = r; continue
        image, valid, grid = old.image_only(row)
        # Predictor receives only BGR imagery. Labels are read AFTER the full
        # frozen CNN pass; selecting support cannot change a pointwise score.
        scene = features.scene_features(model, image, native_side=32)
        with np.load(PRIOR/'imagenet_normalized'/f'seed{seed}'/(c+'_probabilities.npz')) as z:
            error = float(np.max(np.abs(scene['tta']-z['tta5'])))
        if error > 2e-6: raise ValueError('New pretrained feature path does not reproduce cached probabilities')
        _, truth, normal, reference_valid, _ = exp.old_rgb.load_mine(row)
        if not np.array_equal(reference_valid, valid): raise ValueError('Reference/image validity mismatch')
        positive, negative = features.supervision(truth, normal, valid); definitions = xj['normal_regions'] if c == 'CN_XJ2023' else review['cases'][c]['regions']
        partition = audit.normal_partitions(truth.shape, normal, valid, definitions); region = np.zeros(truth.shape, np.uint16)
        for i, (_, _, mask) in enumerate(partition, 1): region[mask] = i
        index = np.flatnonzero((positive | negative).ravel()); x = features.matrix(scene)[index]
        pack = {'x': x, 'index': index, 'label': positive.ravel()[index].astype(np.uint8), 'region': region.ravel()[index], 'probability': scene['tta'].ravel()[index]}
        np.savez_compressed(path, **pack); r = {'case': c, 'seed': seed, 'sha256': base.sha(path), 'grid': grid, 'backbone_config_sha256': base.sha(config),
            'positive_pixels': int(positive.sum()), 'normal_pixels': int(negative.sum()), 'region_names': [r[0] for r in partition], 'probability_max_error': error,
            'cnn_extraction_label_free': True, 'only_explicit_reference_support_retained': True}
        base.write(metadata, r); packs[c] = pack; meta[c] = r; del scene
        print('PRETRAINED_SEMANTIC_SUPPORT', seed, c, len(index), 'replay_error', error, flush=True)
    del model; torch.cuda.empty_cache(); return packs, meta


def policy(out, seed, held, packs, p, prod, persist=True):
    sources = [c for c in CASES if c != held]; s = {c: packs[c] for c in sources}; q = base.read(PRIOR/'imagenet_normalized'/f'seed{seed}'/'candidate.json')['threshold']
    baseline = semantic.rates({c: s[c]['probability'] for c in sources}, s, q); production = {c: prod[c]['0.4']['reviewed_normal']['rate'] for c in sources}
    control_cutoff = semantic.calibration({c: s[c]['probability'] for c in sources}, s, sources, held, production, baseline['normal'])
    control = semantic.rates({c: s[c]['probability'] for c in sources}, s, control_cutoff['threshold'])
    methods = {}; sparse = {}; final = {}
    for arm in ARMS:
        oof = {}; models = {}
        for cal in sources:
            fit = [c for c in sources if c != cal]; models[cal] = semantic.fit({c: s[c] for c in fit}, fit, cal, arm)
            if held in models[cal]['sources']: raise ValueError('Outer held mine entered source head')
            oof[cal] = semantic.predict(models[cal], s[cal]['x'])
        cutoff = semantic.calibration(oof, s, sources, held, production, baseline['normal']); scores = semantic.rates(oof, s, cutoff['threshold'])
        checks = {'core_retained': scores['mean_core'] >= baseline['mean_core']-.01,
                  'each_source_core_retained': max(baseline['core'][c]-scores['core'][c] for c in sources) <= .03,
                  'beats_probability': scores['mean_core'] >= control['mean_core']+.02}
        if arm == 'region_balanced': checks['beats_pixel_head'] = scores['mean_core'] >= methods['pixel']['source_scores']['mean_core']+.01
        methods[arm] = {'calibration': cutoff, 'source_scores': scores, 'checks': checks, 'eligible': all(checks.values()), 'inner_models': models}
        final[arm] = semantic.fit(s, sources, held, arm)
        for c in sources: sparse[arm+'_'+c] = oof[c]
    if final['pixel']['samples'] != final['region_balanced']['samples']: raise ValueError('Semantic head arms did not use the same source pixels')
    eligible = [(a, r) for a, r in methods.items() if r['eligible']]
    chosen = max(eligible, key=lambda x: (x[1]['source_scores']['mean_core'], x[0] == 'pixel'))[0] if eligible else 'unchanged_pretrained'
    d = Path(out)/f'seed{seed}'/f'held_{held}'; d.mkdir(exist_ok=True)
    if persist: np.savez_compressed(d/'source_oof.npz', **sparse)
    else:
        with np.load(d/'source_oof.npz') as z:
            if set(z.files) != set(sparse) or any(not np.array_equal(z[k], v) for k, v in sparse.items()): raise ValueError('Source OOF semantic score replay failed')
    r = {'frozen_at': base.now(), 'held': held, 'seed': seed, 'sources': sources, 'baseline': baseline, 'probability_control': {'calibration': control_cutoff, 'scores': control},
         'methods': methods, 'final_models': final, 'chosen': chosen, 'source_oof_sha256': base.sha(d/'source_oof.npz'), 'protocol_sha256': base.sha(Path(out)/'protocol.json'), 'target_reference_used': False}
    if persist: base.write(d/'policy.json', r); print('PRETRAINED_SEMANTIC_POLICY', seed, held, chosen, {a: r['checks'] for a, r in methods.items()}, flush=True)
    return r


def evaluate(out, seed, held, packs, policy):
    d = Path(out)/f'seed{seed}'/f'held_{held}'; target = {held: packs[held]}
    q = base.read(PRIOR/'imagenet_normalized'/f'seed{seed}'/'candidate.json')['threshold']; baseline = semantic.rates({held: packs[held]['probability']}, target, q)
    chosen = policy['chosen']; candidate = baseline; scores = None
    if chosen != 'unchanged_pretrained':
        scores = semantic.predict(policy['final_models'][chosen], packs[held]['x']); candidate = semantic.rates({held: scores}, target, policy['methods'][chosen]['calibration']['threshold'])
        np.savez_compressed(d/'target_support_scores.npz', score=scores)
    r = {'completed_at': base.now(), 'seed': seed, 'held': held, 'chosen': chosen, 'baseline': baseline, 'candidate': candidate,
         'new_head_target_scoring': scores is not None, 'target_cache_sha256': base.sha(d/'target_support_scores.npz') if scores is not None else None,
         'policy_sha256': base.sha(d/'policy.json')}
    base.write(d/'result.json', r); return r


def run(out):
    out = Path(out); p = register(out); exp.parent.setup_runtime(); prod = base.read(exp.ROOT/'experiments/rgb_native_20261020/mines/results.json')['production_control']
    all_packs = {}; all_policies = {}; metadata = {}; results = []
    for seed in p['seeds']:
        exp.state(out, status='running', stage='new_representation_support', seed=seed); all_packs[seed], metadata[seed] = support(out, seed, p)
        for held in CASES:
            path = out/f'seed{seed}'/f'held_{held}'/'policy.json'
            all_policies[(seed, held)] = base.read(path) if path.exists() else policy(out, seed, held, all_packs[seed], p, prod)
    # Freeze every source-only decision before any new outer-head scoring.
    base.write(out/'source_policies_frozen.json', {'frozen_at': base.now(), 'policies': {f'{s}_{h}': base.sha(out/f'seed{s}'/f'held_{h}'/'policy.json') for s, h in all_policies}})
    for seed in p['seeds']:
        for held in CASES: results.append(evaluate(out, seed, held, all_packs[seed], all_policies[(seed, held)]))
    summaries = []
    for seed in p['seeds']:
        rows = [r for r in results if r['seed'] == seed]; old_core = float(np.mean([r['baseline']['mean_core'] for r in rows])); new_core = float(np.mean([r['candidate']['mean_core'] for r in rows]))
        old_normal = float(np.mean([r['baseline']['mean_normal'] for r in rows])); new_normal = float(np.mean([r['candidate']['mean_normal'] for r in rows]))
        prod_core = float(np.mean([prod[c]['0.4']['positive_core']['rate'] for c in CASES])); prod_normal = float(np.mean([prod[c]['0.4']['reviewed_normal']['rate'] for c in CASES]))
        checks = {'normal_reduced20pct': new_normal <= .8*old_normal, 'core_retained': new_core >= old_core-.01,
                  'each_mine_core_retained': max(r['baseline']['mean_core']-r['candidate']['mean_core'] for r in rows) <= .03, 'near_production_normal': new_normal <= prod_normal+.01}
        joint = {'core_gain5pp': new_core >= prod_core+.05, 'normal_control': new_normal <= prod_normal+.01,
                 'each_mine_core_vs_production': max(prod[r['held']]['0.4']['positive_core']['rate']-r['candidate']['mean_core'] for r in rows) <= .03}
        summaries.append({'seed': seed, 'baseline_core': old_core, 'candidate_core': new_core, 'baseline_normal': old_normal, 'candidate_normal': new_normal,
                          'admitted_heads': sum(r['new_head_target_scoring'] for r in rows), 'checks': checks, 'original_joint_checks': joint})
    # Refit nested source heads and re-score every admitted outer support.
    for seed in p['seeds']:
        for held in CASES:
            old_record = all_policies[(seed, held)]; fresh = policy(out, seed, held, all_packs[seed], p, prod, persist=False)
            for key in fresh:
                if key != 'frozen_at' and old_record[key] != fresh[key]: raise ValueError('Source semantic policy refit differs: '+key)
            r = next(a for a in results if a['seed'] == seed and a['held'] == held)
            if base.sha(out/f'seed{seed}'/f'held_{held}'/'policy.json') != r['policy_sha256']: raise ValueError('Outer policy identity changed')
            if r['new_head_target_scoring']:
                path = out/f'seed{seed}'/f'held_{held}'/'target_support_scores.npz'
                if base.sha(path) != r['target_cache_sha256']: raise ValueError('Outer semantic support cache changed')
                score = semantic.predict(old_record['final_models'][r['chosen']], all_packs[seed][held]['x'])
                with np.load(path) as z:
                    if not np.array_equal(score, z['score']): raise ValueError('Outer semantic pointwise replay failed')
    exp.old_rgb.check_assets(p)
    result = {'completed_at': base.now(), 'by_seed': summaries, 'source_policies': 8, 'new_outer_head_scores': sum(r['new_head_target_scoring'] for r in results),
              'passed_both_seeds': all(v for r in summaries for v in r['checks'].values()), 'original_joint_passed_both_seeds': all(v for r in summaries for v in r['original_joint_checks'].values()),
              'scope': p['scope'], 'production_changed': False, 'new_CNN_training_updates': 0, 'reserved_predictions': 0, 'official_test_predictions': 0}
    base.write(out/'results.json', result); base.write(out/'verification.json', {'verified_at': base.now(), 'status': 'passed', 'source_policies_refitted': 8,
        'sparse_score_replays': result['new_outer_head_scores'], 'old_CNN_probability_replays': 8, 'matched_source_samples': True, 'protected_assets_unchanged': True})
    exp.state(out, status='completed', stage='new_representation_semantic_probe_verified', passed=result['passed_both_seeds'], admitted=result['new_outer_head_scores'])
    print('PRETRAINED_SEMANTICS_COMPLETE', result, flush=True)


if __name__ == '__main__':
    import argparse
    a = argparse.ArgumentParser(); a.add_argument('--output', required=True); args = a.parse_args()
    try: run(Path(args.output))
    except BaseException as e: exp.state(Path(args.output), status='interrupted_or_failed', error=str(e)); raise
