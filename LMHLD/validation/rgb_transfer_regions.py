"""Automatic geographically held-region check of the admitted RGB pipeline.

Reuse the frozen source-only trainer/evaluator through explicit, process-local
model factories. No historical file is changed. Rio is excluded to match the
previous six-region experiment, not called new blind data.
"""
from pathlib import Path
from contextlib import contextmanager
import sys
import time
import numpy as np
sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parent)]
from runtime_paths import configure_runtime_paths
configure_runtime_paths()
import rgb_transfer_study as transfer
import rgb_transfer_model as architecture
exp = transfer.exp; base = exp.base
PRIOR = exp.ROOT/'experiments/rgb_native_20261020'
TRANSFER = exp.ROOT/'experiments/rgb_transfer_20261020'
ARM = 'pretrained_whole'
ORIGINAL_FACTORY = exp.new_model


def wait_for_prior(out):
    while True:
        state = base.read(TRANSFER/'state.json') if (TRANSFER/'state.json').exists() else {}
        if state.get('status') == 'completed': break
        if state.get('status') == 'interrupted_or_failed': raise RuntimeError('Prior transfer stage requires repair before regional training')
        exp.state(out, status='waiting', stage='prior_transfer_verification'); time.sleep(10)
    r = base.read(TRANSFER/'results.json'); v = base.read(TRANSFER/'verification.json')
    if not r['pilot_admitted_both_seeds'] or v['status'] != 'passed': raise ValueError('No admitted verified transfer candidate')


def register(out):
    out = Path(out).resolve()
    if out.drive.upper() != 'D:' or not out.is_relative_to(exp.ROOT/'experiments'): raise ValueError('D experiments only')
    out.mkdir(parents=True, exist_ok=True)
    if (out/'protocol.json').exists():
        p = base.read(out/'protocol.json'); exp.old_rgb.check_assets(p); return p
    previous = base.read(PRIOR/'protocol.json'); frozen = dict(previous['frozen_assets'])
    paths = [Path(__file__), exp.ROOT/'tests/test_rgb_transfer_regions.py', exp.ROOT/'rgb_transfer_model.py',
             exp.ROOT/'validation/rgb_transfer_study.py', exp.ROOT/'validation/rgb_native_training.py',
             transfer.WEIGHTS, TRANSFER/'protocol.json', TRANSFER/'pilot_decision.json', TRANSFER/'results.json', TRANSFER/'verification.json',
             PRIOR/'protocol.json', PRIOR/'data_manifest.json', PRIOR/'results.json', PRIOR/'verification.json']
    for held in previous['held_indices']:
        for seed in previous['seeds']:
            d = exp.folder(PRIOR, 'fixed', held, seed)
            paths += [d/'training.json', d/'result.json', d/'rgb_policy.json']
    frozen.update({str(p): base.sha(p) for p in paths})
    p = {'registered_at': base.now(), 'frozen_assets': frozen, 'arms': [ARM], 'held_indices': previous['held_indices'], 'seeds': previous['seeds'],
         'updates': 4800, 'checkpoints': previous['checkpoints'], 'resume_every': 600, 'source_calibration': previous['source_calibration'],
         'scope': 'Six original LMHLD regions, three geographically held development folds and two seeds. Regions already exposed in prior studies; not a fresh independent blind test. Rio excluded for exact historical data matching.',
         'question': 'Does the admitted pretrained+RGB-normalized pipeline improve geographically held-region detection, and does matched five-view scale screening help on its stronger representations?',
         'training': 'Frozen native RGB whole-view trainer, same full eligible five-source pool per fold, batch20 (four/source), 4800 updates, checkpoint selection and optimizer schedule as historical fixed control. Held region excluded from gradient and all source policy decisions.',
         'initialization': 'Rebuild and verify historical random RGB model; copy its exact decoder/head to RGBTransferModel. Replace encoder by official ImageNet ResNet34, add explicit BGR->RGB mean/std normalization. Historical comparison changes BOTH pretraining and normalization; isolated initialization evidence comes from the preceding matched all-source study.',
         'reuse': 'Temporarily bind frozen trainer/evaluator model factory/loader inside this single Python process, restore bindings in finally. No edits to historical files, sampler, loss, monitoring, source thresholds or anchored area screening.',
         'methods': list(exp.METHODS), 'primary_backbone': 'pretrained TTA5 vs frozen historical fixed TTA5; compare same region/seed, shared decoder and exact parent+augmentation+view draw traces.',
         'backbone_gate': {'min_macro_f1_gain_each_seed': .02, 'max_region_f1_loss': .03, 'max_macro_background_fpr_increase': .002, 'max_macro_small_coverage_loss': .01},
         'limits': previous['limits'], 'extra_limits': previous['extra_limits'], 'absolute_rgb_fpr_limits': previous['absolute_rgb_fpr_limits'],
         'scale_gate': 'Use the unchanged historical area_guard-vs-TTA/mean gates and absolute background FPR limits, separately in both seeds. Do not choose weights or source cutoffs from target results.',
         'retention': 'Six source-selected states and necessary policies/target caches. Delete regenerable training cache and each optimizer resume after fit. No raw source copies.',
         'production_changed': False, 'mine_predictions': 0, 'official_test_predictions': 0, 'reserved_predictions': 0, 'external_model_calls': 0}
    exp.old_rgb.check_assets(p); base.write(out/'protocol.json', p); return p


def load_model(d):
    import torch
    d = Path(d); r = base.read(d/'training.json')
    if base.sha(d/'source_selected.pth') != r['selected_weight_sha256']: raise ValueError('Regional transfer weights changed')
    m = architecture.RGBTransferModel().cuda(); m.load_state_dict(torch.load(d/'source_selected.pth', map_location='cpu', weights_only=True)); m.eval()
    if exp.parent.short.state_hash(m) != r['source_selected_state_sha256']: raise ValueError('Regional selected model state changed')
    return m


@contextmanager
def factories(factory):
    old_factory, old_loader = exp.new_model, exp.load_model
    exp.new_model, exp.load_model = factory, load_model
    try: yield
    finally: exp.new_model, exp.load_model = old_factory, old_loader


def model_factory(out):
    def create(held, seed):
        import torch
        original, old_hash = ORIGINAL_FACTORY(held, seed); decoder = architecture.decoder_hash(original)
        model = architecture.RGBTransferModel(transfer.WEIGHTS).cuda()
        model.decoder.load_state_dict(original.decoder.state_dict()); model.segmentation_head.load_state_dict(original.segmentation_head.state_dict())
        if architecture.decoder_hash(model) != decoder: raise ValueError('Historical decoder was not matched')
        r = {'original_four_input_random_init_sha256': old_hash, 'shared_decoder_sha256': decoder, 'official_encoder_sha256': transfer.WEIGHT_SHA,
             'initial_state_sha256': exp.parent.short.state_hash(model)}
        path = exp.folder(out, ARM, held, seed)/'initialization.json'
        if path.exists() and base.read(path) != r: raise ValueError('Regional initialization changed')
        base.write(path, r); del original; torch.cuda.empty_cache(); return model, old_hash
    return create


def paired_training(current, previous):
    keys = ('parent_schedule_sha256', 'view_draw_sha256', 'visits_per_region', 'unique_seen_per_region', 'available_per_region', 'updates', 'original_four_input_random_init_sha256')
    return all(current[k] == previous[k] for k in keys)


def backbone_gate(current, previous, p):
    metrics = ('f1', 'small_cover50', 'small_iou50', 'background_fpr', 'false_components_per_mpix')
    a = {m: float(np.mean([r[m] for r in current])) for m in metrics}; b = {m: float(np.mean([r[m] for r in previous])) for m in metrics}
    worst = max(y['f1']-x['f1'] for x, y in zip(current, previous)); l = p['backbone_gate']
    checks = {'f1_gain': a['f1']-b['f1'] >= l['min_macro_f1_gain_each_seed'], 'each_region_f1': worst <= l['max_region_f1_loss'],
              'fpr_retained': a['background_fpr']-b['background_fpr'] <= l['max_macro_background_fpr_increase'],
              'small_coverage_retained': b['small_cover50']-a['small_cover50'] <= l['max_macro_small_coverage_loss']}
    return {'current': a, 'historical': b, 'deltas': {m: a[m]-b[m] for m in metrics}, 'worst_region_f1_loss': worst, 'checks': checks, 'passed': all(checks.values())}


def verify_fit(out, held, seed, data, manifest, p):
    import torch
    d = exp.folder(out, ARM, held, seed); policy = base.read(d/'rgb_policy.json'); result = base.read(d/'result.json')
    for path, digest in [(d/'rgb_policy.json', result['policy_sha256']), (d/'target_grids.npz', result['target_grids_sha256']),
                         (d/'source_components.npz', policy['source_components_sha256']), (d/'source_histograms.npz', policy['source_histograms_sha256']),
                         (d/'training.json', policy['training_sha256']), (d/'source_selected.pth', policy['weight_sha256']),
                         (Path(out)/'protocol.json', policy['protocol_sha256']), (Path(out)/'data_manifest.json', policy['manifest_sha256'])]:
        if base.sha(path) != digest: raise ValueError('Regional verification identity mismatch')
    with np.load(d/'source_histograms.npz') as h:
        for key, name in [('global', 'anchor_threshold'), ('mean', 'pixel_threshold'), ('tta', 'tta_threshold')]:
            value, cal = exp.parent.threshold_from_histograms({i: h[f'{key}_g{i}'] for i in policy['source_groups']}, held, .02)
            if value != policy[name] or cal != policy['pixel_calibrations'][key]: raise ValueError('Regional source threshold replay differs')
    with np.load(d/'source_components.npz') as z:
        cases = {i: [] for i in policy['source_groups']}
        for rec in policy['source_records']:
            i, pos, key = rec['group'], rec['position'], rec['key']; lab = z[key+'_labels']; areas = np.bincount(lab.ravel().astype(np.int64))[1:]
            pred = exp.filt.PredictionComponents(lab, z[key+'_features'], z[key+'_anchor'], areas)
            cases[i].append(exp.filt.observed(pred, data[i]['labels'][pos], data[i]['components'][pos], i, pos))
        pool = exp.filt.CalibrationPool(cases, held); baseline = exp.screen.full_calibration_metrics(pool, np.ones(len(pool.area), bool))
        actual = exp.screen.calibrate(pool, exp.screen.score_cases(cases, None, 'area_guard'), True, baseline, p['source_calibration'])
        if actual != policy['area_policy']: raise ValueError('Regional area policy replay differs')
    model = load_model(d)
    with np.load(d/'target_grids.npz') as z:
        if not np.array_equal(z['original_target_indices'], data[held]['indices']): raise ValueError('Target ordering changed')
        if exp.score_target(z, data[held], manifest['regions'][held]['gsd_m'], policy) != result['methods']: raise ValueError('Regional target scores replay differs')
        fresh = exp.native.infer_grids(model, data[held]['views'][:3])
        for key, a in fresh.items():
            if not np.allclose(a, z[key][:3], atol=2e-6, rtol=0): raise ValueError('Regional actual forward replay differs')
    historical = base.read(exp.folder(PRIOR, 'fixed', held, seed)/'training.json'); current = base.read(d/'training.json')
    if not paired_training(current, historical): raise ValueError('Historical training sampling/budget is not matched')
    if current['visits_per_region'][held] or current['unique_seen_per_region'][held]: raise ValueError('Held region entered gradient')
    del model; torch.cuda.empty_cache()
    print('TRANSFER_REGION_VERIFIED', held, seed, flush=True)


def summarize(out, p):
    rows = {(g, s): base.read(exp.folder(out, ARM, g, s)/'result.json') for g in p['held_indices'] for s in p['seeds']}
    backbone, scales = [], []
    for seed in p['seeds']:
        current = [rows[(g, seed)]['methods']['tta5']['held_out'] for g in p['held_indices']]
        old = [base.read(exp.folder(PRIOR, 'fixed', g, seed)/'result.json')['methods']['tta5']['held_out'] for g in p['held_indices']]
        backbone.append({'seed': seed, **backbone_gate(current, old, p)})
        means = {m: {k: float(np.mean([rows[(g, seed)]['methods'][m]['held_out'][k] for g in p['held_indices']])) for k in exp.METRICS} for m in exp.METHODS}
        loss = max(rows[(g, seed)]['methods']['multiscale_raw']['held_out']['small_cover50']-rows[(g, seed)]['methods']['area_guard']['held_out']['small_cover50'] for g in p['held_indices'])
        reduction, checks = exp.old_rgb.acceptance.assess(means, loss, p['limits'], p['extra_limits'])
        absolute = {'macro_fpr': means['area_guard']['background_fpr'] <= .03,
                    'each_region_fpr': max(rows[(g, seed)]['methods']['area_guard']['held_out']['background_fpr'] for g in p['held_indices']) <= .05}
        scales.append({'seed': seed, 'means': means, 'false_component_reduction': reduction, 'checks': checks, 'absolute_checks': absolute, 'passed': all(checks.values()) and all(absolute.values())})
    result = {'completed_at': base.now(), 'backbone_by_seed': backbone, 'scale_by_seed': scales,
              'backbone_passed_both_seeds': all(r['passed'] for r in backbone), 'scale_passed_both_seeds': all(r['passed'] for r in scales),
              'fits': 6, 'total_updates': 28800, 'production_changed': False, 'official_test_predictions': 0, 'reserved_predictions': 0,
              'scope': p['scope']}
    base.write(Path(out)/'results.json', result); return result


def run(out):
    import torch
    out = Path(out).resolve(); out.mkdir(parents=True, exist_ok=True); wait_for_prior(out); p = register(out); exp.parent.setup_runtime()
    exp.state(out, status='running', stage='matched_regional_source_cache')
    x, y, groups, manifest = exp.prepare_data(out); data = exp.load_validation(manifest)
    previous = base.read(PRIOR/'data_manifest.json')
    if manifest['kept_train'] != previous['kept_train']: raise ValueError('Historical native train pool changed')
    gx = torch.as_tensor(np.asarray(x[:, :1]), device='cuda'); gy = torch.as_tensor(np.asarray(y[:, :1]), device='cuda'); del x, y
    with factories(model_factory(out)):
        for held in p['held_indices']:
            for seed in p['seeds']:
                exp.train_fit(out, ARM, held, seed, gx, gy, groups, data, p)
                exp.evaluate_fit(out, ARM, held, seed, data, manifest, p)
    del gx, gy; torch.cuda.empty_cache(); exp.cleanup_cache(out)
    for held in p['held_indices']:
        for seed in p['seeds']: verify_fit(out, held, seed, data, manifest, p)
    exp.old_rgb.check_assets(p); r = summarize(out, p)
    base.write(out/'verification.json', {'verified_at': base.now(), 'status': 'passed', 'source_policies_rebuilt': 6,
        'target_results_rebuilt': 6, 'forward_replay_parents': 18, 'historical_sampling_matched': True, 'protected_assets_unchanged': True})
    exp.state(out, status='completed', stage='all_geographic_folds_verified', backbone_passed=r['backbone_passed_both_seeds'], scale_passed=r['scale_passed_both_seeds'])


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser(); p.add_argument('--output', required=True); args = p.parse_args()
    try: run(Path(args.output))
    except BaseException as e: exp.state(Path(args.output), status='interrupted_or_failed', error=str(e)); raise
