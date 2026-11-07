"""Final seven-source engineering candidates; not a held-region benchmark.

Fixed whole-view recipe chosen from completed development comparisons. Rio
source roles change explicitly here; preceding Rio validation remains frozen.
"""
from pathlib import Path
from collections import defaultdict
import sys
import hashlib
import time
import numpy as np
sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parent)]
import rgb_native_training as exp
import rgb_native_new_region as rio
import rgb_native_detector as runtime
PARENT = exp.ROOT/'experiments/rgb_native_20261020'
RIO = exp.ROOT/'experiments/rgb_native_rio_validation_20261020'


def threshold(hist, budget=.02):
    if set(hist) != set(range(7)): raise ValueError('All seven explicit SOURCE regions required')
    tails = []
    for g in range(7):
        h = np.asarray(hist[g])
        if h.shape != (1001,) or not np.issubdtype(h.dtype, np.integer) or h.sum() <= 0 or np.any(h < 0): raise ValueError('Invalid source histogram')
        tails.append(np.r_[np.cumsum(h[::-1])[::-1]/h.sum(), 0.])
    mean = np.mean(tails, axis=0); i = int(np.flatnonzero(mean <= budget)[0])
    return i/1000, {'source_macro_fpr': float(mean[i]), 'source_fprs': [float(x[i]) for x in tails], 'target_used': False}


def sample(groups, rng):
    ids = np.concatenate([rng.choice(np.flatnonzero(groups == g), 3, replace=True) for g in range(7)])
    rng.shuffle(ids); return ids, int(rng.integers(4)), bool(rng.integers(2))


def register(out):
    out = Path(out).resolve()
    if out.drive.upper() != 'D:' or not out.is_relative_to(exp.ROOT/'experiments'): raise ValueError('Use D experiments')
    out.mkdir(parents=True, exist_ok=True)
    if (out/'protocol.json').exists():
        p = exp.base.read(out/'protocol.json'); exp.old_rgb.check_assets(p); return p
    paths = [Path(__file__), exp.ROOT/'rgb_native_detector.py', exp.ROOT/'tests/test_rgb_final.py',
             PARENT/'protocol.json', PARENT/'data_manifest.json', PARENT/'results.json', PARENT/'verification.json', PARENT/'paired_comparisons.json',
             RIO/'protocol.json', RIO/'results.json', RIO/'verification.json', exp.ROOT/'model_registry.json', exp.ROOT/'datasets/mine_benchmark_20261002/mine_case_registry.json']
    paths += [rio.SOURCE/(s+'_'+kind+'.npy') for s in ('train', 'val') for kind in ('images', 'labels')]
    p = {'registered_at': exp.base.now(), 'frozen_assets': {str(p): exp.base.sha(p) for p in paths},
         'purpose': 'Train final RGB engineering candidates on all seven available LMHLD train domains after the frozen cross-region study. All seven val domains are SOURCE selection/calibration here; this stage provides no new geographic test score.',
         'recipe_choice': 'Fixed whole-view native RGB + TTA5, selected from exposed development results; no scale jitter/area filter automatically promoted.',
         'source_groups': 7, 'seeds': [42, 43], 'updates': 4800, 'checkpoints': [1200, 2400, 3600, 4800], 'resume_every': 600,
         'batch': '21, exactly3 parents/source/update; five percent more examples than previous batch20; this is final candidate training, not a data-volume-only causal comparison.',
         'architecture': 'Random scratch Unet++/ResNet34,3 BGR inputs, parent minmax0..1,64 input, BCE+.5Dice; AdamW and1200+3600 LR schedule unchanged.',
         'selection': 'Equal-source TTA5 native F1@.5 on source selection halves, ties earliest. Other halves calibrate source equal-region mean background FPR<=2%.',
         'rio_input_assumption': exp.base.read(RIO/'protocol.json')['input_mapping'],
         'rio_role_change': 'Rio train478 now added as source; Rio val136 now source selection/calibration. Previous validation remains a frozen record of models with no Rio gradients; scores from this new stage cannot be called new-region/independent Rio acceptance.',
         'mine_cases': exp.base.read(PARENT/'protocol.json')['mine_cases'], 'mine_limits': exp.base.read(PARENT/'protocol.json')['mine_development_gate'],
         'deployment': 'Standalone explicit3-band candidate module/config only. Default registry, detector service and web app unchanged. No automatic promotion on SOURCE validation.',
         'official_test_files_opened': 0, 'reserved_mine_predictions': 0, 'external_model_calls': 0, 'production_changed': False}
    exp.old_rgb.check_assets(p); exp.base.write(out/'protocol.json', p); return p


def prepare(out):
    out = Path(out); old = exp.base.read(PARENT/'data_manifest.json'); regions = old['regions']+[{'region': rio.REGION, 'gsd_m': 5.,
        'splits': {'train': {'image_path': str(rio.SOURCE/'train_images.npy'), 'label_path': str(rio.SOURCE/'train_labels.npy')},
                   'val': {'image_path': str(rio.SOURCE/'val_images.npy'), 'label_path': str(rio.SOURCE/'val_labels.npy'), 'indices': list(range(136))}}}]
    raw = {}; labels = {}; val = {}; validation_hashes = set(); entries = []
    for g, region in enumerate(regions):
        t, v = region['splits']['train'], region['splits']['val']
        raw[g] = np.load(t['image_path'], mmap_mode='r'); labels[g] = np.load(t['label_path'], mmap_mode='r')
        vx = np.load(v['image_path'], mmap_mode='r'); vy = np.load(v['label_path'], mmap_mode='r')
        vh = [exp.base.array_sha(a[:3]) for a in vx]
        if g < 6:
            if vh != old['all_official_val_bgr_sha256'][str(g)]: raise ValueError('Original validation BGR changed')
            for pos, i in enumerate(v['indices']):
                if exp.base.array_sha(vy[i]) != v['label_parent_sha256'][pos]: raise ValueError('Original validation truth changed')
        validation_hashes.update(vh)
        val[g] = {'views': np.stack([exp.native.make_views(vx[i, :3])[0][0] for i in v['indices']]),
                  'labels': [vy[i, 0].astype(np.uint8) for i in v['indices']], 'indices': v['indices']}
    for r in old['kept_train']:
        g, i = r['group'], r['index']
        if exp.base.array_sha(raw[g][i, :3]) != r['sha256'] or exp.base.array_sha(labels[g][i]) != r['label_sha256']: raise ValueError('Prior RGB source changed')
        entries.append(dict(r))
    for i, (x, y) in enumerate(zip(raw[6], labels[6])):
        if not np.isfinite(x[:3]).all() or not np.isin(y, [0, 1]).all(): raise ValueError('Invalid added Rio source')
        entries.append({'group': 6, 'index': i, 'sha256': exp.base.array_sha(x[:3]), 'label_sha256': exp.base.array_sha(y), 'legacy_selected': False})
    kept, removed = exp.parent.quarantine_train(entries, validation_hashes)
    manifest = {'regions': regions, 'kept_train': kept, 'removed': removed, 'all_val_bgr_exact_hashes_count': len(validation_hashes),
                'source_train_parents': len(kept), 'source_selected_val_parents': sum(len(v['labels']) for v in val.values()), 'official_test_files_opened': 0}
    path = out/'data_manifest.json'
    if path.exists() and exp.base.read(path) != manifest: raise ValueError('Final source manifest changed')
    exp.base.write(path, manifest); cache = out/'training_cache'; cache.mkdir(exist_ok=True)
    x = np.lib.format.open_memmap(cache/'views.npy', mode='w+', dtype=np.float16, shape=(len(kept), 1, 3, 64, 64))
    y = np.lib.format.open_memmap(cache/'labels.npy', mode='w+', dtype=np.float16, shape=(len(kept), 1, 1, 64, 64))
    for j, r in enumerate(kept):
        g, i = r['group'], r['index']; bgr = raw[g][i, :3]
        a, b = exp.parent.compact_whole_view(np.concatenate([bgr, bgr[2:3]]), labels[g][i]); x[j] = a[:, :3]; y[j] = b
    x.flush(); y.flush()
    return x, y, np.array([r['group'] for r in kept]), val, manifest


def source_score(model, data, split):
    rows = {}
    for g, d in sorted(data.items()):
        pos = exp.parent.gate.meta_split(len(d['labels']), g)[split]
        probs = runtime.predict_tta(model, d['views'][pos]); truth = [d['labels'][i] for i in pos]
        rows[str(g)] = exp.parent.short.pixel_metrics(exp.parent.short.native_probabilities(probs, truth), truth)
    return {'macro_f1': float(np.mean([v['f1'] for v in rows.values()])), 'regions': rows}


def train(out, seed, x, y, groups, data, p):
    import torch
    import torch.nn.functional as F
    import segmentation_models_pytorch as smp
    d = Path(out)/f'seed{seed}'; d.mkdir(exist_ok=True)
    if (d/'training.json').exists(): return
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    model = smp.UnetPlusPlus(encoder_name='resnet34', encoder_weights=None, in_channels=3, classes=1).cuda()
    opt = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.0001); scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, 1200, eta_min=.00001)
    rng = np.random.default_rng(seed); curve = []; visits = np.zeros(7, np.int64); seen = np.zeros(len(groups), bool); losses = []; step0 = 0; phase = 'short'
    resume = d/'resume_training.pth'
    if resume.exists():
        cp = torch.load(resume, map_location='cpu', weights_only=True)
        if cp['protocol_sha256'] != exp.base.sha(Path(out)/'protocol.json'): raise ValueError('Final resume protocol changed')
        model.load_state_dict(cp['model']); phase = cp['phase']
        if phase == 'continuation': scheduler = exp.parent.short.start_continuation(opt)
        opt.load_state_dict(cp['optimizer']); scheduler.load_state_dict(cp['scheduler']); rng.bit_generator.state = cp['rng']
        curve, losses, step0 = cp['curve'], cp['losses'], cp['step']; visits = np.array(cp['visits']); seen = np.array(cp['seen'], bool)
        torch.set_rng_state(cp['torch_rng']); torch.cuda.set_rng_state_all(cp['cuda_rng'])
    model.train()
    for step in range(step0, p['updates']):
        ids, k, flip = sample(groups, rng); visits += np.bincount(groups[ids], minlength=7); seen[ids] = True
        a, b = x[ids, 0].float().rot90(k, (-2, -1)), y[ids, 0].float().rot90(k, (-2, -1))
        if flip: a, b = a.flip(-1), b.flip(-1)
        opt.zero_grad(set_to_none=True); z = model(a); prob = z.sigmoid()
        dice = (1-(2*(prob*b).sum((2, 3))+1e-6)/(prob.sum((2, 3))+b.sum((2, 3))+1e-6)).mean()
        loss = F.binary_cross_entropy_with_logits(z, b)+.5*dice
        if not torch.isfinite(loss): raise ValueError('Nonfinite final RGB loss')
        loss.backward(); opt.step(); scheduler.step(); losses.append(float(loss.detach())); u = step+1
        if u % 300 == 0:
            exp.state(out, status='running', stage='all_seven_sources_training', seed=seed, update=u)
            print('FINAL_RGB_TRAIN', seed, u, float(np.mean(losses[-100:])), flush=True)
        if u in p['checkpoints']:
            model.eval(); result = source_score(model, data, 0)
            curve.append({'update': u, 'source_selection': result, 'model_state_sha256': exp.parent.short.state_hash(model)})
            if exp.parent.short.choose_source_checkpoint(curve)['update'] == u: exp.parent.short.save_torch(d/'source_selected.pth', {k: v.detach().cpu() for k, v in model.state_dict().items()})
            exp.base.write(d/'source_curve.json', curve); print('FINAL_RGB_SOURCE', seed, u, result['macro_f1'], flush=True)
            if u == 1200: scheduler = exp.parent.short.start_continuation(opt); phase = 'continuation'
            model.train()
        if u % p['resume_every'] == 0:
            exp.parent.short.save_torch(resume, {'protocol_sha256': exp.base.sha(Path(out)/'protocol.json'), 'model': model.state_dict(), 'optimizer': opt.state_dict(), 'scheduler': scheduler.state_dict(),
                'phase': phase, 'rng': rng.bit_generator.state, 'curve': curve, 'losses': losses, 'step': u, 'visits': visits.tolist(), 'seen': seen.tolist(), 'torch_rng': torch.get_rng_state(), 'cuda_rng': torch.cuda.get_rng_state_all()})
    if visits.tolist() != [14400]*7: raise ValueError('Final source balancing failed')
    best = exp.parent.short.choose_source_checkpoint(curve)
    exp.base.write(d/'training.json', {'completed_at': exp.base.now(), 'seed': seed, 'updates': 4800, 'batch': 21,
        'selected_update': best['update'], 'source_macro_f1': best['source_selection']['macro_f1'], 'state_sha256': best['model_state_sha256'], 'weight_sha256': exp.base.sha(d/'source_selected.pth'),
        'source_visits': visits.tolist(), 'source_unique_seen': [int(seen[groups == g].sum()) for g in range(7)], 'source_available': [int((groups == g).sum()) for g in range(7)], 'held_out_regions': []})
    resume.unlink(missing_ok=True); del model, opt, scheduler; torch.cuda.empty_cache()


def calibrate(out, seed, data):
    import torch
    import segmentation_models_pytorch as smp
    d = Path(out)/f'seed{seed}'; record = exp.base.read(d/'training.json')
    if exp.base.sha(d/'source_selected.pth') != record['weight_sha256']: raise ValueError('Final weights changed')
    model = smp.UnetPlusPlus(encoder_name='resnet34', encoder_weights=None, in_channels=3, classes=1).cuda()
    model.load_state_dict(torch.load(d/'source_selected.pth', map_location='cpu', weights_only=True)); model.eval()
    if exp.parent.short.state_hash(model) != record['state_sha256']: raise ValueError('Selected final state changed')
    hist = {}; grids = {}
    for g, v in sorted(data.items()):
        pos = exp.parent.gate.meta_split(len(v['labels']), g)[1]; grids[g] = runtime.predict_tta(model, v['views'][pos])
        truth = [v['labels'][i] for i in pos]; native = exp.parent.short.native_probabilities(grids[g], truth)
        hist[g] = exp.parent.source_histogram(native, truth)
    t, cal = threshold(hist)
    np.savez_compressed(d/'source_calibration.npz', **{f'g{g}': a for g, a in grids.items()}, **{f'h{g}': h for g, h in hist.items()})
    config = {'schema_version': 1, 'input': 'BGR3_parent_minmax01_64_TTA5', 'weight': 'source_selected.pth', 'weight_sha256': record['weight_sha256'], 'threshold': t,
              'numerical_protocol': {'dtype': 'float32', 'cuda_tf32': False, 'cudnn_deterministic': True},
              'source_calibration': cal, 'source_calibration_cache_sha256': exp.base.sha(d/'source_calibration.npz'), 'selection_source_regions': 7,
              'source_validation_only': True, 'production_registered': False, 'official_test_predictions': 0}
    exp.base.write(d/'candidate.json', config); del model; torch.cuda.empty_cache()


def mines(out, seed, p):
    out = Path(out); d = out/f'seed{seed}'; detector = runtime.load_candidate(d/'candidate.json', device='cuda'); rows = []
    ledger = out/'exposure_ledger.json'; old = exp.base.read(ledger) if ledger.exists() else {'predictions': [], 'reserved_cases_opened_or_predicted': []}
    for case in p['mine_cases']:
        image, truth, normal, valid, _ = exp.old_rgb.load_mine(case)
        prob = detector.detect_image(image[:3].transpose(1, 2, 0), max_batch=64); path = d/(case['mine']+'_probability.npz'); np.savez_compressed(path, probability=prob)
        row = {'case': case['mine'], 'seed': seed, 'threshold': detector.threshold,
               'scores': exp.rgb.partial_reference_counts(prob.astype(np.float64) >= detector.threshold, truth, normal, valid),
               'probability_sha256': exp.base.sha(path)}
        rows.append(row); old['predictions'].append({'seed': seed, 'case': case['mine'], 'completed_at': exp.base.now()}); exp.base.write(ledger, old)
        print('FINAL_RGB_MINE', seed, case['mine'], row['scores'], flush=True)
    exp.base.write(d/'mine_results.json', rows); return rows


def main(out):
    import torch
    out = Path(out).resolve(); p = register(out); exp.parent.setup_runtime()
    exp.state(out, status='running', stage='seven_source_audit')
    x, y, groups, data, manifest = prepare(out); gx = torch.as_tensor(np.asarray(x), device='cuda'); gy = torch.as_tensor(np.asarray(y), device='cuda'); del x, y
    for seed in p['seeds']: train(out, seed, gx, gy, groups, data, p); calibrate(out, seed, data)
    del gx, gy; torch.cuda.empty_cache(); exp.cleanup_cache(out)
    all_rows = []
    for seed in p['seeds']: all_rows.extend(mines(out, seed, p))
    control = exp.base.read(PARENT/'mines/results.json')['production_control']; by_seed = []
    for seed in p['seeds']:
        sub = [r for r in all_rows if r['seed'] == seed]; gains = []; normal = []; losses = []
        for r in sub:
            a = r['scores']; b = control[r['case']]['0.4']; gains.append(a['positive_core']['rate']-b['positive_core']['rate']); normal.append(a['reviewed_normal']['rate']-b['reviewed_normal']['rate']); losses.append(-gains[-1])
        limits = p['mine_limits']; checks = {'core_gain': float(np.mean(gains)) >= limits['macro_core_gain_each_seed'],
            'normal_control': float(np.mean(normal)) <= limits['macro_reviewed_normal_increase_max_each_seed'], 'no_large_case_loss': max(losses) <= limits['any_mine_core_loss_max']}
        by_seed.append({'seed': seed, 'macro_core_gain': float(np.mean(gains)), 'macro_normal_change': float(np.mean(normal)), 'worst_core_loss': max(losses), 'checks': checks})
    # Rebuild thresholds and all mine scores; replay a real inference per seed.
    for seed in p['seeds']:
        d = out/f'seed{seed}'; config = exp.base.read(d/'candidate.json')
        with np.load(d/'source_calibration.npz') as z:
            rebuilt = {}
            for g in range(7):
                pos = exp.parent.gate.meta_split(len(data[g]['labels']), g)[1]
                truth = [data[g]['labels'][i] for i in pos]
                rebuilt[g] = exp.parent.source_histogram(exp.parent.short.native_probabilities(z[f'g{g}'], truth), truth)
                if not np.array_equal(rebuilt[g], z[f'h{g}']): raise ValueError('Final source background histogram replay failed')
            t, cal = threshold(rebuilt)
            if t != config['threshold'] or cal != config['source_calibration']: raise ValueError('Final source threshold replay failed')
        detector = runtime.load_candidate(d/'candidate.json', device='cuda')
        for case in p['mine_cases']:
            image, truth, normal, valid, _ = exp.old_rgb.load_mine(case); path = d/(case['mine']+'_probability.npz')
            with np.load(path) as z: prob = z['probability']
            row = next(r for r in all_rows if r['seed'] == seed and r['case'] == case['mine'])
            if exp.base.sha(path) != row['probability_sha256'] or exp.rgb.partial_reference_counts(prob.astype(np.float64) >= detector.threshold, truth, normal, valid) != row['scores']: raise ValueError('Final mine score replay failed')
            if case == p['mine_cases'][0]:
                fresh = detector.detect_image(image[:3].transpose(1, 2, 0), max_batch=17)
                if not np.allclose(fresh, prob, atol=2e-6, rtol=0): raise ValueError('Final streaming batch replay failed')
        del detector; torch.cuda.empty_cache()
    exp.old_rgb.check_assets(p)
    result = {'completed_at': exp.base.now(), 'fits': 2, 'source_regions': 7, 'source_train_parents': manifest['source_train_parents'], 'source_validation_parents': manifest['source_selected_val_parents'],
              'by_seed': by_seed, 'mine_development_gate_passed': all(v for r in by_seed for v in r['checks'].values()),
              'independent_geographic_test_in_this_stage': False, 'production_changed': False, 'official_test_files_opened': 0}
    exp.base.write(out/'results.json', result); exp.base.write(out/'verification.json', {'verified_at': exp.base.now(), 'status': 'passed', 'source_thresholds_rebuilt': 2, 'mine_scores_rebuilt': 8, 'streaming_inference_replays': 2, 'protected_assets_unchanged': True})
    exp.state(out, status='completed', stage='verified', fits=2, source_regions=7, mine_development_gate_passed=result['mine_development_gate_passed'])
    print('FINAL_RGB_COMPLETE', result, flush=True)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(); parser.add_argument('--output', required=True); a = parser.parse_args()
    try: main(Path(a.output))
    except BaseException as error:
        exp.state(Path(a.output), status='interrupted_or_failed', error=str(error), resumable=True); raise
