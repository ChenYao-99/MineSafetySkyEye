"""Autonomous bounded RGB training -> scale screening -> seen-mine regression.

All stages are local, resumable and preregistered. No production promotion,
reserved-test admission, provider API or dataset download is performed.
"""
from __future__ import annotations
import argparse
import copy
import hashlib
from pathlib import Path
import sys
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'validation')]
from runtime_paths import configure_runtime_paths
configure_runtime_paths()
import rgb_native as native
import rgb_scale as rgb
import component_filter as filt
import rgb_scale_regression as old_rgb

parent = old_rgb.parent
screen = old_rgb.screen
base = old_rgb.base
FULL = old_rgb.FULL
PREVIOUS = ROOT/'experiments/rgb_scale_regression_20261020'
ARMS = ('fixed', 'scale_jitter')
METHODS = ('tta5', 'multiscale_raw', 'area_guard')
METRICS = old_rgb.METRICS


def folder(out, arm, held, seed):
    return Path(out)/arm/f'fold_{held}_{base.REGIONS[held][0]}'/f'seed{seed}'


def state(out, **fields):
    base.write(Path(out)/'state.json', {'updated_at': base.now(), 'production_changed': False, **fields})


def register(out):
    out = Path(out).resolve()
    if out.drive.upper() != 'D:' or not out.is_relative_to(ROOT/'experiments'):
        raise ValueError('Use D:/LMHLD/experiments')
    out.mkdir(parents=True, exist_ok=True)
    if (out/'protocol.json').exists():
        p = base.read(out/'protocol.json'); old_rgb.check_assets(p); return p
    predecessor = base.read(PREVIOUS/'protocol.json')
    frozen = dict(predecessor['frozen_assets'])
    for path in [Path(__file__), ROOT/'rgb_native.py', ROOT/'tests/test_rgb_native.py',
                 PREVIOUS/'protocol.json', PREVIOUS/'results.json', PREVIOUS/'verification.json',
                 PREVIOUS/'preprocessing_parity.json', ROOT/'validation/rgb_scale_regression.py']:
        frozen[str(path)] = base.sha(path)
    p = {'registered_at': base.now(), 'frozen_assets': frozen, 'arms': list(ARMS),
         'held_indices': predecessor['held_indices'], 'seeds': predecessor['seeds'],
         'question': 'Does genuinely three-channel source-only training restore RGB transfer, and does paired scale-jitter training improve small-object detection under fixed inference and false-positive controls?',
         'updates': 4800, 'checkpoints': [1200, 2400, 3600, 4800], 'resume_every': 600,
         'architecture': 'Unet++/ResNet34, encoder_weights=None, in_channels=3. BGR only, no NIR and no fourth-channel proxy.',
         'initialization': 'Reproduce the old four-input random initialization for each fold/seed, verify it, copy all common tensors and only truncate encoder.conv1.weight to first3 channels. No trained weights reused; same3-channel initialization in both arms.',
         'training_pool': 'All eligible parents of five source regions. Start from the frozen full-source pool; re-audit raw/label identity and BGR-only exact duplicates against ALL original official val images and other regions. Same train parents in both arms.',
         'leakage_scope': 'BGR exact hashes do not establish spatial/near-duplicate independence. Original train/val splits, exposed development regions, no officialtest. No claim of unseen-sensor or mine-boundary acceptance.',
         'normalization': 'Once per native parent, per BGR band minmax0..1 before crop/resize64, constant band0, identical fractional label resizing and loss. Discard actual NIR before image validation/normalization.',
         'sampling': 'Batch20, exactly4 parents/source, with replacement. Both arms share parent sampling, view draws and paired legacy flips/rotations. fixed always uses global view; scale_jitter uses global with p.5, otherwise one uniform quadrant.',
         'loss': 'BCE with logits +0.5 per-parent softDice, fractional foreground occupancy.',
         'optimizer': 'AdamW lr.001 weight_decay.0001 cosine1200 to.00001; preserve moments, restart lr.0003 and cosine3600 to.00001. FP32 deterministic/noTF32.',
         'checkpoint_selection': 'Max equal-source-region native TTA5 F1@.5 on original source selection halves at1200/2400/3600/4800; ties earliest. No held-out or calibration-half checkpoint selection.',
         'source_calibration': dict(screen.CAL), 'pixel_budget': .02,
         'calibration': 'Other source validation halves independently calibrate TTA/global/mean pixel thresholds and anchored area guard using unchanged predecessor constraints. Calibration reused across pixel/object selection; no formal risk guarantee.',
         'methods': list(METHODS), 'primary_rgb_training_comparison': 'fixed TTA5 vs previous actual-BGRN-trained RGB TTA5, both source recalibrated.',
         'primary_scale_comparison': 'scale_jitter area_guard vs fixed TTA5; also each arm area_guard vs its own TTA5 and raw multiscale. Never choose an arm or cutoff using held-out results.',
         'limits': predecessor['limits'], 'extra_limits': predecessor['extra_limits'],
         'absolute_rgb_fpr_limits': predecessor['absolute_rgb_fpr_limits'],
         'mine_cases': predecessor['mine_cases'], 'mine_primary_fold': 2,
         'mine_development_gate': predecessor['mine_development_gate'],
         'mine_scoring': predecessor['mine_scoring'],
         'mine_pipeline': 'Both trained arms, first fixed fold2, both seeds. Only BGR from four already-seen development scenes, native32 tiles, global/quadrants64, per-parent screening. Image-only NoData fill, independent partial/normal scopes. Read-only prior production control probabilities, fixed.4/.5.',
         'automatic_order': ['RGB data audit', 'train fixed6 fits with source checkpoint selection',
                             'source-calibrated scale/object comparison for fixed6',
                             'train paired scale_jitter6 fits without reading held-out scores for training choices',
                             'source-calibrated scale/object comparison for scale_jitter6',
                             'four-mine regression for both arms and two seeds', 'verify caches and preserve necessary states'],
         'user_authorization': '2026-10-20 user explicitly requested native RGB full-source training followed by multiscale validation and automatic subsequent experiments without per-round chat prompts.',
         'retention': 'At most12 source-selected weights and one current optimizer resume. D-only temporary memorymapped train views, deleted after all training. No raw image copies, target reserve access or automatic production promotion.',
         'production_changed': False, 'official_test_predictions': 0, 'reserved_mine_predictions': 0,
         'external_calls': 0, 'downloads': 0}
    old_rgb.check_assets(p); base.write(out/'protocol.json', p); return p


def prepare_data(out):
    """D-only regenerable mmap cache; BGR exact dedup before any gradient."""
    out = Path(out); prior = base.read(FULL/'data_manifest.json')
    path = out/'data_manifest.json'; cache = out/'training_cache'; cache.mkdir(exist_ok=True)
    if path.exists():
        manifest = base.read(path)
        if base.sha(FULL/'data_manifest.json') != manifest['parent_manifest_sha256']:
            raise ValueError('Parent manifest changed')
        complete = cache/'complete.json'
        if complete.exists():
            stamp = base.read(complete)
            if stamp['manifest_sha256'] == base.sha(path):
                x = np.load(cache/'views.npy', mmap_mode='r'); y = np.load(cache/'labels.npy', mmap_mode='r')
                if list(x.shape) == stamp['x_shape'] and list(y.shape) == stamp['y_shape']:
                    return x, y, np.array([r['group'] for r in manifest['kept_train']]), manifest
    else:
        raw = {}; truth = {}; val_rgb = set(); entries = []; all_val = {}
        for g, region in enumerate(prior['regions']):
            t = region['splits']['train']; v = region['splits']['val']
            raw[g] = np.load(t['image_path'], mmap_mode='r'); truth[g] = np.load(t['label_path'], mmap_mode='r')
            vx = np.load(v['image_path'], mmap_mode='r')
            hashes = [base.array_sha(a[:3]) for a in vx]
            all_val[str(g)] = hashes; val_rgb.update(hashes)
            print('RGB_AUDIT_VAL', g, len(hashes), flush=True)
        for previous in prior['kept_train']:
            g = previous['group']; i = previous['index']; a = raw[g][i]; y = truth[g][i]
            if base.array_sha(a) != previous['sha256'] or base.array_sha(y) != previous['label_sha256']:
                raise ValueError('Frozen train image/label changed')
            entries.append({**previous, 'full_bgrn_sha256': previous['sha256'], 'sha256': base.array_sha(a[:3])})
        kept, removed = parent.quarantine_train(entries, val_rgb)
        manifest = {'registered_at': base.now(), 'parent_manifest_sha256': base.sha(FULL/'data_manifest.json'),
                    'regions': prior['regions'], 'kept_train': kept, 'removed_for_rgb': removed,
                    'all_official_val_bgr_sha256': all_val, 'original_train_parents': len(entries),
                    'rgb_train_parents': len(kept), 'all_val_rgb_hashes': sum(map(len, all_val.values())),
                    'test_files_opened': 0}
        base.write(path, manifest)
        del raw, truth
    n = len(manifest['kept_train'])
    x = np.lib.format.open_memmap(cache/'views.npy', mode='w+', dtype=np.float16, shape=(n, 5, 3, 64, 64))
    y = np.lib.format.open_memmap(cache/'labels.npy', mode='w+', dtype=np.float16, shape=(n, 5, 1, 64, 64))
    pools = {}
    for g, r in enumerate(manifest['regions']):
        t = r['splits']['train']; pools[g] = (np.load(t['image_path'], mmap_mode='r'), np.load(t['label_path'], mmap_mode='r'))
    for j, rec in enumerate(manifest['kept_train']):
        xx, yy = pools[rec['group']]; i = rec['index']
        if base.array_sha(xx[i, :3]) != rec['sha256'] or base.array_sha(yy[i]) != rec['label_sha256']:
            raise ValueError('RGB cache source changed')
        x[j], y[j] = native.make_views(xx[i], yy[i])
        if j % 3000 == 0: print('RGB_PACK', j, '/', n, flush=True)
    x.flush(); y.flush()
    base.write(cache/'complete.json', {'manifest_sha256': base.sha(path), 'x_shape': list(x.shape), 'y_shape': list(y.shape),
                                     'bytes': x.nbytes+y.nbytes, 'regenerable': True})
    return x, y, np.array([r['group'] for r in manifest['kept_train']]), manifest


def load_validation(manifest):
    data = {}
    for g, row in enumerate(manifest['regions']):
        d = old_rgb.load_group(manifest, g, 'rgb_proxy')
        d['views'] = np.ascontiguousarray(d['views'][:, :, :3]); data[g] = d
    return data


def new_model(held, seed):
    import torch
    import segmentation_models_pytorch as smp
    torch.manual_seed(seed+held*100); torch.cuda.manual_seed_all(seed+held*100)
    control = smp.UnetPlusPlus(encoder_name='resnet34', encoder_weights=None, in_channels=4, classes=1)
    h = hashlib.sha256()
    for a in control.state_dict().values(): h.update(a.numpy().tobytes())
    old = base.read(old_rgb.folder(FULL, held, seed)/'result.json')
    if h.hexdigest() != old['training']['initial_state_sha256']: raise ValueError('Paired initialization changed')
    weights = control.state_dict(); weights['encoder.conv1.weight'] = weights['encoder.conv1.weight'][:, :3].clone()
    model = smp.UnetPlusPlus(encoder_name='resnet34', encoder_weights=None, in_channels=3, classes=1)
    model.load_state_dict(weights); del control, weights
    return model.cuda(), h.hexdigest()


def draw_step(groups, held, rng, paired):
    ids = base.sample_batch(groups, held, rng, 4)
    views = np.where(rng.random(len(ids)) < .5, 0, rng.integers(1, 5, len(ids)))
    rng.integers(0, 4); rng.integers(0, 2)  # match the old full-pool RNG consumption
    _, k, flip = parent.legacy_augmentation(np.repeat(np.arange(6), 512), held, paired)
    return ids, views, k, flip


def monitor(model, x, y, groups, data, held):
    result = {}; losses = []
    for g in range(6):
        if g == held: continue
        ids = np.flatnonzero(groups == g)[:64]
        a = x[ids, 0].cpu().numpy(); b = y[ids, 0, 0].cpu().numpy()
        full = parent.short.forward(model, a, batch=32)
        train_loss = parent.short.soft_loss(full, b); losses.append(train_loss)
        pos = parent.gate.meta_split(len(data[g]['labels']), g)[0]
        labels = [data[g]['labels'][i] for i in pos]; images = data[g]['views'][pos, 0]
        full = parent.short.forward(model, images, batch=32); tta = full.copy()
        for k in (1, 2, 3, 4): tta += parent.short.forward(model, images, batch=32, kind=k)
        prob = parent.short.native_probabilities(tta/5, labels)
        result[str(g)] = {**parent.short.pixel_metrics(prob, labels), 'selection_positions': pos.tolist(),
                          'soft_loss': parent.short.soft_loss(full, parent.short.label_grid(labels)), 'train_monitor_loss': train_loss}
    filt.source_keys(map(int, result), held)
    return {'regions': result, 'macro_f1': float(np.mean([r['f1'] for r in result.values()])),
            'macro_train_monitor_loss': float(np.mean(losses))}


def train_fit(out, arm, held, seed, x, y, groups, data, protocol):
    import torch
    import torch.nn.functional as F
    d = folder(out, arm, held, seed); d.mkdir(parents=True, exist_ok=True)
    if (d/'training.json').exists(): return
    model, original_hash = new_model(held, seed); initial_hash = parent.short.state_hash(model)
    opt = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.0001)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, 1200, eta_min=.00001)
    rng = np.random.default_rng(seed+held*100); paired = np.random.default_rng(seed+held*100)
    seen = np.zeros(len(groups), bool); visits = np.zeros(6, np.int64)
    curve = []; losses = []; trace = bytearray(); view_trace = bytearray(); start_step = 0; seconds = 0.; phase = 'short'
    resume = d/'resume_training.pth'
    if resume.exists():
        cp = torch.load(resume, map_location='cpu', weights_only=True)
        if cp['protocol_sha256'] != base.sha(Path(out)/'protocol.json'): raise ValueError('Resume protocol changed')
        model.load_state_dict(cp['model']); phase = cp['phase']
        if phase == 'continuation': scheduler = parent.short.start_continuation(opt)
        opt.load_state_dict(cp['optimizer']); scheduler.load_state_dict(cp['scheduler'])
        rng.bit_generator.state = cp['rng']; paired.bit_generator.state = cp['paired_rng']
        seen = np.array(cp['seen'], bool); visits = np.array(cp['visits'], np.int64)
        curve, losses = cp['curve'], cp['losses']; trace = bytearray(cp['trace']); view_trace = bytearray(cp['view_trace'])
        start_step, seconds = cp['step'], cp['seconds']
        torch.set_rng_state(cp['torch_rng']); torch.cuda.set_rng_state_all(cp['cuda_rng'])
        print('RGB_RESUME', arm, held, seed, start_step, flush=True)
    model.train(); torch.cuda.synchronize(); started = time.perf_counter()
    for step in range(start_step, protocol['updates']):
        ids, views, k, flip = draw_step(groups, held, rng, paired)
        trace.extend(np.asarray([*ids, k, int(flip)], np.int64).tobytes()); view_trace.extend(np.asarray(views, np.int64).tobytes())
        seen[ids] = True; visits += np.bincount(groups[ids], minlength=6)
        chosen = views if arm == 'scale_jitter' else np.zeros(len(ids), np.int64)
        a = x[ids, chosen].float().rot90(k, (-2, -1)); b = y[ids, chosen].float().rot90(k, (-2, -1))
        if flip: a, b = a.flip(-1), b.flip(-1)
        opt.zero_grad(set_to_none=True); logits = model(a); p = logits.sigmoid()
        dice = (1-(2*(p*b).sum((2, 3))+1e-6)/(p.sum((2, 3))+b.sum((2, 3))+1e-6)).mean()
        loss = F.binary_cross_entropy_with_logits(logits, b)+.5*dice
        if not torch.isfinite(loss): raise ValueError('Nonfinite loss')
        loss.backward(); opt.step(); scheduler.step(); losses.append(float(loss.detach())); update = step+1
        if update % 300 == 0:
            state(out, status='running', stage='training', arm=arm, held_index=held, seed=seed, update=update, max_updates=4800)
            print('RGB_TRAIN', arm, held, seed, update, round(float(np.mean(losses[-100:])), 5), flush=True)
        if update in protocol['checkpoints']:
            model.eval(); selection = monitor(model, x, y, groups, data, held)
            curve.append({'update': update, 'model_state_sha256': parent.short.state_hash(model),
                          'source_selection': selection, 'mean_train_loss_last100': float(np.mean(losses[-100:])), 'completed_at': base.now()})
            if parent.short.choose_source_checkpoint(curve)['update'] == update:
                parent.short.save_torch(d/'source_selected.pth', {k: a.detach().cpu() for k, a in model.state_dict().items()})
            base.write(d/'source_curve.json', curve)
            print('RGB_SOURCE_CHECKPOINT', arm, held, seed, update, selection['macro_f1'], flush=True)
            if update == 1200: scheduler = parent.short.start_continuation(opt); phase = 'continuation'
            model.train()
        if update % protocol['resume_every'] == 0:
            torch.cuda.synchronize(); seconds += time.perf_counter()-started
            parent.short.save_torch(resume, {'protocol_sha256': base.sha(Path(out)/'protocol.json'),
                'model': model.state_dict(), 'optimizer': opt.state_dict(), 'scheduler': scheduler.state_dict(),
                'phase': phase, 'rng': rng.bit_generator.state, 'paired_rng': paired.bit_generator.state,
                'seen': seen.tolist(), 'visits': visits.tolist(), 'curve': curve, 'losses': losses,
                'trace': bytes(trace), 'view_trace': bytes(view_trace), 'step': update, 'seconds': seconds,
                'torch_rng': torch.get_rng_state(), 'cuda_rng': torch.cuda.get_rng_state_all()})
            started = time.perf_counter()
    if visits.tolist() != [0 if g == held else 19200 for g in range(6)]: raise ValueError('Source balancing failed')
    best = parent.short.choose_source_checkpoint(curve)
    record = {'completed_at': base.now(), 'arm': arm, 'held_index': held, 'seed': seed, 'updates': 4800,
              'selected_update': best['update'], 'source_macro_f1': best['source_selection']['macro_f1'],
              'source_selected_state_sha256': best['model_state_sha256'], 'selected_weight_sha256': base.sha(d/'source_selected.pth'),
              'original_four_input_random_init_sha256': original_hash, 'initial_rgb_state_sha256': initial_hash,
              'parent_schedule_sha256': hashlib.sha256(trace).hexdigest(), 'view_draw_sha256': hashlib.sha256(view_trace).hexdigest(),
              'visits_per_region': visits.tolist(), 'unique_seen_per_region': [int(seen[groups == g].sum()) for g in range(6)],
              'available_per_region': [int((groups == g).sum()) for g in range(6)], 'seconds_including_monitor': seconds,
              'target_used_for_selection': False, 'calibration_used_for_selection': False}
    if arm == 'scale_jitter':
        fixed = base.read(folder(out, 'fixed', held, seed)/'training.json')
        for k in ('parent_schedule_sha256', 'view_draw_sha256', 'initial_rgb_state_sha256', 'visits_per_region', 'unique_seen_per_region'):
            if record[k] != fixed[k]: raise ValueError(f'Paired-arm mismatch: {k}')
    base.write(d/'training.json', record); resume.unlink(missing_ok=True)
    del model, opt, scheduler; torch.cuda.empty_cache()


def load_model(d):
    import torch
    import segmentation_models_pytorch as smp
    training = base.read(Path(d)/'training.json'); path = Path(d)/'source_selected.pth'
    if base.sha(path) != training['selected_weight_sha256']: raise ValueError('Selected RGB weight changed')
    model = smp.UnetPlusPlus(encoder_name='resnet34', encoder_weights=None, in_channels=3, classes=1).cuda()
    model.load_state_dict(torch.load(path, map_location='cpu', weights_only=True)); model.eval()
    if parent.short.state_hash(model) != training['source_selected_state_sha256']: raise ValueError('Selected state differs')
    return model


def evaluate_fit(out, arm, held, seed, data, manifest, protocol):
    import torch
    d = folder(out, arm, held, seed)
    if (d/'result.json').exists(): return
    state(out, status='running', stage='source_calibration_and_scale', arm=arm, held_index=held, seed=seed)
    model = load_model(d); maps = {}; hists = {k: {} for k in ('global', 'mean', 'tta')}
    for g in range(6):
        if g == held: continue
        grids = native.infer_grids(model, data[g]['views'])
        m = rgb.native_maps(grids, len(data[g]['labels'][0])); maps[g] = (m['global'], m['local'])
        pos = parent.gate.meta_split(len(data[g]['labels']), g)[1]
        labels = [data[g]['labels'][i] for i in pos]
        for key, probabilities in [('global', m['global']), ('mean', [.5*(a+b) for a, b in zip(*maps[g])]), ('tta', m['tta5'])]:
            hists[key][g] = parent.source_histogram([probabilities[i] for i in pos], labels)
    thresholds = {}; calibrations = {}
    for key, hist in hists.items(): thresholds[key], calibrations[key] = parent.threshold_from_histograms(hist, held, .02)
    source = {g: a for g, a in data.items() if g != held}
    _, cal, packed, records = screen.build_source_cases(source, maps, manifest, held, thresholds['mean'], thresholds['global'])
    pool = filt.CalibrationPool(cal, held); baseline = screen.full_calibration_metrics(pool, np.ones(len(pool.area), bool))
    area = screen.calibrate(pool, screen.score_cases(cal, None, 'area_guard'), True, baseline, protocol['source_calibration'])
    records = [r for r in records if r['split'] == 'calibration']
    keys = {r['key']+suffix for r in records for suffix in ('_labels', '_features', '_anchor')}
    np.savez_compressed(d/'source_components.npz', **{k: a for k, a in packed.items() if k in keys})
    np.savez_compressed(d/'source_histograms.npz', **{f'{k}_g{g}': a for k, hist in hists.items() for g, a in hist.items()})
    policy = {'frozen_at': base.now(), 'pixel_threshold': thresholds['mean'], 'anchor_threshold': thresholds['global'],
              'tta_threshold': thresholds['tta'], 'pixel_calibrations': calibrations, 'area_policy': area,
              'source_groups': sorted(source), 'source_records': records, 'held_index': held, 'seed': seed, 'arm': arm,
              'source_components_sha256': base.sha(d/'source_components.npz'), 'source_histograms_sha256': base.sha(d/'source_histograms.npz'),
              'training_sha256': base.sha(d/'training.json'), 'weight_sha256': base.sha(d/'source_selected.pth'),
              'protocol_sha256': base.sha(Path(out)/'protocol.json'), 'manifest_sha256': base.sha(Path(out)/'data_manifest.json'),
              'target_used': False}
    base.write(d/'rgb_policy.json', policy)
    del maps, packed, pool, source, cal
    opened = base.now(); grids = native.infer_grids(model, data[held]['views'])
    np.savez_compressed(d/'target_grids.npz', **grids, original_target_indices=np.array(data[held]['indices']))
    methods = score_target(grids, data[held], manifest['regions'][held]['gsd_m'], policy)
    record = {'completed_at': base.now(), 'arm': arm, 'held_index': held, 'seed': seed,
              'policy_frozen_at': policy['frozen_at'], 'target_inferred_after_policy_at': opened,
              'policy_sha256': base.sha(d/'rgb_policy.json'), 'target_grids_sha256': base.sha(d/'target_grids.npz'),
              'target_rows': len(data[held]['labels']), 'methods': methods}
    base.write(d/'result.json', record)
    print('RGB_SCALE_COMPLETE', arm, held, seed, {k: a['held_out']['f1'] for k, a in methods.items()}, flush=True)
    del model, grids; torch.cuda.empty_cache()


def score_target(grids, data, gsd, policy):
    m = rgb.native_maps(grids, len(data['labels'][0])); counts = {k: [] for k in METHODS}
    for i, truth in enumerate(data['labels']):
        masks, _, _ = rgb.masks_from_policy(m['global'][i], m['local'][i], m['tta5'][i], gsd, policy)
        for key, mask in masks.items(): counts[key].append(base.parent_counts(mask, truth, data['components'][i], .5))
    return {k: {'held_out': base.aggregate_counts(a), 'per_parent_counts': a} for k, a in counts.items()}


def run_mines(out, protocol):
    import torch
    out = Path(out); directory = out/'mines'; directory.mkdir(exist_ok=True)
    prior = base.read(PREVIOUS/'mines/results.json')
    ledger_path = out/'exposure_ledger.json'
    if ledger_path.exists(): ledger = base.read(ledger_path)
    else:
        ledger = {'registered_at': base.now(), 'case_ids': [r['mine'] for r in protocol['mine_cases']],
                  'scope': 'Repeated development regression only, partial references, no untouched mines.',
                  'reserved_cases_opened_or_predicted': [], 'predictions': []}
        base.write(ledger_path, ledger)
    rows = []
    for arm in protocol['arms']:
        for seed in protocol['seeds']:
            held = protocol['mine_primary_fold']; d = folder(out, arm, held, seed); model = load_model(d)
            policy = base.read(d/'rgb_policy.json')
            for case in protocol['mine_cases']:
                name = f"{arm}_seed{seed}_{case['mine']}"; result_path = directory/f'{name}.json'
                if result_path.exists(): rows.append(base.read(result_path)); continue
                state(out, status='running', stage='mine_development_regression', arm=arm, seed=seed, case=case['mine'])
                image, truth, normal, valid, gsd = old_rgb.load_mine(case)
                grids = native.scene_grids(model, image[:3]); path = directory/f'{name}.npz'
                np.savez_compressed(path, **grids)
                masks = native.scene_masks(grids, truth.shape, gsd, policy)
                row = {'arm': arm, 'seed': seed, 'case': case['mine'], 'completed_at': base.now(),
                       'native_cache_path': str(path), 'native_cache_sha256': base.sha(path), 'policy_sha256': base.sha(d/'rgb_policy.json'),
                       'methods': {k: rgb.partial_reference_counts(v, truth, normal, valid) for k, v in masks.items()}}
                base.write(result_path, row); rows.append(row)
                ledger['predictions'].append({k: row[k] for k in ('arm', 'seed', 'case', 'completed_at', 'native_cache_path', 'native_cache_sha256')})
                base.write(ledger_path, ledger)
                print('RGB_NATIVE_MINE', arm, seed, case['mine'], row['methods']['area_guard'], flush=True)
            del model; torch.cuda.empty_cache()
    base.write(directory/'results.json', {'completed_at': base.now(), 'research': rows,
                                         'production_control': {k: v['rgb_proxy'] for k, v in prior['production_control'].items()},
                                         'production_control_source_sha256': base.sha(PREVIOUS/'mines/results.json')})


def summarize(out, protocol):
    out = Path(out); rows = [base.read(folder(out, a, g, s)/'result.json') for a in protocol['arms'] for g in protocol['held_indices'] for s in protocol['seeds']]
    def macro(sub, method):
        return {m: float(np.mean([r['methods'][method]['held_out'][m] for r in sub])) for m in METRICS}
    results = {a: {m: macro([r for r in rows if r['arm'] == a], m) for m in METHODS} for a in protocol['arms']}
    checks = []
    for arm in protocol['arms']:
        for seed in protocol['seeds']:
            sub = [r for r in rows if r['arm'] == arm and r['seed'] == seed]; means = {m: macro(sub, m) for m in METHODS}
            loss = max(r['methods']['multiscale_raw']['held_out']['small_cover50']-r['methods']['area_guard']['held_out']['small_cover50'] for r in sub)
            reduction, gate = old_rgb.acceptance.assess(means, loss, protocol['limits'], protocol['extra_limits'])
            abs_gate = {'macro_fpr': means['area_guard']['background_fpr'] <= .03,
                        'all_region_fpr': max(r['methods']['area_guard']['held_out']['background_fpr'] for r in sub) <= .05}
            checks.append({'arm': arm, 'seed': seed, 'means': means, 'false_component_reduction': reduction,
                           'filter_checks': gate, 'absolute_fpr_checks': abs_gate})
    mine = base.read(out/'mines/results.json'); mine_gates = []
    for arm in protocol['arms']:
        for seed in protocol['seeds']:
            sub = [r for r in mine['research'] if r['arm'] == arm and r['seed'] == seed]
            core = []; normal = []; losses = []
            for r in sub:
                a = r['methods']['area_guard']; b = mine['production_control'][r['case']]['0.4']
                core.append(a['positive_core']['rate']-b['positive_core']['rate'])
                normal.append(a['reviewed_normal']['rate']-b['reviewed_normal']['rate']); losses.append(-core[-1])
            limits = protocol['mine_development_gate']
            gate = {'core_gain': float(np.mean(core)) >= limits['macro_core_gain_each_seed'],
                    'normal_control': float(np.mean(normal)) <= limits['macro_reviewed_normal_increase_max_each_seed'],
                    'no_large_case_loss': max(losses) <= limits['any_mine_core_loss_max']}
            mine_gates.append({'arm': arm, 'seed': seed, 'macro_core_gain': float(np.mean(core)),
                               'macro_normal_change': float(np.mean(normal)), 'worst_core_loss': max(losses), 'checks': gate})
    for g in protocol['held_indices']:
        for seed in protocol['seeds']:
            a = base.read(folder(out, 'fixed', g, seed)/'training.json'); b = base.read(folder(out, 'scale_jitter', g, seed)/'training.json')
            for key in ('initial_rgb_state_sha256', 'parent_schedule_sha256', 'view_draw_sha256'):
                if a[key] != b[key]: raise ValueError('Paired experiment changed')
    old_rgb.check_assets(protocol)
    result = {'completed_at': base.now(), 'status': 'completed', 'fits': 12, 'updates_per_fit': 4800,
              'total_updates': 57600, 'target_macro': results, 'by_arm_seed': checks, 'mine_by_arm_seed': mine_gates,
              'previous_rgb_projection_macro': base.read(PREVIOUS/'results.json')['rgb_target_macro'],
              'different_target_parents': 768, 'target_parent_arm_seed_predictions': sum(r['target_rows'] for r in rows),
              'source_selected_weights': 12, 'mine_scene_predictions': len(mine['research']),
              'paired_initialization_sampling_and_augmentation_verified': True,
              'official_test_predictions': 0, 'reserved_mine_predictions': 0, 'external_calls': 0, 'production_changed': False}
    base.write(out/'results.json', result); state(out, status='completed', stage='verification_pending', fits=12, total_updates=57600)
    return result


def verify(out):
    import torch
    out = Path(out); protocol = register(out); manifest = base.read(out/'data_manifest.json'); data = load_validation(manifest)
    checked = 0
    for arm in protocol['arms']:
        for g in protocol['held_indices']:
            for seed in protocol['seeds']:
                d = folder(out, arm, g, seed); policy = base.read(d/'rgb_policy.json'); r = base.read(d/'result.json')
                for path, digest in [(d/'rgb_policy.json', r['policy_sha256']), (d/'target_grids.npz', r['target_grids_sha256']),
                                     (d/'source_components.npz', policy['source_components_sha256']), (d/'source_histograms.npz', policy['source_histograms_sha256']),
                                     (d/'training.json', policy['training_sha256']), (d/'source_selected.pth', policy['weight_sha256']),
                                     (out/'protocol.json', policy['protocol_sha256']), (out/'data_manifest.json', policy['manifest_sha256'])]:
                    if base.sha(path) != digest: raise ValueError('Verified asset changed')
                with np.load(d/'source_histograms.npz') as h:
                    for key, name in [('global', 'anchor_threshold'), ('mean', 'pixel_threshold'), ('tta', 'tta_threshold')]:
                        value, cal = parent.threshold_from_histograms({i: h[f'{key}_g{i}'] for i in policy['source_groups']}, g, .02)
                        if value != policy[name] or cal != policy['pixel_calibrations'][key]: raise ValueError('Pixel calibration changed')
                with np.load(d/'source_components.npz') as p:
                    cases = {i: [] for i in policy['source_groups']}
                    for rec in policy['source_records']:
                        i = rec['group']; pos = rec['position']; key = rec['key']; labels = p[key+'_labels']
                        areas = np.bincount(labels.ravel().astype(np.int64))[1:]
                        pred = filt.PredictionComponents(labels, p[key+'_features'], p[key+'_anchor'], areas)
                        cases[i].append(filt.observed(pred, data[i]['labels'][pos], data[i]['components'][pos], i, pos))
                    pool = filt.CalibrationPool(cases, g); baseline = screen.full_calibration_metrics(pool, np.ones(len(pool.area), bool))
                    calibrated = screen.calibrate(pool, screen.score_cases(cases, None, 'area_guard'), True, baseline, protocol['source_calibration'])
                    if calibrated != policy['area_policy']: raise ValueError('Area calibration changed')
                with np.load(d/'target_grids.npz') as grids:
                    if not np.array_equal(grids['original_target_indices'], data[g]['indices']): raise ValueError('Target ordering changed')
                    if score_target(grids, data[g], manifest['regions'][g]['gsd_m'], policy) != r['methods']: raise ValueError('Target metrics changed')
                    model = load_model(d); fresh = native.infer_grids(model, data[g]['views'][:3])
                    for k in fresh:
                        if not np.allclose(fresh[k], grids[k][:3], atol=2e-6, rtol=0): raise ValueError('Forward replay differs')
                    del model; torch.cuda.empty_cache()
                checked += 1; print('RGB_NATIVE_VERIFIED', arm, g, seed, flush=True)
    mine = base.read(out/'mines/results.json'); cases = {r['mine']: r for r in protocol['mine_cases']}
    for r in mine['research']:
        _, truth, normal, valid, gsd = old_rgb.load_mine(cases[r['case']]); path = Path(r['native_cache_path'])
        if base.sha(path) != r['native_cache_sha256']: raise ValueError('Mine cache changed')
        policy = base.read(folder(out, r['arm'], protocol['mine_primary_fold'], r['seed'])/'rgb_policy.json')
        with np.load(path) as grids: masks = native.scene_masks(grids, truth.shape, gsd, policy)
        scores = {k: rgb.partial_reference_counts(v, truth, normal, valid) for k, v in masks.items()}
        if scores != r['methods']: raise ValueError('Mine metrics differ')
    old_rgb.check_assets(protocol)
    base.write(out/'verification.json', {'verified_at': base.now(), 'status': 'passed', 'source_policies_rebuilt': checked,
                                        'target_results_rebuilt': checked, 'forward_replays': checked*3,
                                        'mine_results_rebuilt': len(mine['research']), 'protected_assets_unchanged': True})
    state(out, status='completed', stage='verified', fits=12, total_updates=57600, verification_status='passed')


def cleanup_cache(out):
    # Only named, regenerable artifacts inside THIS experiment are removed.
    root = Path(out).resolve(); cache = (root/'training_cache').resolve()
    if not root.is_relative_to(ROOT/'experiments') or not cache.is_relative_to(root):
        raise ValueError('Invalid cleanup target')
    for name in ('views.npy', 'labels.npy', 'complete.json'):
        target = (cache/name).resolve()
        if target.parent != cache: raise ValueError('Cache escaped experiment')
        target.unlink(missing_ok=True)
    if cache.exists() and not any(cache.iterdir()): cache.rmdir()


def run(out, protocol):
    import torch
    parent.setup_runtime(); out = Path(out)
    if (out/'results.json').exists(): verify(out); return
    state(out, status='running', stage='rgb_data_audit')
    x, y, groups, manifest = prepare_data(out); data = load_validation(manifest)
    # GPU cache avoids exhausting the currently small available host RAM.
    gpu_x = torch.as_tensor(np.asarray(x), device='cuda'); gpu_y = torch.as_tensor(np.asarray(y), device='cuda')
    del x, y
    for arm in protocol['arms']:
        for g in protocol['held_indices']:
            for seed in protocol['seeds']: train_fit(out, arm, g, seed, gpu_x, gpu_y, groups, data, protocol)
        for g in protocol['held_indices']:
            for seed in protocol['seeds']: evaluate_fit(out, arm, g, seed, data, manifest, protocol)
    del gpu_x, gpu_y, data; torch.cuda.empty_cache(); cleanup_cache(out)
    run_mines(out, protocol); summarize(out, protocol); verify(out)


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--output', required=True)
    parser.add_argument('--stage', choices=('run', 'register', 'prepare', 'verify'), default='run'); args = parser.parse_args()
    out = Path(args.output).resolve(); protocol = register(out)
    try:
        if args.stage == 'register': return
        if args.stage == 'prepare': prepare_data(out); return
        if args.stage == 'verify': parent.setup_runtime(); verify(out); return
        run(out, protocol)
    except BaseException as error:
        state(out, status='interrupted_or_failed', error_type=type(error).__name__, error=str(error), resumable=True)
        raise


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8'); main()
