"""Bounded decoder-only resolution adaptation, with matched source controls."""
from pathlib import Path
import sys
import hashlib
import numpy as np
import cv2
sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parent)]
from runtime_paths import configure_runtime_paths
configure_runtime_paths()
import rgb_resolution_study as detail
import rgb_transfer_study as transfer
import rgb_transfer_regions as regions
import rgb_transfer_model as architecture
import rgb_resolution as resolution
exp = transfer.exp; base = exp.base
SOURCE = exp.ROOT/'experiments/rgb_transfer_20261020'
GEOGRAPHIC = exp.ROOT/'experiments/rgb_transfer_regions_20261020'
ARMS = ('decoder64', 'decoder128')


def initial_path(held, seed):
    return (SOURCE/'imagenet_normalized'/f'seed{seed}'/'source_selected.pth') if held is None else exp.folder(GEOGRAPHIC, regions.ARM, held, seed)/'source_selected.pth'


def directory(out, held, arm, seed):
    d = Path(out)/('all_sources' if held is None else f'fold{held}')/arm/f'seed{seed}'; d.mkdir(parents=True, exist_ok=True); return d


def register(out):
    out = Path(out).resolve()
    if out.drive.upper() != 'D:' or not out.is_relative_to(exp.ROOT/'experiments'): raise ValueError('D experiments only')
    out.mkdir(parents=True, exist_ok=True)
    if (out/'protocol.json').exists():
        p = base.read(out/'protocol.json'); exp.old_rgb.check_assets(p); return p
    old = base.read(GEOGRAPHIC/'protocol.json'); frozen = dict(old['frozen_assets'])
    paths = [Path(__file__), exp.ROOT/'tests/test_rgb_detail_adaptation.py', exp.ROOT/'rgb_resolution.py', exp.ROOT/'rgb_transfer_model.py',
             exp.ROOT/'validation/rgb_resolution_study.py', exp.ROOT/'validation/rgb_final_training.py',
             SOURCE/'results.json', SOURCE/'verification.json', GEOGRAPHIC/'results.json', GEOGRAPHIC/'verification.json',
             exp.ROOT/'experiments/rgb_resolution_20261020/protocol.json', exp.ROOT/'experiments/rgb_resolution_20261020/results.json',
             exp.ROOT/'experiments/rgb_resolution_20261020/verification.json']
    for seed in old['seeds']:
        paths += [initial_path(None, seed), SOURCE/'imagenet_normalized'/f'seed{seed}'/'full4800_source.json']
        for g in old['held_indices']: paths += [initial_path(g, seed), exp.folder(GEOGRAPHIC, regions.ARM, g, seed)/'training.json', exp.folder(GEOGRAPHIC, regions.ARM, g, seed)/'rgb_policy.json']
    frozen.update({str(a): base.sha(a) for a in paths})
    p = {'registered_at': base.now(), 'frozen_assets': frozen, 'seeds': old['seeds'], 'held_indices': old['held_indices'], 'arms': list(ARMS),
         'hypothesis': 'Adapt a frozen stronger RGB representation to a spatially richer feature grid. Direct128 inference improved object IoU but lost F1; decoder supervision may resolve the mismatch.',
         'updates': 600, 'resume_every': 300, 'selection': 'Fixed final600 updates; no checkpoint/target selection.',
         'training': 'Both arms start from identical source-selected normalized pretrained model per seed/fold. Freeze entire encoder; optimize decoder/head only. Keep all BN running statistics fixed in BOTH arms. Parent/augmentation traces identical.',
         'optimizer': 'Fresh AdamW decoder/head lr.0001 wd.0001 cosine600 to.00001; BCE+.5Dice with fractional occupancy. FP32 deterministic/noTF32.',
         'sampling': 'Three parents per eligible source/update. Split the shuffled batch into exactly3 equal microbatches, accumulate arithmetic mean of losses before one optimizer update. Same BN freeze and microbatch schedule in both arms.',
         'resolution': 'decoder64 trains64 and evaluates TTA5; decoder128 trains128 and evaluates one identity view. All parents have per-band minmax before resize; constant bands zero. Larger interpolation does not create sensor information.',
         'budget': 'Matched updates/parent examples, not matched training FLOPs:128 costs roughly4x spatial convolution work. Inference input-pixel budget128^2 is .8*(5*64^2), not an exact latency guarantee.',
         'source_gate': {'min_macro_small_cover_gain': .02, 'min_macro_small_iou_gain': .02, 'max_macro_f1_loss': .005, 'max_region_f1_loss': .02, 'max_macro_fpr_increase': .002, 'max_region_fpr_increase': .01},
         'original_guard': '128 candidate must additionally retain frozen original64 TTA source macroF1 within.005 and each source within.02, at separately source-calibrated2% background FPR.',
         'automatic_order': 'Run four600-update all-seven-source pilots. If BOTH seeds pass all gates, adapt both arms in all3 historical geographic folds/two seeds, excluding target and Rio. Calibrate/screen source only. Only if all6 regional policies pass may held-region predictions run. Otherwise stop this resolution route without new target predictions.',
         'scope': 'Historical development data, no new blind set. All-source pilots include Rio as explicit source; geographic continuation uses only five of original six regions per fold to match geographic controls.',
         'retention': 'At most16 final decoder/head deltas, not duplicate full models; delete optimizer resumes and regenerable D-only cache. Retain sparse masks, source policies and necessary replay evidence.',
         'production_changed': False, 'official_test_predictions': 0, 'reserved_predictions': 0, 'mine_predictions': 0, 'external_model_calls': 0}
    exp.old_rgb.check_assets(p); base.write(out/'protocol.json', p); return p


def draw(groups, sources, rng):
    ids = np.concatenate([rng.choice(np.flatnonzero(groups == g), 3, replace=True) for g in sources]); rng.shuffle(ids)
    return ids, int(rng.integers(4)), bool(rng.integers(2))


def freeze_representation(model):
    model.eval(); model.encoder.requires_grad_(False)
    model.decoder.requires_grad_(True); model.segmentation_head.requires_grad_(True)
    return [v for v in model.parameters() if v.requires_grad]


def encoder_hash(model):
    h = hashlib.sha256()
    for k, v in model.encoder.state_dict().items(): h.update(k.encode()); h.update(v.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def delta(model):
    return {k: v.detach().cpu() for k, v in model.state_dict().items() if not k.startswith('encoder.')}


def load(out, held, arm, seed):
    import torch
    d = directory(out, held, arm, seed); r = base.read(d/'training.json')
    if base.sha(d/'decoder.pth') != r['decoder_sha256'] or base.sha(initial_path(held, seed)) != r['initial_weight_sha256']: raise ValueError('Adapted state provenance changed')
    model = architecture.RGBTransferModel(); model.load_state_dict(torch.load(initial_path(held, seed), map_location='cpu', weights_only=True))
    state = model.state_dict(); update = torch.load(d/'decoder.pth', map_location='cpu', weights_only=True)
    if set(update) != {k for k in state if not k.startswith('encoder.')}: raise ValueError('Wrong decoder delta keys')
    state.update(update); model.load_state_dict(state); model.cuda().eval()
    if exp.parent.short.state_hash(model) != r['final_state_sha256'] or encoder_hash(model) != r['encoder_sha256']: raise ValueError('Adapted model state replay failed')
    return model


def prepare(out):
    x64, y64, groups, data, manifest = transfer.final.prepare(out)
    cache = Path(out)/'training_cache'; xp, yp = cache/'detail_views.npy', cache/'detail_labels.npy'
    x128 = np.lib.format.open_memmap(xp, mode='w+', dtype=np.float16, shape=(len(groups), 1, 3, 128, 128))
    y128 = np.lib.format.open_memmap(yp, mode='w+', dtype=np.float16, shape=(len(groups), 1, 1, 128, 128))
    pools = {g: (np.load(r['splits']['train']['image_path'], mmap_mode='r'), np.load(r['splits']['train']['label_path'], mmap_mode='r')) for g, r in enumerate(manifest['regions'])}
    for j, r in enumerate(manifest['kept_train']):
        raw, labels = pools[r['group']]; image, truth = raw[r['index'], :3], labels[r['index'], 0]
        x128[j, 0] = resolution.whole_view(image)
        inter = cv2.INTER_AREA if len(truth) > 128 else cv2.INTER_LINEAR
        y128[j, 0, 0] = cv2.resize(truth.astype(np.float32), (128, 128), interpolation=inter)
    x128.flush(); y128.flush(); return {ARMS[0]: (x64, y64), ARMS[1]: (x128, y128)}, groups, data, manifest


def train(out, held, arm, seed, x, y, groups, sources, p):
    import torch
    import torch.nn.functional as F
    d = directory(out, held, arm, seed)
    if (d/'training.json').exists(): return
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    model = architecture.RGBTransferModel(); model.load_state_dict(torch.load(initial_path(held, seed), map_location='cpu', weights_only=True)); model.cuda()
    before = encoder_hash(model); opt = torch.optim.AdamW(freeze_representation(model), lr=.0001, weight_decay=.0001)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, p['updates'], eta_min=.00001)
    rng = np.random.default_rng(20261020+seed+(0 if held is None else 100*held)); start = 0; trace = bytearray(); visits = np.zeros(7, np.int64); losses = []
    resume = d/'resume_training.pth'
    if resume.exists():
        cp = torch.load(resume, map_location='cpu', weights_only=True)
        if cp['protocol_sha256'] != base.sha(Path(out)/'protocol.json'): raise ValueError('Adaptation resume protocol changed')
        state = model.state_dict(); state.update(cp['decoder']); model.load_state_dict(state)
        opt.load_state_dict(cp['optimizer']); sched.load_state_dict(cp['scheduler']); rng.bit_generator.state = cp['rng']
        start, trace, visits, losses = cp['step'], bytearray(cp['trace']), np.array(cp['visits']), cp['losses']
    n = len(sources)
    for step in range(start, p['updates']):
        ids, k, flip = draw(groups, sources, rng); visits += np.bincount(groups[ids], minlength=7); trace.extend(np.asarray([*ids, k, int(flip)], np.int64).tobytes()); opt.zero_grad(set_to_none=True)
        total = 0.
        for chunk in range(3):
            pos = ids[chunk*n:(chunk+1)*n]; a, b = x[pos, 0].float().rot90(k, (-2, -1)), y[pos, 0].float().rot90(k, (-2, -1))
            if flip: a, b = a.flip(-1), b.flip(-1)
            z = model(a); prob = z.sigmoid(); dice = (1-(2*(prob*b).sum((2, 3))+1e-6)/(prob.sum((2, 3))+b.sum((2, 3))+1e-6)).mean()
            loss = F.binary_cross_entropy_with_logits(z, b)+.5*dice
            if not torch.isfinite(loss): raise ValueError('Nonfinite detail adaptation loss')
            (loss/3).backward(); total += float(loss.detach())/3
        opt.step(); sched.step(); losses.append(total); u = step+1
        if u % 100 == 0:
            exp.state(out, status='running', stage='decoder_resolution_adaptation', held=held, arm=arm, seed=seed, update=u, target=p['updates'])
            print('DETAIL_ADAPT', held, arm, seed, u, float(np.mean(losses[-100:])), flush=True)
        if u % p['resume_every'] == 0:
            exp.parent.short.save_torch(resume, {'protocol_sha256': base.sha(Path(out)/'protocol.json'), 'decoder': delta(model), 'optimizer': opt.state_dict(), 'scheduler': sched.state_dict(), 'rng': rng.bit_generator.state,
                'step': u, 'trace': bytes(trace), 'visits': visits.tolist(), 'losses': losses})
    if encoder_hash(model) != before or any(a.grad is not None for a in model.encoder.parameters()): raise ValueError('Frozen representation changed')
    if visits.tolist() != [p['updates']*3 if g in sources else 0 for g in range(7)]: raise ValueError('Adaptation source balancing/leakage failure')
    exp.parent.short.save_torch(d/'decoder.pth', delta(model))
    base.write(d/'training.json', {'completed_at': base.now(), 'held': held, 'arm': arm, 'seed': seed, 'updates': p['updates'], 'sources': sources,
        'initial_weight_sha256': base.sha(initial_path(held, seed)), 'encoder_sha256': before, 'decoder_sha256': base.sha(d/'decoder.pth'), 'final_state_sha256': exp.parent.short.state_hash(model),
        'parent_augmentation_sha256': hashlib.sha256(trace).hexdigest(), 'visits': visits.tolist(), 'loss_last100': float(np.mean(losses[-100:])), 'all_bn_statistics_frozen': True})
    resume.unlink(missing_ok=True); del model, opt, sched; torch.cuda.empty_cache()


def probabilities(model, arm, data, manifest, g):
    if arm == ARMS[0]: return exp.parent.short.native_probabilities(transfer.runtime.predict_tta(model, data[g]['views']), data[g]['labels'])
    split = manifest['regions'][g]['splits']['val']; raw = np.load(split['image_path'], mmap_mode='r')
    views = np.stack([resolution.whole_view(raw[i, :3]) for i in split['indices']]); pred = resolution.predict(model, views)
    return exp.parent.short.native_probabilities(pred, data[g]['labels'])


def calibrate(out, held, arm, seed, data, manifest, sources):
    import torch
    d = directory(out, held, arm, seed)
    if (d/'source.json').exists(): return base.read(d/'source.json')
    model = load(out, held, arm, seed); hist = {}; temp = d/'temporary'; temp.mkdir(exist_ok=True)
    for g in sources:
        probs = probabilities(model, arm, data, manifest, g); pos = exp.parent.gate.meta_split(len(data[g]['labels']), g)[1]
        hist[g] = exp.parent.source_histogram([probs[i] for i in pos], [data[g]['labels'][i] for i in pos]); np.savez_compressed(temp/f'g{g}.npz', probability=np.stack(probs))
    q, cal = transfer.final.threshold(hist) if held is None else exp.parent.threshold_from_histograms(hist, held, .02)
    np.savez_compressed(d/'source_histograms.npz', **{f'g{g}': h for g, h in hist.items()}); masks = {}; scores = {'selection': {}, 'calibration': {}}
    for g in sources:
        path = temp/f'g{g}.npz'
        with np.load(path) as z: probs = z['probability']
        components = [base.gt_components(y, manifest['regions'][g]['gsd_m']) for y in data[g]['labels']]
        for i, a in enumerate(probs): masks[f'g{g}_p{i}'] = np.packbits((a.astype(np.float64) >= q).ravel())
        for split, ids in zip(('selection', 'calibration'), exp.parent.gate.meta_split(len(data[g]['labels']), g)):
            scores[split][str(g)] = detail.metrics([probs[i] for i in ids], [data[g]['labels'][i] for i in ids], [components[i] for i in ids], q)
        path.unlink()
    temp.rmdir(); np.savez_compressed(d/'source_masks.npz', **masks)
    r = {'frozen_at': base.now(), 'held': held, 'seed': seed, 'arm': arm, 'threshold': q, 'source_calibration': cal, 'sources': sources, **scores,
         'source_masks_sha256': base.sha(d/'source_masks.npz'), 'source_histograms_sha256': base.sha(d/'source_histograms.npz'), 'training_sha256': base.sha(d/'training.json'), 'target_used': False}
    base.write(d/'source.json', r); del model; torch.cuda.empty_cache(); return r


def policy(out, held, seed, data, manifest, sources, p):
    import torch
    records = [calibrate(out, held, a, seed, data, manifest, sources) for a in ARMS]
    d = directory(out, held, ARMS[1], seed); a, b = records
    t = [base.read(directory(out, held, arm, seed)/'training.json') for arm in ARMS]
    for key in ('parent_augmentation_sha256', 'visits', 'initial_weight_sha256', 'encoder_sha256'):
        if t[0][key] != t[1][key]: raise ValueError('Resolution adaptation is not paired')
    gate = detail.source_gate(a['selection'], b['selection'], p['source_gate'])
    if held is None: original = base.read(SOURCE/'imagenet_normalized'/f'seed{seed}'/'full4800_source.json')['methods']['tta5']['selection']
    else:
        model = architecture.RGBTransferModel(); model.load_state_dict(torch.load(initial_path(held, seed), map_location='cpu', weights_only=True)); model.cuda().eval()
        q = base.read(exp.folder(GEOGRAPHIC, regions.ARM, held, seed)/'rgb_policy.json')['tta_threshold']; original = {}
        for g in sources:
            pos = exp.parent.gate.meta_split(len(data[g]['labels']), g)[0]; labels = [data[g]['labels'][i] for i in pos]
            pred = exp.parent.short.native_probabilities(transfer.runtime.predict_tta(model, data[g]['views'][pos]), labels)
            original[str(g)] = detail.metrics(pred, labels, [base.gt_components(y, manifest['regions'][g]['gsd_m']) for y in labels], q)
        del model; torch.cuda.empty_cache()
    extra = {'original_macro_f1_retained': detail.macro(b['selection'])['f1'] >= detail.macro(original)['f1']-.005,
        'original_each_source_f1_retained': max(original[g]['f1']-b['selection'][g]['f1'] for g in original) <= .02}
    r = {'decided_at': base.now(), 'held': held, 'seed': seed, 'paired_gate': gate, 'original_checks': extra, 'passed': gate['passed'] and all(extra.values()), 'source_only': True}
    base.write(d.parent.parent/f'policy_seed{seed}.json', r); print('DETAIL_ADAPT_POLICY', held, seed, r, flush=True); return r


def verify_source(out, held, arm, seed, data, manifest):
    import torch
    d = directory(out, held, arm, seed); r = base.read(d/'source.json')
    if base.sha(d/'source_masks.npz') != r['source_masks_sha256'] or base.sha(d/'source_histograms.npz') != r['source_histograms_sha256'] or base.sha(d/'training.json') != r['training_sha256']: raise ValueError('Adaptation source proof changed')
    with np.load(d/'source_histograms.npz') as h, np.load(d/'source_masks.npz') as z:
        hist = {g: h[f'g{g}'] for g in r['sources']}; q, cal = transfer.final.threshold(hist) if held is None else exp.parent.threshold_from_histograms(hist, held, .02)
        if q != r['threshold'] or cal != r['source_calibration']: raise ValueError('Adaptation calibration replay failed')
        for g in r['sources']:
            components = [base.gt_components(y, manifest['regions'][g]['gsd_m']) for y in data[g]['labels']]
            for split, ids in zip(('selection', 'calibration'), exp.parent.gate.meta_split(len(data[g]['labels']), g)):
                masks = [transfer.unpack_mask(z[f'g{g}_p{i}'], data[g]['labels'][i].shape) for i in ids]
                if detail.metrics(masks, [data[g]['labels'][i] for i in ids], [components[i] for i in ids], .5) != r[split][str(g)]: raise ValueError('Adaptation source mask replay failed')
    model = load(out, held, arm, seed); del model; torch.cuda.empty_cache()


def target(out, held, arm, seed, data, manifest):
    import torch
    d = directory(out, held, arm, seed); policy_record = base.read(d/'source.json'); model = load(out, held, arm, seed)
    probs = probabilities(model, arm, data, manifest, held); truth = data[held]['labels']; components = [base.gt_components(y, manifest['regions'][held]['gsd_m']) for y in truth]
    scores = detail.metrics(probs, truth, components, policy_record['threshold'])
    np.savez_compressed(d/'target_masks.npz', **{f'p{i}': np.packbits((a.astype(np.float64) >= policy_record['threshold']).ravel()) for i, a in enumerate(probs)})
    with np.load(d/'target_masks.npz') as z:
        masks = [transfer.unpack_mask(z[f'p{i}'], y.shape) for i, y in enumerate(truth)]
        if detail.metrics(masks, truth, components, .5) != scores: raise ValueError('Adaptation target mask replay failed')
    r = {'completed_at': base.now(), 'held': held, 'arm': arm, 'seed': seed, 'scores': scores, 'source_policy_sha256': base.sha(d/'source.json'), 'target_masks_sha256': base.sha(d/'target_masks.npz')}
    base.write(d/'target.json', r); del model; torch.cuda.empty_cache(); return r


def run(out):
    import torch
    out = Path(out); p = register(out); exp.parent.setup_runtime(); exp.state(out, status='running', stage='detail_training_cache')
    arrays, groups, data, manifest = prepare(out)
    gpu = {a: (torch.as_tensor(np.asarray(x), device='cuda'), torch.as_tensor(np.asarray(y), device='cuda')) for a, (x, y) in arrays.items()}; del arrays
    completed = []; pilots = []
    def stage(held):
        sources = list(range(7)) if held is None else [g for g in range(6) if g != held]
        records = []
        for seed in p['seeds']:
            for arm in ARMS:
                train(out, held, arm, seed, *gpu[arm], groups, sources, p); completed.append((held, arm, seed))
            records.append(policy(out, held, seed, data, manifest, sources, p))
        return records
    pilots = stage(None); passed = all(r['passed'] for r in pilots); base.write(out/'pilot_decision.json', {'decided_at': base.now(), 'seeds': pilots, 'continue_to_regions': passed})
    regional = []; targets = []
    if passed:
        for held in p['held_indices']: regional.extend(stage(held))
    del gpu; torch.cuda.empty_cache()
    cache = (out/'training_cache').resolve()
    if not cache.is_relative_to(out.resolve()): raise ValueError('Unsafe cache cleanup target')
    for name in ('detail_views.npy', 'detail_labels.npy'): (cache/name).unlink(missing_ok=True)
    exp.cleanup_cache(out)
    for held, arm, seed in completed: verify_source(out, held, arm, seed, data, manifest)
    if regional and all(r['passed'] for r in regional):
        for held in p['held_indices']:
            for seed in p['seeds']:
                for arm in ARMS: targets.append(target(out, held, arm, seed, data, manifest))
    exp.old_rgb.check_assets(p)
    result = {'completed_at': base.now(), 'pilot_passed_both_seeds': passed, 'pilots': pilots, 'regional_source_policies': regional,
        'fits': len(completed), 'total_updates': len(completed)*p['updates'], 'target_fits_predicted': len(targets), 'targets': targets,
        'production_changed': False, 'official_test_predictions': 0, 'reserved_predictions': 0, 'mine_predictions': 0}
    base.write(out/'results.json', result); base.write(out/'verification.json', {'verified_at': base.now(), 'status': 'passed', 'source_fits_rebuilt': len(completed), 'target_scores_rebuilt': len(targets), 'protected_assets_unchanged': True})
    exp.state(out, status='completed', stage='decoder_detail_adaptation_verified', pilot_passed=passed, target_fits=len(targets))


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser(); p.add_argument('--output', required=True); args = p.parse_args()
    try: run(Path(args.output))
    except BaseException as e: exp.state(Path(args.output), status='interrupted_or_failed', error=str(e)); raise
