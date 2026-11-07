"""Source-gated paired ImageNet initialization study, with automatic continuation."""
from pathlib import Path
import sys
import hashlib
import time
import numpy as np
sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parent)]
from runtime_paths import configure_runtime_paths
configure_runtime_paths()
import rgb_final_training as final
import rgb_transfer_model as architecture
import rgb_native_detector as runtime
exp = final.exp; base = exp.base
PRIOR = exp.ROOT/'experiments/rgb_final_all_regions_20261020'
WEIGHTS = exp.ROOT/'cache/pretrained/resnet34-b627a593.pth'
WEIGHT_SHA = 'b627a593bcbe140c234610266fe4f8ae95ea42fc881d091c9b6052e6b1d0590f'
ARMS = ('scratch_normalized', 'imagenet_normalized')


def register(out):
    out = Path(out).resolve()
    if out.drive.upper() != 'D:' or not out.is_relative_to(exp.ROOT/'experiments'): raise ValueError('Use D experiments')
    out.mkdir(parents=True, exist_ok=True)
    if (out/'protocol.json').exists():
        p = base.read(out/'protocol.json'); exp.old_rgb.check_assets(p); return p
    if base.sha(WEIGHTS) != WEIGHT_SHA: raise ValueError('Official pretrained weight identity mismatch')
    original = base.read(PRIOR/'protocol.json')
    paths = [Path(__file__), exp.ROOT/'rgb_transfer_model.py', exp.ROOT/'tests/test_rgb_transfer.py', WEIGHTS,
             exp.ROOT/'validation/rgb_final_training.py', exp.ROOT/'rgb_native_detector.py',
             PRIOR/'protocol.json', PRIOR/'data_manifest.json', PRIOR/'results.json', PRIOR/'verification.json',
             exp.ROOT/'experiments/mine_rgb_embedding_20261020/results.json',
             exp.ROOT/'experiments/mine_rgb_embedding_20261020/verification.json',
             exp.ROOT/'model_registry.json', exp.ROOT/'datasets/input_reserve_registry.json',
             exp.ROOT/'datasets/mine_benchmark_20261002/mine_case_registry.json']
    paths += [Path(c[k+'_path']) for c in original['mine_cases'] for k in ('image', 'label', 'normal')]
    protected = dict(original['frozen_assets'])
    protected.update(base.read(exp.ROOT/'experiments/rgb_native_20261020/protocol.json')['frozen_assets'])
    protected.update({str(a): base.sha(a) for a in paths})
    protected[str(exp.ROOT/'experiments/rgb_native_20261020/mines/results.json')] = base.sha(exp.ROOT/'experiments/rgb_native_20261020/mines/results.json')
    p = {'registered_at': base.now(), 'frozen_assets': protected,
         'question': 'Does public encoder pretraining improve RGB landslide capability at matched data/normalization/decoder initialization/optimizer/budget, and do new representations improve the operation/accident tradeoff?',
         'arms': list(ARMS), 'seeds': [42, 43], 'pilot_updates': 1200, 'max_updates': 4800, 'checkpoints': [1200, 2400, 3600, 4800],
         'pretrained': {'source': 'https://download.pytorch.org/models/resnet34-b627a593.pth', 'sha256': WEIGHT_SHA,
                        'documentation': 'https://docs.pytorch.org/vision/main/models/generated/torchvision.models.resnet34.html',
                        'local_file': str(WEIGHTS), 'downloaded_on_D_only': True, 'project_images_labels_code_or_secrets_uploaded': False},
         'single_training_difference': 'Encoder initial state: official ImageNet1K_V1 vs random. Exact same random decoder/head tensors, BGR parents, ImageNet RGB normalization, parent/rotation/flip schedules, losses and optimizer in both arms.',
         'normalization': 'Native BGR parent minmax01 then64. In BOTH encoders reorderBGR->RGB and normalize mean[.485,.456,.406]/std[.229,.224,.225]. Keep dense64 views, not ImageNet classification resize256/center crop224.',
         'architecture': 'SMP0.5 Unet++/ResNet34, three input channels, same decoder. Pretraining is not a new/deeper architecture or claimed original algorithm.',
         'sampling': 'Same seven-source train pool as previous final study, batch21 with3/source, whole-parent view only, paired RNG and initialization seed42/43.',
         'optimizer': 'Both arms AdamW lr.001 wd.0001; cosine1200 to1e-5, then retain moments/restart.0003 and cosine3600 to1e-5. FP32 deterministic/noTF32.',
         'selection': 'At each registered checkpoint maximize source selection-half TTA5 native F1@.5; ties earliest. Other halves calibrate each method to2% equal-source background FPR.',
         'pilot_admission': {'min_pretrained_source_f1_gain_at_calibrated2pct': .02, 'max_any_source_region_f1_loss': .03, 'both_seeds_required': True},
         'automatic_branch': 'Complete matched1200-step pilots for both arms/seeds, source-calibrate/verify, and only if both seeds pass source admission continue ALL4 fits to4800 with optimizer/RNG resume. No mine prediction before source admission; otherwise stop the initialization route without target tuning.',
         'inference_methods': ['tta5', 'mean_multiscale5'],
         'method_selection': 'Prefer TTA5 unless source selection multiscale gains>=2pp coverage AND strictIoU while macroF1 loss<=.005, each-regionF1 loss<=.02, macroFPR increase<=.002. Same5 per-parent CNN-view budget for either method; shared benchmarking computes9 views.',
         'mine_cases': original['mine_cases'], 'mine_limits': original['mine_limits'],
         'scope': 'All seven LMHLD regions are SOURCE training/selection/calibration. Four mines are repeatedly observed development only. No new blind/geographic/complete-boundary or professional trial claim.',
         'retention': 'At most4 current source-best states, source probability histograms/packed masks and necessary scene predictions. Intermediate1200 snapshots are reproducible from the fixed recipe, not permanently duplicated. Delete optimizer resumes and D-only train/view temporary caches on completion.',
         'official_test_predictions': 0, 'reserved_predictions': 0, 'production_changed': False,
         'external_model_calls': 0, 'public_weight_download_bytes': WEIGHTS.stat().st_size}
    exp.old_rgb.check_assets(p); base.write(out/'protocol.json', p); return p


def directory(out, arm, seed):
    d = Path(out)/arm/f'seed{seed}'; d.mkdir(parents=True, exist_ok=True); return d


def make_model(seed, arm):
    import torch
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    model = architecture.RGBTransferModel(WEIGHTS if arm == 'imagenet_normalized' else None)
    return model.cuda()


def train(out, arm, seed, x, y, groups, data, target_updates, p):
    import torch
    import torch.nn.functional as F
    d = directory(out, arm, seed); status_path = d/'training_status.json'
    if status_path.exists() and base.read(status_path)['completed_updates'] >= target_updates: return
    model = make_model(seed, arm); init_decoder = architecture.decoder_hash(model)
    init_state = exp.parent.short.state_hash(model)
    opt = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.0001)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, 1200, eta_min=.00001)
    rng = np.random.default_rng(seed); curve = []; losses = []; seen = np.zeros(len(groups), bool); visits = np.zeros(7, np.int64)
    trace = bytearray(); step0 = 0; phase = 'short'; resume = d/'resume_training.pth'
    if resume.exists():
        cp = torch.load(resume, map_location='cpu', weights_only=True)
        allowed = {base.sha(Path(out)/'protocol.json')}
        if 'implementation_amendment' in p: allowed.add(p['implementation_amendment']['initial_protocol_sha256'])
        if cp['protocol_sha256'] not in allowed: raise ValueError('Transfer resume protocol changed')
        model.load_state_dict(cp['model']); phase = cp['phase']
        if phase == 'continuation': scheduler = exp.parent.short.start_continuation(opt)
        opt.load_state_dict(cp['optimizer']); scheduler.load_state_dict(cp['scheduler']); rng.bit_generator.state = cp['rng']
        curve, losses, step0 = cp['curve'], cp['losses'], cp['step']; seen = np.array(cp['seen'], bool); visits = np.array(cp['visits']); trace = bytearray(cp['trace'])
        torch.set_rng_state(cp['torch_rng']); torch.cuda.set_rng_state_all(cp['cuda_rng'])
    model.train()
    for step in range(step0, target_updates):
        ids, k, flip = final.sample(groups, rng); seen[ids] = True; visits += np.bincount(groups[ids], minlength=7)
        trace.extend(np.asarray([*ids, k, int(flip)], np.int64).tobytes())
        a, b = x[ids, 0].float().rot90(k, (-2, -1)), y[ids, 0].float().rot90(k, (-2, -1))
        if flip: a, b = a.flip(-1), b.flip(-1)
        opt.zero_grad(set_to_none=True); logits = model(a); prob = logits.sigmoid()
        dice = (1-(2*(prob*b).sum((2, 3))+1e-6)/(prob.sum((2, 3))+b.sum((2, 3))+1e-6)).mean()
        loss = F.binary_cross_entropy_with_logits(logits, b)+.5*dice
        if not torch.isfinite(loss): raise ValueError('Nonfinite transfer loss')
        loss.backward(); opt.step(); scheduler.step(); losses.append(float(loss.detach())); update = step+1
        if update % 300 == 0:
            exp.state(out, status='running', stage='transfer_training', arm=arm, seed=seed, update=update, target=target_updates)
            print('TRANSFER_TRAIN', arm, seed, update, float(np.mean(losses[-100:])), flush=True)
        if update in p['checkpoints']:
            model.eval(); score = final.source_score(model, data, 0)
            curve.append({'update': update, 'source_selection': score, 'model_state_sha256': exp.parent.short.state_hash(model)})
            if exp.parent.short.choose_source_checkpoint(curve)['update'] == update:
                exp.parent.short.save_torch(d/'source_selected.pth', {k: v.detach().cpu() for k, v in model.state_dict().items()})
            base.write(d/'source_curve.json', curve); print('TRANSFER_SOURCE', arm, seed, update, score['macro_f1'], flush=True)
            if update == 1200: scheduler = exp.parent.short.start_continuation(opt); phase = 'continuation'
            model.train()
        if update % 600 == 0 or update == target_updates:
            exp.parent.short.save_torch(resume, {'protocol_sha256': base.sha(Path(out)/'protocol.json'), 'model': model.state_dict(), 'optimizer': opt.state_dict(), 'scheduler': scheduler.state_dict(),
                'phase': phase, 'rng': rng.bit_generator.state, 'curve': curve, 'losses': losses, 'seen': seen.tolist(), 'visits': visits.tolist(), 'trace': bytes(trace), 'step': update,
                'torch_rng': torch.get_rng_state(), 'cuda_rng': torch.cuda.get_rng_state_all()})
    best = exp.parent.short.choose_source_checkpoint(curve)
    if visits.tolist() != [target_updates*3]*7: raise ValueError('Transfer source balance failed')
    status = {'completed_updates': target_updates, 'seed': seed, 'arm': arm, 'initial_state_sha256': init_state, 'initial_decoder_sha256': init_decoder,
              'parent_augmentation_sha256': hashlib.sha256(trace).hexdigest(), 'source_visits': visits.tolist(),
              'source_unique_seen': [int(seen[groups == g].sum()) for g in range(7)], 'source_available': [int((groups == g).sum()) for g in range(7)],
              'selected_update': best['update'], 'selected_source_f1_at_half': best['source_selection']['macro_f1'], 'weight_sha256': base.sha(d/'source_selected.pth'), 'selected_state_sha256': best['model_state_sha256']}
    base.write(status_path, status); del model, opt, scheduler; torch.cuda.empty_cache()


def macro(rows):
    metrics = ('f1', 'small_cover50', 'small_iou50', 'background_fpr', 'false_components_per_mpix')
    return {k: float(np.mean([r[k] for r in rows.values() if r[k] is not None])) for k in metrics}


def add_source_statistics(result, thresholds, calibrations):
    for method, record in result.items():
        record['threshold'] = thresholds[method]
        # Keep calibration-half outcome metrics separate from threshold metadata.
        record['source_calibration'] = calibrations[method]
        for split in ('selection', 'calibration'): record[split+'_macro'] = macro(record[split])


def source_phase(out, arm, seed, data, manifest, phase):
    import torch
    d = directory(out, arm, seed); dest = d/f'{phase}_source.json'
    if dest.exists(): return base.read(dest)
    training = base.read(d/'training_status.json'); model = architecture.RGBTransferModel().cuda()
    model.load_state_dict(torch.load(d/'source_selected.pth', map_location='cpu', weights_only=True)); model.eval()
    if exp.parent.short.state_hash(model) != training['selected_state_sha256']: raise ValueError('Transfer state identity mismatch')
    temp = d/'source_view_temporary'; temp.mkdir(exist_ok=True); hists = {m: {} for m in ('tta5', 'mean_multiscale5')}
    for g, region in enumerate(manifest['regions']):
        v = region['splits']['val']; raw = np.load(v['image_path'], mmap_mode='r')
        views = np.stack([exp.native.make_views(raw[i, :3])[0] for i in v['indices']]); grids = exp.native.infer_grids(model, views)
        np.savez_compressed(temp/f'g{g}.npz', **grids)
        native = exp.rgb.native_maps(grids, len(data[g]['labels'][0])); pos = exp.parent.gate.meta_split(len(data[g]['labels']), g)[1]
        labels = [data[g]['labels'][i] for i in pos]
        for method, probs in [('tta5', native['tta5']), ('mean_multiscale5', [.5*(a+b) for a, b in zip(native['global'], native['local'])])]:
            hists[method][g] = exp.parent.source_histogram([probs[i] for i in pos], labels)
    thresholds = {}; calibrations = {}; masks = {}; result = {m: {'selection': {}, 'calibration': {}} for m in hists}
    for m, h in hists.items(): thresholds[m], calibrations[m] = final.threshold(h)
    np.savez_compressed(d/f'{phase}_source_histograms.npz', **{f'{m}_g{g}': h for m, a in hists.items() for g, h in a.items()})
    for g, region in enumerate(manifest['regions']):
        with np.load(temp/f'g{g}.npz') as grids: maps = exp.rgb.native_maps(grids, len(data[g]['labels'][0]))
        splits = exp.parent.gate.meta_split(len(data[g]['labels']), g); fit = set(splits[0].tolist())
        counts = {m: {'selection': [], 'calibration': []} for m in hists}
        for i, truth in enumerate(data[g]['labels']):
            components = base.gt_components(truth, region['gsd_m']); split = 'selection' if i in fit else 'calibration'
            for m, prob in [('tta5', maps['tta5'][i]), ('mean_multiscale5', .5*(maps['global'][i]+maps['local'][i]))]:
                mask = np.asarray(prob, np.float64) >= thresholds[m]; key = f'g{g}_p{i}_{m}'
                masks[key] = np.packbits(mask.ravel()); counts[m][split].append(base.parent_counts(mask, truth, components, .5))
        for m in hists:
            for split in ('selection', 'calibration'): result[m][split][str(g)] = base.aggregate_counts(counts[m][split])
        (temp/f'g{g}.npz').unlink()
    temp.rmdir(); np.savez_compressed(d/f'{phase}_source_masks.npz', **masks)
    add_source_statistics(result, thresholds, calibrations)
    t, ms = result['tta5']['selection_macro'], result['mean_multiscale5']['selection_macro']
    checks = {'coverage_gain': ms['small_cover50'] >= t['small_cover50']+.02, 'iou_gain': ms['small_iou50'] >= t['small_iou50']+.02,
              'macro_f1_retained': ms['f1'] >= t['f1']-.005, 'region_f1_retained': max(result['tta5']['selection'][g]['f1']-result['mean_multiscale5']['selection'][g]['f1'] for g in result['tta5']['selection']) <= .02,
              'selection_fpr_control': ms['background_fpr'] <= t['background_fpr']+.002}
    record = {'completed_at': base.now(), 'phase': phase, 'arm': arm, 'seed': seed, 'training_snapshot': training,
              'methods': result, 'multiscale_source_checks': checks, 'chosen_method': 'mean_multiscale5' if all(checks.values()) else 'tta5',
              'source_masks_sha256': base.sha(d/f'{phase}_source_masks.npz'), 'source_histograms_sha256': base.sha(d/f'{phase}_source_histograms.npz'),
              'no_mine_or_target_used': True}
    base.write(dest, record); del model; torch.cuda.empty_cache(); return record


def pilot_gate(scratch, pretrained):
    a, b = scratch['methods']['tta5'], pretrained['methods']['tta5']
    gain = b['selection_macro']['f1']-a['selection_macro']['f1']
    worst = max(a['selection'][g]['f1']-b['selection'][g]['f1'] for g in a['selection'])
    return {'source_macro_f1_gain_at2pct': gain, 'worst_source_f1_loss': worst, 'passed': gain >= .02 and worst <= .03}


def unpack_mask(packed, shape):
    a = np.asarray(packed); n = int(np.prod(shape))
    if a.dtype != np.uint8 or a.ndim != 1 or len(a) != (n+7)//8: raise ValueError('Malformed packed source mask')
    if n % 8 and int(a[-1]) & ((1 << (8-n % 8))-1): raise ValueError('Nonzero mask padding')
    return np.unpackbits(a, count=n).reshape(shape).astype(bool)


def verify_source(out, p, data, manifest, phase):
    import torch
    checks = []
    for arm in ARMS:
        for seed in p['seeds']:
            d = directory(out, arm, seed); r = base.read(d/f'{phase}_source.json')
            mp, hp = d/f'{phase}_source_masks.npz', d/f'{phase}_source_histograms.npz'
            if base.sha(mp) != r['source_masks_sha256'] or base.sha(hp) != r['source_histograms_sha256']: raise ValueError('Source replay caches changed')
            with np.load(mp) as masks, np.load(hp) as hist:
                expected_keys = set()
                for method, result in r['methods'].items():
                    q, cal = final.threshold({g: hist[f'{method}_g{g}'] for g in range(7)})
                    if q != result['threshold'] or cal != result['source_calibration']: raise ValueError('Source threshold replay failed')
                    observed = []
                    for g, region in enumerate(manifest['regions']):
                        fit = set(exp.parent.gate.meta_split(len(data[g]['labels']), g)[0].tolist())
                        counts = {'selection': [], 'calibration': []}
                        for i, truth in enumerate(data[g]['labels']):
                            key = f'g{g}_p{i}_{method}'; expected_keys.add(key); mask = unpack_mask(masks[key], truth.shape)
                            split = 'selection' if i in fit else 'calibration'
                            counts[split].append(base.parent_counts(mask, truth, base.gt_components(truth, region['gsd_m']), .5))
                        for split in counts:
                            if base.aggregate_counts(counts[split]) != result[split][str(g)]: raise ValueError('Source packed-mask score replay failed')
                        observed.append(result['calibration'][str(g)]['background_fpr'])
                    if not np.allclose(observed, cal['source_fprs'], atol=1e-12, rtol=0): raise ValueError('Histogram/mask source rates differ')
                    for split in ('selection', 'calibration'):
                        if macro(result[split]) != result[split+'_macro']: raise ValueError('Source macro replay failed')
                if set(masks.files) != expected_keys: raise ValueError('Unexpected source mask keys')
                forward_replay = 0
                if base.sha(d/'source_selected.pth') == r['training_snapshot']['weight_sha256']:
                    model = architecture.RGBTransferModel().cuda()
                    model.load_state_dict(torch.load(d/'source_selected.pth', map_location='cpu', weights_only=True)); model.eval()
                    if exp.parent.short.state_hash(model) != r['training_snapshot']['selected_state_sha256']: raise ValueError('Transfer selected state changed')
                    region = manifest['regions'][0]['splits']['val']; raw = np.load(region['image_path'], mmap_mode='r')
                    views = np.stack([exp.native.make_views(raw[i, :3])[0] for i in region['indices'][:3]])
                    grids = exp.native.infer_grids(model, views, batch=3); native = exp.rgb.native_maps(grids, len(data[0]['labels'][0]))
                    for i, truth in enumerate(data[0]['labels'][:3]):
                        for method, prob in [('tta5', native['tta5'][i]), ('mean_multiscale5', .5*(native['global'][i]+native['local'][i]))]:
                            old = unpack_mask(masks[f'g0_p{i}_{method}'], truth.shape)
                            if not np.array_equal(old, prob.astype(np.float64) >= r['methods'][method]['threshold']): raise ValueError('Source actual forward replay failed')
                            forward_replay += 1
                    del model; torch.cuda.empty_cache()
            checks.append({'arm': arm, 'seed': seed, 'phase': phase, 'thresholds_rebuilt': 2, 'packed_masks_rebuilt': len(expected_keys), 'fresh_forward_masks_checked': forward_replay})
    base.write(Path(out)/f'{phase}_verification.json', {'verified_at': base.now(), 'status': 'passed', 'fits': checks})
    return checks


def stitch(native, shape):
    h, w = shape; result = {m: np.empty((h, w), np.float32) for m in ('tta5', 'mean_multiscale5')}
    for i, (y, x) in enumerate(exp.rgb.tile_positions(h, w, 32)):
        hh, ww = min(32, h-y), min(32, w-x)
        result['tta5'][y:y+hh, x:x+ww] = native['tta5'][i][:hh, :ww]
        result['mean_multiscale5'][y:y+hh, x:x+ww] = .5*(native['global'][i][:hh, :ww]+native['local'][i][:hh, :ww])
    return result


def evaluate_mines(out, p):
    import torch
    out = Path(out); control = base.read(exp.ROOT/'experiments/rgb_native_20261020/mines/results.json')['production_control']
    ledger_path = out/'exposure_ledger.json'; ledger = base.read(ledger_path) if ledger_path.exists() else {'predictions': [], 'reserved_cases_opened_or_predicted': []}
    all_results = []; verification = []
    for arm in ARMS:
        for seed in p['seeds']:
            d = directory(out, arm, seed); source = base.read(d/'full4800_source.json'); method = source['chosen_method']; training = source['training_snapshot']
            if base.sha(d/'source_selected.pth') != training['weight_sha256']: raise ValueError('Transfer final weights changed')
            config = {'schema_version': 1, 'model_type': 'UnetPlusPlus_resnet34_BGR_to_ImageNetRGB', 'weight': 'source_selected.pth', 'weight_sha256': training['weight_sha256'],
                      'method': method, 'threshold': source['methods'][method]['threshold'], 'source_only_selection': True,
                      'numerical_protocol': {'dtype': 'float32', 'cuda_tf32': False, 'cudnn_deterministic': True}, 'production_registered': False}
            base.write(d/'candidate.json', config); detector = architecture.load_candidate(d/'candidate.json', device='cuda'); model = detector.model
            rows_path = d/'mine_results.json'; old = base.read(rows_path) if rows_path.exists() else []
            rows = []
            for case in p['mine_cases']:
                image, truth, normal, valid, _ = exp.old_rgb.load_mine(case); path = d/(case['mine']+'_probabilities.npz')
                previous = next((r for r in old if r['case'] == case['mine']), None)
                if path.exists() and previous and base.sha(path) == previous['probabilities_sha256']:
                    with np.load(path) as z: probs = {m: z[m] for m in ('tta5', 'mean_multiscale5')}
                else:
                    probs = stitch(exp.native.scene_grids(model, image[:3]), truth.shape); np.savez_compressed(path, **probs)
                    ledger['predictions'].append({'arm': arm, 'seed': seed, 'case': case['mine'], 'methods': ['tta5', 'mean_multiscale5'], 'completed_at': base.now()}); base.write(ledger_path, ledger)
                scores = {m: exp.rgb.partial_reference_counts(prob.astype(np.float64) >= source['methods'][m]['threshold'], truth, normal, valid) for m, prob in probs.items()}
                row = {'case': case['mine'], 'arm': arm, 'seed': seed, 'source_chosen_method': method, 'methods': scores, 'probabilities_sha256': base.sha(path)}
                if previous and previous != row: raise ValueError('Mine score replay failed')
                rows.append(row); base.write(rows_path, rows)
                if case == p['mine_cases'][0]:
                    fresh = detector.detect_image(image[:3].transpose(1, 2, 0), max_batch=17)
                    error = float(np.max(np.abs(fresh-probs[method])))
                    if error > 2e-6: raise ValueError('Transfer streaming replay failed')
                    verification.append({'arm': arm, 'seed': seed, 'case': case['mine'], 'max_probability_error': error, 'method': method})
                print('TRANSFER_MINE', arm, seed, case['mine'], scores[method]['positive_core']['rate'], scores[method]['reviewed_normal']['rate'], flush=True)
            gains = [r['methods'][method]['positive_core']['rate']-control[r['case']]['0.4']['positive_core']['rate'] for r in rows]
            normals = [r['methods'][method]['reviewed_normal']['rate']-control[r['case']]['0.4']['reviewed_normal']['rate'] for r in rows]
            limits = p['mine_limits']; checks = {'core_gain': float(np.mean(gains)) >= limits['macro_core_gain_each_seed'],
                'normal_control': float(np.mean(normals)) <= limits['macro_reviewed_normal_increase_max_each_seed'], 'no_large_case_loss': max(-g for g in gains) <= limits['any_mine_core_loss_max']}
            all_results.append({'arm': arm, 'seed': seed, 'chosen_method': method, 'macro_core_gain': float(np.mean(gains)), 'macro_normal_change': float(np.mean(normals)), 'worst_core_loss': max(-g for g in gains), 'checks': checks, 'passed': all(checks.values())})
            del detector, model; torch.cuda.empty_cache()
    result = {'completed_at': base.now(), 'by_arm_seed': all_results, 'streaming_replays': verification,
              'pretrained_both_seeds_pass': all(r['passed'] for r in all_results if r['arm'] == 'imagenet_normalized'),
              'scope': 'Four repeatedly exposed development mines, partial core and reviewed-normal evidence; not a complete-boundary or blind test.'}
    base.write(out/'mine_regression.json', result); return result


def check_pairing(out, p, phase=None):
    for seed in p['seeds']:
        a, b = [base.read(directory(out, arm, seed)/f'{phase}_source.json')['training_snapshot'] if phase else base.read(directory(out, arm, seed)/'training_status.json') for arm in ARMS]
        for key in ('initial_decoder_sha256', 'parent_augmentation_sha256', 'source_visits', 'source_unique_seen', 'completed_updates'):
            if a[key] != b[key]: raise ValueError('Matched transfer control differs: '+key)


def run(out):
    import torch
    out = Path(out); p = register(out); exp.parent.setup_runtime(); exp.state(out, status='running', stage='paired_transfer_data')
    x, y, groups, data, manifest = final.prepare(out)
    gx = torch.as_tensor(np.asarray(x), device='cuda'); gy = torch.as_tensor(np.asarray(y), device='cuda'); del x, y
    pilots = {}
    for arm in ARMS:
        for seed in p['seeds']:
            train(out, arm, seed, gx, gy, groups, data, 1200, p); pilots[(arm, seed)] = source_phase(out, arm, seed, data, manifest, 'pilot1200')
    check_pairing(out, p, 'pilot1200'); gates = []
    for seed in p['seeds']:
        a = pilots[(ARMS[0], seed)]['training_snapshot']; b = pilots[(ARMS[1], seed)]['training_snapshot']
        for key in ('initial_decoder_sha256', 'parent_augmentation_sha256', 'source_visits', 'source_unique_seen'):
            if a[key] != b[key]: raise ValueError('Matched transfer control differs')
        gates.append({'seed': seed, **pilot_gate(pilots[(ARMS[0], seed)], pilots[(ARMS[1], seed)])})
    passed = all(r['passed'] for r in gates); base.write(out/'pilot_decision.json', {'decided_at': base.now(), 'by_seed': gates, 'continue_to4800': passed, 'source_only': True})
    print('TRANSFER_PILOT_DECISION', gates, passed, flush=True)
    if passed:
        for arm in ARMS:
            for seed in p['seeds']:
                train(out, arm, seed, gx, gy, groups, data, 4800, p); source_phase(out, arm, seed, data, manifest, 'full4800')
        check_pairing(out, p)
    del gx, gy; torch.cuda.empty_cache(); exp.cleanup_cache(out)
    for arm in ARMS:
        for seed in p['seeds']: (directory(out, arm, seed)/'resume_training.pth').unlink(missing_ok=True)
    finish(out, p, data, manifest, passed)


def finish(out, p, data, manifest, admitted):
    verify_source(out, p, data, manifest, 'pilot1200')
    mines = None
    if admitted:
        verify_source(out, p, data, manifest, 'full4800'); mines = evaluate_mines(out, p)
    exp.old_rgb.check_assets(p)
    result = {'completed_at': base.now(), 'pilot_admitted_both_seeds': admitted, 'completed_fits': 4,
              'updates_per_fit': 4800 if admitted else 1200, 'production_changed': False,
              'official_test_predictions': 0, 'reserved_predictions': 0}
    result['mine_regression'] = mines
    base.write(Path(out)/'results.json', result)
    base.write(Path(out)/'verification.json', {'verified_at': base.now(), 'status': 'passed', 'source_phases': 2 if admitted else 1,
        'source_fits_verified': 8 if admitted else 4, 'mine_streaming_replays': len(mines['streaming_replays']) if mines else 0,
        'paired_sampling_and_decoder_checked': True, 'protected_assets_unchanged': True})
    exp.state(out, status='completed', stage='all_stages_verified', source_pilot_passed=admitted, fits=4, updates=result['updates_per_fit'])


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(); parser.add_argument('--output', required=True); a = parser.parse_args()
    try: run(Path(a.output))
    except BaseException as error: exp.state(Path(a.output), status='interrupted_or_failed', error=str(error)); raise
