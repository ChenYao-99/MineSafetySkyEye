"""Bounded source-only BatchNorm attribution for the128 input mismatch."""
from pathlib import Path
import sys
import hashlib
import numpy as np
sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parent)]
from runtime_paths import configure_runtime_paths
configure_runtime_paths()
import rgb_detail_adaptation as adapt
import rgb_resolution_study as detail
import rgb_transfer_study as transfer
import rgb_transfer_model as architecture
import rgb_resolution as resolution
exp = transfer.exp; base = exp.base
PRIOR = exp.ROOT/'experiments/rgb_detail_adaptation_20261020'
SOURCE = adapt.SOURCE


def parameter_hash(model):
    h = hashlib.sha256()
    for name, p in model.named_parameters(): h.update(name.encode()); h.update(p.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def recalibrate(model, batches):
    import torch
    before = parameter_hash(model); model.eval(); modules = []
    for name, m in model.named_modules():
        if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
            if not m.track_running_stats: raise ValueError('Running-stat BN required')
            modules.append((name, m, m.momentum)); m.reset_running_stats(); m.momentum = None; m.train()
    with torch.inference_mode():
        for x in batches: model(x)
    for _, m, momentum in modules: m.eval(); m.momentum = momentum
    if parameter_hash(model) != before: raise ValueError('Statistics-only calibration modified parameters')
    delta = {name+'.'+key: getattr(m, key).detach().cpu() for name, m, _ in modules for key in ('running_mean', 'running_var', 'num_batches_tracked')}
    return delta


def source_inputs(manifest):
    data = {}
    for g, r in enumerate(manifest['regions']):
        v = r['splits']['val']; raw = np.load(v['image_path'], mmap_mode='r'); y = np.load(v['label_path'], mmap_mode='r')
        labels = [y[i, 0].astype(np.uint8) for i in v['indices']]
        data[g] = {'views': np.stack([resolution.whole_view(raw[i, :3]) for i in v['indices']]), 'labels': labels,
                   'components': [base.gt_components(a, r['gsd_m']) for a in labels]}
    return data


def register(out):
    out = Path(out).resolve()
    if out.drive.upper() != 'D:' or not out.is_relative_to(exp.ROOT/'experiments'): raise ValueError('D experiments only')
    out.mkdir(parents=True, exist_ok=True)
    if (out/'protocol.json').exists():
        p = base.read(out/'protocol.json'); exp.old_rgb.check_assets(p); return p
    frozen = dict(base.read(PRIOR/'protocol.json')['frozen_assets'])
    paths = [Path(__file__), exp.ROOT/'tests/test_rgb_resolution_bn.py', PRIOR/'protocol.json', PRIOR/'results.json', PRIOR/'verification.json', SOURCE/'data_manifest.json']
    frozen.update({str(a): base.sha(a) for a in paths})
    p = {'registered_at': base.now(), 'frozen_assets': frozen, 'seeds': [42, 43], 'source_gate': base.read(PRIOR/'protocol.json')['source_gate'],
         'question': 'Is the128 mismatch removable by source-only BN statistics rather than decoder/encoder gradient fitting?',
         'calibration': 'Exactly63 eligible train parents/source, without replacement, seed20261020+modelseed.21 batches each with3/source. Resize parents128 using frozen minmax/interpolation. Reset all BN running stats and use cumulative average; all parameters fixed; eval-mode for non-BN layers. No validation parents in BN updates.',
         'comparison': 'Before/after BN128 source masks at independently calibrated2% backgroundFPR; final source gate compares BN128 to frozen original64 TTA5, not just to a weak128 control.',
         'scope': 'All seven LMHLD regions are explicit sources. No held-region, mine, official-test or reserved-case prediction. No independent acceptance or promotion.',
         'stop': 'If either seed fails the source gate, close this128 statistics branch. If both pass, retain candidate BN buffers for a separately frozen geographic continuation.',
         'retention': 'Tiny BN buffers, chosen train-parent identifiers, sparse source masks and histograms. No full weight copies or raw-image caches.',
         'training_updates': 0, 'production_changed': False, 'target_predictions': 0, 'external_model_calls': 0}
    exp.old_rgb.check_assets(p); base.write(out/'protocol.json', p); return p


def train_views(manifest, seed):
    rng = np.random.default_rng(20261020+seed); choices = {}; arrays = {}
    for g, region in enumerate(manifest['regions']):
        eligible = [r for r in manifest['kept_train'] if r['group'] == g]
        chosen = [eligible[i] for i in rng.choice(len(eligible), 63, replace=False)]; raw = np.load(region['splits']['train']['image_path'], mmap_mode='r')
        for r in chosen:
            if base.array_sha(raw[r['index'], :3]) != r['sha256']: raise ValueError('BN source parent changed')
        choices[str(g)] = chosen; arrays[g] = np.stack([resolution.whole_view(raw[r['index'], :3]) for r in chosen])
    batches = [np.concatenate([arrays[g][i:i+3] for g in range(7)]) for i in range(0, 63, 3)]
    return batches, choices


def evaluate(out, model, data, seed, name):
    d = Path(out)/f'seed{seed}'; d.mkdir(exist_ok=True); hist = {}; probabilities = {}
    for g, v in data.items():
        probs = exp.parent.short.native_probabilities(resolution.predict(model, v['views']), v['labels']); probabilities[g] = probs
        pos = exp.parent.gate.meta_split(len(v['labels']), g)[1]
        hist[g] = exp.parent.source_histogram([probs[i] for i in pos], [v['labels'][i] for i in pos])
    q, cal = transfer.final.threshold(hist); scores = {'selection': {}, 'calibration': {}}; masks = {}
    for g, v in data.items():
        probs = probabilities[g]
        for i, a in enumerate(probs): masks[f'g{g}_p{i}'] = np.packbits((a.astype(np.float64) >= q).ravel())
        for split, ids in zip(('selection', 'calibration'), exp.parent.gate.meta_split(len(v['labels']), g)):
            scores[split][str(g)] = detail.metrics([probs[i] for i in ids], [v['labels'][i] for i in ids], [v['components'][i] for i in ids], q)
    np.savez_compressed(d/f'{name}_masks.npz', **masks); np.savez_compressed(d/f'{name}_histograms.npz', **{f'g{g}': h for g, h in hist.items()})
    r = {'threshold': q, 'source_calibration': cal, **scores, 'masks_sha256': base.sha(d/f'{name}_masks.npz'), 'histograms_sha256': base.sha(d/f'{name}_histograms.npz')}
    base.write(d/f'{name}.json', r); return r


def verify(out, data, seed, name):
    d = Path(out)/f'seed{seed}'; r = base.read(d/f'{name}.json')
    if base.sha(d/f'{name}_masks.npz') != r['masks_sha256'] or base.sha(d/f'{name}_histograms.npz') != r['histograms_sha256']: raise ValueError('BN diagnostic proof changed')
    with np.load(d/f'{name}_masks.npz') as z, np.load(d/f'{name}_histograms.npz') as h:
        q, cal = transfer.final.threshold({g: h[f'g{g}'] for g in range(7)})
        if q != r['threshold'] or cal != r['source_calibration']: raise ValueError('BN threshold replay differs')
        for g, v in data.items():
            for split, ids in zip(('selection', 'calibration'), exp.parent.gate.meta_split(len(v['labels']), g)):
                masks = [transfer.unpack_mask(z[f'g{g}_p{i}'], v['labels'][i].shape) for i in ids]
                if detail.metrics(masks, [v['labels'][i] for i in ids], [v['components'][i] for i in ids], .5) != r[split][str(g)]: raise ValueError('BN source score replay differs')


def run(out):
    import torch
    out = Path(out); p = register(out); exp.parent.setup_runtime(); exp.state(out, status='running', stage='source128_BN_attribution')
    manifest = base.read(SOURCE/'data_manifest.json'); data = source_inputs(manifest); records = []
    for seed in p['seeds']:
        d = out/f'seed{seed}'; d.mkdir(exist_ok=True); model = architecture.RGBTransferModel(); model.load_state_dict(torch.load(adapt.initial_path(None, seed), map_location='cpu', weights_only=True)); model.cuda().eval()
        before = evaluate(out, model, data, seed, 'before'); batches, choices = train_views(manifest, seed); base.write(d/'source_parents.json', choices)
        params = parameter_hash(model); buffers = recalibrate(model, (torch.as_tensor(a, device='cuda').float() for a in batches)); del batches
        exp.parent.short.save_torch(d/'bn_buffers.pth', buffers); state = exp.parent.short.state_hash(model)
        after = evaluate(out, model, data, seed, 'after'); original = base.read(SOURCE/'imagenet_normalized'/f'seed{seed}'/'full4800_source.json')['methods']['tta5']['selection']
        gate = detail.source_gate(original, after['selection'], p['source_gate'])
        # Reconstruct exactly from the immutable original state and tiny buffers.
        model.load_state_dict(torch.load(adapt.initial_path(None, seed), map_location='cpu', weights_only=True)); rebuilt = model.state_dict(); rebuilt.update(torch.load(d/'bn_buffers.pth', map_location='cpu', weights_only=True)); model.load_state_dict(rebuilt)
        if exp.parent.short.state_hash(model) != state or parameter_hash(model) != params: raise ValueError('BN state/parameter replay failed')
        for name in ('before', 'after'): verify(out, data, seed, name)
        row = {'seed': seed, 'before_macro': detail.macro(before['selection']), 'after_macro': detail.macro(after['selection']), 'gate_vs_original64': gate,
               'source_parents_sha256': base.sha(d/'source_parents.json'), 'bn_buffers_sha256': base.sha(d/'bn_buffers.pth'), 'parameter_sha256': params, 'recalibrated_state_sha256': state}
        records.append(row); print('RESOLUTION_BN_SOURCE', row, flush=True); del model; torch.cuda.empty_cache()
    exp.old_rgb.check_assets(p); passed = all(r['gate_vs_original64']['passed'] for r in records)
    base.write(out/'results.json', {'completed_at': base.now(), 'seeds': records, 'passed_both_seeds': passed, 'production_changed': False, 'target_predictions': 0})
    base.write(out/'verification.json', {'verified_at': base.now(), 'status': 'passed', 'source_masks_rebuilt': 4, 'parameter_and_state_replays': 2, 'protected_assets_unchanged': True})
    exp.state(out, status='completed', stage='source128_BN_attribution_verified', passed=passed, target_predictions=0)


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser(); p.add_argument('--output', required=True); args = p.parse_args()
    try: run(Path(args.output))
    except BaseException as e: exp.state(Path(args.output), status='interrupted_or_failed', error=str(e)); raise
