"""Source-only BN-statistics diagnostic after the fixed paired RGB study.

No held-out predictions, gradients, new CNN parameters or NIR inputs. A failed
source gate stops this route before further cross-region evaluation.
"""
from pathlib import Path
import sys
import hashlib
import numpy as np
sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parent)]
import rgb_native_training as exp
PARENT = exp.ROOT/'experiments/rgb_native_20261020'


def parameter_hash(model):
    h = hashlib.sha256()
    for key, value in model.named_parameters(): h.update(key.encode()); h.update(value.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def source_score(model, data, held):
    rows = {}
    for g, d in sorted(data.items()):
        if g == held: raise ValueError('Held-out region cannot enter BN source screen')
        pos = exp.parent.gate.meta_split(len(d['labels']), g)[0]
        x = d['views'][pos, 0]; labels = [d['labels'][i] for i in pos]
        p = exp.parent.short.forward(model, x, batch=32)
        for k in (1, 2, 3, 4): p += exp.parent.short.forward(model, x, batch=32, kind=k)
        rows[str(g)] = exp.parent.short.pixel_metrics(exp.parent.short.native_probabilities(p/5, labels), labels)
    exp.filt.source_keys(map(int, rows), held)
    return {'macro_f1': float(np.mean([r['f1'] for r in rows.values()])), 'regions': rows}


def gate(before, after):
    gain = after['macro_f1']-before['macro_f1']
    worst = max(before['regions'][g]['f1']-after['regions'][g]['f1'] for g in before['regions'])
    return {'macro_f1_gain': gain, 'worst_source_f1_loss': worst,
            'checks': {'macro_gain_at_least_one_pp': gain >= .01, 'no_source_loss_over_two_pp': worst <= .02},
            'passed': gain >= .01 and worst <= .02}


def register(out):
    out = Path(out).resolve()
    if out.drive.upper() != 'D:' or not out.is_relative_to(exp.ROOT/'experiments'): raise ValueError('Use D experiments')
    out.mkdir(parents=True, exist_ok=True)
    if (out/'protocol.json').exists():
        p = exp.base.read(out/'protocol.json'); exp.old_rgb.check_assets(p); return p
    paths = [Path(__file__), exp.ROOT/'tests/test_rgb_bn_source_pilot.py', PARENT/'protocol.json', PARENT/'data_manifest.json',
             PARENT/'results.json', PARENT/'verification.json']
    for seed in (42, 43):
        for name in ('source_selected.pth', 'source_curve.json', 'training.json'):
            paths.append(exp.folder(PARENT, 'scale_jitter', 2, seed)/name)
    p = {'registered_at': exp.base.now(), 'frozen_assets': {str(p): exp.base.sha(p) for p in paths},
         'hypothesis': 'Source global-view performance may be affected by mixed global/quadrant BN statistics. Reestimate BN running moments using source training global views, without changing CNN parameters.',
         'cohort': 'First fixed source fold2 only, seeds42/43; source-only diagnostic, not held-out acceptance.',
         'sampling': 'Uniform64 original training parents per source, fixed label-free seed20261020+17*held+131*group; same parents and balanced20 batches in both model seeds.',
         'mechanism': 'Reset all BN running moments/counters, cumulative momentumNone, BN layers train and other layers eval, no_grad. Sixteen batches of4 images from each of five sources; no target or val inputs in moment fitting.',
         'source_gate': {'min_macro_f1_gain': .01, 'max_any_source_f1_loss': .02, 'both_seeds_required': True},
         'source_scoring': 'Same original source selection halves, native TTA5 F1@.5 as checkpoint selection. Source val not used to fit BN moments; no cutoff fitting in this pilot.',
         'decisions': 'If either seed fails source gate, stop without target evaluation. If both pass, a new frozen full-cohort evaluation is required before deployment.',
         'training_updates': 0, 'held_out_predictions': 0, 'mine_predictions': 0, 'production_changed': False,
         'documentation': 'https://docs.pytorch.org/docs/stable/generated/torch.optim.swa_utils.update_bn.html'}
    exp.base.write(out/'protocol.json', p); return p


def run(out):
    import torch
    out = Path(out); p = register(out); exp.parent.setup_runtime(); manifest = exp.base.read(PARENT/'data_manifest.json'); held = 2
    source = {g: exp.old_rgb.load_group(manifest, g, 'rgb_proxy') for g in range(6) if g != held}
    for d in source.values(): d['views'] = np.ascontiguousarray(d['views'][:, :, :3])
    selected = {}; batches = []
    for g in source:
        pool = [r for r in manifest['kept_train'] if r['group'] == g]
        ids = np.random.default_rng(20261020+17*held+131*g).choice(len(pool), 64, replace=False)
        selected[g] = [pool[i] for i in ids]
    raw = {g: np.load(manifest['regions'][g]['splits']['train']['image_path'], mmap_mode='r') for g in source}
    for start in range(0, 64, 4):
        images = []
        for g in source:
            for r in selected[g][start:start+4]:
                a = raw[g][r['index']]
                if exp.base.array_sha(a[:3]) != r['sha256']: raise ValueError('Source BN input changed')
                images.append(exp.native.make_views(a[:3])[0][0])
        batches.append(np.stack(images))
    exp.base.write(out/'source_parents.json', {str(g): [{'index': r['index'], 'bgr_sha256': r['sha256']} for r in rows] for g, rows in selected.items()})
    rows = []
    for seed in (42, 43):
        model = exp.load_model(exp.folder(PARENT, 'scale_jitter', held, seed)); before = source_score(model, source, held)
        expected = exp.base.read(exp.folder(PARENT, 'scale_jitter', held, seed)/'training.json')['source_macro_f1']
        if abs(before['macro_f1']-expected) > 1e-12: raise ValueError('Original source score did not reproduce')
        identity = parameter_hash(model); model.eval(); layers = []
        for layer in model.modules():
            if isinstance(layer, torch.nn.modules.batchnorm._BatchNorm):
                layers.append((layer, layer.momentum)); layer.reset_running_stats(); layer.momentum = None; layer.train()
        with torch.no_grad():
            for a in batches: model(torch.as_tensor(a, device='cuda').float())
        for layer, momentum in layers: layer.momentum = momentum
        model.eval()
        if parameter_hash(model) != identity: raise ValueError('BN calibration changed model parameters')
        after = source_score(model, source, held); report = gate(before, after)
        buffers = {key: value.detach().cpu() for key, value in model.named_buffers() if key.endswith(('running_mean', 'running_var', 'num_batches_tracked'))}
        exp.parent.short.save_torch(out/f'seed{seed}_bn_buffers.pth', buffers)
        rows.append({'seed': seed, 'before': before, 'after': after, 'parameter_hash_unchanged': identity,
                     'bn_layers': len(layers), 'calibration_batches': len(batches), **report})
        exp.base.write(out/f'seed{seed}_result.json', rows[-1]); print('BN_SOURCE_PILOT', seed, report, flush=True)
        del model; torch.cuda.empty_cache()
    exp.old_rgb.check_assets(p); passed = all(r['passed'] for r in rows)
    result = {'completed_at': exp.base.now(), 'source_gate_passed_both_seeds': passed, 'by_seed': rows,
              'next_action': 'Freeze full-cohort evaluation before any target predictions' if passed else 'Stop BN-global recalibration route; no source evidence strong enough to justify target tests.',
              'held_out_predictions': 0, 'mine_predictions': 0, 'training_updates': 0, 'production_changed': False}
    exp.base.write(out/'results.json', result); print('BN_SOURCE_PILOT_COMPLETE', passed, flush=True)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(); parser.add_argument('--output', required=True); a = parser.parse_args(); run(Path(a.output))
