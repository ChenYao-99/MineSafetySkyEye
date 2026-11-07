"""Frozen supplementary Rio validation; no fitting, mining reserve or test files."""
from pathlib import Path
import sys
import time
import numpy as np
sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parent)]
import rgb_native_training as exp
import rgb_native_analysis as analysis

PARENT = exp.ROOT/'experiments/rgb_native_20261020'
REGION = 'Rio de Janeiro_Brazil_64'
SOURCE = Path(r'F:\LMHLD数据集\LMHLD_dataset_different_patch_sizes')/REGION


def admit(images, labels, excluded_hashes):
    if images.ndim != 4 or images.shape[1:] != (5, 64, 64) or labels.shape != (len(images), 1, 64, 64):
        raise ValueError('Expected original five-band Rio64 validation parents')
    records = []; excluded = []; seen = {}
    for i, (x, y) in enumerate(zip(images, labels)):
        hx = exp.base.array_sha(x[:3]); hy = exp.base.array_sha(y)
        if not np.isfinite(x[:3]).all() or not np.isin(y, [0, 1]).all():
            excluded.append({'index': i, 'reason': 'invalid_input_or_truth'}); continue
        if hx in excluded_hashes:
            excluded.append({'index': i, 'reason': 'matches_any_prior_train_or_val_bgr'}); continue
        if hx in seen:
            if seen[hx] != hy: raise ValueError('Conflicting truth for duplicate Rio BGR image')
            excluded.append({'index': i, 'reason': 'same_region_bgr_duplicate'}); continue
        seen[hx] = hy
        records.append({'index': i, 'bgr_sha256': hx, 'raw_sha256': exp.base.array_sha(x), 'label_sha256': hy})
    if not records: raise ValueError('No disjoint new-region validation parents')
    return records, excluded


def register(out):
    out = Path(out).resolve()
    if out.drive.upper() != 'D:' or not out.is_relative_to(exp.ROOT/'experiments'):
        raise ValueError('Use D:/LMHLD/experiments')
    out.mkdir(parents=True, exist_ok=True)
    path = out/'protocol.json'
    if path.exists():
        p = exp.base.read(path); exp.old_rgb.check_assets(p); return p
    parent = exp.base.read(PARENT/'protocol.json'); manifest = exp.base.read(PARENT/'data_manifest.json')
    excluded = {r['sha256'] for r in manifest['kept_train']}
    excluded.update(h for rows in manifest['all_official_val_bgr_sha256'].values() for h in rows)
    image_path = SOURCE/'val_images.npy'; label_path = SOURCE/'val_labels.npy'
    images = np.load(image_path, mmap_mode='r'); labels = np.load(label_path, mmap_mode='r')
    records, rejected = admit(images, labels, excluded)
    frozen = {str(p): exp.base.sha(p) for p in [Path(__file__), exp.ROOT/'tests/test_rgb_new_region.py', PARENT/'protocol.json',
                                              PARENT/'data_manifest.json', PARENT/'primary_comparison_protocol.json', image_path, label_path]}
    p = {'registered_at': exp.base.now(), 'frozen_assets': frozen, 'region': REGION, 'nominal_gsd_m': 5.,
         'image_path': str(image_path), 'label_path': str(label_path), 'records': records, 'excluded': rejected,
         'parents': len(records), 'model_protocol': str(PARENT/'protocol.json'), 'model_arms': parent['arms'],
         'original_held_indices': parent['held_indices'], 'seeds': parent['seeds'], 'primary_source_fold': 2,
         'input_mapping': 'Take NumPy channels0,1,2 as BGR, discard red-edge/NIR before normalization. Mapping follows documented standard RapidEye Analytic band order; NPY lacks its own semantic channel labels and exporter mapping has not been independently verified.',
         'sensor_documentation': 'https://assets.planet.com/docs/combined-imagery-product-spec-april-2019.pdf#page=80',
         'nominal_gsd_documentation': 'https://arxiv.org/pdf/2502.19866 Table1 (preprint); project formal paper DOI10.1109/TGRS.2025.3619062 uses same region.',
         'dataset_source': 'https://zenodo.org/records/11519933',
         'model_rule': 'ALL twelve predetermined source-selected models; freeze actual weight/policy hashes before first Rio inference. No model, checkpoint, method or threshold selection on Rio.',
         'threshold_rule': 'Exactly original source-only TTA/global/mean thresholds and area guard. No Rio calibration, labels or normalization statistics shared across parents.',
         'primary_comparison': 'jitter area_guard vs fixed TTA5, first preregistered original fold2, require both seeds. Other source folds are descriptive repetitions on SAME parents, not additional independent samples.',
         'primary_limits': exp.base.read(PARENT/'primary_comparison_protocol.json')['limits'],
         'previous_access': 'Only shape, finiteness, binary labels, raw band value summaries and exact hashes examined for admission before scoring. Rio train/val were not in this series model gradients/checkpoint selection/calibration. Earlier global-project exposure cannot be independently guaranteed.',
         'scope': 'Supplementary new geographic/sensor region relative to the current six-region series, under explicit band-order assumption. Same public LMHLD dataset. No spatial-adjacency audit, complete mining accident boundary or field blind acceptance claim.',
         'official_test_files_opened': 0, 'training_updates': 0, 'production_changed': False, 'external_model_calls': 0}
    exp.old_rgb.check_assets(p); exp.base.write(path, p); return p


def run(out, p):
    import torch
    exp.parent.setup_runtime(); out = Path(out)
    verification = exp.base.read(PARENT/'verification.json')
    if verification['status'] != 'passed': raise ValueError('Complete and verify all twelve models first')
    snapshot_path = out/'model_snapshot.json'
    frozen = {}
    for arm in p['model_arms']:
        for g in p['original_held_indices']:
            for seed in p['seeds']:
                d = exp.folder(PARENT, arm, g, seed)
                for name in ('source_selected.pth', 'training.json', 'rgb_policy.json'):
                    frozen[str(d/name)] = exp.base.sha(d/name)
    snapshot = {'frozen_assets': frozen, 'parent_verification_sha256': exp.base.sha(PARENT/'verification.json')}
    if snapshot_path.exists():
        old = exp.base.read(snapshot_path)
        if old['frozen_assets'] != frozen: raise ValueError('Rio model identity changed')
    else: exp.base.write(snapshot_path, {'frozen_at': exp.base.now(), **snapshot})
    images = np.load(p['image_path'], mmap_mode='r'); labels = np.load(p['label_path'], mmap_mode='r')
    views = []; truth = []
    for r in p['records']:
        x, y = images[r['index']], labels[r['index']]
        if exp.base.array_sha(x) != r['raw_sha256'] or exp.base.array_sha(y) != r['label_sha256']: raise ValueError('Rio parents changed')
        views.append(exp.native.make_views(x[:3])[0]); truth.append(y[0].astype(np.uint8))
    data = {'views': np.stack(views), 'labels': truth, 'components': [exp.base.gt_components(y, 5.) for y in truth]}
    rows = []
    for arm in p['model_arms']:
        for g in p['original_held_indices']:
            for seed in p['seeds']:
                d = out/arm/f'fold_{g}'/f'seed{seed}'; d.mkdir(parents=True, exist_ok=True)
                result_path = d/'result.json'
                if result_path.exists(): rows.append(exp.base.read(result_path)); continue
                source = exp.folder(PARENT, arm, g, seed); policy = exp.base.read(source/'rgb_policy.json'); model = exp.load_model(source)
                grids = exp.native.infer_grids(model, data['views']); path = d/'probabilities.npz'
                np.savez_compressed(path, **grids)
                methods = exp.score_target(grids, data, 5., policy)
                row = {'completed_at': exp.base.now(), 'arm': arm, 'source_fold': g, 'seed': seed,
                       'methods': methods, 'probabilities_sha256': exp.base.sha(path), 'policy_sha256': exp.base.sha(source/'rgb_policy.json')}
                exp.base.write(result_path, row); rows.append(row); del model; torch.cuda.empty_cache()
                print('RIO_VALIDATION', arm, g, seed, {k: a['held_out']['f1'] for k, a in methods.items()}, flush=True)
    macro = {a: {m: {k: float(np.mean([r['methods'][m]['held_out'][k] for r in rows if r['arm'] == a])) for k in exp.METRICS} for m in exp.METHODS} for a in p['model_arms']}
    checks = []
    for seed in p['seeds']:
        fixed = next(r for r in rows if r['arm'] == 'fixed' and r['source_fold'] == 2 and r['seed'] == seed)
        jitter = next(r for r in rows if r['arm'] == 'scale_jitter' and r['source_fold'] == 2 and r['seed'] == seed)
        checks.append({'seed': seed, **analysis.assess([(fixed['methods']['tta5']['held_out'], jitter['methods']['area_guard']['held_out'])], p['primary_limits'])})
    exp.old_rgb.check_assets(p); exp.old_rgb.check_assets(snapshot)
    exp.base.write(out/'results.json', {'completed_at': exp.base.now(), 'parents': len(truth), 'parent_model_predictions': len(truth)*len(rows),
                                       'target_macro': macro, 'primary_by_seed': checks, 'primary_passed_both_seeds': all(r['passed'] for r in checks),
                                       'scope': p['scope'], 'input_mapping': p['input_mapping'], 'production_changed': False})
    # Recompute every mask/count from preserved probabilities; no policy refit.
    for r in rows:
        d = out/r['arm']/f"fold_{r['source_fold']}"/f"seed{r['seed']}"
        policy = exp.base.read(exp.folder(PARENT, r['arm'], r['source_fold'], r['seed'])/'rgb_policy.json')
        if exp.base.sha(d/'probabilities.npz') != r['probabilities_sha256']: raise ValueError('Rio cache changed')
        with np.load(d/'probabilities.npz') as grids:
            if exp.score_target(grids, data, 5., policy) != r['methods']: raise ValueError('Rio metric replay differs')
    exp.base.write(out/'verification.json', {'verified_at': exp.base.now(), 'status': 'passed', 'metric_replays': len(rows),
                                            'training_or_calibration_on_rio': False, 'official_test_files_opened': 0})
    print('RIO_COMPLETE', macro, flush=True)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(); parser.add_argument('--output', required=True); parser.add_argument('--register', action='store_true'); parser.add_argument('--wait', action='store_true')
    args = parser.parse_args(); out = Path(args.output); p = register(out)
    if not args.register:
        if args.wait:
            started = time.monotonic(); print('RIO_QUEUED_AFTER_TRAINING_VERIFICATION', flush=True)
            while not (PARENT/'verification.json').exists():
                if time.monotonic()-started > 21600: raise TimeoutError('Training queue exceeded six hours')
                time.sleep(15)
        run(out, p)
