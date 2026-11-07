"""Full-source, compute-matched training and source-chosen scale revalidation.

Offline development experiment. Frozen predecessors and production are read only.
The full-data arm changes the training pool; architecture, update budget, optimizer,
learning-rate schedule, validation rows and augmentation schedule stay matched.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import copy
import hashlib
from pathlib import Path
import sys
import time

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from runtime_paths import configure_runtime_paths
configure_runtime_paths()
import cross_region_scale as base
import training_sufficiency as short
import validate_scale_selector as gate

PRIOR = ROOT / 'experiments/training_sufficiency_20261019'
CHECKPOINTS = (1200, 2400, 3600, 4800)
HELD = (2, 4, 3)
BUDGETS = (.02, .01, .005)


def register(out):
    out = Path(out).resolve()
    if out.drive.upper() != 'D:' or not out.is_relative_to(ROOT / 'experiments'):
        raise ValueError('Use D:/LMHLD/experiments')
    out.mkdir(parents=True, exist_ok=True)
    parent = base.read(PRIOR / 'protocol.json')
    frozen = dict(parent['frozen_assets'])
    for p in (Path(__file__), ROOT / 'tests/test_full_source_training.py',
              PRIOR / 'protocol.json', PRIOR / 'data_manifest.json', PRIOR / 'results.json'):
        frozen[str(p)] = base.sha(p)
    for held in HELD:
        for seed in (42, 43):
            folder = PRIOR / f'fold_{held}_{base.REGIONS[held][0]}' / f'seed{seed}'
            for name in ('result.json', 'source_selected.pth'):
                frozen[str(folder / name)] = base.sha(folder / name)
    for p, digest in frozen.items():
        if base.sha(p) != digest:
            raise ValueError(f'Frozen predecessor changed: {p}')
    path = out / 'protocol.json'
    if path.exists():
        result = base.read(path)
        if result['frozen_assets'] != frozen:
            raise ValueError('Frozen experiment changed; use a new directory')
        return result
    parent_gate = base.read(gate.PRIOR.parent / 'source_scale_selector_20261019/protocol.json')
    result = {
        'registered_at': base.now(), 'frozen_assets': frozen,
        'question': 'Does expanding actual source train parents improve transfer at the same training budget, and does scale routing help on a source-preferred stronger backbone?',
        'held_indices': list(HELD), 'seeds': [42, 43], 'fits': 6,
        'updates': 4800, 'checkpoints': list(CHECKPOINTS),
        'training_change': 'All eligible source train parents instead of each-source512 cap; native validation rows unchanged. No architecture change.',
        'audit_rules': 'Hash all train and all official val images, open no test. Exclude nonfinite/nonbinary train rows, any train image matching any val image, cross-region duplicates, conflicting-label duplicates; deduplicate same-region identical rows, preferring prior selected rows.',
        'comparison_guard': 'All old kept train parents must survive unchanged; otherwise stop rather than claim a data-volume-only comparison.',
        'compact_inputs': 'Train stores one global64 view per parent, bit-identical to the first of the old five views. Validation keeps all five views. No persistent data copy.',
        'fixed_recipe': 'Unet++/ResNet34 random init, actual BGRN, parent minmax0..1, float16 storage/float32 training, BCE+.5softDice, batch20 with4 per source, AdamW .001/weight_decay.0001; cosine1200 to1e-5, retain moments, restart.0003 and cosine3600 to1e-5.',
        'augmentation_pairing': 'Use the same fold seed and sampling/unused-view RNG calls in both pools; separately replay old512 RNG to guarantee every applied flip/rotation is identical. Verify old sampling hashes at1200 and4800 and initial weights. If pools are identical, the full sampling schedule is exactly identical too.',
        'training_monitor': 'Identical prior first64 source parents and preprocessing, no augmentation; original old512 parents ordered first in each full pool.',
        'selection': 'Highest five-source mean native-grid TTA5 pixel F1@.5 on original source selection half, ties earliest. Calibration half disjoint and unchanged. Target not used.',
        'backbone_for_scales': 'Per fit, choose full or prior512 source-selected weight using only identical source selection F1, strict tie prefers prior512; persist before any target inference.',
        'scale_controls': list(gate.METHODS),
        'scale_training': {k: parent_gate[k] for k in ('head_steps', 'head_per_region_batch', 'head_optimizer', 'head_regularization', 'head_initialization', 'head_objective', 'pixel_sampling')},
        'head_input_integrity': 'Fit fresh source-only constant/full/noGSD heads on chosen backbone; old coefficients never reused. Source heads persisted before this scale-stage target inference. Previous development target baselines are exposed but cannot select a component.',
        'calibration': 'For each method separately, source calibration half only; lowest .001-grid threshold with equal-source-region background FPR <= budget. Primary2%; fixed secondary1%/.5%, never choose best target operating point.',
        'fpr_budgets': list(BUDGETS), 'primary_fpr_budget': .02,
        'scale_diagnostic_checks': {'min_small_gain_each_seed': .02, 'positive_regions_both_seeds': 2,
             'min_f1_delta_each_seed': -.005, 'max_macro_fpr_delta_each_seed': .002,
             'max_region_fpr_delta': .01, 'max_candidate_macro_fpr_each_seed': .03,
             'max_candidate_region_fpr': .05, 'min_strict_iou_delta_each_seed': 0.,
             'min_small_gain_vs_constant_each_seed': .01},
        'scope': 'Three historically exposed development domains x2 seeds. Not all-six-region stability, new blind data, mine field validation or architecture novelty.',
        'retention': 'Six selected full weights, temporary optimizer resume checkpoint deleted per completed fit, FP32 compact target grids, small calibration histograms, protocol/state/results. D only; original data read only.',
        'production_changed': False, 'official_test_predictions': 0,
        'new_mine_predictions': 0, 'external_calls': 0, 'downloads': 0,
    }
    base.write(path, result)
    return result


def compact_whole_view(image, label):
    image = np.asarray(image, np.float32)
    if image.ndim != 3 or image.shape[0] != 4 or not np.isfinite(image).all():
        raise ValueError('Expected actual finite four-band CHW image')
    side = image.shape[-1]
    if image.shape[-2] != side or side % 2:
        raise ValueError('Expected even square parent')
    lo = image.min((1, 2), keepdims=True)
    span = image.max((1, 2), keepdims=True) - lo
    image = np.divide(image-lo, span, out=np.zeros_like(image), where=span > 0)
    interpolation = cv2.INTER_AREA if side > 64 else cv2.INTER_LINEAR
    xx = cv2.resize(image.transpose(1, 2, 0), (64, 64), interpolation=interpolation).transpose(2, 0, 1)
    yy = cv2.resize(np.asarray(label, np.float32).reshape(side, side), (64, 64), interpolation=interpolation)
    return xx[None].astype(np.float16), yy[None, None].astype(np.float16)


def quarantine_train(entries, val_hashes):
    grouped = defaultdict(list)
    for row in entries:
        grouped[row['sha256']].append(row)
    keep, removed = [], []
    for digest, rows in grouped.items():
        reason = None
        if digest in val_hashes:
            reason = 'matches_official_val'
        elif len({r['group'] for r in rows}) > 1:
            reason = 'cross_region_duplicate'
        elif len({r['label_sha256'] for r in rows}) > 1:
            reason = 'conflicting_labels'
        ordered = sorted(rows, key=lambda r: (not r['legacy_selected'], r['index']))
        if reason:
            removed.extend({**r, 'reason': reason} for r in rows)
        else:
            keep.append(ordered[0])
            removed.extend({**r, 'reason': 'same_region_duplicate'} for r in ordered[1:])
    keep.sort(key=lambda r: (r['group'], not r['legacy_selected'], r['index']))
    return keep, removed


def prepare_data(out):
    from data_paths import resolve_data_root
    old = base.read(PRIOR / 'data_manifest.json')
    root = Path(resolve_data_root('native'))
    legacy = {(r['region'], r['index']): r['sha256'] for r in old['kept_train']}
    legacy_labels = {(r['region'], idx): hy for r in old['regions']
                     for idx,hy in zip(r['splits']['train']['indices'],r['splits']['train']['label_parent_sha256'])}
    if len(legacy)!=3072 or set(old['regional_train_counts'].values())!={512}:
        raise ValueError('Expected original six512-parent control')
    entries, regions, data, invalid, val_hashes = [], [], {}, [], set()
    pools = {}
    for g, (name, gsd) in enumerate(base.REGIONS):
        folder = root / name
        tx = np.load(folder / 'train_images.npy', mmap_mode='r')
        ty = np.load(folder / 'train_labels.npy', mmap_mode='r')
        vx = np.load(folder / 'val_images.npy', mmap_mode='r')
        vy = np.load(folder / 'val_labels.npy', mmap_mode='r')
        for xx, yy in ((tx, ty), (vx, vy)):
            if xx.ndim != 4 or xx.shape[1] != 4 or yy.shape != (len(xx), 1, xx.shape[2], xx.shape[3]):
                raise ValueError('Native image/label layout changed')
        pools[g] = (tx, ty)
        val = copy.deepcopy(old['regions'][g]['splits']['val'])
        if list(vx.shape) != val['image_shape']:
            raise ValueError('Validation shape changed')
        all_val = [base.array_sha(row) for row in vx]
        val_hashes.update(all_val)
        views, labels = [], []
        for idx, hx, hy in zip(val['indices'], val['raw_parent_sha256'], val['label_parent_sha256']):
            if all_val[idx] != hx or base.array_sha(vy[idx]) != hy:
                raise ValueError('Frozen selected validation row changed')
            views.append(base.make_views(vx[idx])[0]); labels.append(vy[idx, 0].astype(np.uint8))
        data[g] = {'views': np.stack(views), 'labels': labels, 'indices': val['indices'], 'gsd': gsd,
                   'components': [base.gt_components(y, gsd) for y in labels]}
        rows = []
        for idx in range(len(tx)):
            x, y = np.asarray(tx[idx]), np.asarray(ty[idx])
            if not np.isfinite(x).all() or not np.isfinite(y).all() or not ((y == 0) | (y == 1)).all():
                invalid.append({'group': g, 'index': idx, 'reason': 'nonfinite_or_nonbinary'}); continue
            row = {'group': g, 'index': idx, 'sha256': base.array_sha(x), 'label_sha256': base.array_sha(y),
                   'legacy_selected': (name, idx) in legacy}
            if row['legacy_selected'] and row['sha256'] != legacy[(name, idx)]:
                raise ValueError('Legacy train image changed')
            if row['legacy_selected'] and row['label_sha256'] != legacy_labels[(name, idx)]:
                raise ValueError('Legacy train label changed')
            rows.append(row); entries.append(row)
        train = {'image_path': str(folder/'train_images.npy'), 'label_path': str(folder/'train_labels.npy'),
                 'image_shape': list(tx.shape), 'indices': [r['index'] for r in rows],
                 'raw_parent_sha256': [r['sha256'] for r in rows], 'label_parent_sha256': [r['label_sha256'] for r in rows]}
        regions.append({'region': name, 'gsd_m': gsd, 'splits': {'train': train, 'val': val},
                        'all_official_val_image_sha256': all_val})
        print('AUDIT_FULL', name, 'train', len(tx), 'all_val_hashes', len(vx), 'selected_val', len(labels), flush=True)
    kept, removed = quarantine_train(entries, val_hashes)
    keys = {(base.REGIONS[r['group']][0], r['index']) for r in kept}
    if not set(legacy).issubset(keys):
        base.write(Path(out)/'audit_blocker.json', {'missing_legacy': [list(v) for v in set(legacy)-keys],
                                                  'invalid': invalid, 'removed': removed})
        raise ValueError('Prior512 rows excluded by broader audit; need corrected baseline before training')
    # Compact all-parent views have a singleton view axis, preserving old monitor API.
    xx = np.empty((len(kept), 1, 4, 64, 64), np.float16)
    yy = np.empty((len(kept), 1, 1, 64, 64), np.float16)
    groups = np.array([r['group'] for r in kept], np.int64)
    for j, row in enumerate(kept):
        x, y = pools[row['group']]
        xx[j], yy[j] = compact_whole_view(x[row['index']], y[row['index']])
    for g in range(6):
        prefix = [r['index'] for r in kept if r['group'] == g][:512]
        expected = [r['index'] for r in old['kept_train'] if r['region'] == base.REGIONS[g][0]]
        if prefix != expected:
            raise ValueError('Legacy monitor parent ordering changed')
    manifest = {'registered_at': base.now(), 'regions': regions, 'kept_train': kept,
                'removed_train': removed, 'invalid_train': invalid,
                'regional_train_counts': dict(Counter(base.REGIONS[g][0] for g in groups)),
                'legacy_kept_train_count': len(legacy), 'legacy_parent_prefix_unchanged': True,
                'all_official_val_images_hash_audited': sum(len(r['all_official_val_image_sha256']) for r in regions),
                'test_files_opened': 0, 'compact_train_bytes': xx.nbytes+yy.nbytes}
    path = Path(out)/'data_manifest.json'
    if path.exists():
        previous = base.read(path)
        if {k:v for k,v in previous.items() if k!='registered_at'} != {k:v for k,v in manifest.items() if k!='registered_at'}:
            raise ValueError('Frozen full-data manifest changed')
    else:
        base.write(path, manifest)
    return xx, yy, groups, data, manifest


def legacy_augmentation(groups, held, rng):
    ids = base.sample_batch(groups, held, rng, 4)
    np.where(rng.random(len(ids)) < .5, 0, rng.integers(1, 5, len(ids)))
    k, flip = int(rng.integers(0, 4)), bool(rng.integers(0, 2))
    return ids, k, flip


def source_backbone_choice(full_selection, prior_selection):
    # Only named source statistics are read; neither target metrics nor thresholds.
    return 'full' if full_selection['source_macro_f1'] > prior_selection['source_macro_f1'] else 'prior512'


def train(directory, x, y, groups, data, held, seed, protocol):
    import torch
    import torch.nn.functional as F
    import segmentation_models_pytorch as smp
    fold_seed = seed+held*100
    torch.manual_seed(fold_seed); torch.cuda.manual_seed_all(fold_seed)
    model = smp.UnetPlusPlus(encoder_name='resnet34', encoder_weights=None, in_channels=4, classes=1).cuda()
    initial = hashlib.sha256()
    for value in model.state_dict().values(): initial.update(value.detach().cpu().numpy().tobytes())
    old_dir = PRIOR / directory.parent.name / directory.name
    original = base.read(short.PRIOR / directory.parent.name / directory.name / 'result.json')
    if initial.hexdigest() != original['training']['initial_state_sha256']:
        raise ValueError('Initialization differs from the paired512 control')
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.0001)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 1200, eta_min=.00001)
    rng = np.random.default_rng(fold_seed); paired = np.random.default_rng(fold_seed)
    legacy_groups = np.repeat(np.arange(6), 512)
    visits = np.zeros(6, np.int64); seen = np.zeros(len(groups), bool)
    schedule = bytearray(); paired_schedule = bytearray(); curve = []; losses = []; step0 = 0
    resume = directory/'resume_training.pth'; best_path = directory/'source_selected.pth'
    seconds = 0.; phase = 'short'
    if resume.exists():
        cp = torch.load(resume, map_location='cpu', weights_only=True)
        if cp['protocol_sha256'] != base.sha(directory.parents[1]/'protocol.json'):
            raise ValueError('Resume protocol changed')
        model.load_state_dict(cp['model']); phase = cp['phase']
        if phase == 'continuation': scheduler = short.start_continuation(optimizer)
        optimizer.load_state_dict(cp['optimizer']); scheduler.load_state_dict(cp['scheduler'])
        rng.bit_generator.state = cp['rng']; paired.bit_generator.state = cp['paired_rng']
        visits = np.array(cp['visits'], np.int64); seen = np.array(cp['seen'], bool)
        schedule = bytearray(cp['schedule']); paired_schedule = bytearray(cp['paired_schedule'])
        curve = cp['curve']; losses = cp['losses']; step0 = cp['step']; seconds = cp['seconds']
        torch.set_rng_state(cp['torch_rng']); torch.cuda.set_rng_state_all(cp['cuda_rng'])
        if not best_path.exists(): raise ValueError('Resume selected state missing')
        print('RESUME_FULL', base.REGIONS[held][0], seed, step0, flush=True)
    model.train(); torch.cuda.synchronize(); started = time.perf_counter()
    for step in range(step0, protocol['updates']):
        ids, _, _ = legacy_augmentation(groups, held, rng)
        old_ids, k, flip = legacy_augmentation(legacy_groups, held, paired)
        schedule.extend(np.asarray([*ids, k, int(flip)], np.int64).tobytes())
        paired_schedule.extend(np.asarray([*old_ids, k, int(flip)], np.int64).tobytes())
        visits += np.bincount(groups[ids], minlength=6); seen[ids] = True
        a = x[ids, 0].float().rot90(k, (-2,-1)); b = y[ids, 0].float().rot90(k, (-2,-1))
        if flip: a, b = a.flip(-1), b.flip(-1)
        optimizer.zero_grad(set_to_none=True); logits = model(a); p = logits.sigmoid()
        dice = (1-(2*(p*b).sum((2,3))+1e-6)/(p.sum((2,3))+b.sum((2,3))+1e-6)).mean()
        loss = F.binary_cross_entropy_with_logits(logits, b)+.5*dice
        if not torch.isfinite(loss): raise ValueError('Nonfinite full training loss')
        loss.backward(); optimizer.step(); scheduler.step(); losses.append(float(loss.detach()))
        update = step+1
        if update % 300 == 0:
            print('TRAIN_FULL', base.REGIONS[held][0], seed, update, 'loss100', round(float(np.mean(losses[-100:])), 5), flush=True)
        if update in protocol['checkpoints']:
            model.eval()
            if update == 1200 and hashlib.sha256(paired_schedule).hexdigest() != original['training']['parent_schedule_sha256']:
                raise ValueError('Legacy1200 augmentation schedule not reproduced')
            if update == 4800 and hashlib.sha256(paired_schedule).hexdigest() != base.read(old_dir/'result.json')['training']['sampling_sha256']:
                raise ValueError('Legacy4800 augmentation schedule not reproduced')
            monitor = short.source_monitor(model, x, y, groups, data, held)
            curve.append({'update': update, 'model_state_sha256': short.state_hash(model),
                          'mean_train_loss_last100': float(np.mean(losses[-100:])),
                          'source_selection': monitor, 'completed_at': base.now()})
            if short.choose_source_checkpoint(curve)['update'] == update:
                short.save_torch(best_path, {k:v.detach().cpu() for k,v in model.state_dict().items()})
            base.write(directory/'source_curve.json', curve)
            print('SOURCE_FULL', base.REGIONS[held][0], seed, update, 'macroF1', round(monitor['macro_f1'],5), flush=True)
            if update == 1200:
                scheduler = short.start_continuation(optimizer); phase = 'continuation'
            torch.cuda.synchronize(); seconds += time.perf_counter()-started
            short.save_torch(resume, {'protocol_sha256': base.sha(directory.parents[1]/'protocol.json'),
                'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(),
                'phase': phase, 'rng': rng.bit_generator.state, 'paired_rng': paired.bit_generator.state,
                'visits': visits.tolist(), 'seen': seen.tolist(), 'schedule': bytes(schedule),
                'paired_schedule': bytes(paired_schedule), 'curve': curve, 'losses': losses, 'step': update,
                'torch_rng': torch.get_rng_state(), 'cuda_rng': torch.cuda.get_rng_state_all(), 'seconds': seconds})
            model.train(); started = time.perf_counter()
    if visits.tolist() != [0 if g==held else 19200 for g in range(6)]:
        raise ValueError('Source balancing/target exclusion failed')
    return model.eval(), curve, {'updates':4800, 'visits_per_region':visits.tolist(),
        'unique_seen_per_region':[int(seen[groups==g].sum()) for g in range(6)],
        'available_per_region': [int((groups==g).sum()) for g in range(6)],
        'source_parent_schedule_sha256': hashlib.sha256(schedule).hexdigest(),
        'paired_legacy_schedule_sha256': hashlib.sha256(paired_schedule).hexdigest(),
        'initial_state_sha256': initial.hexdigest(), 'seconds_including_monitor': seconds,
        'augmentations_exactly_paired': True}


def selection_record(directory, curve, held):
    best = short.choose_source_checkpoint(curve)
    return {'selected_at': base.now(), 'selected_update': best['update'],
            'source_macro_f1': best['source_selection']['macro_f1'],
            'selected_state_sha256': best['model_state_sha256'],
            'weights_sha256': base.sha(directory/'source_selected.pth'),
            'selection_regions': [base.REGIONS[g][0] for g in range(6) if g!=held],
            'target_used': False, 'calibration_half_used_for_selection': False}


def compare_summary(out, protocol):
    rows = [base.read(p) for p in sorted(Path(out).glob('fold_*/seed*/result.json'))]
    if len(rows)!=6: raise ValueError('Require all six full-source fits')
    names = ('prior512', 'full_source')
    metrics = ('f1','small_cover50','small_iou50','background_fpr')
    macro = {name:{m:float(np.mean([r['methods'][name]['held_out'][m] for r in rows])) for m in metrics} for name in names}
    source = [{'region':r['held_out'], 'seed':r['seed'],
        'prior512_source_f1':r['backbone_choice']['prior512_source_f1'],
        'full_source_f1':r['selection']['source_macro_f1'], 'scale_backbone':r['backbone_choice']['chosen'],
        'selected_update':r['selection']['selected_update'],
        'target_delta':{m:r['methods']['full_source']['held_out'][m]-r['methods']['prior512']['held_out'][m] for m in metrics}}
        for r in rows]
    for p,digest in protocol['frozen_assets'].items():
        if base.sha(p)!=digest: raise ValueError('Frozen asset changed')
    result = {'completed_at':base.now(), 'status':'completed', 'fits':6, 'updates':28800,
        'target_macro':macro, 'source_choices_and_target_diagnostics':source,
        'source_preferred_full_fits':sum(r['backbone_choice']['chosen']=='full' for r in rows),
        'production_changed':False, 'official_test_predictions':0,'new_mine_predictions':0,'external_calls':0}
    base.write(Path(out)/'results.json',result)
    return result


def setup_runtime():
    import torch
    torch.set_num_threads(4); torch.backends.cudnn.benchmark=False; torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
    if not torch.cuda.is_available(): raise RuntimeError('CUDA unavailable')
    return torch


def run_training(out):
    torch = setup_runtime(); protocol = register(out)
    x,y,groups,data,manifest = prepare_data(out)
    x,y = torch.from_numpy(x).cuda(),torch.from_numpy(y).cuda()
    base.write(Path(out)/'runtime.json',{'torch':torch.__version__,'cuda':torch.version.cuda,
        'device':torch.cuda.get_device_name(0),'threads':4,'deterministic_cudnn':True,'tf32':False,'cache_drive':'D:'})
    completed = 0
    for held in protocol['held_indices']:
        region = base.REGIONS[held][0]
        for seed in protocol['seeds']:
            directory = Path(out)/f'fold_{held}_{region}'/f'seed{seed}'; directory.mkdir(parents=True,exist_ok=True)
            dest = directory/'result.json'
            if dest.exists():
                if base.sha(directory/'source_selected.pth')!=base.read(dest)['selected_weights_sha256']:
                    raise ValueError('Finished model changed')
                completed+=1;continue
            base.write(Path(out)/'state.json',{'status':'running','stage':'full_training','updated_at':base.now(),
                'held_out':region,'seed':seed,'completed':completed,'total':6,'production_changed':False})
            model,curve,training = train(directory,x,y,groups,data,held,seed,protocol)
            selection = selection_record(directory,curve,held)
            base.write(directory/'checkpoint_selection.json',selection)
            old = base.read(PRIOR/directory.parent.name/directory.name/'result.json')
            chosen = source_backbone_choice(selection,old['selection'])
            choice = {'chosen_at':base.now(),'chosen':chosen,'full_source_f1':selection['source_macro_f1'],
                'prior512_source_f1':old['selection']['source_macro_f1'],'target_used':False,
                'chosen_weight':str(directory/'source_selected.pth') if chosen=='full' else str(PRIOR/directory.parent.name/directory.name/'source_selected.pth')}
            base.write(directory/'scale_backbone_choice.json',choice)
            # Both source-only checkpoint and backbone decisions precede target inference.
            model.load_state_dict(torch.load(directory/'source_selected.pth',map_location='cpu',weights_only=True));model.eval()
            if short.state_hash(model)!=selection['selected_state_sha256']:raise ValueError('Selected state differs')
            method,grids = short.evaluate(model,data,held)
            cache = {'target':grids[held],'original_target_indices':np.array(data[held]['indices'],np.int64)}
            cache.update({f'source_calibration_{g}':grids[g] for g in range(6) if g!=held})
            np.savez_compressed(directory/'probability_grids.npz',**cache)
            base.write(dest,{'held_index':held,'held_out':region,'seed':seed,'completed_at':base.now(),
                'training':training,'selection':selection,'backbone_choice':choice,'source_curve':curve,
                'methods':{'prior512':old['methods']['selected_source'],'full_source':method},
                'selected_weights_sha256':base.sha(directory/'source_selected.pth'),
                'probability_grids_sha256':base.sha(directory/'probability_grids.npz'),
                'target_inferred_after_source_decisions':True})
            (directory/'resume_training.pth').unlink(missing_ok=True)
            del model,grids;torch.cuda.empty_cache();completed+=1
            print('FULL_COMPLETED',completed,'/6',region,seed,'scale_backbone',chosen,flush=True)
    result = compare_summary(out,protocol)
    base.write(Path(out)/'state.json',{'status':'training_completed','stage':'scale_revalidation_pending',
                                     'updated_at':base.now(),'completed':6,'total':6,'production_changed':False})
    print('FULL_TRAINING_COMPLETE',result['target_macro'],flush=True)
    del x,y;torch.cuda.empty_cache()
    return data,manifest


def component_grids(model, data):
    started=time.perf_counter()
    full=short.forward(model,data['views'][:,0]);tta=full.copy()
    for kind in (1,2,3,4):tta+=short.forward(model,data['views'][:,0],kind=kind)
    tta/=5
    local=short.forward(model,data['views'][:,1:].reshape(-1,4,64,64)).reshape(-1,4,64,64)
    grids={'global_grid':full,'local_grids':local,'tta_grid':tta}
    native={'global':[],'local':[],'tta5':[]}
    side=len(data['labels'][0]);half=side//2
    for p,q,t in zip(full,local,tta):
        lp=np.empty((side,side),np.float32)
        for i in range(4):
            y,x=(i//2)*half,(i%2)*half
            lp[y:y+half,x:x+half]=cv2.resize(q[i],(half,half),interpolation=cv2.INTER_LINEAR)
        native['global'].append(cv2.resize(p,(side,side),interpolation=cv2.INTER_LINEAR))
        native['local'].append(lp)
        native['tta5'].append(cv2.resize(t,(side,side),interpolation=cv2.INTER_LINEAR))
    return native,grids,time.perf_counter()-started


def source_histogram(probabilities, labels):
    histogram=np.zeros(1001,np.int64)
    if len(probabilities)!=len(labels):raise ValueError('Calibration row counts differ')
    for p,y in zip(probabilities,labels):
        p,y=np.asarray(p),np.asarray(y)
        if p.shape!=y.shape or not np.isfinite(p).all() or np.any(p<0) or np.any(p>1):
            raise ValueError('Invalid calibration probability grid')
        bins=np.minimum((p[y==0].astype(np.float64)*1000).astype(np.int32),1000)
        histogram+=np.bincount(bins,minlength=1001)
    if not histogram.sum():raise ValueError('No calibration background')
    return histogram


def threshold_from_histograms(histograms, held, budget):
    gate.source_keys(histograms,held)
    if not 0<budget<1:raise ValueError('Invalid source FPR budget')
    tails=[]
    for g in sorted(histograms):
        h=np.asarray(histograms[g])
        if h.shape!=(1001,) or not np.issubdtype(h.dtype,np.integer) or np.any(h<0) or h.sum()==0:
            raise ValueError('Invalid source histogram')
        tails.append(np.r_[np.cumsum(h[::-1])[::-1]/h.sum(),0.])
    macro=np.mean(tails,axis=0);idx=int(np.flatnonzero(macro<=budget)[0])
    return idx/1000,{'source_regions':[base.REGIONS[g][0] for g in sorted(histograms)],
        'source_macro_fpr':float(macro[idx]),'source_region_fprs':[float(t[idx]) for t in tails],
        'held_out_in_calibration':False}


def scale_metrics(components, heads, data, manifest, held):
    maps={name:{} for name in gate.METHODS}
    for g,d in data.items():
        positions=np.arange(len(d['labels'])) if g==held else gate.meta_split(len(d['labels']),g)[1]
        raw=np.load(manifest['regions'][g]['splits']['val']['image_path'],mmap_mode='r')
        for name in gate.METHODS:maps[name][g]=[]
        for i in positions:
            p,q=components[g]['global'][i],components[g]['local'][i]
            maps['tta5'][g].append(components[g]['tta5'][i])
            maps['mean_multiscale5'][g].append(.5*(p+q))
            features=gate.feature_map(raw[d['indices'][i]],p,q,d['gsd'])
            for arm in gate.HEADS:
                maps[arm][g].append(gate.fuse(features,p,q,heads[arm])[0])
    results={};hist_cache={}
    for name in gate.METHODS:
        histograms={}
        for g in range(6):
            if g==held:continue
            positions=gate.meta_split(len(data[g]['labels']),g)[1]
            histograms[g]=source_histogram(maps[name][g],[data[g]['labels'][i] for i in positions])
            hist_cache[f'{name}_source_{g}']=histograms[g]
        pairs=list(zip(maps[name][held],data[held]['labels'],data[held]['components']))
        operating={}
        for budget in BUDGETS:
            threshold,calibration=threshold_from_histograms(histograms,held,budget)
            rows=[base.parent_counts(p,y,c,threshold) for p,y,c in pairs]
            operating[f'{budget:g}']={'threshold':threshold,'calibration':calibration,
                                      'held_out':base.aggregate_counts(rows),'per_parent_counts':rows}
        fixed=[base.parent_counts(p,y,c,.5) for p,y,c in pairs]
        results[name]={**operating['0.02'],'held_out_at_fixed_05':base.aggregate_counts(fixed),
                       'operating_points':operating}
    return results,hist_cache


def scale_summary(out, protocol):
    rows=[base.read(p) for p in sorted(Path(out).glob('fold_*/seed*/scale_result.json'))]
    if len(rows)!=6:raise ValueError('Require six scale fits')
    metrics=('f1','small_cover50','small_iou50','background_fpr')
    comparisons={}
    for budget in BUDGETS:
        key=f'{budget:g}';comparisons[key]={}
        for comparator in ('tta5','constant_gate','mean_multiscale5','feature_gate_no_gsd'):
            by_seed=[];regional=[]
            for seed in protocol['seeds']:
                selected=[r for r in rows if r['seed']==seed]
                deltas=[]
                for r in selected:
                    a=r['methods']['feature_gate']['operating_points'][key]['held_out']
                    b=r['methods'][comparator]['operating_points'][key]['held_out']
                    delta={m:a[m]-b[m] for m in metrics};deltas.append(delta)
                    regional.append({'region':r['held_out'],'seed':seed,'delta':delta})
                by_seed.append({'seed':seed,'macro_delta':{m:float(np.mean([d[m] for d in deltas])) for m in metrics},
                    'candidate_macro':{m:float(np.mean([r['methods']['feature_gate']['operating_points'][key]['held_out'][m] for r in selected])) for m in metrics},
                    'worst_candidate_fpr':max(r['methods']['feature_gate']['operating_points'][key]['held_out']['background_fpr'] for r in selected),
                    'worst_fpr_increase':max(d['background_fpr'] for d in deltas)})
            comparisons[key][comparator]={'by_seed':by_seed,'regions':regional}
    primary=comparisons['0.02'];thresholds=protocol['scale_diagnostic_checks']
    positive=sum(all(next(r for r in primary['tta5']['regions'] if r['region']==base.REGIONS[g][0] and r['seed']==seed)['delta']['small_cover50']>0
                     for seed in protocol['seeds']) for g in HELD)
    tt=primary['tta5']['by_seed'];cc=primary['constant_gate']['by_seed']
    checks={
        'small_gain_each_seed':all(r['macro_delta']['small_cover50']>=thresholds['min_small_gain_each_seed'] for r in tt),
        'two_regions_positive_both_seeds':positive>=thresholds['positive_regions_both_seeds'],
        'f1_preserved_vs_tta':all(r['macro_delta']['f1']>=thresholds['min_f1_delta_each_seed'] for r in tt),
        'strict_iou_preserved_vs_tta':all(r['macro_delta']['small_iou50']>=thresholds['min_strict_iou_delta_each_seed'] for r in tt),
        'macro_fpr_delta_bounded':all(r['macro_delta']['background_fpr']<=thresholds['max_macro_fpr_delta_each_seed'] for r in tt),
        'region_fpr_delta_bounded':all(r['worst_fpr_increase']<=thresholds['max_region_fpr_delta'] for r in tt),
        'candidate_macro_fpr_bounded':all(r['candidate_macro']['background_fpr']<=thresholds['max_candidate_macro_fpr_each_seed'] for r in tt),
        'candidate_region_fpr_bounded':all(r['worst_candidate_fpr']<=thresholds['max_candidate_region_fpr'] for r in tt),
        'beats_constant_small':all(r['macro_delta']['small_cover50']>=thresholds['min_small_gain_vs_constant_each_seed'] for r in cc),
        'f1_preserved_vs_constant':all(r['macro_delta']['f1']>=thresholds['min_f1_delta_each_seed'] for r in cc),
        'fpr_preserved_vs_constant':all(r['macro_delta']['background_fpr']<=thresholds['max_macro_fpr_delta_each_seed'] for r in cc),
    }
    macros={f'{budget:g}':{name:{m:float(np.mean([r['methods'][name]['operating_points'][f'{budget:g}']['held_out'][m] for r in rows])) for m in metrics}
                                      for name in gate.METHODS} for budget in BUDGETS}
    for p,digest in protocol['frozen_assets'].items():
        if base.sha(p)!=digest:raise ValueError('Frozen asset changed')
    result={'completed_at':base.now(),'status':'completed','fits':6,'head_fits':18,
        'target_macro_by_source_budget':macros,'comparisons':comparisons,
        'diagnostic_checks':checks,'diagnostic_gate_passed':all(checks.values()),
        'regions_positive_both_seeds':positive,'source_budgets_not_target_selected':True,
        'scope':protocol['scope'],'production_changed':False,'official_test_predictions':0,
        'new_mine_predictions':0,'external_calls':0}
    base.write(Path(out)/'scale_results.json',result)
    base.write(Path(out)/'state.json',{'status':'completed','stage':'full_and_scale_completed','completed_at':base.now(),
        'full_training_fits':6,'scale_fits':6,'diagnostic_gate_passed':all(checks.values()),'production_changed':False})
    return result


def load_validation(out):
    manifest=base.read(Path(out)/'data_manifest.json');data={}
    for g,row in enumerate(manifest['regions']):
        v=row['splits']['val'];raw=np.load(v['image_path'],mmap_mode='r');gt=np.load(v['label_path'],mmap_mode='r')
        views=[];labels=[]
        for i,hx,hy in zip(v['indices'],v['raw_parent_sha256'],v['label_parent_sha256']):
            if base.array_sha(raw[i])!=hx or base.array_sha(gt[i])!=hy:raise ValueError('Validation row changed')
            views.append(base.make_views(raw[i])[0]);labels.append(gt[i,0].astype(np.uint8))
        data[g]={'views':np.stack(views),'labels':labels,'indices':v['indices'],'gsd':row['gsd_m'],
                 'components':[base.gt_components(y,row['gsd_m']) for y in labels]}
    return data,manifest


def run_scales(out, data=None, manifest=None):
    torch=setup_runtime();protocol=register(out)
    if not (Path(out)/'results.json').exists():raise ValueError('Complete full-source training first')
    if data is None:data,manifest=load_validation(out)
    import segmentation_models_pytorch as smp
    completed=0
    for held in protocol['held_indices']:
        for seed in protocol['seeds']:
            directory=Path(out)/f'fold_{held}_{base.REGIONS[held][0]}'/f'seed{seed}'
            dest=directory/'scale_result.json'
            if dest.exists():
                previous=base.read(dest)
                if base.sha(directory/'scale_heads.json')!=previous['head_snapshot_sha256']:raise ValueError('Finished scale heads changed')
                completed+=1;continue
            base.write(Path(out)/'state.json',{'status':'running','stage':'scale_revalidation','updated_at':base.now(),
                'held_out':base.REGIONS[held][0],'seed':seed,'completed':completed,'total':6,'production_changed':False})
            full=base.read(directory/'result.json');choice=base.read(directory/'scale_backbone_choice.json')
            old=base.read(PRIOR/directory.parent.name/directory.name/'result.json')
            chosen_result=full if choice['chosen']=='full' else old
            expected=chosen_result['selection']
            if base.sha(choice['chosen_weight'])!=expected['weights_sha256']:raise ValueError('Chosen backbone weight identity changed')
            model=smp.UnetPlusPlus(encoder_name='resnet34',encoder_weights=None,in_channels=4,classes=1).cuda()
            model.load_state_dict(torch.load(choice['chosen_weight'],map_location='cpu',weights_only=True));model.eval()
            digest=short.state_hash(model)
            if digest!=expected['selected_state_sha256']:raise ValueError('Chosen model state differs')
            components={};times={}
            for g in range(6):
                if g==held:continue
                components[g],_,times[base.REGIONS[g][0]]=component_grids(model,data[g])
            pools,split_rows=gate.make_source_pools(data,components,manifest,held,seed)
            heads={};training={};hp=protocol['scale_training']
            for arm in gate.HEADS:
                heads[arm],training[arm]=gate.fit_gate(pools,held,seed,arm,hp)
                heads[arm]['base_model_state_sha256']=digest
            if len({r['sample_schedule_sha256'] for r in training.values()})!=1:raise ValueError('Head sampling not paired')
            snapshot={'frozen_at':base.now(),'backbone_choice':choice,'base_state_sha256':digest,
                'weights_sha256':expected['weights_sha256'],'coefficients':heads,'head_training':training,
                'source_meta_split':{base.REGIONS[g][0]:v for g,v in split_rows.items()},'target_used':False}
            base.write(directory/'scale_heads.json',snapshot)
            components[held],target_grids,times[base.REGIONS[held][0]]=component_grids(model,data[held])
            if short.state_hash(model)!=digest:raise ValueError('Scale inference mutated base model')
            methods,histograms=scale_metrics(components,heads,data,manifest,held)
            baseline=full['methods']['full_source'] if choice['chosen']=='full' else old['methods']['selected_source']
            if methods['tta5']['threshold']!=baseline['threshold'] or methods['tta5']['held_out']['counts']!=baseline['held_out']['counts'] or methods['tta5']['held_out_at_fixed_05']['counts']!=baseline['held_out_at_fixed_05']['counts']:
                raise ValueError('Chosen-backbone TTA baseline was not reproduced')
            target_grids['original_target_indices']=np.array(data[held]['indices'],np.int64)
            np.savez_compressed(directory/'scale_target_grids.npz',**target_grids)
            np.savez_compressed(directory/'scale_calibration_histograms.npz',**histograms)
            base.write(dest,{'held_index':held,'held_out':base.REGIONS[held][0],'seed':seed,'completed_at':base.now(),
                'backbone_choice':choice,'methods':methods,'head_snapshot_sha256':base.sha(directory/'scale_heads.json'),
                'target_grids_sha256':base.sha(directory/'scale_target_grids.npz'),
                'calibration_histograms_sha256':base.sha(directory/'scale_calibration_histograms.npz'),
                'base_state_sha256':digest,'baseline_reproduced':True,'prediction_seconds':times,
                'target_scale_inference_after_heads_persisted':True})
            del model,components,pools,target_grids;torch.cuda.empty_cache();completed+=1
            print('FULL_SCALE_COMPLETED',completed,'/6',base.REGIONS[held][0],seed,flush=True)
    result=scale_summary(out,protocol)
    print('FULL_SCALE_COMPLETE',{'passed':result['diagnostic_gate_passed'],'macro':result['target_macro_by_source_budget']},flush=True)


if __name__=='__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True)
    parser.add_argument('--stage',choices=('register','train','scales','all'),default='all')
    args=parser.parse_args()
    if args.stage=='register':register(args.output)
    elif args.stage=='train':run_training(args.output)
    elif args.stage=='scales':run_scales(args.output)
    else:
        validation,manifest=run_training(args.output)
        run_scales(args.output,validation,manifest)
