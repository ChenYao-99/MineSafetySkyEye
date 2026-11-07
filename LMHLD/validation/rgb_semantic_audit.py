"""Read-only attribution of the stronger RGB model's partial-reference errors.

Existing reviewed polygons are an imperfect semantic taxonomy, not new expert
labels. Predictions never create or expand supervision. All cases DEVELOPMENT.
"""
from pathlib import Path
import sys
import numpy as np
import cv2
sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parent)]
from runtime_paths import configure_runtime_paths
configure_runtime_paths()
import rgb_transfer_study as transfer
exp = transfer.exp; base = exp.base
PRIOR = exp.ROOT/'experiments/rgb_transfer_20261020'
DATA = exp.ROOT/'datasets/mine_benchmark_20261002'
REVIEW = DATA/'review_v2_20261003/review_manifest.json'
XJ = DATA/'development/Xinjing/reference_v1_20261004/label_reference.json'
WORK_ACCEPTANCE = exp.ROOT/'datasets/slats_operation_development_20261005/work_reference_acceptance.json'


def category(region_id):
    if 'dump' in region_id: return 'dump_access_only'
    if 'plant' in region_id: return 'plant_access_surface'
    if 'bench' in region_id: return 'bench_or_bench_access'
    if 'platform' in region_id: return 'constructed_platform'
    if 'road' in region_id or 'access' in region_id: return 'road_or_constructed_access'
    raise ValueError('Unreviewed semantic category: '+region_id)


def normal_partitions(shape, normal, valid, definitions):
    normal = (np.asarray(normal) == 1) & np.asarray(valid, bool)
    used = np.zeros(shape, bool); parts = []
    for r in definitions:
        polygon = np.zeros(shape, np.uint8); cv2.fillPoly(polygon, [np.asarray(r['polygon_pixel_xy'], np.int32)], 1)
        mask = (polygon == 1) & normal
        if np.any(used & mask): raise ValueError('Overlapping normal polygons would double-count support')
        expected = r.get('retained_pixels', r.get('accepted_pixels'))
        if int(mask.sum()) != expected: raise ValueError('Normal region differs from its frozen accepted pixel count')
        used |= mask; parts.append((r['id'], category(r['id']), mask))
    if not np.array_equal(used, normal): raise ValueError('Normal polygons do not explain the whole explicit reviewed mask')
    return parts


def activation(mask, predictions):
    n = int(mask.sum())
    hits = [int((p & mask).sum()) for p in predictions]
    return {'pixels': n, 'seed42_activated': hits[0], 'seed43_activated': hits[1],
            'seed42_rate': hits[0]/n if n else None, 'seed43_rate': hits[1]/n if n else None,
            'both_activated': int((predictions[0] & predictions[1] & mask).sum()),
            'both_missed': int((~predictions[0] & ~predictions[1] & mask).sum()),
            'disagreement': int(((predictions[0] != predictions[1]) & mask).sum())}


def register(out):
    out = Path(out).resolve()
    if out.drive.upper() != 'D:' or not out.is_relative_to(exp.ROOT/'experiments'): raise ValueError('D experiments only')
    out.mkdir(parents=True, exist_ok=True)
    if (out/'protocol.json').exists():
        p = base.read(out/'protocol.json'); exp.old_rgb.check_assets(p); return p
    prior = base.read(PRIOR/'protocol.json'); frozen = {str(a): base.sha(a) for a in [Path(__file__), exp.ROOT/'tests/test_rgb_semantic_audit.py', PRIOR/'protocol.json', PRIOR/'results.json', PRIOR/'verification.json', REVIEW, XJ, WORK_ACCEPTANCE,
        exp.ROOT/'model_registry.json', exp.ROOT/'datasets/input_reserve_registry.json', DATA/'mine_case_registry.json']}
    for r in prior['mine_cases']:
        for name in ('image', 'label', 'normal'): frozen[r[name+'_path']] = r[name+'_sha256']
        for seed in (42, 43):
            d = PRIOR/'imagenet_normalized'/f'seed{seed}'
            for name in ('candidate.json', 'mine_results.json', r['mine']+'_probabilities.npz'): frozen[str(d/name)] = base.sha(d/name)
    p = {'registered_at': base.now(), 'frozen_assets': frozen, 'mine_cases': prior['mine_cases'], 'seeds': [42, 43],
         'scope': 'Read-only reuse of four historically seen DEVELOPMENT mines and existing prediction caches; no new prediction, strong label, independent mine or expert review.',
         'partition': 'Use only original project-reviewed polygon IDs intersected with the exact existing normal mask. Require disjoint masks, exact recorded per-region counts and complete coverage. Categories are names of reviewed constructed surfaces, not inferred geology or active-operation/safety truth.',
         'positive': 'Only conservative known cores. Eight-connected CORE pieces and pixel response are not complete landslide instances or incident recall. Unknown never becomes background.',
         'tile_edge_diagnostic': 'Compare existing positive/core and normal response within2 native pixels of a32-tile edge versus interior. Descriptive association only, not causal attribution.',
         'review_selection': 'One normal-region contact sheet, all17 original regions. Seed-common false activations displayed as red; no new annotation from those overlays.',
         'expected_work': 'Government annual mine-clearing class plus visible change remains weak expected-work context. Exclude it from binary accident-negative truth in this stage; it cannot establish accident absence or daily safe operations.',
         'training_next': 'If references validate, test whether the NEW pretrained representation supports the same bounded source-only weak-supervision head previously rejected on scratch RGB. Same fixed classifier and nested whole-mine isolation; do not simply lower a cutoff or claim new data.',
         'production_changed': False, 'new_model_predictions': 0, 'training_updates': 0, 'reserved_or_official_test_predictions': 0, 'external_calls': 0}
    exp.old_rgb.check_assets(p); base.write(out/'protocol.json', p); return p


def contact_sheet(out, panels):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    n = len(panels); fig, axes = plt.subplots((n+3)//4, 4, figsize=(13, 3*((n+3)//4)), dpi=140)
    for ax in np.asarray(axes).ravel(): ax.axis('off')
    for ax, (title, image, mask, common) in zip(np.asarray(axes).ravel(), panels):
        yy, xx = np.where(mask); y0, y1, x0, x1 = max(0, yy.min()-10), min(mask.shape[0], yy.max()+11), max(0, xx.min()-10), min(mask.shape[1], xx.max()+11)
        rgb = image[:3, y0:y1, x0:x1][[2, 1, 0]].transpose(1, 2, 0).astype(np.float32)
        lo, hi = np.percentile(rgb, [2, 98], axis=(0, 1)); rgb = np.clip((rgb-lo)/np.maximum(hi-lo, 1e-6), 0, 1)
        ax.imshow(rgb); local = mask[y0:y1, x0:x1]; overlay = np.zeros((*local.shape, 4), np.float32)
        overlay[common[y0:y1, x0:x1] & local] = [1, 0, 0, .65]
        ax.imshow(overlay); ax.contour(local, levels=[.5], colors=['cyan'], linewidths=.65)
        ax.set_title(title, fontsize=7)
    fig.suptitle('Existing weak normal surfaces: cyan=reviewed support; red=both-seed activations\nDevelopment diagnosis only; not expert labels or field safety truth', fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, .96)); fig.savefig(Path(out)/'normal_region_review.png'); plt.close(fig)


def run(out):
    out = Path(out); p = register(out); review = base.read(REVIEW); xj = base.read(XJ); rows = []; panels = []; totals = {}
    for case in p['mine_cases']:
        name = case['mine']; image, truth, normal, valid, _ = exp.old_rgb.load_mine(case)
        definitions = xj['normal_regions'] if name == 'CN_XJ2023' else review['cases'][name]['regions']
        partitions = normal_partitions(truth.shape, normal, valid, definitions)
        pred = []; q = []
        for seed in p['seeds']:
            d = PRIOR/'imagenet_normalized'/f'seed{seed}'; config = base.read(d/'candidate.json')
            if config['method'] != 'tta5': raise ValueError('Audit expects the frozen TTA5 selection')
            record = next(r for r in base.read(d/'mine_results.json') if r['case'] == name)
            path = d/(name+'_probabilities.npz')
            if base.sha(path) != record['probabilities_sha256']: raise ValueError('Frozen prediction cache changed')
            with np.load(path) as z: prob = z['tta5']
            mask = prob.astype(np.float64) >= config['threshold']
            if exp.rgb.partial_reference_counts(mask, truth, normal, valid) != record['methods']['tta5']: raise ValueError('Original weak-reference score did not reproduce')
            pred.append(mask); q.append(config['threshold'])
        pos = (truth == 1) & valid; neg = (normal == 1) & valid
        if np.any(pos & neg): raise ValueError('Positive/normal conflict')
        rr = []
        for identity, semantic, mask in partitions:
            scores = activation(mask, pred); rr.append({'id': identity, 'category': semantic, **scores})
            panels.append((f'{name}: {identity}\nactivated42/43={scores["seed42_rate"]:.1%}/{scores["seed43_rate"]:.1%}', image, mask, pred[0] & pred[1]))
            acc = totals.setdefault(semantic, dict(pixels=0, seed42_activated=0, seed43_activated=0, both_activated=0, regions=0, mines=[]))
            for k in ('pixels', 'seed42_activated', 'seed43_activated', 'both_activated'): acc[k] += scores[k]
            acc['regions'] += 1
            if name not in acc['mines']: acc['mines'].append(name)
        n, lab, stats, _ = cv2.connectedComponentsWithStats(pos.astype(np.uint8), 8)
        pieces = [{'piece': i, 'core_pixels': int(stats[i, cv2.CC_STAT_AREA]), **activation(lab == i, pred)} for i in range(1, n)]
        yy, xx = np.indices(truth.shape); edge = (np.minimum(yy % 32, 31-yy % 32) < 2) | (np.minimum(xx % 32, 31-xx % 32) < 2)
        row = {'case': name, 'source_thresholds': q, 'known_core': activation(pos, pred), 'reviewed_normal': activation(neg, pred), 'normal_regions': rr,
               'core_pieces_not_full_instances': pieces, 'positive_near_tile_edge': activation(pos & edge, pred), 'positive_tile_interior': activation(pos & ~edge, pred),
               'normal_near_tile_edge': activation(neg & edge, pred), 'normal_tile_interior': activation(neg & ~edge, pred),
               'unknown_pixels_excluded': int(((truth == 255) & valid).sum()), 'invalid_pixels_excluded': int((~valid).sum())}
        rows.append(row); print('SEMANTIC_AUDIT', name, 'normal_both', row['reviewed_normal']['both_activated'], 'core_both_missed', row['known_core']['both_missed'], flush=True)
    for v in totals.values():
        for seed in (42, 43): v[f'seed{seed}_rate'] = v[f'seed{seed}_activated']/v['pixels']
    acceptance = base.read(WORK_ACCEPTANCE)
    work = {k: {'pixels': v['pixels'], 'allowed_use': v['allowed_use'], 'binary_accident_negative_admitted': False} for k, v in acceptance['accepted'].items()}
    result = {'completed_at': base.now(), 'cases': rows, 'semantic_normal_totals': totals, 'annual_work_context': work,
              'unique_development_mines': 4, 'existing_normal_regions': sum(len(r['normal_regions']) for r in rows),
              'known_core_pixels': sum(r['known_core']['pixels'] for r in rows), 'reviewed_normal_pixels': sum(r['reviewed_normal']['pixels'] for r in rows),
              'new_strong_labels': 0, 'new_independent_events': 0, 'binary_supervision_scope': 'Existing conservative positive and explicit weak reviewed-normal only; unknown and expected-work-only pixels ignored.',
              'next_admission': 'Allow ONE bounded new-representation/source-only semantic probe, not a broad CNN sweep. Four mines remain development; normal surfaces do not establish safe active mining.',
              'production_changed': False, 'new_predictions': 0}
    base.write(out/'results.json', result); contact_sheet(out, panels); exp.old_rgb.check_assets(p)
    base.write(out/'verification.json', {'verified_at': base.now(), 'status': 'passed', 'original_predictions_and_scores_rebuilt': 8, 'normal_partitions_complete_disjoint': True,
               'normal_reference_regions': result['existing_normal_regions'], 'protected_assets_unchanged': True})
    exp.state(out, status='completed', stage='semantic_attribution_verified', normal_regions=result['existing_normal_regions'])


if __name__ == '__main__':
    import argparse
    a = argparse.ArgumentParser(); a.add_argument('--output', required=True); args = a.parse_args()
    try: run(Path(args.output))
    except BaseException as e: exp.state(Path(args.output), status='interrupted_or_failed', error=str(e)); raise
