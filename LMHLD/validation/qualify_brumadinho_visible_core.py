"""Positive-only auxiliary plume concurrence; never a complete accident truth.

Date-uncertain official boundary is corroborating context only. Fixed conservative
spatial/radiometric filters are registered before reading imagery. No model run.
"""
from pathlib import Path
import sys
import numpy as np
import cv2
sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parent)]
from runtime_paths import configure_runtime_paths
configure_runtime_paths()
import rgb_semantic_audit as audit
exp = audit.exp; base = audit.base
DATA = audit.DATA
RAW = DATA/'raw_sources/dlr/s1s2_landslide_reference_data/original_scenes/BR0001'
QUALITY = DATA/'development/Brumadinho/exact_date_quality_20261004'
BOUNDARY = DATA/'development/Brumadinho/official_boundary_review_v1_20261004'


def select_core(inventory, corroboration, visible, pre, post):
    inventory, corroboration, visible = map(lambda x: np.asarray(x, bool), (inventory, corroboration, visible))
    pre, post = np.asarray(pre, np.float32), np.asarray(post, np.float32)
    if pre.shape != post.shape or pre.shape != (*inventory.shape, 4) or corroboration.shape != inventory.shape or visible.shape != inventory.shape:
        raise ValueError('Exactly aligned measured BGRN pairs required')
    if not np.isfinite(pre[visible]).all() or not np.isfinite(post[visible]).all(): raise ValueError('Invalid visible observations')
    overlap = inventory & corroboration
    core = cv2.erode(overlap.astype(np.uint8), np.ones((11, 11), np.uint8)) > 0
    def ndvi(a):
        den = a[..., 3]+a[..., 2]
        return np.divide(a[..., 3]-a[..., 2], den, out=np.zeros_like(den), where=np.abs(den)>1e-6)
    before, after = ndvi(pre), ndvi(post)
    observed_loss = (before > .35) & (after < .20) & ((before-after) > .25)
    bright = np.zeros(inventory.shape, bool)
    for a in (pre, post): bright |= (a[..., :3].mean(-1) > .35) & (a[..., 0] > .20)
    # This also removes bright ground: conservative exclusion, not a certified
    # cloud classifier. SCL alone previously missed a visible cloud in this AOI.
    bright = cv2.dilate(bright.astype(np.uint8), np.ones((21, 21), np.uint8)) > 0
    selected = core & visible & ~bright & observed_loss
    labels = np.full(inventory.shape, 255, np.uint8); labels[selected] = 1
    weight = np.zeros(inventory.shape, np.float32); weight[selected] = .25
    return labels, weight, {'research_inventory_pixels': int(inventory.sum()), 'corroborating_boundary_pixels': int(corroboration.sum()),
        'overlap_pixels': int(overlap.sum()), 'after50m_erosion_pixels': int(core.sum()), 'joint_visible_core_pixels': int((core & visible).sum()),
        'bright_excluded_core_pixels': int((core & bright & visible).sum()), 'positive_auxiliary_pixels': int(selected.sum()), 'negative_pixels': 0,
        'outside_auxiliary_positive_unknown': True}


def register(out):
    from mine_data_protocol import assert_development_cases
    out = Path(out).resolve()
    if out.drive.upper() != 'D:' or not out.is_relative_to(exp.ROOT/'datasets'): raise ValueError('Use D datasets only')
    out.mkdir(parents=True, exist_ok=True); assert_development_cases(['BR0001'])
    if (out/'protocol.json').exists():
        p = base.read(out/'protocol.json'); exp.old_rgb.check_assets(p); return p
    files = [Path(__file__), exp.ROOT/'tests/test_brumadinho_visible_core.py', DATA/'mine_case_registry.json', QUALITY/'manifest.json', BOUNDARY/'boundary_audit.json', BOUNDARY/'boundary_date_verification.json',
             BOUNDARY/'government_impact_outline_on_original_grid.tif', RAW/'BR0001__None__None__MASK.tif']
    files += [RAW/f'BR0001__S2L2A__{day}__B{band:02d}.tif' for day in ('2019-01-07', '2019-02-01') for band in (2, 3, 4, 8)]
    files += [QUALITY/f'{period}_SCL_original_grid.tif' for period in ('pre', 'post')]
    p = {'registered_at': base.now(), 'frozen_assets': {str(f): base.sha(f) for f in files}, 'case': 'BR0001', 'role': 'auxiliary_development',
         'scope': 'Weak positive-only visible vegetation-burial/debris-plume concurrence, not open-pit-wall failure labels, complete accident extent, dated official boundary or no-accident negative truth.',
         'task_change': 'Earlier full-boundary training/scoring remains NOT accepted. This narrower task admits only image-observed changed interiors of two source footprints, retains all other pixels unknown and supplies no negatives. Old reference products and registry remain frozen.',
         'dates': {'pre': '2019-01-07', 'event': '2019-01-25', 'post': '2019-02-01'},
         'fixed_filters': 'Intersection published DLR mask / official impact polygon,50m erosion; valid original grids; joint SCL excludes0,1,3,6,7,8,9,10,11. Exclude meanBGRreflectance>.35 and blue>.20 with100m buffer in either date. Require preNDVI>.35,postNDVI<.20,drop>.25. No threshold search.',
         'confidence': '.25 algorithmic weak supervision weight, not calibrated correctness probability.',
         'source_limitations': 'Official Pleiades capture date unverified; underlying image/geometry independence not proven. Its boundary is broad corroboration, not assigned the IBGE2019-01-29 date. SCL is approximate and bright exclusions are conservative. Vegetation-burial sample bias does not solve bare-bench/active-excavation semantics.',
         'admission': 'At least1000 retained positive-only pixels plus manual project image review. No blind or professional validation claim. A positive pixel count is not a count of independent events.',
         'training_priority': 'Do not broaden the rejected four-mine head merely because auxiliary positives exist; primary missing contrast remains bare failure vs verified work. Keep this as explicit lower-priority auxiliary supervision.',
         'new_model_predictions': 0, 'production_changed': False, 'raw_images_changed': False}
    exp.old_rgb.check_assets(p); base.write(out/'protocol.json', p); return p


def preview(out, pre, post, labels, rawmask):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    yy, xx = np.where(rawmask); y0, y1, x0, x1 = max(0, yy.min()-30), min(labels.shape[0], yy.max()+31), max(0, xx.min()-30), min(labels.shape[1], xx.max()+31)
    fig, axes = plt.subplots(1, 3, figsize=(15, 6), dpi=120)
    for ax, a, title in zip(axes, (pre, post, post), ('2019-01-07: before event', '2019-02-01: after event', 'Cyan: weak positive-only core')):
        rgb = a[y0:y1, x0:x1, :3][..., [2, 1, 0]]; lo, hi = np.percentile(rgb, [2, 98], axis=(0, 1)); rgb = np.clip((rgb-lo)/np.maximum(hi-lo, 1e-6), 0, 1)
        ax.imshow(rgb); ax.axis('off'); ax.set_title(title)
    core = labels[y0:y1, x0:x1] == 1; overlay = np.zeros((*core.shape, 4), np.float32); overlay[core] = [0, 1, 1, .65]; axes[-1].imshow(overlay)
    fig.suptitle('Brumadinho auxiliary debris-burial observation; no negatives, no complete-boundary claim, no model outputs')
    fig.tight_layout(); fig.savefig(Path(out)/'visible_core_review.png'); plt.close(fig)


def run(out):
    import rasterio
    out = Path(out); p = register(out); arrays = {}; reference = None; valid = None
    quality = base.read(QUALITY/'manifest.json')
    for period in ('pre', 'post'):
        calibration = quality['images'][period]['calibration']
        if calibration['processing_baseline'] != '02.12' or calibration['scales'] != [.0001]*4 or calibration['DN_offsets'] != [0.]*4:
            raise ValueError('Expected measured original DN with verified baseline02.12 scaling')
    for period, date in [('pre', '2019-01-07'), ('post', '2019-02-01')]:
        bands = []
        for band in (2, 3, 4, 8):
            with rasterio.open(RAW/f'BR0001__S2L2A__{date}__B{band:02d}.tif') as f:
                if f.dtypes != ('uint16',) or f.crs is None or not f.crs.is_projected or f.crs.linear_units != 'metre' or f.res != (10., 10.) or f.transform.b or f.transform.d:
                    raise ValueError('Expected original measured uint16 native10m grid')
                signature = (f.shape, f.transform, f.crs)
                if reference is None: reference = signature; profile = f.profile; valid = np.ones(f.shape, bool)
                if signature != reference: raise ValueError('Source band grids differ; do not resample')
                raw = f.read(1); valid &= (f.read_masks(1) > 0) & (raw > 0); bands.append(raw.astype(np.float32)*.0001)
        arrays[period] = np.stack(bands, -1)
        with rasterio.open(QUALITY/f'{period}_SCL_original_grid.tif') as f:
            if (f.shape, f.transform, f.crs) != reference: raise ValueError('Quality grid differs')
            valid &= ~np.isin(f.read(1), [0, 1, 3, 6, 7, 8, 9, 10, 11])
    masks = []
    for path in (RAW/'BR0001__None__None__MASK.tif', BOUNDARY/'government_impact_outline_on_original_grid.tif'):
        with rasterio.open(path) as f:
            if (f.shape, f.transform, f.crs) != reference: raise ValueError('Reference grids differ')
            masks.append(f.read(1) > 0)
    labels, confidence, counts = select_core(*masks, valid, arrays['pre'], arrays['post'])
    if counts['positive_auxiliary_pixels'] < 1000: raise ValueError('Insufficient conservative auxiliary support; do not relax filters')
    profile.update(count=1, compress='deflate')
    for name, a, dtype, nodata in [('labels.tif', labels, 'uint8', 255), ('confidence.tif', confidence, 'float32', None)]:
        profile.update(dtype=dtype, nodata=nodata)
        with rasterio.open(out/name, 'w', **profile) as f: f.write(a, 1)
    preview(out, arrays['pre'], arrays['post'], labels, masks[0])
    replay, weights, rebuilt = select_core(*masks, valid, arrays['pre'], arrays['post'])
    if not np.array_equal(replay, labels) or not np.array_equal(weights, confidence) or counts != rebuilt or np.any(labels == 0) or np.any(confidence[labels == 255]): raise ValueError('Auxiliary support replay/unknown contract failure')
    exp.old_rgb.check_assets(p)
    result = {'completed_at': base.now(), 'case': 'BR0001', 'role': 'auxiliary_development', 'status': 'pending_project_visual_review', 'counts': counts,
              'semantic_class': 'postevent_visible_tailings_debris_burial', 'binary_pitwall_labels_admitted': False, 'normal_negative_labels_admitted': False,
              'independent_expert_or_field_review': False, 'new_independent_primary_mines': 0, 'new_model_predictions': 0,
              'labels_sha256': base.sha(out/'labels.tif'), 'confidence_sha256': base.sha(out/'confidence.tif')}
    base.write(out/'reference.json', result); base.write(out/'verification.json', {'verified_at': base.now(), 'status': 'passed', 'same_original_grid': True, 'deterministic_filter_replay': True, 'no_negative_or_unknown_supervision': True, 'protected_assets_unchanged': True})
    print('BRUMADINHO_VISIBLE_CORE', result, flush=True)


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser(); p.add_argument('--output', required=True); args = p.parse_args(); run(Path(args.output))
