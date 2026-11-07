"""Checked positive-only auxiliary reference; never accident-negative or pit-wall truth."""
from pathlib import Path
import sys
import hashlib
import json
import numpy as np
from runtime_paths import configure_runtime_paths
configure_runtime_paths()


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_reference(folder, task):
    import rasterio
    if task != 'auxiliary_tailings_runout_positive_only': raise ValueError('Auxiliary tailings support cannot become pitwall, negative or full-boundary truth')
    folder = Path(folder).resolve(); r = json.loads((folder/'reviewed_reference.json').read_text(encoding='utf-8'))
    if r['status'] != 'accepted_project_weak_positive_only' or r['allowed_task'] != task or r['expert_or_field_validation']:
        raise ValueError('Unaccepted or misrepresented auxiliary reference')
    for p, h in r['frozen_inputs'].items():
        if sha(p) != h: raise ValueError('Auxiliary provenance changed')
    source_protocol = json.loads((folder/'protocol.json').read_text(encoding='utf-8'))
    for p, h in source_protocol['frozen_assets'].items():
        if sha(p) != h: raise ValueError('Original auxiliary source changed')
    with rasterio.open(folder/'reviewed_labels.tif') as f: labels = f.read(1)
    with rasterio.open(folder/'reviewed_confidence.tif') as f: weight = f.read(1)
    if sha(folder/'reviewed_labels.tif') != r['labels_sha256'] or sha(folder/'reviewed_confidence.tif') != r['confidence_sha256']:
        raise ValueError('Auxiliary masks changed')
    if not np.isin(labels, [1, 255]).all() or np.any(weight[labels == 255]) or not np.allclose(weight[labels == 1], .25):
        raise ValueError('Auxiliary unknown/weight semantics changed')
    return labels, weight, r


def apply_review(folder):
    import rasterio
    sys.path.insert(0, str(Path(__file__).resolve().parent/'validation'))
    import qualify_brumadinho_visible_core as proposal
    from mine_data_protocol import assert_development_cases
    assert_development_cases(['BR0001']); folder = Path(folder).resolve()
    root = Path(__file__).resolve().parent/'datasets'
    if not folder.is_relative_to(root): raise ValueError('Auxiliary outputs must remain in D datasets')
    review_path = folder/'project_visual_review.json'; review = json.loads(review_path.read_text(encoding='utf-8')); initial = json.loads((folder/'reference.json').read_text(encoding='utf-8'))
    if initial['status'] != 'pending_project_visual_review' or review['reviewed_using_model_predictions'] or review['allowed_task'] != 'auxiliary_tailings_runout_positive_only': raise ValueError('Explicit original-image project review required')
    if sha(folder/'labels.tif') != initial['labels_sha256'] or sha(folder/'confidence.tif') != initial['confidence_sha256']: raise ValueError('Proposal changed before review')
    with rasterio.open(folder/'labels.tif') as f: original = f.read(1); profile = f.profile
    row = review['conservative_subset']['keep_original_raster_rows_at_or_below']
    if type(row) is not int or not 0 < row < len(original): raise ValueError('Invalid original-grid review subset')
    labels = original.copy(); labels[:row] = 255; weights = np.zeros(labels.shape, np.float32); weights[labels == 1] = .25
    retained = int((labels == 1).sum())
    if retained < review['minimum_retained_positive_pixels']: raise ValueError('Reviewed support insufficient; do not relax subset')
    if np.any((labels == 1) & (original != 1)): raise ValueError('Visual review must only narrow, never create support')
    for name, a, dtype, nodata in [('reviewed_labels.tif', labels, 'uint8', 255), ('reviewed_confidence.tif', weights, 'float32', None)]:
        profile.update(dtype=dtype, nodata=nodata, count=1, compress='deflate')
        with rasterio.open(folder/name, 'w', **profile) as f: f.write(a, 1)
    prepost = []
    for day in ('2019-01-07', '2019-02-01'):
        bands = []
        for b in (2, 3, 4, 8):
            with rasterio.open(proposal.RAW/f'BR0001__S2L2A__{day}__B{b:02d}.tif') as f: bands.append(f.read(1).astype(np.float32)*.0001)
        prepost.append(np.stack(bands, -1))
    with rasterio.open(proposal.RAW/'BR0001__None__None__MASK.tif') as f: inventory = f.read(1) > 0
    proposal.preview(folder, *prepost, labels, inventory)
    (folder/'visible_core_review.png').replace(folder/'reviewed_visible_core.png')
    inputs = [Path(__file__), review_path, folder/'reference.json', folder/'labels.tif', folder/'confidence.tif', folder/'protocol.json']
    r = {'status': 'accepted_project_weak_positive_only', 'case': 'BR0001', 'role': 'auxiliary_development', 'allowed_task': review['allowed_task'],
         'semantic_class': 'visibly_changed_downstream_tailings_runout_core', 'positive_pixels': retained, 'removed_preexisting_or_ambiguous_source_pixels': int((original == 1).sum())-retained,
         'negative_pixels': 0, 'unknown_pixels': int((labels == 255).sum()), 'weak_loss_weight': .25, 'expert_or_field_validation': False,
         'binary_pitwall_labels_admitted': False, 'normal_or_safe_mining_labels_admitted': False, 'complete_accident_extent_claimed': False, 'new_independent_primary_mines': 0,
         'dates': {'pre': '2019-01-07', 'event': '2019-01-25', 'post': '2019-02-01'}, 'frozen_inputs': {str(p): sha(p) for p in inputs},
         'labels_sha256': sha(folder/'reviewed_labels.tif'), 'confidence_sha256': sha(folder/'reviewed_confidence.tif'), 'review_image_sha256': sha(folder/'reviewed_visible_core.png'),
         'rationale': 'Narrow auxiliary supervision scope with explicit visible runout concurrence; older full-boundary/date/field-validation restrictions remain. Foreground pixel count is not independent event count.'}
    (folder/'reviewed_reference.json').write_text(json.dumps(r, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    load_reference(folder, r['allowed_task']); print(json.dumps({'retained_positive': retained, 'removed_ambiguous_source': r['removed_preexisting_or_ambiguous_source_pixels'], 'negative': 0, 'allowed_task': r['allowed_task']}, ensure_ascii=False))


if __name__ == '__main__':
    import argparse
    a = argparse.ArgumentParser(); a.add_argument('--folder', required=True); args = a.parse_args(); apply_review(Path(args.folder))
