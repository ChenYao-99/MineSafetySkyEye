"""Load scoped accident references without laundering unknown pixels or reserved mines."""
from pathlib import Path
import json
import numpy as np
import rasterio
from model_registry import file_sha256
from validation.mine_data_protocol import registry,assert_development_cases,assert_mine_group_role,DATA

def load_partial_reference(folder,use='quality_review'):
    folder=Path(folder).resolve();root=DATA.parents[1].resolve()
    if not folder.is_relative_to(root/'datasets'):raise ValueError('Reference must stay within D: datasets')
    contract=json.loads((folder/'usage_contract.json').read_text(encoding='utf-8'))
    case_id=contract['case_id'];value=registry();case=value['cases'].get(case_id)
    if case is None or case['role']!=contract['role']:raise ValueError('Reference split differs from canonical mine registry')
    assert_mine_group_role(case_id,case['role'],value)
    reference_path=Path(contract['reference']).resolve()
    if reference_path!=folder/'reference.json' or case.get('partial_pixel_reference')!=str(reference_path):raise ValueError('Reference identity/path is not registered')
    digest=file_sha256(reference_path)
    if digest!=contract['reference_sha256'] or digest!=case.get('partial_pixel_reference_sha256'):raise ValueError('Frozen reference manifest changed')
    reference=json.loads(reference_path.read_text(encoding='utf-8'))
    if reference['role']!=case['role'] or reference['case_id']!=case_id:raise ValueError('Reference event/role mismatch')
    if use=='development_training':
        assert_development_cases([case_id],value)
        if not contract['training_allowed']:raise ValueError('Reference not permitted for development training')
    elif use=='known_core_recall':
        if case['role']=='reserved_final_test':
            lock=DATA/'final_evaluation_once.json'
            if not lock.exists():raise ValueError('Reserved reference needs an authorised frozen final evaluation')
            final=json.loads(lock.read_text(encoding='utf-8'))
            if case_id not in final['case_ids'] or case['quality_review_status']!='accepted':raise ValueError('Reserved case not admitted to frozen final evaluation')
    elif use=='full_accident_iou':
        raise ValueError('Partial positive-core reference cannot support complete accident IoU')
    elif use!='quality_review':raise ValueError('Unsupported reference use')
    if contract['full_accident_IoU_allowed'] or contract['unknown_pixels_may_become_background'] or reference['accepted_for_full_accident_IoU_or_area_accuracy']:
        raise ValueError('Unknown/partial references must not become full ground truth')
    for path,expected in reference['frozen_dependencies'].items():
        if file_sha256(path)!=expected:raise ValueError('Frozen source dependency changed')
    for name,expected in reference['frozen_assets'].items():
        path=(folder/name).resolve()
        if not path.is_relative_to(folder) or file_sha256(path)!=expected:raise ValueError('Frozen partial reference asset changed')
    with rasterio.open(folder/'labels_partial.tif') as f:
        labels=f.read(1);grid=(f.crs,f.transform,f.shape)
        if f.res!=(10,10) or f.nodata!=255 or f.tags().get('role')!=case['role']:raise ValueError('Invalid native partial-reference grid/role')
    with rasterio.open(folder/'confidence_partial.tif') as f:
        confidence=f.read(1)
        if (f.crs,f.transform,f.shape)!=grid:raise ValueError('Confidence grid mismatch')
    if not set(np.unique(labels)).issubset({1,255}) or int((labels==1).sum())!=reference['positive_pixels'] or (labels==0).any():raise ValueError('Unknown pixels were relabeled or accident core changed')
    if not np.isfinite(confidence).all() or (confidence[labels==255]!=0).any() or not np.allclose(confidence[labels==1],reference['reference_confidence']):raise ValueError('Unknown pixels must carry no supervision weight')
    return {'case_id':case_id,'role':case['role'],'labels':labels,'confidence':confidence,'grid':grid,'reference':reference,'contract':contract}

def observed_core_recall(probabilities,folder,threshold=.5):
    data=load_partial_reference(folder,use='known_core_recall');p=np.asarray(probabilities)
    if p.shape!=data['labels'].shape or not np.isfinite(p).all() or p.min()<0 or p.max()>1:raise ValueError('Finite native probability grid in[0,1] required')
    if not isinstance(threshold,(int,float)) or not 0<threshold<1:raise ValueError('Fixed finite threshold in(0,1) required')
    known=data['labels']==1
    return {'metric':'recall_on_observed_conservative_positive_core','case_id':data['case_id'],'positive_pixels':int(known.sum()),'recall':float((p[known]>=threshold).mean()),'full_accident_IoU':None,'unknown_pixels_scored':0}
