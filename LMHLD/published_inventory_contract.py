"""Scoped external benchmark loading/scoring with an unopened-test execution gate."""
from pathlib import Path
import json
import numpy as np,rasterio
from model_registry import file_sha256
from validation.mine_data_protocol import registry,assert_mine_group_role,DATA

def load_inventory_reference(case_id,scope,input_mode,use='quality_review'):
    v=registry();case=v['cases'].get(case_id,{})
    if case.get('role')!='reserved_final_test' or case.get('quality_review_status')!='accepted':raise ValueError('Benchmark case is not accepted/reserved')
    assert_mine_group_role(case_id,'reserved_final_test',v)
    path=Path(case['benchmark_usage_contract']);contract=json.loads(path.read_text(encoding='utf-8'))
    if file_sha256(path)!=case['benchmark_usage_contract_sha256']:raise ValueError('Frozen benchmark contract changed')
    if scope not in contract['scopes'] or input_mode not in contract['scopes'][scope]['input_modes']:raise ValueError('Model input scope incompatible with scored support')
    refpath=Path(contract['reference_path']);ref=json.loads(refpath.read_text(encoding='utf-8'))
    if file_sha256(refpath)!=contract['reference_sha256'] or file_sha256(refpath)!=case['benchmark_reference_sha256']:raise ValueError('Frozen accepted reference changed')
    if contract['unknown_scored_as_background'] or contract['outside_mapped_event_is_geotechnically_safe'] or contract['source_original_field_survey']:raise ValueError('Reference scope was falsely upgraded')
    if use=='development_training':raise ValueError('Reserved inventory may not train or tune')
    if use=='evaluation':
        lock=DATA/'final_evaluation_once.json'
        if not lock.exists():raise ValueError('Unopened independent benchmark requires frozen final evaluation gate')
        plan=json.loads(lock.read_text(encoding='utf-8'))['plan']
        if case_id not in plan['case_ids'] or plan.get('reference_scopes',{}).get(case_id)!=scope or plan.get('input_modes',{}).get(case_id)!=input_mode:raise ValueError('Scope/input was not frozen before predictions')
    elif use!='quality_review':raise ValueError('Unsupported reference use')
    folder=refpath.parent
    for p,digest in ref['frozen_dependencies'].items():
        if file_sha256(p)!=digest:raise ValueError('Reference source dependency changed')
    for name,digest in ref['frozen_assets'].items():
        p=(folder/name).resolve()
        if not p.is_relative_to(folder.resolve()) or file_sha256(p)!=digest:raise ValueError('Reference data changed')
    mask=folder/contract['scopes'][scope]['mask']
    with rasterio.open(mask) as f:y=f.read(1);grid=(f.crs,f.transform,f.shape)
    if not set(np.unique(y)).issubset({0,1,255}):raise ValueError('Invalid reference classes')
    expected=contract['scopes'][scope]
    if int((y==1).sum())!=expected['positive_pixels'] or int((y==0).sum())!=expected['annotation_background_pixels']:raise ValueError('Reference support changed')
    return {'labels':y,'grid':grid,'reference':ref,'contract':contract,'scope':scope,'input_mode':input_mode}

def reference_metrics(probabilities,labels,threshold):
    p=np.asarray(probabilities);y=np.asarray(labels)
    if p.shape!=y.shape or not np.isfinite(p).all() or p.min()<0 or p.max()>1 or not 0<threshold<1:raise ValueError('Finite native probabilities/frozen threshold required')
    if not set(np.unique(y)).issubset({0,1,255}) or not (y==1).any() or not (y==0).any():raise ValueError('Foreground and annotation background required;partial positives cannot give IoU')
    valid=y!=255;positive=p>=threshold;tp=int((positive & (y==1)).sum());fn=int((~positive & (y==1)).sum());fp=int((positive & (y==0)).sum());tn=int((~positive & (y==0)).sum())
    return {'TP':tp,'FP':fp,'FN':fn,'TN':tn,'precision_on_scored_published_reference':tp/max(tp+fp,1),'recall_on_scored_published_reference':tp/max(tp+fn,1),'IoU_on_scored_published_reference':tp/max(tp+fn+fp,1),
            'unknown_pixels_scored':0,'scored_pixels':int(valid.sum()),'unmasked_actual_field_IoU':None,'geotechnical_safety_accuracy':None}
