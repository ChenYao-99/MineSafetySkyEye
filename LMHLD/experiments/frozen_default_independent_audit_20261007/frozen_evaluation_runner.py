"""One logical prediction on the frozen independent case; never rerun inference."""
from datetime import datetime
from pathlib import Path
import hashlib,json,sys,time

ML=Path('D:/LMHLD');EVAL=ML/'experiments/frozen_default_independent_audit_20261007'
OUT=Path('D:/GeoMMAgent/cache/independent_acceptance_20261007')


def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024**2),b''):digest.update(block)
    return digest.hexdigest()


def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write(path,value):
    with Path(path).open('x',encoding='utf-8') as stream:json.dump(value,stream,ensure_ascii=False,indent=2)


def main():
    freeze=read(EVAL/'freeze.json');protocol=read(EVAL/'input_protocol.json');plan=read(EVAL/'plan.json')
    assert sha(__file__)==freeze['runner_sha256']
    assert sha(EVAL/'input_protocol.json')==freeze['protocol_sha256'] and sha(EVAL/'plan.json')==freeze['plan_sha256']
    lock=ML/'datasets/mine_benchmark_20261002/final_evaluation_once.json'
    assert sha(lock)==freeze['one_shot_lock_sha256'] and read(lock)['case_ids']==['CL0002']
    manifest=read(OUT/'portable_release_manifest.json')
    assert sha(OUT/'portable_release_manifest.json')==freeze['release_manifest_sha256']
    release=Path(manifest['release']);frozen_ml=release/'LMHLD'
    for relative,digest in manifest['files'].items():assert sha(release/relative)==digest,relative
    for key in ['input','labels','reference','contract']:assert sha(protocol[key]['path'])==protocol[key]['sha256'],key
    # This exclusive journal consumes the one-shot before model construction.
    write(EVAL/'inference_started.json',{'started_at':datetime.now().astimezone().isoformat(),
        'case_id':'CL0002','allowed_logical_predictions':1,'resume_may_only_score_saved_probability':True})
    sys.path.insert(0,str(frozen_ml))
    from runtime_paths import configure_runtime_paths
    configure_runtime_paths()
    import numpy as np,rasterio,torch
    from detector import LandslideDetector
    from upload_utils import decode_upload
    from raster_upload import observed_mask
    from geo_geometry import label_instances
    from model_registry import ModelRegistry
    import base64
    weight=Path(plan['weights'][0]['path'])
    artifact=ModelRegistry(frozen_ml/'model_registry.json',frozen_ml/'models_prior').verify('upload:mine',weight,
        {'patch_size':32,'threshold':.4,'patch_norm':'minmax','input_scale':1.0})
    image,meta=decode_upload(base64.b64encode(Path(protocol['input']['path']).read_bytes()).decode(),'AUTO')
    with rasterio.open(protocol['labels']['path']) as source:
        labels=source.read(1);profile=source.profile.copy();label_grid=(source.crs,source.transform,source.shape)
    with rasterio.open(protocol['input']['path']) as source:
        assert (source.crs,source.transform,source.shape)==label_grid
    valid=observed_mask(meta,image.shape[:2]);scored=labels!=255
    assert np.all(valid[scored]) and set(np.unique(labels).tolist())=={0,1,255}
    assert int((labels==1).sum())==2163 and int((labels==0).sum())==243973
    torch.manual_seed(42);torch.backends.cudnn.benchmark=False
    detector=LandslideDetector(str(weight),patch_size=32,threshold=.4)
    begin=time.monotonic()
    probability=detector.detect_image(image,patch_norm='minmax',max_batch=128)
    elapsed=time.monotonic()-begin
    assert probability.shape==labels.shape and np.isfinite(probability).all()
    np.savez_compressed(EVAL/'probability_once.npz',probability=probability)
    raw=(probability>=.4).astype(np.uint8)*valid.astype(np.uint8)
    objects,_=label_instances(raw,min_area=10);reported=(objects>0).astype(np.uint8)
    profile.update(dtype='uint8',count=1,nodata=None,compress='deflate')
    for name,array in [('raw_threshold_mask.tif',raw),('reported_mask.tif',reported)]:
        with rasterio.open(EVAL/name,'w',**profile) as target:target.write(array,1)
    def metrics(mask):
        positive=labels==1;background=labels==0
        tp=int((mask.astype(bool)&positive).sum());fn=int(((mask==0)&positive).sum())
        fp=int((mask.astype(bool)&background).sum());tn=int(((mask==0)&background).sum())
        return {'TP':tp,'FN':fn,'FP':fp,'TN':tn,'precision_on_scored_published_reference':tp/max(tp+fp,1),
            'recall_on_scored_published_reference':tp/max(tp+fn,1),'IoU_on_scored_published_reference':tp/max(tp+fn+fp,1),
            'activation_on_annotation_background_not_field_FPR':fp/max(fp+tn,1)}
    primary=metrics(reported);secondary=metrics(raw)
    targets=plan['performance_reference_targets']
    passes={name:primary[name+'_on_scored_published_reference']>=target for name,target in targets.items()}
    result={'completed_at':datetime.now().astimezone().isoformat(),'independent_measurement_completed':True,
        'case_id':'CL0002','mine':'Ministro Hales, Chile','evaluation_purpose':plan['evaluation_purpose'],
        'logical_predictions':1,'frozen_version':freeze,'model_artifact':artifact,'inference_seconds_excluding_load':elapsed,
        'input_metadata':{k:v for k,v in meta.items() if not k.startswith('_')},
        'primary_reported_production_mask':primary,'secondary_raw_mask_diagnostic_only':secondary,
        'scored_positive_pixels':2163,'scored_annotation_background_pixels':243973,'unknown_pixels_excluded':int((labels==255).sum()),
        'whole_published_footprint_observed_fraction':read(protocol['reference']['path'])['post_mapped_foreground_observed_fraction'],
        'performance_reference_targets':targets,'reference_target_passes':passes,'all_internal_reference_targets_met':all(passes.values()),
        'probability_diagnostics':{'positive_quantiles_10_50_90':np.quantile(probability[labels==1],[.1,.5,.9]).tolist(),
            'annotation_background_quantiles_10_50_90':np.quantile(probability[labels==0],[.1,.5,.9]).tolist()},
        'removed_small_component_pixels':int(raw.sum()-reported.sum()),
        'new_fit_or_threshold_tuning':0,'other_reserved_case_predictions':0,
        'limits':['single previously unseen mine only;not stable multi-mine acceptance',
            'agreement with scoped published inventory,not surveyed full ground boundary accuracy',
            'annotation background is not verified normal mining or absence of all other hazards',
            'default model retained despite failed development joint gate;no passing status fabricated',
            'all future versions have seen this result;CL0002 cannot be reused as an untouched blind test']}
    result['artifacts']={name:sha(EVAL/name) for name in ['probability_once.npz','raw_threshold_mask.tif','reported_mask.tif']}
    write(EVAL/'results.json',result)
    registry_path=ML/'datasets/mine_benchmark_20261002/mine_case_registry.json';inventory=read(registry_path)
    assert inventory['cases']['CL0002']['model_predictions_run']==0
    inventory['cases']['CL0002'].update(model_predictions_run=1,model_evaluation_readiness='frozen_version_independent_audit_completed_not_available_as_future_untouched_test',
        independent_evaluation_result=str(EVAL/'results.json'),evaluation_version_weight_sha256=artifact['weights_sha256'])
    inventory['new_test_model_predictions_run']=inventory.get('new_test_model_predictions_run',0)+1
    registry_path.write_text(json.dumps(inventory,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    state_path=ML/'datasets/mine_benchmark_20261002/STATE.json';state=read(state_path)
    state.update(updated_at=result['completed_at'],independent_frozen_version_audit_completed=True,
        latest_independent_frozen_version_audit=str(EVAL/'results.json'),independent_audit_cases_completed=['CL0002'],
        new_final_test_predictions=1,default_model_changed=False,phase='frozen_default_independent_audit_completed_followup_different_mine_data_and_task_benefit_pending')
    state_path.write_text(json.dumps(state,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    for relative,digest in manifest['files'].items():assert sha(release/relative)==digest,relative
    print(json.dumps({'completed':True,'case':'CL0002','primary':primary,'targets_met':all(passes.values()),'logical_predictions':1},ensure_ascii=False),flush=True)


if __name__=='__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    main()
