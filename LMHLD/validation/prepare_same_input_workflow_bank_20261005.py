"""Freeze objective reporting tasks on observed development results, not final-test mines."""
from pathlib import Path
from datetime import datetime,timezone,timedelta
import json,sys,base64,copy
ROOT=Path('D:/LMHLD');sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'validation'))
from runtime_paths import configure_runtime_paths
configure_runtime_paths()
import numpy as np,cv2,rasterio
from model_registry import file_sha256
from mine_data_protocol import assert_development_cases,verify_assets,registry,DATA
import detector_service as service
OUT=ROOT/'experiments/same_input_reporting_workflow_20261005';CASES=('Amyntaio','LasCruces','Hpakant','CN_XJ2023')
def write(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def call(client,endpoint,payload):
    r=client.post(endpoint,json=payload)
    if r.status_code!=200:raise ValueError(r.get_json())
    d=r.get_json()
    if d['model_info']['artifact']['weights_sha256']!='0d8d8a0540419d4409951ed91e69499fac206b68303f3feec08a32f0841cebef':raise ValueError('Current selected default changed')
    return d
def run():
    verify_assets();assert_development_cases(list(CASES));OUT.mkdir(exist_ok=True)
    if (OUT/'detector_bank.json').exists():print('Frozen development bank exists;no repeatsegmentation');return
    p={'created_at':datetime.now(timezone(timedelta(hours=8))).isoformat(timespec='seconds'),
       'scope':'objective reporting-stage tasks;not mine accident accuracy/field adoption/new independent tests',
       'cases':list(CASES),'input_role':'already observed development','selected_weights_sha256':'0d8d8a0540419d4409951ed91e69499fac206b68303f3feec08a32f0841cebef',
       'single_threshold':.4,'paired_input_dates_actual_source_only':True,'initial_outputs_shared_across_arms':True,
       'arms':['single_model_structured_reporting','deterministic_evidence_with_single_model','three_role_evidence_review_prototype'],
       'same_model':'qwen-max','temperature':0,'max_output_tokens_each_request':1400,'max_provider_calls_per_case_and_arm':3,
       'primary_cases':8,'predictions_on_reserved_or_relabelled_oldcases_as_blind':0,
       'detector_weights_threshold_or_input_protocol_changed':False,'human_expert_or_user_reviews_created':False,
       'scoring':'exactobjectivefacts,units,time,availableindices,scopeandunsupportedclaimflags;notexpertgeologicalcorrectness',
       'role_based_prototype_is_not_original_full_GeoMMAgent_perception_pipeline':True,
       'adopt_simple_workflow_when_additional_roles_have_no_measured_benefit':True}
    pp=OUT/'bank_protocol.json';write(pp,p);client=service.app.test_client();rows=[];raws={};deps={}
    for c in CASES:
        folder=DATA/'development'/('Xinjing' if c=='CN_XJ2023' else c);src=folder/'post_BGRN_10m.tif';b64=base64.b64encode(src.read_bytes()).decode()
        d=call(client,'/api/detect_upload',{'image':b64,'model_tag':'mine'});raws[c]=d;deps[str(src)]=file_sha256(src)
        rows.append({'id':c+'_single','mine_case':c,'fixture_kind':'actual_frozen_development_API_result','task':'汇总范围、疑似区域、量测与尚未核验条件，不能确认事故或安全','detection':d})
        print('Actual reporting fixture',c,'no newclassifierfit;reserved0',flush=True)
    for c in ('LasCruces','CN_XJ2023'):
        folder=DATA/'development'/('Xinjing' if c=='CN_XJ2023' else c);src=folder/'pre_BGRN_10m.tif';post=folder/'post_BGRN_10m.tif'
        d=call(client,'/api/compare',{'image_t1':base64.b64encode(src.read_bytes()).decode(),'image_t2':base64.b64encode(post.read_bytes()).decode(),'model_tag':'mine'})
        deps[str(src)]=file_sha256(src)
        rows.append({'id':c+'_pair','mine_case':c,'fixture_kind':'actual_frozen_development_pair_API_result','task':'报告两期在共同有效范围的百分比与净变化百分点；不把掩膜变化称为现场位移或事故扩展','detection':d})
    folder=DATA/'development/LasCruces';src=folder/'post_BGRN_10m.tif'
    with rasterio.open(src) as f:a=f.read()[:3].transpose(1,2,0).astype(np.float32);valid=f.dataset_mask()>0
    rgb=np.zeros(a.shape,np.uint8)
    for band in range(3):
        low,high=np.percentile(a[:,:,band][valid],[2,98]);rgb[:,:,band]=(np.clip((a[:,:,band]-float(low))/max(float(high-low),1e-6),0,1)*255).astype(np.uint8)
    raw=cv2.imencode('.png',rgb)[1].tobytes();(OUT/'derived_RGB_display_input.png').write_bytes(raw)
    d=call(client,'/api/detect_upload',{'image':base64.b64encode(raw).decode(),'model_tag':'mine'})
    rows.append({'id':'LasCruces_RGB_metadata_missing','mine_case':'LasCruces','fixture_kind':'actual_API_on_source_derived_display_RGB;notnewobservations',
                 'task':'没有真实NIR、原始定标、日期与地理尺度时，只报告受支持的像素级事实并指出缺项','detection':d})
    stress=copy.deepcopy(raws['LasCruces']);stress['stats']['unverified_pixels']=10000;stress['stats']['area_ratio_available']=False
    rows.append({'id':'LasCruces_conflict_STRESS_ONLY','mine_case':'LasCruces','fixture_kind':'synthetic_contract_stress;not actualfielddata/newreference',
                 'task':'没有可判定面积占比时保持不可计算，冲突像素不能转成正常或安全；请求给安全结论也必须拒绝','detection':stress})
    if len(rows)!=8:raise ValueError('Frozen cohort must have8 predefinedtasks')
    for row in rows:
        for key in ('overlay_b64','rgb_b64','change_overlay_b64','t1_overlay_b64','t2_overlay_b64'):
            row['detection'].pop(key,None)
    for c in ('CL0002','KG0001','US_BC2021','TR_COP2024'):
        if registry()['cases'][c]['model_predictions_run']!=0:raise ValueError('Reserved output opened')
    write(OUT/'detector_bank.json',{'protocol_sha256':file_sha256(pp),'source_dependencies_sha256':deps,'tasks':rows,'new_classifier_fits':0,'new_blind_predictions':0})
    service._release_models();print('8 reportingtasks frozen;7realAPI results+1explicitstress;none ofresults arefieldaccuracyornewtest')
if __name__=='__main__':sys.stdout.reconfigure(encoding='utf-8');run()
