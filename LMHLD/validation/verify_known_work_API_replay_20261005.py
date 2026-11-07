"""Exercise updated real API routing with a frozen actual probability, without GPU rerun."""
from pathlib import Path
from unittest.mock import patch
import base64,copy,json,sys
ROOT=Path('D:/LMHLD');sys.path.insert(0,str(ROOT))
from runtime_paths import configure_runtime_paths
configure_runtime_paths()
import numpy as np
import detector_service as service
OUT=ROOT/'experiments/government_work_runtime_validation_20261005'

def run():
    old=json.loads((OUT/'default_mine_API_on_Amrun_yearly_work.json').read_text(encoding='utf-8'))
    loaded=service._load_upload_detector('mine');detector,patch_norm,input_scale=loaded
    probability=np.load(OUT/'default_mine_Amrun_yearly_probability.npy')
    with patch.object(detector,'detect_image',return_value=probability.copy()) as replay:
        response=service.app.test_client().post('/api/detect_upload',json={
            'image':base64.b64encode((OUT/'post_original_BGRN_with_original_SCL.tif').read_bytes()).decode(),
            'model_tag':'mine','channel_order':'auto','max_batch':16})
    assert response.status_code==200 and replay.call_count==1
    result=response.get_json();context=result['work_context']
    assert context['status']=='matched_public_expected_work_reference'
    assert result['risk_level']==old['actual_detector_result']['risk_level']
    before=old['actual_detector_result']['stats'];after=result['stats']
    for key in before:
        if key!='regions':assert after[key]==before[key],key
    for a,b in zip(after['regions'],before['regions']):
        for key in b:assert a[key]==b[key],key
    assert not context['geotechnical_safety_or_accident_absence_verified']
    result.pop('overlay_b64',None);result.pop('rgb_b64',None)
    record={'scope':'updated actual API pipeline replay of one earlier real probability;not a new model accuracy run',
            'new_GPU_forward_calls':0,'cloud_requests':0,'all_existing_stats_regions_and_level_preserved':True,
            'actual_detector_result_with_public_context':result}
    (OUT/'API_context_replay_verification.json').write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print('API preserved all raw stats/geometry and level;workoverlap',context['predicted_pixels_overlapping_work_reference'],
          'regioncues',len(context['same_request_region_overlaps']),'source dated annual;GPUcalls0,cloudcalls0')

if __name__=='__main__':sys.stdout.reconfigure(encoding='utf-8');run()
