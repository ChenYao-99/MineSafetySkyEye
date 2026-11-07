"""Current real default-model API fixture, on an already observed development mine."""
from pathlib import Path
import sys,json,base64,time
ROOT=Path('D:/LMHLD');sys.path.insert(0,str(ROOT))
from runtime_paths import configure_runtime_paths
configure_runtime_paths()
from validation.mine_data_protocol import assert_development_cases,verify_assets,registry,DATA
from model_registry import file_sha256
import detector_service as service
OUT=Path('D:/GeoMMAgent/cache/competition_readiness_20261005')
def run():
    assert_development_cases(['LasCruces']);verify_assets();before=registry()
    source=DATA/'development/LasCruces/post_BGRN_10m.tif';raw=base64.b64encode(source.read_bytes()).decode()
    started=time.perf_counter();response=service.app.test_client().post('/api/detect_upload',json={'image':raw,'model_tag':'mine','roi':[.25,.25,.9,.9]})
    if response.status_code!=200:raise ValueError(response.get_json())
    body=response.get_json()
    expected='0d8d8a0540419d4409951ed91e69499fac206b68303f3feec08a32f0841cebef'
    if body['model_info']['artifact']['weights_sha256']!=expected or body['request_id']!=response.headers['X-KAT-Request-ID']:raise ValueError('Model/result trace changed')
    if body['band_meta']['nir_source']!='measured' or body['scale_meta']['meters_per_pixel']!=10:raise ValueError('Native spectral/geographic metadata lost')
    if body['stats']['area_ratio']!=body['roi']['risk_inside']:raise ValueError('Observation scope mismatch')
    OUT.mkdir(exist_ok=True);path=OUT/'real_detection.json'
    path.write_text(json.dumps(body,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    record={'source':str(source),'source_sha256':file_sha256(source),'case':'LasCruces','role':'already observed development',
            'default_weights_sha256':expected,'request_id':body['request_id'],'full_API_fixture_sha256':file_sha256(path),
            'elapsed_s_includes_initial_model_load':time.perf_counter()-started,'field_accuracy_or_new_blind_test':False,
            'additional_model_fit':0,'all_reserved_predictions':{c:registry()['cases'][c]['model_predictions_run'] for c in ('CL0002','KG0001','US_BC2021','TR_COP2024')}}
    if any(record['all_reserved_predictions'].values()) or before['new_test_model_predictions_run']!=0:raise ValueError('Reserved test opened')
    (OUT/'fixture_provenance.json').write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8')
    service._release_models();print(json.dumps(record))
if __name__=='__main__':sys.stdout.reconfigure(encoding='utf-8');run()
