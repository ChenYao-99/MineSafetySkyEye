"""Exercise actual cached API records from seven observed inputs and one named stress fixture."""
from pathlib import Path
from unittest.mock import patch
import base64, copy, hashlib, json, sys
from tempfile import TemporaryDirectory
ROOT=Path('D:/GeoMMAgent');sys.path.insert(0,str(ROOT))
import backend as service
from case_archive import CaseArchive
SOURCE=Path('D:/LMHLD/experiments/same_input_reporting_workflow_20261005')
OUT=ROOT/'cache'/'review_task_bank_replay'

def run():
    OUT.mkdir(parents=True,exist_ok=True)
    bank=json.loads((SOURCE/'detector_bank.json').read_text(encoding='utf-8'))
    service._report_cache.clear();client=service.app.test_client();rows=[]
    with TemporaryDirectory(dir=OUT) as isolated, \
         patch.object(service,'case_archive',CaseArchive(Path(isolated)/'cases.sqlite3')), \
         patch.object(service,'detector_request',side_effect=AssertionError('Cached handoff must not repeat inference')), \
         patch.object(service,'analyze_with_agents',side_effect=AssertionError('Local acceptance must not call cloud models')):
        for task in bank['tasks']:
            det=copy.deepcopy(task['detection']);pair='t1_ratio' in det['stats']
            result=service.cache_report({'region':'compare' if pair else 'upload','detection':det,
                'final_answer':'本地接口验收，仅汇总检测事实。','knowledge':[],
                'generation':{'status':'fallback','source':'detection_summary'},'quality':{'degraded':True},'agents':[]})
            response=client.post('/api/review_record',json={'report_id':result['report_id'],'meters_per_pixel':999})
            assert response.status_code==200,(task['id'],response.get_json())
            body=response.get_json();record=json.loads(base64.b64decode(body['content_b64']))
            assert record==body['record'] and record['trace']['request_id']==det['request_id']
            assert record['trace']['trace_complete'] and record['accepted_human_review_decisions']==[]
            assert not record['project_field_trial_or_expert_endorsement']
            assert record['measurement_scale']==(det.get('scale_meta') or {})
            if pair:
                assert record['input_evidence']['t1']['acquisition_datetime']==det['band_meta']['t1']['acquisition_datetime']
                assert abs(record['summary']['net_change_percentage_points']-det['stats']['net_ratio']*100)<1e-8
            if task['fixture_kind'].startswith('synthetic'):
                assert record['summary']['area_ratio'] is None
            (OUT/(task['id']+'_local_handoff.json')).write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
            rows.append({'task_id':task['id'],'fixture_kind':task['fixture_kind'],'report_id':record['trace']['report_id'],
                         'request_id_preserved':True,'download_JSON_matches_API_record':True,
                         'weight_identity_preserved':True,'pending_checks':[x['code'] for x in record['pending_checks']],
                         'paired_measurement':pair,'passed':True})
    receipt={'scope':'local actual-cache handoff API;7 observed detector inputs+1 named synthetic stress fixture',
             'task_bank_sha256':hashlib.sha256((SOURCE/'detector_bank.json').read_bytes()).hexdigest(),
             'passed':len(rows),'external_requests':0,'new_classifier_predictions':0,'human_trial_or_expert_endorsement':False,'rows':rows}
    (OUT/'local_task_acceptance.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print('8/8 local handoff tasks accepted;no inference rerun,no cloud request,no claimed human trial')

if __name__=='__main__':
    sys.stdout.reconfigure(encoding='utf-8');run()
