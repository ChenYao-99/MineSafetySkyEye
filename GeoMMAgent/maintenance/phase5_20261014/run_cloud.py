"""Run the separately approved, capped report-role candidate acceptance."""
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from contextlib import ExitStack
from datetime import datetime
import argparse,base64,copy,hashlib,io,json,os,sys,time
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent
SCRATCH=OUT/'_scratch';SCRATCH.mkdir(exist_ok=True)
sys.path.insert(0,str(ROOT))
from maintenance.phase4_20261013 import run_cloud_acceptance as audit_module
audit_module.OUT=OUT
os.environ.update(KAT_LOCAL_FACTS_ONLY='0',KAT_CASE_ARCHIVE_PATH=str(SCRATCH/'candidate.sqlite3'),TEMP=str(SCRATCH),TMP=str(SCRATCH))

def read(path):return json.loads(path.read_text('utf-8'))
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def write(path,value):path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n','utf-8')

class CandidateAudit(audit_module.ProviderAudit):
    def send(self,original,client,request,*args,**kwargs):
        if self.count>=14:raise RuntimeError('Separately approved 14-request cap reached')
        return super().send(original,client,request,*args,**kwargs)


def main(approved):
    manifest_path=OUT/'outbound_review_manifest.json';manifest=read(manifest_path)
    if sha(manifest_path)!=approved:raise RuntimeError('Approval manifest changed')
    for name,digest in manifest['exact_source_hashes'].items():
        if sha(ROOT/name)!=digest:raise RuntimeError('Candidate changed before acceptance: '+name)
    if (OUT/'cloud_acceptance.json').exists():raise RuntimeError('Existing cloud result must be preserved')
    import backend as service
    import httpx
    from PIL import Image
    from report_grounding import check_source_detail_claims,check_observation_claims,check_detection_numbers
    audit=CandidateAudit(manifest);rows=[]
    cfg=copy.deepcopy(service._cfg);cfg['report_workflow']['enabled']=True
    client=service.app.test_client();service._llm_check_cache.update(t=0,ok=None)
    original=httpx.Client.send
    def blocked(*args,**kwargs):raise RuntimeError('Extra external search/download disabled')
    with ExitStack() as stack:
        stack.enter_context(patch.object(service,'_cfg',cfg))
        stack.enter_context(patch('httpx.Client.send',lambda c,r,*a,**kw:audit.send(original,c,r,*a,**kw)))
        stack.enter_context(patch('httpx.AsyncClient.send',side_effect=blocked))
        stack.enter_context(patch('requests.sessions.Session.request',side_effect=blocked))
        stack.enter_context(patch('urllib.request.urlopen',side_effect=blocked))
        for packet in manifest['tasks']:
            task=packet['task_id'];audit.current_task=task
            image_path=OUT/packet['image_file']
            if sha(image_path)!=packet['image_sha256']:raise RuntimeError('Approved picture changed')
            im=Image.open(image_path).convert('RGB');audit.current_images={hashlib.sha256(im.tobytes()).hexdigest()}
            for _ in range(3):
                buffer=io.BytesIO();im.save(buffer,format='JPEG')
                im=Image.open(io.BytesIO(buffer.getvalue())).convert('RGB');audit.current_images.add(hashlib.sha256(im.tobytes()).hexdigest())
            det=read(ROOT/'maintenance/phase4_20261013/cloud_inputs'/(task+'_detector.json'))
            paired=packet['kind']=='paired'
            if paired:det['change_overlay_b64']=det['overlay_b64']
            cid=packet['case_id']
            with patch('case_archive.uuid.uuid4',return_value=SimpleNamespace(hex=cid)):
                service.case_archive.create_case('公开Las Cruces专用报告角色验收',task,'工程验收，非专业用户或盲测')
            payload={'case_id':cid,'model_tag':'mine'}
            payload.update({'image_t1':'local_fixture','image_t2':'local_fixture'} if paired else {'image':'local_fixture'})
            before=audit.count;started=time.perf_counter()
            with patch.object(service,'detector_request',return_value=det):
                response=client.post('/api/pipeline_compare' if paired else '/api/pipeline_upload',json=payload)
            result=response.get_json();errors=[]
            if response.status_code!=200:errors.append('HTTP'+str(response.status_code))
            if result.get('generation',{}).get('status')!='live':errors.append('report_fallback')
            names=[a['name'] for a in result.get('agents',[])]
            if names!=packet['role_route']['roles']:errors.append('expected_roles_not_completed')
            workflow=result.get('quality',{}).get('workflow',{})
            if workflow.get('interrupted'):errors.append('workflow_interrupted')
            facts={**service.report_facts(det),'case_id':cid,'kind':'compare' if paired else 'single'}
            errors+=check_observation_claims(result.get('final_answer'),facts)
            errors+=check_detection_numbers(result.get('final_answer'),facts)
            errors+=check_source_detail_claims(result.get('final_answer'),packet['question'])
            scope=result.get('quality',{}).get('evidence_context') or {}
            if scope.get('case_id')!=cid or scope.get('request_id')!=det['request_id']:errors.append('scope_mismatch')
            rid=result.get('report_id')
            if rid:
                archived=service.case_archive.get_report(rid)
                if archived['objective_facts']!=result['objective_facts'] or archived['final_answer']!=result['final_answer']:errors.append('archive_mismatch')
                exported=client.post('/api/review_record',json={'report_id':rid,'format':'json'})
                if exported.status_code!=200:errors.append('export_failed')
                else:
                    packet_export=json.loads(base64.b64decode(exported.get_json()['content_b64']))
                    if packet_export['automatic_analysis']['answer']!=result['final_answer']:errors.append('export_mismatch')
            captured={k:v for k,v in result.items() if k not in ('detection','image_path')}
            rows.append({'task_id':task,'passed':not errors,'errors':errors,'roles':names,
                'provider_requests':audit.count-before,'elapsed_seconds':round(time.perf_counter()-started,3),'response':captured})
            print(json.dumps({'task':task,'passed':not errors,'errors':errors,'roles':names,'calls':audit.count-before},ensure_ascii=False),flush=True)
    value={'client_work_date':'2026-10-14','completed_at':datetime.now().astimezone().isoformat(),
        'manifest_sha256':approved,'user_approved_cap':14,'actual_provider_requests':audit.count,
        'total_provider_tokens':audit.total_tokens,'all_passed':all(r['passed'] for r in rows),
        'tasks':len(rows),'passed':sum(r['passed'] for r in rows),'rows':rows,
        'production_archive_untouched':True,'new_CNN_predictions_during_cloud_stage':0,
        'existing_GPU_outputs_replayed':True,'new_mine_or_blind_accuracy_scored':False,
        'general_multiagent_advantage_proved':False,'external_expert_review':False}
    write(OUT/'cloud_acceptance.json',value)
    print(json.dumps({k:v for k,v in value.items() if k!='rows'},ensure_ascii=False),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--approved-manifest-sha256',required=True)
    main(parser.parse_args().approved_manifest_sha256)
