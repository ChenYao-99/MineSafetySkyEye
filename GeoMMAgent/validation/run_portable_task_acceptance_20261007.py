"""Actual GPU/public-image tasks in a copied deployment; never a professional trial."""
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
import base64, hashlib, json, os, sqlite3, statistics, subprocess, sys, time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from project_runtime import configure_runtime_paths
configure_runtime_paths()
import psutil,requests

OUT=Path(os.environ.get('KAT_ENGINEERING_OUT',str(ROOT/'cache/portable_task_acceptance_20261007')));RELEASE=OUT/'迁移验收 空格'
AGENT=RELEASE/'GeoMMAgent';ML=RELEASE/'LMHLD'
AGENT_PY=ROOT/'cache/deployment_soak_20261006/envs/GeoMMA/Scripts/python.exe'
ML_PY=ROOT/'cache/deployment_soak_20261006/envs/lmhld/Scripts/python.exe'
BASE='http://127.0.0.1:5118';DET='http://127.0.0.1:5119'
session=requests.Session();session.trust_env=False


def write(name,value):
    (OUT/name).write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def get(path,host=BASE):
    response=session.get(host+path,timeout=15);response.raise_for_status();return response.json()


def post(path,data):
    response=session.post(BASE+path,json=data,timeout=180)
    return response.status_code,response.json()


def launch(environment):
    command=['powershell.exe','-NoProfile','-ExecutionPolicy','Bypass','-File',str(AGENT/'start_demo.ps1'),
      '-NoBrowser','-LocalFactsOnly','-AgentPython',str(AGENT_PY),'-DetectorPython',str(ML_PY),
      '-DetectorDirectory',str(ML),'-BackendPort','5118','-DetectorPort','5119']
    # Descendant Windows workers may inherit PIPE handles and keep communicate()
    # waiting after the launcher exits. Write directly to an owned D: log instead.
    with (OUT/('startup_'+str(time.time_ns())+'.log')).open('wb') as log:
        result=subprocess.run(command,env=environment,stdout=log,stderr=subprocess.STDOUT,timeout=120,creationflags=subprocess.CREATE_NO_WINDOW)
    assert result.returncode==0,'Portable launcher failed;inspect startup logs'
    health=get('/api/health');det=get('/api/health',DET)
    assert health['runtime_root']==str(AGENT) and det['runtime_root']==str(ML)
    assert health['runtime_python_prefix']==str(AGENT_PY.parent.parent)
    assert det['runtime_python_prefix']==str(ML_PY.parent.parent)
    assert health['reporting_mode']=='local_facts_only' and health['detector_runtime_compatible']
    assert health['http_runtime']=='waitress' and det['http_runtime']=='waitress'
    return health,det


def stop():
    result=subprocess.run(['powershell.exe','-NoProfile','-ExecutionPolicy','Bypass','-File',str(AGENT/'stop_demo.ps1'),
        '-DetectorDirectory',str(ML),'-BackendPort','5118','-DetectorPort','5119'],capture_output=True,timeout=40,creationflags=subprocess.CREATE_NO_WINDOW)
    assert result.returncode==0,'Owned deployment stop failed'


def main():
    OUT.mkdir(exist_ok=True)
    from build_portable_release_20261007 import build
    build(OUT)
    state=Path('D:/LMHLD/datasets/mine_benchmark_20261002/STATE.json');state_before=sha(state)
    manifest=json.loads((OUT/'portable_release_manifest.json').read_text(encoding='utf-8'))
    protocol={'registered_at':datetime.now().astimezone().isoformat(),'uninterrupted_workload_seconds':int(os.environ.get('KAT_ACCEPTANCE_SECONDS','1200')),
        'fixed_default_weight_sha256':manifest['default_weight_SHA256'],'production_archive_written':False,
        'scope':'actual public/development-image tasks,local facts reporting,copied source/default assets,new dependency environments on same machine',
        'not_another_machine_or_professional_user_trial_or_independent_accident_test':True,
        'new_training_or_threshold_tuning':0,'cloud_calls_allowed':0,'heldout_auxiliary_inputs_not_used':True}
    write('task_protocol.json',protocol)
    write('qualified_workload_release_manifest.json',manifest)
    environment=os.environ.copy();environment.update(KAT_MODELS_ROOT=str(ML/'models_prior'),KAT_MODEL_REGISTRY=str(ML/'model_registry.json'),
        TEMP=str(OUT/'tmp'),TMP=str(OUT/'tmp'),PYTHONIOENCODING='utf-8',YOLO_AUTOINSTALL='false')
    environment['KAT_CASE_ARCHIVE_PATH']=str(OUT/('task_archive_'+str(time.time_ns())+'.sqlite3'))
    qualification=json.loads(Path('D:/LMHLD/datasets/data_conditions_20261007/MineNetCD_qualification.json').read_text(encoding='utf-8'))
    original=Path('D:/LMHLD/datasets/mine_benchmark_20261002/development')
    inputs={name:original/folder/'post_BGRN_10m.tif' for name,folder in [('LasCruces','LasCruces'),('Amyntaio','Amyntaio'),('Hpakant','Hpakant'),('CN_XJ2023','Xinjing')]}
    inputs['LasCruces_pre']=original/'LasCruces/pre_BGRN_10m.tif'
    inputs['CN_XJ2023_pre']=original/'Xinjing/pre_BGRN_10m.tif'
    for site in ['Austria_01','Namibia2']:
        for epoch in ['im1','im2']:
            row=next(x for x in qualification['engineering_inputs'] if x['site']==site and x['epoch']==epoch)
            inputs[site+'_'+epoch]=Path(row['path'])
    data={name:base64.b64encode(path.read_bytes()).decode() for name,path in inputs.items()}
    write('task_inputs.json',{'sources':{name:{'path':str(path),'sha256':sha(path),'role':'development engineering task'} for name,path in inputs.items()},
        'derivative_provenance':qualification['engineering_inputs'],'mine_change_labels_not_scored_as_accident_truth':True})
    summaries=[];memory=[];first=None;running=False
    try:
        health,det=launch(environment);running=True
        write('startup_identity.json',{'backend':health,'detector':det})
        # Same launcher twice must reuse our deployment, rather than spawn another worker.
        reused=launch(environment)
        assert reused[0]['runtime_pid']==health['runtime_pid'] and reused[1]['runtime_pid']==det['runtime_pid']
        code,case=post('/api/cases',{'mine_name':'公开影像任务工程验收','title':'跨目录部署与任务交付','notes':'内部工程验收，不是矿方试用或专家反馈'})
        assert code==201
        singles=['LasCruces','Amyntaio','Hpakant','CN_XJ2023','Austria_01_im2','Namibia2_im2']
        for name in singles:
            started=time.monotonic();code,result=post('/api/pipeline_upload',{'image':data[name],'model_tag':'mine','case_id':case['case_id']})
            assert code==200,(name,result)
            assert result['review_handoff']['trace']['weights_sha256']==[protocol['fixed_default_weight_sha256']]
            assert result['review_handoff']['input_evidence']['image']['input_sha256']==sha(inputs[name])
            assert result['generation']['status']!='success'
            rid=result['report_id'];frozen=result['objective_facts'];snapshot=result['archive']['snapshot_sha256']
            review={'expected_revision':0,'target_key':'report','status':'insufficient_evidence','reviewer':'内部任务验收',
                'evidence':'公开开发影像和同次工具结果；模拟功能验收意见，非专业用户反馈','note':'量测与来源保留，变化原因仍须进一步核查'}
            code,current=post('/api/archive/reports/'+rid+'/reviews',review);assert code==200
            assert current['objective_facts']==frozen and current['archive']['snapshot_sha256']==snapshot
            assert post('/api/archive/reports/'+rid+'/reviews',review)[0]==409
            code,export=post('/api/review_record',{'report_id':rid});assert code==200
            assert json.loads(base64.b64decode(export['content_b64']))==export['record']
            assert export['record']['review_revision']==1 and not export['record']['project_field_trial_or_expert_endorsement']
            if name.endswith('_im2'):
                assert any(x['code']=='PHYSICAL_INDEX_UNAVAILABLE' for x in export['record']['pending_checks'])
            summaries.append({'task':name,'kind':'single','report_id':rid,'input_sha256':sha(inputs[name]),
                'snapshot_sha256':snapshot,'facts':frozen,'elapsed_seconds':time.monotonic()-started,'passed':True})
            if first is None:first=summaries[-1]
        # A new centre-crop pair may lack stable control points. A clear rejection
        # is the expected safe outcome; never bypass registration to obtain a pass.
        code,rejected=post('/api/pipeline_compare',{'image_t1':data['Austria_01_im1'],'image_t2':data['Austria_01_im2'],'model_tag':'mine','case_id':case['case_id']})
        assert code==400 and '配准' in rejected.get('error',''),rejected
        write('new_RGB_pair_registration_rejection.json',{'passed':True,'status':code,'response':rejected,
            'source_pair':'MineNetCD Austria_01 fixed centre crops','registration_override_used':False,
            'scope':'safe input rejection;no claim that all original publisher pairs are misregistered'})
        for name,left,right in [('LasCruces_pair','LasCruces_pre','LasCruces'),('CN_XJ2023_pair','CN_XJ2023_pre','CN_XJ2023')]:
            code,result=post('/api/pipeline_compare',{'image_t1':data[left],'image_t2':data[right],'model_tag':'mine','case_id':case['case_id']});assert code==200,(name,result)
            rid=result['report_id'];code,export=post('/api/report_compare',{'report_id':rid,'format':'html'});assert code==200
            html=base64.b64decode(export['content_b64']).decode('utf-8');assert rid in html
            (OUT/(name+'.html')).write_text(html,encoding='utf-8')
            summaries.append({'task':name,'kind':'paired','report_id':rid,'snapshot_sha256':result['archive']['snapshot_sha256'],
                'facts':result['objective_facts'],'paired_HTML_same_record':True,'passed':True})
        assert post('/api/detect_upload',{'image':'invalid'})[0]==400
        assert post('/api/pipeline_upload',{'image':data['LasCruces'],'case_id':'not-a-case'})[0]==404
        response=session.post(BASE+'/api/detect_upload',data=b'{"image":"'+b'x'*(48*1024**2)+b'"}',headers={'Content-Type':'application/json'},timeout=30)
        assert response.status_code==413 and response.json().get('error')
        def busy(_):
            s=requests.Session();s.trust_env=False
            try:
                return s.post(BASE+'/api/detect_upload',json={'image':data['LasCruces'],'model_tag':'mine'},timeout=180).status_code
            finally:s.close()
        with ThreadPoolExecutor(max_workers=12) as pool:statuses=list(pool.map(busy,range(12)))
        assert 200 in statuses and 503 in statuses and set(statuses)<={200,503},statuses
        write('task_acceptance_partial.json',{'passed_tasks':len(summaries),'tasks':summaries,'concurrent_statuses':statuses,
            'real_GPU_predictions':True,'user_or_expert_trial':False})
        start=time.monotonic();worker_pids=(health['runtime_pid'],det['runtime_pid']);rounds=[]
        while time.monotonic()-start<protocol['uninterrupted_workload_seconds']:
            name=singles[len(rounds)%len(singles)];code,result=post('/api/pipeline_upload',{'image':data[name],'model_tag':'mine','case_id':case['case_id']})
            assert code==200
            assert result['review_handoff']['trace']['weights_sha256']==[protocol['fixed_default_weight_sha256']]
            cached=get('/api/archive/reports/'+result['report_id']);assert cached['objective_facts']==result['objective_facts']
            h=get('/api/health');d=get('/api/health',DET);assert (h['runtime_pid'],d['runtime_pid'])==worker_pids
            memory.append({'elapsed_seconds':time.monotonic()-start,'backend_RSS':psutil.Process(worker_pids[0]).memory_info().rss,
                           'detector_RSS':psutil.Process(worker_pids[1]).memory_info().rss,'GPU_reserved':d['cuda_memory']['reserved']})
            rounds.append({'input':name,'report_id':result['report_id'],'snapshot_sha256':result['archive']['snapshot_sha256']})
            write('uninterrupted_partial.json',{'elapsed_seconds':time.monotonic()-start,'rounds':rounds,'memory':memory,'worker_pids':worker_pids})
            if len(rounds)%5==0:print('CONTINUOUS',round(time.monotonic()-start),len(rounds),flush=True)
            time.sleep(15)
        continuous_elapsed=time.monotonic()-start
        assert len(memory)>=30
        growth={key:statistics.median(x[key] for x in memory[-10:])-statistics.median(x[key] for x in memory[5:15])
                for key in ['backend_RSS','detector_RSS','GPU_reserved']}
        assert growth['backend_RSS']<=256*1024**2 and growth['detector_RSS']<=256*1024**2 and growth['GPU_reserved']<=512*1024**2
        response=session.post(BASE+'/api/case_archive/backup',json={},timeout=30);assert response.status_code==200
        backup=OUT/'actual_task_backup.sqlite3';backup.write_bytes(response.content)
        with sqlite3.connect(backup) as db:assert db.execute('PRAGMA integrity_check').fetchone()[0]=='ok'
        stop();running=False
        restore=AGENT/'data/restored_case_archive.sqlite3';restore.parent.mkdir(exist_ok=True);restore.write_bytes(backup.read_bytes())
        restored_environment=environment.copy();restored_environment['KAT_CASE_ARCHIVE_PATH']=str(restore)
        restored_health,restored_det=launch(restored_environment);running=True
        original_result=get('/api/archive/reports/'+first['report_id'])
        assert original_result['objective_facts']==first['facts'] and original_result['archive']['snapshot_sha256']==first['snapshot_sha256']
        assert original_result['review_handoff']['review_revision']==1
        assert sha(state)==state_before
        receipt={'passed':True,'verified_at':datetime.now().astimezone().isoformat(),'protocol':protocol,'tasks':summaries,
            'completed_real_input_tasks':len(summaries),'concurrent_statuses':statuses,'unexpected_failures':0,
            'continuous_seconds':continuous_elapsed,'continuous_rounds':len(rounds),'uninterrupted_worker_PIDs_verified':True,
            'memory_growth_bytes':growth,'database_backup_integrity_and_restore_to_different_path_verified':True,
            'idempotent_portable_launcher_verified':True,'Unicode_and_space_paths_verified':True,
            'new_RGB_pair_without_reliable_registration_refused_without_override':True,
            'same_machine_not_clean_second_machine':True,'no_professional_user_trial_or_accuracy_measurement':True,
            'new_cloud_calls_or_training_or_reserved_predictions':0,'benchmark_STATE_unchanged':True,
            'optional_models_and_builtin_source_datasets_not_packaged':True}
        write('verification.json',receipt);print('PASS PORTABLE ACTUAL TASK ACCEPTANCE',flush=True)
    except Exception as error:
        write('failure.json',{'passed':False,'error':str(error),'tasks_completed':len(summaries)})
        raise
    finally:
        stop()


if __name__=='__main__':
    sys.stdout.reconfigure(encoding='utf-8');main()
