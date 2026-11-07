"""Frozen one-hour functional workload; reports correctness, not mining accuracy."""
from pathlib import Path
from datetime import datetime,timezone
from concurrent.futures import ThreadPoolExecutor
import base64,hashlib,json,os,statistics,subprocess,sys,time,urllib.request,urllib.error
ROOT=Path('D:/GeoMMAgent');ML=Path('D:/LMHLD');sys.path.insert(0,str(ROOT))
from project_runtime import configure_runtime_paths
configure_runtime_paths()
import psutil
OUT=ROOT/'cache/deployment_soak_20261006';BASE='http://127.0.0.1:5110';DETECTOR='http://127.0.0.1:5111'


def write(path,value):Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
def get(path,host=BASE):
    with urllib.request.urlopen(host+path,timeout=10) as response:return json.loads(response.read())
def post(path,data):
    req=urllib.request.Request(BASE+path,data=json.dumps(data).encode(),headers={'Content-Type':'application/json'})
    try:
        with urllib.request.urlopen(req,timeout=180) as response:return response.status,json.loads(response.read())
    except urllib.error.HTTPError as error:return error.code,json.loads(error.read())
def launch():
    release=OUT/'release';env=os.environ.copy();env.update(KAT_LMHLD_DIR=str(release/'LMHLD'),DETECTOR_PORT='5111',
        KAT_MODELS_ROOT=str(ML/'models_prior'),KAT_MODEL_REGISTRY=str(release/'LMHLD/model_registry.json'),
        YOLO_AUTOINSTALL='false',PYTHONIOENCODING='utf-8',TEMP=str(OUT/'tmp'),TMP=str(OUT/'tmp'),
        TORCH_HOME=str(OUT/'runtime/torch'),HF_HOME=str(OUT/'runtime/huggingface'),MPLCONFIGDIR=str(OUT/'runtime/matplotlib'))
    procs=[]
    commands=[('detector',[str(OUT/'envs/lmhld/Scripts/python.exe'),'-u',str(release/'LMHLD/detector_service.py'),'5111'],release/'LMHLD'),
      ('backend',[str(OUT/'envs/GeoMMA/Scripts/python.exe'),'-u',str(ROOT/'validation/serve_isolated_release.py'),
                  '--release',str(release),'--db',str(OUT/'soak.sqlite3')],release/'GeoMMAgent')]
    for name,cmd,cwd in commands:
        stamp=str(time.time_ns());log=(OUT/(name+'_'+stamp+'.log')).open('w',encoding='utf-8')
        proc=subprocess.Popen(cmd,cwd=cwd,env=env,stdout=log,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW)
        procs.append((name,proc,log))
    for _ in range(120):
        try:
            health=get('/api/health');d=get('/api/health',DETECTOR)
            if health['detector_online'] and d['runtime_root']==str(release/'LMHLD'):break
        except Exception:pass
        if any(proc.poll() is not None for _,proc,_ in procs):raise RuntimeError('Isolated service startup failed;inspect deployment logs')
        time.sleep(.5)
    else:raise RuntimeError('Isolated startup exceeded60seconds')
    write(OUT/'owned_processes.json',{'launchers':{name:proc.pid for name,proc,_ in procs},
                                    'workers':{'backend':health['runtime_pid'],'detector':d['runtime_pid']}})
    return procs
def stop(procs):
    for _,proc,log in reversed(procs):
        try:
            children=psutil.Process(proc.pid).children(recursive=True)
            for child in reversed(children):
                command=' '.join(child.cmdline())
                if str(OUT) in command or 'serve_isolated_release.py' in command:child.terminate()
            psutil.wait_procs(children,timeout=10)
        except psutil.Error:pass
        if proc.poll() is None:proc.terminate()
        try:proc.wait(timeout=15)
        except subprocess.TimeoutExpired:proc.kill();proc.wait(timeout=10)
        log.close()
def assert_module_isolation(identity):
    prefix=Path(identity['prefix']).resolve()
    assert str(prefix).startswith(str(OUT/'envs'))
    for name,path in identity['module_paths'].items():assert Path(path).resolve().is_relative_to(prefix),(name,path)
    assert Path(identity['source']).resolve().is_relative_to((OUT/'release').resolve())
def measurements(procs):
    backend=get('/api/health');detector=get('/api/health',DETECTOR)
    values={'backend':psutil.Process(backend['runtime_pid']).memory_info().rss,
            'detector':psutil.Process(detector['runtime_pid']).memory_info().rss,
            'worker_pids':{'backend':backend['runtime_pid'],'detector':detector['runtime_pid']}}
    values['cuda']=detector['cuda_memory'];return values


def main():
    folders={'LasCruces':'LasCruces','Amyntaio':'Amyntaio','Hpakant':'Hpakant','CN_XJ2023':'Xinjing'}
    files={case:ML/'datasets/mine_benchmark_20261002/development'/folder/'post_BGRN_10m.tif' for case,folder in folders.items()}
    files['LasCruces_pre']=files['LasCruces'].with_name('pre_BGRN_10m.tif')
    protocol={'registered_at':datetime.now(timezone.utc).isoformat(),'minimum_wall_seconds':3600,'between_tasks_seconds':15,
      'cases':{name:{'path':str(path),'sha256':sha(path)} for name,path in files.items()},
      'sources_previously_observed_development_not_fresh_test':True,'one_frozen_default_model_no_parameter_search':True,
      'default_weight_sha256':sha(ML/'models_prior/real_ft_T1b_ours_seed42/best_model.pth'),
      'cloud_calls_allowed':0,'restart_services_halfway':True,'busy_and_bad_input_and_snapshot_checks':True,
      'monitored_processes':'actual service worker PIDs reported from inside services,not Windows venv launcher processes',
      'criteria':{'unexpected_failures':0,'lost_records_or_changed_facts':0,'late_RSS_growth_budget_bytes':256*1024**2,
                  'late_GPU_reserved_growth_budget_bytes':512*1024**2},
      'limits':'same host and base interpreter DLLs;fresh dependency environments and copied source;fixed external project assets;not24hourorcleanmachine qualification'}
    write(OUT/'protocol.json',protocol)
    data={name:base64.b64encode(path.read_bytes()).decode() for name,path in files.items()}
    procs=[];rows=[];samples=[];failures=[];restarted=False;record=None
    started=time.monotonic()
    try:
        procs=launch();identity=get('/api/deployment_identity');assert_module_isolation(identity)
        write(OUT/'deployment_identity.json',identity)
        status,case=post('/api/cases',{'mine_name':'固定开发影像持续验收','title':'独立部署运行检查','notes':'内部工程验收，不是矿区试用或模型效果评测'});assert status==201
        while time.monotonic()-started<protocol['minimum_wall_seconds']:
            index=len(rows);case_name=list(files)[:4][index%4]
            payload={'image':data[case_name],'model_tag':'mine','case_id':case['case_id']};t=time.monotonic()
            status,result=post('/api/pipeline_upload',payload)
            if status!=200:raise RuntimeError('Unexpected pipeline failure: '+json.dumps(result,ensure_ascii=False))
            assert result['review_handoff']['trace']['weights_sha256']==[protocol['default_weight_sha256']]
            assert result['review_handoff']['input_evidence']['image']['input_sha256']==protocol['cases'][case_name]['sha256']
            frozen=result['objective_facts'];snapshot=result['archive']['snapshot_sha256'];rid=result['report_id']
            loaded=get('/api/archive/reports/'+rid);assert loaded['objective_facts']==frozen
            if index%8==0:
                status,review=post('/api/archive/reports/'+rid+'/reviews',{'expected_revision':0,'target_key':'report','status':'insufficient_evidence',
                    'reviewer':'内部稳定性验收','evidence':'固定开发影像和同次模型结果；无现场或专家验证','note':'只核验工程运行，不评价事故和安全'});assert status==200
                assert review['objective_facts']==frozen and review['archive']['snapshot_sha256']==snapshot
            if index%12==0:
                status,bad=post('/api/detect_upload',{'image':'invalid'});assert status==400
                status,export=post('/api/review_record',{'report_id':rid});assert status==200
                assert export['record']['objective_facts']==frozen
            if index==8:
                with ThreadPoolExecutor(max_workers=3) as pool:busy=list(pool.map(lambda _:post('/api/detect_upload',{'image':data['LasCruces'],'model_tag':'mine'}),range(3)))
                assert any(status==200 for status,_ in busy) and any(status==503 for status,_ in busy)
                assert all(status in (200,503) for status,_ in busy)
                write(OUT/'busy_verification.json',{'passed':True,'statuses':[status for status,_ in busy],'no_unbounded_queue':True})
            if index==12:
                status,pair=post('/api/pipeline_compare',{'image_t1':data['LasCruces_pre'],'image_t2':data['LasCruces'],'model_tag':'mine','case_id':case['case_id']})
                assert status==200 and pair['objective_facts']['kind']=='paired_mask_change'
                status,export=post('/api/report_compare',{'report_id':pair['report_id'],'format':'html'});assert status==200
                write(OUT/'paired_verification.json',{'passed':True,'report_id':pair['report_id'],'facts':pair['objective_facts']})
            rows.append({'index':index,'case':case_name,'report_id':rid,'elapsed_seconds':round(time.monotonic()-t,3),
                         'snapshot_sha256':snapshot,'objective_facts_unchanged':True,'wall_elapsed':round(time.monotonic()-started,3)})
            sample={'index':index,'epoch':1 if restarted else 0,'wall_elapsed':rows[-1]['wall_elapsed'],**measurements(procs)};samples.append(sample)
            write(OUT/'partial_results.json',{'rows':rows,'memory_samples':samples,'elapsed_seconds':time.monotonic()-started,'restart_completed':restarted})
            if record is None:record={'report_id':rid,'facts':frozen,'snapshot_sha256':snapshot}
            if not restarted and time.monotonic()-started>=1800:
                for name,proc,_ in procs:
                    if name=='detector':
                        worker=psutil.Process(get('/api/health',DETECTOR)['runtime_pid']);worker.terminate();worker.wait(timeout=15)
                status,offline=post('/api/detect_upload',{'image':data['LasCruces'],'model_tag':'mine'})
                assert status==503
                still_readable=get('/api/archive/reports/'+record['report_id'])
                assert still_readable['objective_facts']==record['facts']
                write(OUT/'detector_outage_verification.json',{'passed':True,'live_detector_unavailable_503':True,'archive_readable_without_detector':True})
                stop(procs);procs=[];procs=launch();assert_module_isolation(get('/api/deployment_identity'))
                restored=get('/api/archive/reports/'+record['report_id']);assert restored['objective_facts']==record['facts'] and restored['archive']['snapshot_sha256']==record['snapshot_sha256']
                assert restored['review_handoff']['review_revision']==1
                restarted=True;write(OUT/'restart_verification.json',{'passed':True,'preserved_report_id':record['report_id'],'facts_and_review_preserved':True})
            time.sleep(15)
        memory={};qualified=True
        for epoch in (0,1):
            group=[sample for sample in samples if sample['epoch']==epoch]
            assert len(group)>=20
            warm=group[8:18];late=group[-10:]
            metrics={name:statistics.median([r[name] for r in late])-statistics.median([r[name] for r in warm]) for name in ['backend','detector']}
            metrics['GPU_reserved']=statistics.median([r['cuda']['reserved'] for r in late])-statistics.median([r['cuda']['reserved'] for r in warm])
            passed=all(metrics[name]<=protocol['criteria']['late_RSS_growth_budget_bytes'] for name in ['backend','detector']) and metrics['GPU_reserved']<=protocol['criteria']['late_GPU_reserved_growth_budget_bytes']
            memory[str(epoch)]={'growth_bytes':metrics,'gate_pass':passed};qualified &= passed
        receipt={'passed':qualified and restarted,'elapsed_seconds':time.monotonic()-started,'completed_tasks':len(rows),'restart_completed':restarted,
                 'memory_qualification':memory,'latency_median_seconds':statistics.median(r['elapsed_seconds'] for r in rows),
                 'deployment_identity':identity,'unexpected_failures':0,'new_training_or_parameter_tuning_or_reserved_predictions':0,
                 'cloud_calls':0,'known_development_functional_predictions_only':True,'no_mining_accuracy_or_long_term_field_claim':True}
        write(OUT/'verification.json',receipt);print('ONE HOUR QUALIFICATION COMPLETE',json.dumps(receipt,ensure_ascii=False),flush=True)
    except Exception as error:
        failures.append(str(error));write(OUT/'failure.json',{'errors':failures,'completed_tasks':len(rows),'elapsed_seconds':time.monotonic()-started});raise
    finally:stop(procs)


if __name__=='__main__':main()
