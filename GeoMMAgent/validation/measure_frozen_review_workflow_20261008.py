"""Matched internal file-handoff versus archived-review task benchmark, no human-time claim."""
from datetime import datetime
from pathlib import Path
from unittest.mock import patch
import argparse,base64,copy,hashlib,json,os,statistics,sys,time,zipfile

ROOT=Path('D:/GeoMMAgent');OUT=ROOT/'cache/independent_acceptance_20261007/review_workflow_benchmark'
BANK=Path('D:/LMHLD/experiments/same_input_reporting_workflow_20261005/detector_bank.json')


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def write(path,value):Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def forbidden(*args,**kwargs):raise AssertionError('Matched workflow benchmark permits no inference or cloud call')


def main():
    global OUT
    parser=argparse.ArgumentParser();parser.add_argument('--attempt',type=int,default=1)
    attempt=parser.parse_args().attempt;assert 1<=attempt<=5
    if attempt>1:OUT=OUT.with_name(OUT.name+'_v'+str(attempt))
    assert not OUT.exists(),'Do not replace a completed task benchmark'
    OUT.mkdir()
    manifest=read(ROOT/'cache/independent_acceptance_20261007/portable_release_manifest.json')
    release=Path(manifest['release'])
    for relative,digest in manifest['files'].items():assert sha(release/relative)==digest
    tasks=[task for task in read(BANK)['tasks'] if task['fixture_kind']=='real_observed_input']
    if not tasks:
        tasks=[task for task in read(BANK)['tasks'] if not task['fixture_kind'].startswith('synthetic') and task['id']!='LasCruces_RGB_metadata_missing']
    assert len(tasks)==6,[task['id'] for task in tasks]
    protocol={'registered_at':datetime.now().astimezone().isoformat(),'bank_sha256':sha(BANK),
        'task_ids':[task['id'] for task in tasks],'unique_development_mines':4,
        'arms':['flat JSON plus current review JSON plus ZIP handoff','frozen case archive plus revision-controlled review plus same-record export'],
        'same_detector_facts_and_review_text_in_both_arms':True,
        'order':'alternating arms across task IDs,one execution each',
        'required_tasks':['open same facts','save initial review','handle second writer on stale revision','export current handoff','reopen and recover record'],
        'scored_outcomes':['machine facts unchanged','record recovered','stale write rejected','initial review retained'],
        'latency_scope':'local record handling and export only;excludes inference,cloud calls,and human visual/geological review',
        'participants':'internal automation only','professional_user_time_savings_or_field_effectiveness_measured':False,
        'baseline_definition':'simple editable current-review file;no conflict control or history;not all alternative products',
        'no_claim_of_AI_accuracy_or_novel_algorithm_from_version_checks':True}
    write(OUT/'protocol.json',protocol)
    os.environ['KAT_LOCAL_FACTS_ONLY']='1';os.environ['KAT_CASE_ARCHIVE_PATH']=str(OUT/'archive.sqlite3')
    os.environ['KAT_LMHLD_DIR']=str(release/'LMHLD')
    sys.path[:0]=[str(release/'GeoMMAgent'),str(release/'LMHLD')]
    from project_runtime import configure_runtime_paths
    configure_runtime_paths()
    import backend as service
    from case_archive import CaseArchive
    from evidence_reporting import objective_facts
    assert Path(service.__file__).resolve().is_relative_to(release)
    rows=[]
    with patch.object(service,'detector_request',side_effect=forbidden),patch.object(service,'analyze_with_agents',side_effect=forbidden):
        client=service.app.test_client()
        for index,task in enumerate(tasks):
            det=copy.deepcopy(task['detection']);paired='t1_ratio' in det['stats']
            common={'region':'compare' if paired else 'upload','detection':det,
                'final_answer':'内部流程测量：同次量测保留，事故原因仍需专业核查。','knowledge':[],
                'generation':{'status':'fallback','source':'detection_summary'},'quality':{'degraded':True},'agents':[]}
            expected=objective_facts(det)
            initial={'status':'insufficient_evidence','reviewer':'内部自动化任务A','evidence':'同次冻结开发结果；不代表专家意见',
                'note':'第一条待核查意见；原因没有确认'}
            stale={**initial,'reviewer':'内部自动化任务B','note':'从旧版本写入的第二条意见'}
            def baseline():
                folder=OUT/'flat_reference'/task['id'];folder.mkdir(parents=True)
                start=time.perf_counter()
                record={**copy.deepcopy(common),'objective_facts':expected,'record_id':task['id']}
                write(folder/'result.json',record);write(folder/'review.json',initial)
                # Two scripted writers read the same starting state; baseline has no revision guard.
                write(folder/'review.json',stale)
                current=read(folder/'review.json')
                with zipfile.ZipFile(folder/'handoff.zip','w',compression=zipfile.ZIP_DEFLATED) as archive:
                    archive.write(folder/'result.json','result.json');archive.write(folder/'review.json','review.json')
                with zipfile.ZipFile(folder/'handoff.zip') as archive:
                    recovered=json.loads(archive.read('result.json'));review=json.loads(archive.read('review.json'))
                elapsed=time.perf_counter()-start
                return {'seconds':elapsed,'machine_facts_unchanged':recovered['detection']==det and recovered['objective_facts']==expected,
                    'record_recovered':recovered['record_id']==task['id'],
                    'stale_write_rejected':False,'initial_review_retained':review==initial,
                    'handoff_bytes':(folder/'handoff.zip').stat().st_size}
            def assisted():
                start=time.perf_counter()
                response=client.post('/api/cases',json={'mine_name':'内部流程测量 '+task['id'],'title':'同次结果交接与复核',
                    'notes':'自动化对照，非专业用户任务研究'})
                assert response.status_code in (200,201),response.get_json()
                saved=service.cache_report(copy.deepcopy(common),response.get_json()['case_id']);rid=saved['report_id']
                assert saved['objective_facts']==expected
                one=client.post('/api/archive/reports/'+rid+'/reviews',json={**initial,'expected_revision':0,'target_key':'report'})
                assert one.status_code==200,one.get_json()
                two=client.post('/api/archive/reports/'+rid+'/reviews',json={**stale,'expected_revision':0,'target_key':'report'})
                assert two.status_code==409,two.get_json()
                exported=client.post('/api/review_record',json={'report_id':rid});assert exported.status_code==200,exported.get_json()
                body=exported.get_json();data=base64.b64decode(body['content_b64']);handoff=json.loads(data)
                (OUT/(task['id']+'_review.json')).write_bytes(data)
                # Clear volatile lookup and reopen the persistent store for the last task stage.
                service._report_cache.clear()
                reopened=CaseArchive(OUT/'archive.sqlite3').get_report(rid)
                elapsed=time.perf_counter()-start
                return {'seconds':elapsed,'machine_facts_unchanged':reopened['detection']==det and reopened['objective_facts']==expected,
                    'record_recovered':handoff['trace']['report_id']==rid and reopened['report_id']==rid,
                    'stale_write_rejected':two.status_code==409,
                    'initial_review_retained':handoff['review_revision']==1 and len(handoff['review_history'])==1 and handoff['review_history'][0]['note']==initial['note'],
                    'handoff_bytes':len(data),'report_id':rid,
                    'machine_snapshot_sha256':saved['archive']['snapshot_sha256']}
            if index%2==0:flat=baseline();full=assisted()
            else:full=assisted();flat=baseline()
            assert flat['machine_facts_unchanged'] and flat['record_recovered']
            assert all(full[key] for key in ['machine_facts_unchanged','record_recovered','stale_write_rejected','initial_review_retained'])
            rows.append({'task_id':task['id'],'flat_reference':flat,'archived_workflow':full})
            print('MEASURED',task['id'],'flat',round(flat['seconds'],4),'archive',round(full['seconds'],4),flush=True)
    summaries={}
    for arm in ['flat_reference','archived_workflow']:
        summaries[arm]={key:sum(row[arm][key] for row in rows) for key in ['machine_facts_unchanged','record_recovered','stale_write_rejected','initial_review_retained']}
        summaries[arm].update(tasks=len(rows),median_record_handling_seconds=statistics.median(row[arm]['seconds'] for row in rows),
            total_record_handling_seconds=sum(row[arm]['seconds'] for row in rows))
    result={'completed_at':datetime.now().astimezone().isoformat(),'passed_scoped_measurement':True,
        'protocol_sha256':sha(OUT/'protocol.json'),'rows':rows,'summary':summaries,
        'human_or_geological_review_time_measured':False,'human_time_saving_claim_allowed':False,
        'new_detector_or_cloud_calls':0,'scope':'matched internal record handling benchmark using frozen development outputs',
        'limits':['six tasks on four already observed mines,not independent accident accuracy',
            'baseline is explicitly simple file handoff;other systems can implement version control',
            'measured stale-write loss prevention is engineering workflow value,not geological review accuracy',
            'actual operator cognitive effort,time savings and professional usability remain unmeasured']}
    write(OUT/'results.json',result)
    for relative,digest in manifest['files'].items():assert sha(release/relative)==digest,relative
    print(json.dumps(summaries,ensure_ascii=False),flush=True)


if __name__=='__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    main()
