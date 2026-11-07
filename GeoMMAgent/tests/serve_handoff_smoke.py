"""Ephemeral loopback-only acceptance server. No inference or cloud calls allowed."""
from pathlib import Path
import argparse,copy, json, sys,threading
ROOT=Path('D:/GeoMMAgent');sys.path.insert(0,str(ROOT))
import backend as service
from case_archive import CaseArchive
import tempfile
test_archive_dir=tempfile.TemporaryDirectory(dir=ROOT/'cache')
service.case_archive=CaseArchive(Path(test_archive_dir.name)/'cases.sqlite3')
from flask import jsonify

def forbid(*args,**kwargs):
    raise AssertionError('Inference and cloud requests are disabled for browser handoff acceptance')

service.llm_fast_check=lambda *args,**kwargs:False
service.analyze_with_agents=forbid
service.detector_request=lambda endpoint,*args,**kwargs: {'status':'disabled_for_local_acceptance','detector_loaded':False} if endpoint=='/api/health' else forbid()
service.coordinator=None
bank=json.loads(Path('D:/LMHLD/experiments/same_input_reporting_workflow_20261005/detector_bank.json').read_text(encoding='utf-8'))
KNOWN_WORK=False
SMOKE_SERVER=None

@service.app.route('/api/local_handoff_smoke_fixture')
def fixture():
    rows=[]
    for name,slot in [('LasCruces_single','upload'),('LasCruces_pair','compare')]:
        task=next(t for t in bank['tasks'] if t['id']==name)
        detection=copy.deepcopy(task['detection'])
        if KNOWN_WORK and slot=='upload':
            path=Path('D:/LMHLD/experiments/government_work_runtime_validation_20261005/API_context_replay_verification.json')
            detection=json.loads(path.read_text(encoding='utf-8'))['actual_detector_result_with_public_context']
        result=service.cache_report({'region':'compare' if slot=='compare' else 'upload',
            'detection':detection,'knowledge':[],
            'final_answer':'本地导出验收：模型掩膜仅供筛查，原因与现场状态待人工复核。',
            'agents':[],'generation':{'status':'fallback','source':'detection_summary'},'quality':{'degraded':True}})
        rows.append({'slot':slot,'result':result})
    return jsonify({'scope':'observed development cache acceptance, not user trial',
                    'fixture_version':'authoritative_facts_and_public_work_context_v2','known_work':KNOWN_WORK,'rows':rows})

@service.app.route('/api/local_handoff_smoke_shutdown',methods=['POST'])
def shutdown():
    threading.Thread(target=SMOKE_SERVER.shutdown,daemon=True).start()
    return jsonify({'status':'ephemeral_test_server_stopping'})

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--known-work',action='store_true')
    parser.add_argument('--port',type=int,default=5105);args=parser.parse_args()
    KNOWN_WORK=args.known_work
    from werkzeug.serving import make_server
    SMOKE_SERVER=make_server('127.0.0.1',args.port,service.app,threaded=True)
    timer=threading.Timer(180,SMOKE_SERVER.shutdown);timer.daemon=True;timer.start()
    try:SMOKE_SERVER.serve_forever()
    finally:timer.cancel();SMOKE_SERVER.server_close();test_archive_dir.cleanup()
