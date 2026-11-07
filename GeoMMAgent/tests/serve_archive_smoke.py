"""Restartable loopback-only archive acceptance server; inference/cloud forbidden."""
from pathlib import Path
import argparse,copy,json,sys,threading
ROOT=Path('D:/GeoMMAgent');sys.path.insert(0,str(ROOT))
import backend as service
from case_archive import CaseArchive
from flask import jsonify
from werkzeug.serving import make_server


def forbidden(*args,**kwargs):raise AssertionError('No inference or cloud permitted in archive acceptance')
service.coordinator=None;service.llm_fast_check=lambda *args,**kwargs:False;service.analyze_with_agents=forbidden
service.detector_request=lambda endpoint,*args,**kwargs: {'status':'offline_for_acceptance','detector_loaded':False} if endpoint=='/api/health' else forbidden()
server=None


@service.app.route('/api/local_archive_fixture')
def fixture():
    listing=service.case_archive.list_cases('内部归档验收')
    created=not listing['cases']
    if created:
        case=service.case_archive.create_case('内部归档验收矿区','保存与重启复核','工程测试，不是现场试用')
        bank=json.loads(Path('D:/LMHLD/experiments/same_input_reporting_workflow_20261005/detector_bank.json').read_text(encoding='utf-8'))
        for name,region in [('LasCruces_single','upload'),('LasCruces_pair','compare')]:
            detection=copy.deepcopy(next(row['detection'] for row in bank['tasks'] if row['id']==name))
            service.cache_report({'region':region,'target_index':None,'detection':detection,'agents':[],'knowledge':[],
                'final_answer':'同次检测事实；变化原因仍待核查。','quality':{'degraded':True},
                'generation':{'status':'fallback','source':'detection_summary'}},case['case_id'])
    else:case=listing['cases'][0]
    rows=service.case_archive.list_reports(case['case_id'])
    return jsonify({'case':rows['case'],'reports':rows['reports'],'created_new_fixture':created,
                    'scope':'engineering acceptance only;not a user trial or model evaluation'})


@service.app.route('/api/local_archive_shutdown',methods=['POST'])
def shutdown():
    threading.Thread(target=server.shutdown,daemon=True).start();return jsonify({'status':'stopping'})


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--db',required=True);parser.add_argument('--port',type=int,default=5105);args=parser.parse_args()
    service.case_archive=CaseArchive(args.db)
    server=make_server('127.0.0.1',args.port,service.app,threaded=True)
    timer=threading.Timer(240,server.shutdown);timer.daemon=True;timer.start()
    try:server.serve_forever()
    finally:timer.cancel();server.server_close()
