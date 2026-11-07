"""Isolated browser acceptance server with explicitly labelled scripted role fixtures."""
from pathlib import Path
from types import SimpleNamespace
import copy,json,os,sys,threading
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent
SCRATCH=OUT/'_scratch';SCRATCH.mkdir(exist_ok=True)
os.environ.update(KAT_LOCAL_FACTS_ONLY='1',KAT_CASE_ARCHIVE_PATH=str(SCRATCH/'browser.sqlite3'),TEMP=str(SCRATCH),TMP=str(SCRATCH))
sys.path.insert(0,str(ROOT))
import backend as service
from flask import jsonify
from report_grounding import fallback_report
from report_workflow import workflow_status
from coordinator.coordinator import AgentStep


def forbid(*args,**kwargs):raise AssertionError('No model inference or external network in browser acceptance')
service.llm_fast_check=lambda *args,**kwargs:False
service.analyze_with_agents=forbid
service.detector_request=lambda *args,**kwargs:{'status':'disabled_for_isolated_browser_test','detector_loaded':False}


@service.app.route('/api/competition_fixture/<kind>')
def fixture(kind):
    paired=kind!='single';task='M02' if paired else 'M01'
    det=json.loads((ROOT/('maintenance/phase4_20261013/cloud_inputs/'+task+'_detector.json')).read_text('utf-8'))
    case=service.case_archive.create_case('内部浏览器功能验收',kind,'受控角色输出，非新AI实验、真实用户评价或专家反馈')
    names=['EvidenceAuditAgent','ReasoningAgent']+(['ReportReviewAgent'] if paired else [])
    plan=[{'agent':i,'name':name,'subtask':'本地受控界面检查'} for i,name in enumerate(names)]
    outputs=[json.dumps({'decision':'needs_review','checks':['同次案例、时相与数值已传递'],
                        'unknowns':['现场原因待人工复核'],'issues':[],'source_ids':[]},ensure_ascii=False),
             '受控界面检查：根据本次工具量测生成复核记录，不是云端新分析。',
             '受控界面检查：双期净变化使用百分点，保留原因待复核。']
    steps=[AgentStep(step=i+1,agent_index=i,agent_name=name,agent_description='工程测试角色',
                     subtask=plan[i]['subtask'],output=outputs[i]) for i,name in enumerate(names)]
    if kind=='blocked':
        steps=steps[:1];steps[0].status='invalid';steps[0].validation_errors=['audit_numeric_claim_inconsistent']
    trace=SimpleNamespace(dispatch_plan=plan,agent_steps=steps)
    wf=workflow_status(trace,mode='scripted_browser_test',reason='明确标记的受控输出，仅验收前端和归档交接')
    facts={**service.report_facts(det),'kind':'compare' if paired else 'single'}
    answer=fallback_report(facts,'本地受控演示验收，不是新增AI结果')
    return jsonify(service.cache_report({'region':'compare' if paired else 'upload','detection':det,
        'knowledge':[],'agents':[{'step':s.step,'name':s.agent_name,'subtask':s.subtask,'output':s.output,'status':s.status,'validation_errors':s.validation_errors} for s in steps],
        'final_answer':answer,'is_mock':True,'generation':{'status':'fallback','source':'scripted_browser_test'},
        'quality':{'guard_pass':False,'degraded':True,'missing':['本地界面验收，不作新AI结果'],'workflow':wf,
                   'human_review_required':True}},case['case_id']))


SERVER=None
@service.app.route('/api/competition_fixture_shutdown',methods=['POST'])
def shutdown():
    threading.Thread(target=SERVER.shutdown,daemon=True).start()
    return jsonify({'status':'stopping'})


if __name__=='__main__':
    from werkzeug.serving import make_server
    SERVER=make_server('127.0.0.1',5122,service.app,threaded=True)
    timer=threading.Timer(180,SERVER.shutdown);timer.daemon=True;timer.start()
    try:SERVER.serve_forever()
    finally:timer.cancel();SERVER.server_close()
