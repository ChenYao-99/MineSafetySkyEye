"""Freeze exact report-role inputs for review without any provider requests."""
from pathlib import Path
from datetime import datetime
from unittest.mock import patch
import json,os,sys,hashlib
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent
SCRATCH=OUT/'_scratch';SCRATCH.mkdir(exist_ok=True)
os.environ.update(KAT_LOCAL_FACTS_ONLY='1',KAT_CASE_ARCHIVE_PATH=str(SCRATCH/'prepare.sqlite3'))
sys.path.insert(0,str(ROOT))
import backend as service
from report_workflow import AUDIT_PROMPT,WRITER_PROMPT,REVIEW_PROMPT,choose_report_route
from evidence_reporting import bounded_input
from report_grounding import report_input_context

def read(path):return json.loads(path.read_text('utf-8'))
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def write(path,value):path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n','utf-8')

def main():
    path=OUT/'outbound_review_manifest.json'
    if path.exists():raise RuntimeError('Prepared manifest already exists; do not silently rewrite it')
    old=read(ROOT/'maintenance/phase4_20261013/outbound_review_manifest.json');packets=[]
    client=service.app.test_client()
    for old_packet in old['tasks']:
        task=old_packet['task_id'];paired=old_packet['kind']=='paired'
        det=read(ROOT/'maintenance/phase4_20261013/cloud_inputs'/(task+'_detector.json'))
        if paired:det['change_overlay_b64']=det['overlay_b64']
        case=service.case_archive.create_case('Las Cruces公开影像角色链验收',task,'不是新增矿区、独立盲测或专业用户评估')
        payload={'case_id':case['case_id'],'model_tag':'mine'}
        payload.update({'image_t1':'local_fixture','image_t2':'local_fixture'} if paired else {'image':'local_fixture'})
        with patch.object(service,'detector_request',return_value=det),patch.object(service,'llm_fast_check',return_value=False):
            response=client.post('/api/pipeline_compare' if paired else '/api/pipeline_upload',json=payload)
        assert response.status_code==200,response.get_json()
        body=response.get_json()
        facts={**service.report_facts(det),'case_id':case['case_id'],'kind':'compare' if paired else 'single'}
        packets.append({'task_id':task,'kind':old_packet['kind'],'case_id':case['case_id'],
            'request_id':det['request_id'],'question':body['question']+'\n\n'+report_input_context(facts),
            'tool_fields':bounded_input(facts),'role_route':choose_report_route(facts),
            'image_file':'../phase4_20261013/'+old_packet['image_file'],
            'image_sha256':old_packet['image_sha256'],'image_bytes':old_packet['image_bytes']})
    manifest={'schema_version':1,'client_work_date':'2026-10-14','status':'prepared_not_sent',
        'destination':old['destination'],'models':old['models'],'max_generation_requests':14,
        'max_output_tokens_per_request':2000,'total_provider_token_budget':200000,
        'purpose':'验证专用证据核查—报告—按需复核角色的实际传递、完整性与交付，不作为新盲测或协同优势成绩',
        'images_already_authorized_in_prior_manifest_sha256':sha(ROOT/'maintenance/phase4_20261013/outbound_review_manifest.json'),
        'additional_generation_allowance_required':True,'prior_phase_actual_calls':23,'prior_24_cap_not_reset':True,
        'role_prompts':{'EvidenceAuditAgent':AUDIT_PROMPT,'ReasoningAgent':WRITER_PROMPT,'ReportReviewAgent':REVIEW_PROMPT},
        'exact_source_hashes':{name:sha(ROOT/name) for name in ['backend.py','report_workflow.py','coordinator/coordinator.py','report_grounding.py','configs/GeoMMBench.yaml']},
        'later_role_input_rule':'同次原始工具字段和给定知识片段加前序角色输出；结构化核查错误立即停止后续，未完成步骤如实标记',
        'extra_search_or_downloads':False,'SDK_hidden_retries':0,
        'excluded':old['excluded'],'tasks':packets,
        'scope':'沿用此前批准的3张公开影像覆盖JPEG和事实类型；新增专用角色提示及额外最多14次生成调用',
        'limits':['原有开发影像与已使用任务，不是新盲测','本次只验收链路，不证明独立专家正确率','候选默认关闭，真实验收后决定启用']}
    write(path,manifest)
    write(OUT/'state.json',{'client_work_date':'2026-10-14','status':'candidate_implemented_local_tests_passed_pending_real_validation',
        'root_cause':'通用调度器在工具事实齐全时选择ReasoningAgent，报告缺专用证据与复核角色；前端未漏展示',
        'candidate_default_enabled':False,'local_role_contract_tests':8,'cloud_calls':0,
        'manifest_sha256':sha(path),'submitted_materials_changed':False,'new_CNN_training':0})
    print(json.dumps({'tasks':len(packets),'roles':[p['role_route']['roles'] for p in packets],
        'manifest_sha256':sha(path),'new_cloud_calls':0},ensure_ascii=False))

if __name__=='__main__':main()
