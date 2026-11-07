"""Internal review and local replay of frozen actual provider outputs; no new calls."""
from pathlib import Path
from datetime import datetime
import json
import os
import sys

ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent
os.environ['KAT_LOCAL_FACTS_ONLY']='1'
sys.path.insert(0,str(ROOT))
import backend as service
from report_grounding import check_source_detail_claims,check_observation_claims,check_detection_numbers

manifest=json.loads((OUT/'outbound_review_manifest.json').read_text('utf-8'))
cloud=json.loads((OUT/'cloud_acceptance.json').read_text('utf-8'))
events=[json.loads(line) for line in (OUT/'provider_receipts.jsonl').read_text('utf-8').splitlines() if line.strip()]
raw_calls={e['number']:e for e in events if e['event']=='finished'}
# Latest report/editor output for each task, before optional model self-evaluation.
selected={'M01':16,'M02':19,'M03':22}
rows=[]
for packet in manifest['tasks']:
    task=packet['task_id'];provider=raw_calls[selected[task]]
    text=provider['response']['choices'][0]['message']['content']
    det=json.loads((OUT/'cloud_inputs'/(task+'_detector.json')).read_text('utf-8'))
    facts={**service.report_facts(det),'case_id':packet['case_id'],'kind':'compare' if task=='M02' else 'single'}
    ground=service.build_geometry_quant(det['stats'],(det.get('scale_meta') or {}).get('meters_per_pixel'))[1] if task!='M02' else None
    missing=service.validate_report(text,det['risk_name'],ground,facts)
    source_errors=check_source_detail_claims(text,packet['question'])
    if source_errors:missing.append('引用细节有依据')
    prior=next(row for row in cloud['rows'] if row['task_id']==task)
    rows.append(dict(task_id=task,provider_request_number=selected[task],
        original_API_generation=prior['response']['generation'],
        original_model_self_evaluation=prior['response']['quality'].get('self_eval'),
        local_replay_mechanical_passed=not missing,missing=missing,
        source_detail_errors=source_errors,observation_errors=check_observation_claims(text,facts),
        numeric_errors=check_detection_numbers(text,facts),
        interpretation='内部逐项复核与确定性守卫重放；不是新增云端成功率、独立专家评价或新盲测'))

result=dict(reviewed_at=datetime.now().astimezone().isoformat(),
    reviewer='Codex internal review, not external domain expert',
    real_new_provider_requests=0,rows=rows,
    conclusions=[
        '3个业务请求的执行角色均只有ReasoningAgent；调度器正常参与，但未验证多执行角色协同收益。',
        '原始最终云端回执2/3机械通过，不能改写；修复后本地重放也不是新的模型验收。',
        'M01受到分级阈值区间误拦；实例排序重编号已修复。',
        'M02虽通过旧机械守卫且模型自评通过，仍出现未提供的标准章节、无证据自动降级和置信度混淆；新守卫拦截。',
        '自评角色配置关闭却仍调用的问题已修复；模型自评不作现场正确性或专业认证。',
        '当前故障注入和3条真实影像任务均已用于修复开发，后续不能称未参与迭代的独立验收集。'],
    remaining=['采用专门的证据核查与写作角色，并按任务条件增加协同；新任务冻结后再评估',
               '完整可信事故边界、稳定跨矿识别、真实专业用户验证仍未补足'])
(OUT/'semantic_review.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n','utf-8')
print(json.dumps({'local_replay_passed':sum(r['local_replay_mechanical_passed'] for r in rows),
                  'tasks':len(rows),'new_cloud_calls':0,'rows':rows},ensure_ascii=False))
