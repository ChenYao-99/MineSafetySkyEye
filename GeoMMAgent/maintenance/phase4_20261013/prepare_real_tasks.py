"""Real GPU business acceptance and exact outbound packet preparation; zero cloud calls."""
from pathlib import Path
from datetime import datetime
from unittest.mock import patch
import base64
import copy
import hashlib
import json
import os
import sys
import time

ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent
SCRATCH=OUT/'_scratch';SCRATCH.mkdir(exist_ok=True)
os.environ.update(KAT_LOCAL_FACTS_ONLY='1',KAT_CASE_ARCHIVE_PATH=str(SCRATCH/'real_tasks.sqlite3'),
                  DETECTOR_PORT='5101',TEMP=str(SCRATCH),TMP=str(SCRATCH),YOLO_AUTOINSTALL='false')
sys.path.insert(0,str(ROOT))
import backend as service
from evidence_reporting import bounded_input
from report_grounding import report_input_context,check_detection_numbers,check_observation_claims
client=service.app.test_client()


def write(path,value):
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n','utf-8')


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def post(url,payload):
    response=client.post(url,json=payload);body=response.get_json()
    assert response.status_code in (200,201),(url,response.status_code,body.get('error'))
    assert not body.get('error'),body.get('error')
    return body


def run():
    if (OUT/'real_delivery.json').exists():raise RuntimeError('Receipt already exists; preserve prior results')
    inputs=ROOT/'cache/case_delivery_20261008_v2/inputs'
    images={epoch:base64.b64encode((inputs/(epoch+'_BGRN_10m.tif')).read_bytes()).decode() for epoch in ('pre','post')}
    output_dir=OUT/'cloud_inputs';output_dir.mkdir(exist_ok=True)
    cases=[('M01','single','/api/pipeline_upload',{'image':images['post']}),
           ('M02','paired','/api/pipeline_compare',{'image_t1':images['pre'],'image_t2':images['post']}),
           ('M03','ROI','/api/pipeline_upload',{'image':images['post'],'roi':{'x0':.2,'y0':.2,'x1':.8,'y1':.8}})]
    records=[];packets=[]
    for task_id,kind,url,payload in cases:
        case=post('/api/cases',dict(mine_name='Las Cruces公开开发影像工程验收',title=task_id,notes='隔离测试，非新增矿区或用户评价'))
        started=time.perf_counter();result=post(url,{**payload,'model_tag':'mine','case_id':case['case_id']})
        assert result['generation']['source']=='detection_summary'
        det=result['detection']; facts=result['objective_facts']; rid=result['report_id']
        scope=result['quality']['evidence_context']
        assert scope['case_id']==case['case_id'] and scope['request_id']==det['request_id']
        assert not check_detection_numbers(result['final_answer'],service.report_facts(det))
        snapshot=result['archive']['snapshot_sha256']
        service._report_cache.clear()
        h,w=det['size']
        reviewed=post('/api/archive/reports/'+rid+'/manual_annotations',dict(expected_revision=0,
            box_px=[w*.3,h*.3,w*.4,h*.4],reviewer='内部自动工程验收',
            evidence='原开发影像定位测试，不是专家事故边界',note='与模型预测分层保存'))
        assert reviewed['objective_facts']==facts and reviewed['archive']['snapshot_sha256']==snapshot
        export_hashes={}
        for fmt in ('html','json'):
            body=post('/api/review_record',dict(report_id=rid,format=fmt)); raw=base64.b64decode(body['content_b64'])
            export_hashes[fmt]=hashlib.sha256(raw).hexdigest()
            if fmt=='json':
                exported=json.loads(raw)
                assert exported['objective_facts']==facts and len(exported['manual_annotations'])==1
                assert exported['automatic_analysis']['answer']==result['final_answer']
            else:assert b'<svg id="scene"' in raw
        if kind=='paired':
            for fmt in ('html','json'):assert post('/api/report_compare',dict(report_id=rid,format=fmt))['content_b64']
        else:assert post('/api/report_upload',dict(report_id=rid))['report_b64']
        image_path=output_dir/(task_id+'.jpg')
        image_path.write_bytes(service._prepare_vlm_image(base64.b64decode(det['overlay_b64'])))
        minimal=bounded_input(det)
        question=result['question']+'\n'+report_input_context({**service.report_facts(det),'case_id':case['case_id'],
                                               'kind':'compare' if kind=='paired' else 'single'})
        # No raw TIFF, geospatial coordinates, paths, account identity or archive contents.
        assert 'D:\\' not in question and 'D:/' not in question
        packets.append(dict(task_id=task_id,case_id=case['case_id'],request_id=det['request_id'],kind=kind,
            question=question,tool_fields=minimal,image_file=str(image_path.relative_to(OUT)),
            image_sha256=sha(image_path),image_bytes=image_path.stat().st_size,
            input_description='Las Cruces现有公开Sentinel-2开发影像的检测覆盖图；JPEG无地理标签，含项目预测轮廓'))
        records.append(dict(task_id=task_id,kind=kind,passed=True,request_id=det['request_id'],
            case_id=case['case_id'],scope_matches=True,objective_facts=facts,
            observation_meta=det.get('band_meta'),review_preserves_facts=True,
            offline_exports=export_hashes,elapsed_seconds=round(time.perf_counter()-started,3)))
        # Preserve detector result once for reproducible cloud integration without rerunning inference.
        write(output_dir/(task_id+'_detector.json'),det)
    with service.case_archive.connection() as db:assert db.execute('PRAGMA integrity_check').fetchone()[0]=='ok'
    result=dict(completed_at=datetime.now().astimezone().isoformat(),passed=True,real_GPU=True,cloud_calls=0,
        image_source='existing Las Cruces development imagery from Copernicus/Planetary Computer',
        original_input_hashes={e:sha(inputs/(e+'_BGRN_10m.tif')) for e in images},
        new_mines=0,new_blind_samples=0,professional_user_trials=0,production_archive_untouched=True,records=records)
    write(OUT/'real_delivery.json',result)
    manifest=dict(schema_version=1,status='prepared_not_sent',
        destination='https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions',
        models=['qwen3-vl-plus','qwen-max'],
        purpose='修复后的真实矿山影像报告链路验收；单期、双期和ROI三种任务；不作为跨矿精度或盲测',
        max_generation_requests=24,SDK_hidden_retries=0,
        max_output_tokens_per_request=2000,total_provider_token_budget=200000,
        role_input_rule='仅本清单问题、工具字段、3张JPEG、当前协调器及角色提示；后续角色接收同接口此前输出',
        exact_role_prompt_sources={p:sha(ROOT/p) for p in [
            'configs/loader.py','coordinator/prompts.py','exec_agents/perception/prompts.py',
            'exec_agents/reasoning/prompts.py','backend.py']},
        additional_search_or_external_downloads_allowed=False,
        excluded=['人员姓名、学校、生产案例库、原始TIFF、完整代码、权重、地理坐标、密钥正文'],
        allowed_project_derived_data=['检测覆盖JPEG、检测统计及几何量、影像时间和波段说明、知识片段、隔离测试编号、角色笔记及草稿'],
        tasks=packets,
        limits=['公开开发影像，非新增盲测','三任务不是独立矿区','自然语言只做内部语义核查','最多24次，不无限重试'],
        previous_phase2_32_request_allowance_exhausted=True,new_explicit_outbound_approval_required=True)
    write(OUT/'outbound_review_manifest.json',manifest)
    print(json.dumps(dict(passed=True,tasks=len(records),cloud_calls=0,
        images_bytes=sum(t['image_bytes'] for t in packets),manifest_sha256=sha(OUT/'outbound_review_manifest.json')),ensure_ascii=False))


if __name__=='__main__':run()
