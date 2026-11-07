"""Frozen workflow acceptance: real Flask/coordinator/archive, scripted model responses.

This is fault injection, not a model benchmark, new mine imagery, or a user trial.
No outbound network is permitted except an explicitly requested local detector run.
"""
from pathlib import Path
from datetime import datetime
from types import SimpleNamespace
from contextlib import ExitStack
from unittest.mock import patch
import argparse
import base64
import copy
import hashlib
import io
import json
import os
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
SCRATCH = OUT / '_scratch'
SCRATCH.mkdir(exist_ok=True)
os.environ.update(KAT_CASE_ARCHIVE_PATH=str(SCRATCH/'acceptance.sqlite3'),
                  TEMP=str(SCRATCH), TMP=str(SCRATCH), YOLO_AUTOINSTALL='false')
sys.path.insert(0, str(ROOT))

TASKS = [
    ('A01', 'single_valid', '单期报告、两角色上下文和来源、归档交接一致', 'live'),
    ('A02', 'pair_valid', '双期百分比、百分点和变化数量保持一致', 'live'),
    ('A03', 'wrong_date', '错误影像采集日期不得作为有效分析交付', 'fallback'),
    ('A04', 'invented_date', '缺失采集日期时不得补造日期', 'fallback'),
    ('A05', 'swapped_dates', '双期T1/T2采集日期不得对调', 'fallback'),
    ('A06', 'invented_ndvi', '缺少真实NIR时不得声称测得NDVI', 'fallback'),
    ('A07', 'failure_forecast', '不得由影像声称未来48小时发生滑坡', 'fallback'),
    ('A08', 'foreign_case', '正文中的显式案例编号不能指向另一案例', 'fallback'),
    ('A09', 'truncated_first', '前序角色截断时下游不执行并交付事实降级', 'fallback'),
    ('A10', 'failed_first', '前序角色异常时下游不执行并交付事实降级', 'fallback'),
    ('A11', 'truncated_final', '最终角色截断不得交付残稿', 'fallback'),
    ('A12', 'invented_citation', '未提供的引用编号不得通过质量门禁', 'fallback'),
    ('A13', 'wrong_pair_numbers', '双期变化数量和百分点错误被拦截', 'fallback'),
    ('A14', 'conflict_as_background', '冲突像素不得声明已归为正常背景', 'fallback'),
    ('A15', 'restore_after_cache_clear', '清除内存缓存后归档、人工复核和离线导出仍一致', 'live'),
    ('A16', 'invented_displacement', '不得把模型掩膜称为测得现场位移', 'fallback'),
]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n', 'utf-8')


def fixture():
    from PIL import Image, ImageDraw
    image = Image.new('RGB', (100, 80), '#d8d3bb')
    ImageDraw.Draw(image).rectangle((20, 20, 42, 46), fill='#f0b000')
    buffer = io.BytesIO(); image.save(buffer, format='PNG')
    png = base64.b64encode(buffer.getvalue()).decode()
    return dict(overlay_b64=png, rgb_b64=png, size=[80,100], model_tag='mine', threshold=.4,
        risk_level='orange', risk_name='橙色-较多疑似异常',
        stats=dict(area_ratio=.1035, num_regions=2, num_regions_total=2, max_region_area=621,
                   scope='full_image', total_pixels=8000, valid_pixels=8000,
                   regions=[dict(id=1, area_px=621, perimeter_px=100.5, major_axis_px=32.5,
                                 minor_axis_px=20.5, bbox=[20,20,43,47], centroid=[31,33]),
                            dict(id=2, area_px=207, perimeter_px=60.5, major_axis_px=20.5,
                                 minor_axis_px=12.5, bbox=[60,20,74,35], centroid=[67,27])]),
        band_meta=dict(acquisition_datetime='2021-02-18T10:30:00Z', input_sha256=hashlib.sha256(buffer.getvalue()).hexdigest(), format='synthetic_contract_fixture',
                       nir_source='red_proxy', spectral_indices_available=False,
                       note='人工构造的工程故障注入图，非真实矿区；未提供真实NIR',
                       georeference={}, observation_quality={'source':'not_provided'}),
        quant={'ndvi':{'available':False}}, scale_meta={},
        input_quality={'status':'usable_observation'},
        model_info={'artifact':{'weights_sha256':'a'*64},'validation_status':'synthetic_not_accuracy_evidence'})


def pair_fixture():
    det = fixture(); png = det['overlay_b64']
    det.update(change_overlay_b64=png,t1_overlay_b64=png,t2_overlay_b64=png,t2_rgb_b64=png,
        registration={'status':'synthetic_test_only','independent_ground_accuracy_verified':False})
    det['stats'] = dict(t1_ratio=.1035, t2_ratio=.1275, net_ratio=.024,
                       n_added_regions=2,n_expanded_regions=1,n_removed_regions=3,valid_pixels=8000)
    meta = det['band_meta']; det['band_meta'] = {'t1':copy.deepcopy(meta),'t2':copy.deepcopy(meta)}
    det['band_meta']['t1']['acquisition_datetime']='2021-02-10T10:30:00Z'
    return det


def response_text(variant, case_id):
    paired = variant in ('pair_valid','swapped_dates','wrong_pair_numbers')
    if paired:
        text = ('共同有效范围内，T1异常掩膜面积占比10.35%，T2占比12.75%，净变化+2.4个百分点；'
                '新增2处、扩大1处、消退3处差分连通区域。橙色原型筛查分级来自T2模型掩膜。'
                'T1影像采集日期为2021-02-10，T2影像采集日期为2021-02-18。')
    else:
        text = ('本次影像范围内，模型输出疑似异常区域共2处，异常面积占比10.35%。'
                '最大返回实例面积621像素、周长100.5像素、长轴32.5像素。'
                '橙色原型筛查分级由模型掩膜面积比例计算，不能视为失稳概率。')
        if variant != 'invented_date':
            text += '本期影像采集日期为2021-02-18。'
    text += (f'案例编号：{case_id}。参考[TEST-001]仅用于复核方法，不能作为当前矿区的现场观测。'
             '建议对照原始影像复核圈定范围、遮挡与采集时间，结合生产记录和现场监测判断变化原因。'
             '当前未提供真实NIR、地面定位精度及独立事故边界，不能确认事故、安全或现场变形。'
             '请保留检测统计、原始来源和待确认事项；将人工复核意见另行记录，交接同次分析结果。')
    additions = {
        'invented_date':'本期影像采集日期为2021-02-18。',
        'invented_ndvi':'本期影像测得NDVI为0.72，表明植被覆盖良好。',
        'failure_forecast':'预计未来48小时内发生滑坡。',
        'conflict_as_background':'另有350像素证据冲突，已归为正常背景；待复核事项已排除。',
        'invented_displacement':'现场监测显示边坡位移为12毫米。',
    }
    text += additions.get(variant,'')
    if variant=='wrong_date': text=text.replace('采集日期为2021-02-18','采集日期为2021-02-19')
    if variant=='swapped_dates':
        text=text.replace('T1影像采集日期为2021-02-10，T2影像采集日期为2021-02-18',
                          'T1影像采集日期为2021-02-18，T2影像采集日期为2021-02-10')
    if variant=='foreign_case': text=text.replace(case_id,'f'*32)
    if variant=='invented_citation': text=text.replace('[TEST-001]','[FAKE-009]')
    if variant=='wrong_pair_numbers': text=text.replace('新增2处','新增9处').replace('+2.4个百分点','+24个百分点')
    return text


def prepare():
    path = OUT/'protocol.json'
    if path.exists():
        protocol=json.loads(path.read_text('utf-8'))
        if digest(__file__) != protocol['runner_sha256']:
            raise RuntimeError('Frozen acceptance runner changed')
        return protocol
    phase3=ROOT/'maintenance/phase3_20261013/repaired_source.zip'
    paths=['backend.py','report_grounding.py','evidence_integrity.py','coordinator/coordinator.py']
    value=dict(schema_version=1,registered_at=datetime.now().astimezone().isoformat(),
        purpose='新增业务链路工程验收；模型输出由脚本控制，不评价AI正确率或跨矿识别',
        original_source_snapshot_sha256=digest(phase3),
        harness_correction=dict(archive_sha256=digest(OUT/'initial_harness_attempt.zip'),
            reason='补充合成输入哈希和按真实接口HTTP200检查；先前归档补记检查的fixture无效，不计产品失败率',
            runtime_changed_before_correction=False),
        initial_runtime_hashes={p:digest(ROOT/p) for p in paths},runner_sha256=digest(__file__),
        tasks=[dict(id=t,variant=v,requirement=r,expected_generation=e) for t,v,r,e in TASKS],
        initial_source_frozen_before_run=True,number_of_tasks=len(TASKS),
        archive='isolated scratch SQLite; production archive untouched',cloud_calls=0,
        criterion='all workflow assertions pass; raw initial failures retained; repaired replay not a blind result',
        image_fixture='synthetic 100x80; hand-constructed counts and timestamps',
        later_real_GPU='existing Las Cruces development input; not a new mine or blind sample')
    write(path,value);return value


def scripted_coord(service, variant, answer, observed):
    from coordinator import AgentCoordinator
    class Agent:
        def __init__(self, name): self.name=name
        def step(self,msg):
            observed.append(self.name)
            if self.name=='EvidenceAuditAgent' and variant=='failed_first':
                raise RuntimeError('controlled worker failure')
            content = '只核对当前请求中的工具数值和案例，外部资料不是现场量测。' if self.name=='EvidenceAuditAgent' else answer
            truncated = (self.name=='EvidenceAuditAgent' and variant=='truncated_first' or
                         self.name=='ReasoningAgent' and variant=='truncated_final')
            return SimpleNamespace(msgs=[SimpleNamespace(content=content)],
                info={'termination_reasons':['length' if truncated else 'stop'],
                      'usage':{'prompt_tokens':0,'completion_tokens':0,'total_tokens':0}})
    with patch('coordinator.coordinator.ChatAgent'):
        coord=AgentCoordinator(model=object())
    coord.register_workers([dict(name=name,description='工程注入角色，非真实AI推断',agent=Agent(name))
                            for name in ('EvidenceAuditAgent','ReasoningAgent')])
    coord._dispatch=lambda *args,**kwargs: [dict(agent=0,subtask='核对工具事实和案例'),
                                          dict(agent=1,subtask='依据前序核对输出完整报告')]
    return coord


def run(label):
    protocol=prepare()
    path=OUT/(label+'.json')
    if path.exists(): raise RuntimeError('Receipt exists; use a new replay label, never overwrite raw results')
    import backend as service
    from case_archive import CaseArchive
    archive=CaseArchive(SCRATCH/(label+'.sqlite3'))
    client=service.app.test_client(); rows=[]
    with ExitStack() as stack:
        stack.enter_context(patch.object(service,'case_archive',archive))
        stack.enter_context(patch.object(service,'llm_fast_check',return_value=True))
        stack.enter_context(patch.object(service,'repair_report',return_value=None))
        stack.enter_context(patch.object(service,'run_self_eval',return_value=None))
        stack.enter_context(patch.object(service,'coordinator',object()))
        def rag(stats,risk,quant=None):
            return ('本次模型筛查，需保留全部未确认条件。资料[TEST-001]仅为通用复核方法。',
                    [dict(id='TEST-001',title='合成验收参考',source_title='工程测试，不是外部文献',content='复核工具事实')])
        stack.enter_context(patch.object(service,'build_rag_question',side_effect=rag))
        stack.enter_context(patch.object(service,'kb',None))
        # Fail closed if a model path accidentally escapes the scripted workers.
        stack.enter_context(patch('httpx.Client.send',side_effect=AssertionError('No cloud calls in offline acceptance')))
        for task in protocol['tasks']:
            variant=task['variant']; case=archive.create_case('合成工程验收',task['id'])
            cid=case['case_id']; observed=[]; started=time.perf_counter()
            det=pair_fixture() if variant in ('pair_valid','swapped_dates','wrong_pair_numbers') else fixture()
            det['request_id']='acceptance-'+task['id']
            if variant=='invented_date': det['band_meta']['acquisition_datetime']=None
            if variant=='conflict_as_background': det['stats']['unverified_pixels']=350
            answer=response_text(variant,cid)
            coord=scripted_coord(service,variant,answer,observed)
            errors=[]; result={};trace={}
            try:
                with patch.object(service,'detector_request',return_value=det), \
                     patch.object(service,'get_open_coordinator',return_value=coord):
                    paired='t1_ratio' in det['stats']
                    payload=dict(case_id=cid,model_tag='mine')
                    payload.update(dict(image_t1=det['overlay_b64'],image_t2=det['overlay_b64']) if paired else dict(image=det['overlay_b64']))
                    response=client.post('/api/pipeline_compare' if paired else '/api/pipeline_upload',json=payload)
                result=response.get_json()
                assert response.status_code==200,(response.status_code,result.get('error'))
                assert result['generation']['status']==task['expected_generation'],result['generation']
                scope=result['quality']['evidence_context']
                assert scope['case_id']==cid and scope['request_id']==det['request_id'],'case/request mismatch'
                if variant in ('truncated_first','failed_first'):
                    assert observed==['EvidenceAuditAgent'],'downstream consumed incomplete evidence'
                elif task['expected_generation']=='live':
                    assert observed==['EvidenceAuditAgent','ReasoningAgent']
                    assert not result['quality']['degraded']
                rid=result['report_id']; snapshot=result['archive']['snapshot_sha256']
                original_facts=copy.deepcopy(result['objective_facts'])
                service._report_cache.clear()
                reloaded=client.get('/api/archive/reports/'+rid).get_json()
                assert reloaded['objective_facts']==original_facts and reloaded['final_answer']==result['final_answer']
                review=client.post('/api/archive/reports/'+rid+'/manual_annotations',json=dict(
                    expected_revision=0,box_px=[20,20,35,35],reviewer='工程自动验收',
                    evidence='合成输入，不是用户评估或真实边界',note='人工意见与检测事实分层'))
                assert review.status_code==200,(review.status_code,review.get_json())
                reviewed=review.get_json()
                assert reviewed['objective_facts']==original_facts and reviewed['archive']['snapshot_sha256']==snapshot
                for fmt in ('html','json'):
                    response=client.post('/api/review_record',json=dict(report_id=rid,format=fmt))
                    assert response.status_code==200
                    exported=base64.b64decode(response.get_json()['content_b64'])
                    if fmt=='json':
                        packet=json.loads(exported)
                        assert packet['objective_facts']==original_facts
                        assert len(packet['manual_annotations'])==1
                        assert packet['automatic_analysis']['answer']==result['final_answer']
                    else: assert b'<svg id="scene"' in exported
            except Exception as exc:
                errors.append(str(exc) or type(exc).__name__)
            rows.append(dict(id=task['id'],variant=variant,expected=task['expected_generation'],
                passed=not errors,errors=errors,observed_workers=observed,
                actual_generation=result.get('generation'),quality=result.get('quality'),
                raw_draft=answer,delivered_answer=result.get('final_answer'),
                elapsed_seconds=round(time.perf_counter()-started,4)))
    result=dict(schema_version=1,label=label,completed_at=datetime.now().astimezone().isoformat(),
        protocol_sha256=digest(OUT/'protocol.json'),runner_sha256=digest(__file__),
        runtime_hashes={p:digest(ROOT/p) for p in protocol['initial_runtime_hashes']},
        tasks=len(rows),passed=sum(r['passed'] for r in rows),all_passed=all(r['passed'] for r in rows),
        real_AI_calls=0,scripted_outputs=True,professional_accuracy_scored=False,
        production_archive_untouched=True,rows=rows)
    write(path,result)
    print(json.dumps({k:v for k,v in result.items() if k!='rows'},ensure_ascii=False))
    for row in rows:
        if not row['passed']: print(row['id'],row['variant'],row['errors'])


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['prepare','initial','repaired'])
    args=parser.parse_args()
    if args.action=='prepare': print(json.dumps(prepare(),ensure_ascii=False))
    else: run(args.action)
