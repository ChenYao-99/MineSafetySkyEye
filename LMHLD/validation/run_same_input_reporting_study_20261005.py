"""Paired objective-reporting study; all arms share the exact initial model output.

This evaluates a new reporting-stage role prototype, not the old full GeoMMAgent
perception pipeline, geological correctness, or operational adoption.
"""
from pathlib import Path
from datetime import datetime,timezone,timedelta
import argparse,os,sys,json,time,hashlib,copy
ROOT=Path('D:/LMHLD');APP=Path('D:/GeoMMAgent');sys.path.insert(0,str(APP));sys.path.insert(0,str(ROOT))
from project_runtime import configure_runtime_paths
configure_runtime_paths();os.environ['MODEL_TIMEOUT']='35'
from dotenv import load_dotenv
load_dotenv(APP/'.env')
from camel.models import ModelFactory
from camel.types import ModelPlatformType
from model_registry import file_sha256
from knowledge_base import KnowledgeBase
from evidence_reporting import objective_facts,bounded_input,parse_report,fixed_evidence_report,score_objective_report
OUT=ROOT/'experiments/same_input_reporting_workflow_20261005';MODEL='qwen-max'
def now():return datetime.now(timezone(timedelta(hours=8))).isoformat(timespec='seconds')
def write(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def load(p):return json.loads(p.read_text(encoding='utf-8'))
SCHEMA={'analysis_scope':'whole|roi|common_valid_pixels|not_declared;use exact supplied scope','kind':'single_epoch_mask|paired_mask_change',
 'reported_regions':'number or null;paired counts only in change','area_percent':'area_ratio*100 or null if unavailable','largest_area_px':'number or null',
 'largest_area_m2':'number or null;do not pretend pixel units are m2','largest_perimeter_px':'number or null','largest_major_axis_px':'number or null',
 'physical_spectral_indices_available':'boolean from supplied usable spectral-index flag for each epoch','weather_quality_provided':'boolean;pixelreadabilitydoesnotcertifyweather',
 'capture_times':'list of actual supplied UTC capture strings or null per epoch','unverified_pixels':'number;missing means0',
 'independent_ground_accuracy_verified':'boolean only when explicitly verified','accident_confirmed_from_mask':'false',
 'site_stability_confirmed':'false','failure_time_predicted':'false','ground_displacement_measured':'false',
 'change':{'t1_percent':'fraction*100or null','t2_percent':'fraction*100or null','net_percentage_points':'net_ratio*100or null;notdisplacement',
 'added_regions':'numberor null','expanded_regions':'numberor null','removed_regions':'numberor null'}}
def request(model,task_id,role,system,prompt):
    path=OUT/(task_id+'_'+role+'_provider.json')
    if path.exists():
        record=load(path)
        if record['prompt_sha256']!=hashlib.sha256((system+'\n'+prompt).encode()).hexdigest():raise ValueError('Frozen model input changed')
        return record
    started=time.perf_counter()
    response=model.run(messages=[{'role':'system','content':system},{'role':'user','content':prompt}])
    answer=response.choices[0].message.content or ''
    usage=response.usage.model_dump() if response.usage else None
    record={'task_id':task_id,'role':role,'model_requested':MODEL,'model_reported':response.model,'temperature':0,
            'completed_at':now(),'elapsed_s':time.perf_counter()-started,'prompt_sha256':hashlib.sha256((system+'\n'+prompt).encode()).hexdigest(),
            'system':system,'prompt':prompt,'answer':answer,'usage':usage,'provider_response_id':response.id}
    write(path,record);print(task_id,role,'tokens',usage.get('total_tokens') if usage else 'notprovided','seconds',round(record['elapsed_s'],2),flush=True)
    return record
def run(approved_manifest_sha256):
    bank=load(OUT/'detector_bank.json')
    if bank['new_blind_predictions']!=0 or len(bank['tasks'])!=8:raise ValueError('Explicit observed bank only')
    manifest_path=OUT/'outbound_review_manifest.json'
    if not approved_manifest_sha256 or file_sha256(manifest_path)!=approved_manifest_sha256:
        raise ValueError('Explicit approval for the exact outbound review manifest is required')
    manifest=load(manifest_path);kb=KnowledgeBase()
    chunks=kb.retrieve('光学筛查 监测范围 人工复核 掩膜变化 量测',k=4)
    if manifest['detector_bank_sha256']!=file_sha256(OUT/'detector_bank.json') or manifest['public_knowledge_summary']!=kb.format_context(chunks):
        raise ValueError('Proposed outbound evidence changed after review')
    for task,proposed in zip(bank['tasks'],manifest['proposed_tasks']):
        if task['id']!=proposed['local_task_id'] or bounded_input(task['detection'])!=proposed['tool_fields_to_send']:
            raise ValueError('Minimum outbound task fields changed after review')
    protocol={'created_at':now(),'bank_sha256':file_sha256(OUT/'detector_bank.json'),
       'explicitly_approved_outbound_manifest_sha256':approved_manifest_sha256,
       'bank_protocol_sha256':file_sha256(OUT/'bank_protocol.json'),'implementation_sha256':file_sha256(__file__),
       'objective_reporting_code_sha256':file_sha256(APP/'evidence_reporting.py'),'knowledge_chunks_sha256':file_sha256(ROOT/'knowledge_chunks.json'),
       'model':MODEL,'temperature':0,'max_tokens':1400,'arms':['single_model','fixed_evidence','role_review_prototype'],
       'initial_writer_response_exactly_shared':True,'fixed_evidence_fields_are_deterministically_copied_from_tool_result':True,
       'role_review_prototype_auditor_then_optional_editor_max3calls':True,'original_full_GeoMMAgent_perception_benchmark':False,
       'objective_fields_and_provided_citation_IDs_only_not_professional_geological_correctness':True,
       'same_mechanical_summary_guard_all_arms_not_semantic_or_expert_score':True,
       'provider_usage_missing_not_estimated':True,'syntheticstress_case_explicit':True,'fieldtrials_expertendorsements_or_newblinddata':0,
       'provider_model_alias_revision_opaque':True,'hypothesis_not_claimed_in_advance':'additionalroles maycostmore withoutfactualbenefit'}
    pp=OUT/'reporting_protocol.json'
    if pp.exists():
        old=load(pp)
        if any(old[k]!=v for k,v in protocol.items() if k!='created_at'):raise ValueError('Frozen paired study changed')
    else:write(pp,protocol)
    model=ModelFactory.create(model_platform=ModelPlatformType.QWEN,model_type=MODEL,model_config_dict={'temperature':0,'max_tokens':1400})
    if hasattr(model,'_client'):model._client.max_retries=0
    rows=[]
    for task in bank['tasks']:
        det=task['detection'];expected=objective_facts(det)
        chunks=kb.retrieve('光学筛查 监测范围 人工复核 掩膜变化 量测',k=4);allowed=[c['id'] for c in chunks]
        context=kb.format_context(chunks)
        source={'task':task['task'],'fixture_kind':task['fixture_kind'],'tool_result':bounded_input(det),'provided_knowledge':context,'allowed_citation_IDs':allowed}
        stimulus=json.dumps(source,ensure_ascii=False,indent=2)
        writer_prompt='根据下列同次工具结果，输出JSON对象，字段facts、summary、citations。facts必须完整遵循给定结构，summary为简短中文核查说明；引用仅限提供的ID。字段不可得必须显式null，不得确认事故、安全、失稳概率或现场位移。\n结构说明：'+json.dumps(SCHEMA,ensure_ascii=False)+'\n输入：'+stimulus
        writer=request(model,task['id'],'writer','你是影像核查报告助理。精确整理已提供事实，不编造缺失信息。',writer_prompt)
        draft=parse_report(writer['answer']);fixed=fixed_evidence_report(draft,det,allowed)
        audit_prompt='审核以下报告与同次工具输入是否一致，重点检查指标、单位、两期净变化百分点、缺测以及未核实的结论。只输出JSON：{"needs_revision":true或false,"issues":[具体问题]}。你不是现场专家，不能生成新的地质事实。\n输入：'+stimulus+'\n结构说明：'+json.dumps(SCHEMA,ensure_ascii=False)+'\n报告：'+writer['answer']
        auditor=request(model,task['id'],'auditor','你是证据一致性审阅角色，只核查给定信息。',audit_prompt)
        review=parse_report(auditor['answer']);editor=None;final=draft
        needs=not isinstance(review,dict) or review.get('needs_revision') is not False
        if needs:
            edit_prompt='根据同次原始输入和审阅意见修订报告，仍只输出包含facts、summary、citations的完整JSON，不能增加未经提供的信息。\n输入：'+stimulus+'\n结构说明：'+json.dumps(SCHEMA,ensure_ascii=False)+'\n初稿：'+writer['answer']+'\n审阅意见：'+auditor['answer']
            editor=request(model,task['id'],'editor','你是核查报告修订角色，精确引用工具结果与已提供资料。',edit_prompt);final=parse_report(editor['answer'])
        def tokens(receipts):
            return sum(r['usage']['total_tokens'] for r in receipts) if all(r.get('usage') and r['usage'].get('total_tokens') is not None for r in receipts) else None
        modes=[('single_model',draft,[writer]),('fixed_evidence',fixed,[writer]),('role_review_prototype',final,[writer,auditor]+([editor] if editor else []))]
        scores={}
        for arm,report,receipts in modes:
            score=score_objective_report(report,expected,allowed,det=det);scores[arm]=score
            rows.append({'task_id':task['id'],'arm':arm,'mine_case':task['mine_case'],'fixture_kind':task['fixture_kind'],
                         'expected_objective_facts':expected,'report':report,'score':score,'logical_model_calls':len(receipts),
                         'logical_usage_total_tokens':tokens(receipts),'logical_model_elapsed_s':sum(r['elapsed_s'] for r in receipts),
                         'initial_response_id':writer['provider_response_id'],'auditor_or_editor_is_not_humanfeedback':True})
        write(OUT/'reporting_results_partial.json',{'protocol_sha256':file_sha256(pp),'models_and_fixed_outputs':rows})
        print(task['id'],'wrongfields',{a:s['wrong_fields'] for a,s in scores.items()},flush=True)
    summary={}
    for arm in ('single_model','fixed_evidence','role_review_prototype'):
        r=[x for x in rows if x['arm']==arm]
        summary[arm]={'tasks':len(r),'all_objective_fields_correct_tasks':sum(x['score']['all_objective_fields_correct'] for x in r),
                      'report_contract_passed_tasks':sum(x['score'].get('report_contract_passed',False) for x in r),
                      'summary_contract_error_cases':sum(bool(x['score'].get('summary_mechanical_contract_errors')) for x in r),
                      'wrong_field_count':sum(x['score']['wrong_fields'] for x in r),'unprovided_citation_cases':sum(bool(x['score']['unprovided_citations']) for x in r),
                      'logical_model_calls':sum(x['logical_model_calls'] for x in r),
                      'logical_token_total':sum(x['logical_usage_total_tokens'] for x in r) if all(x['logical_usage_total_tokens'] is not None for x in r) else None,
                      'logical_elapsed_s':sum(x['logical_model_elapsed_s'] for x in r)}
    paid=[load(p) for p in OUT.glob('*_provider.json')]
    result={'completed_at':now(),'protocol_sha256':file_sha256(pp),'models_and_fixed_outputs':rows,'summary':summary,
            'actual_paid_calls_initialwriter_shared_not_triplecounted':len(paid),
            'actual_recorded_provider_tokens':sum(x['usage']['total_tokens'] for x in paid) if all(x.get('usage') for x in paid) else None,
            'interpretation':'objective reporting stage only;fixedfields exactbyconstruction;neither professionalcorrectness norindustrybenefit claimed',
            'new_classifier_fits':0,'reserved_predictions':0,'production_multimodel_pipeline_changed_from_this_study':False}
    write(OUT/'reporting_results.json',result);print(json.dumps(summary,ensure_ascii=False),flush=True)
if __name__=='__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    parser=argparse.ArgumentParser()
    parser.add_argument('--approved-outbound-manifest-sha256',required=True,
                        help='Only supply after user explicitly authorizes this outbound data transfer')
    args=parser.parse_args();run(args.approved_outbound_manifest_sha256)
