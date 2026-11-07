"""Mine report roles with explicit evidence checks and conditional review.

Routing uses declared tool conditions, not a learned confidence score. The general
image-QA/benchmark dispatcher remains separate. All roles are real model calls.
"""
import copy,json
from coordinator import AgentCoordinator
from camel.agents import ChatAgent
from configs import create_model_from_config
from configs.loader import DEFAULT_MODEL
from evidence_reporting import bounded_input, workflow_status

AUDIT_PROMPT = (
    '你是EvidenceAuditAgent，负责核对当前矿山报告的证据，不撰写报告。'
    '只依据同次工具事实、案例范围、给定知识摘要作答。检测掩膜不等于事故真值，面积分级不等于模型置信度。'
    '保留未知条件，参考资料不得补成现场量测；不得编造全文章节、时间阈值、因果或安全证明。'
    '未知资料应needs_review，只有确实串案、输入矛盾或无法使用时才block。'
    '以objective_facts中的百分数与净变化百分点为准，不把比例小数、相对变化率和百分点混用。'
    'roi_selected已声明时不要再问是否已圈定，可核查其与现场工程分区是否一致。'
    '实测NIR、日期和verified_local_product_XML已提供时，不得列作缺失；地面精度和现场稳定性仍未知。'
    'issues只列输入中的具体矛盾，不能虚构用户已误用规则，也不能把引用预测几何本身说成违规。'
    '输出完整JSON对象，不用Markdown：case_id、request_id必须照抄当前证据范围；'
    'decision为continue/needs_review/block；checks为最多5个简短字符串；'
    'unknowns和issues各最多4条；source_ids只列当前允许的资料编号。总计不超过350字。')

WRITER_PROMPT = (
    '你是ReasoningAgent，负责根据当前工具事实和前序证据核查结果撰写中文影像复核报告。'
    '先检测事实与范围，再解释与限制，最后复核和监测建议，正文约400至650字。'
    '沿用原始实例ID，逐项保持数值、单位、时相和案例归属；净变化使用百分点。'
    '优先使用objective_facts的数值，净变化保留两位小数；不同舍入值不能写成严格相等。'
    'ROI选择、实测NIR与产品定标声明属于已知输入条件，不要重复列为未知；工程分区适合性和地面精度另行复核。'
    '按analysis_scope区分整幅、ROI及共同有效范围，不把通用ROI资料当成当前圈定证明。'
    '只用提供来源的方括号编号；只有标题或摘要时不得虚构标准章节及原文条款。'
    '不要编造事故、安全、现场位移、失稳时间或固定处置期限；未获得现场证据不能作为自动降级理由。'
    '筛查分级由掩膜面积比例计算，不是置信度或失稳概率。'
    '明确保留前序角色指出的未知和冲突，不输出选项字母或模型自评分数。')

REVIEW_PROMPT = (
    '你是ReportReviewAgent，负责对照原始工具事实与证据核查结果审查前序报告，输出修订后的完整中文报告。'
    '逐项核对案例、日期、分析范围、实例ID、数值单位、引用编号和知识摘要的支持范围。'
    '同时纠正前序角色把已提供条件写成缺失的错误；净变化依objective_facts按百分点保留两位小数。'
    '删除虚构章节、事故确认、稳定证明、置信度混淆、无依据处置时限和缺少证据自动降级规则。'
    '检查差分连通区域不是事故数量或匹配实例数量，模型面积变化不是位移。'
    '无法确认的条件保持未知，不补造实测或专业背书。正文约400至650字。'
    '若存在无法修正的串案或关键矛盾，输出[REVIEW_BLOCKED]及简短原因；否则只输出完整报告。')


def choose_report_route(facts):
    facts=facts or {};stats=facts.get('stats') or {};meta=facts.get('band_meta') or {}
    paired=facts.get('kind')=='compare' or 't1_ratio' in stats
    epochs=[meta.get(k) or {} for k in ('t1','t2')] if paired else [meta]
    reasons=[]
    if paired:reasons.append('双期时相、变化口径与原因需要复核')
    if stats.get('unverified_pixels',0):reasons.append('包含尚未判定的冲突像素')
    if stats.get('area_ratio_available',True) is False:reasons.append('面积占比不可计算')
    if any(not item.get('acquisition_datetime') for item in epochs):reasons.append('采集日期未确认')
    if (facts.get('work_context') or {}).get('status')=='matched_public_expected_work_reference':
        reasons.append('年度作业参考与事故判断需要区分')
    names=['EvidenceAuditAgent','ReasoningAgent']
    if reasons:names.append('ReportReviewAgent')
    return {'mode':'evidence_report','roles':names,'review_reasons':reasons,
            'reason':'证据核查后撰写报告'+('；复杂条件增加独立角色复核' if reasons else '；工具信息完整，不增加第三个角色'),
            'routing_method':'declared_tool_conditions_not_learned_confidence'}


class ReportCoordinator(AgentCoordinator):
    def __init__(self,cfg,facts):
        self.facts=copy.deepcopy(facts or {})
        self.route=choose_report_route(facts)
        model_name=cfg.get('coordinator',{}).get('model',DEFAULT_MODEL)
        super().__init__(create_model_from_config(model_name,max_tokens=768))
        settings=cfg.get('report_workflow') or {}
        self.model_name=settings.get('model',model_name)
        prompts={'EvidenceAuditAgent':AUDIT_PROMPT,'ReasoningAgent':WRITER_PROMPT,'ReportReviewAgent':REVIEW_PROMPT}
        for name in self.route['roles']:
            cap=768 if name=='EvidenceAuditAgent' else 1600
            agent=ChatAgent(system_message=prompts[name],retry_attempts=1,token_limit=32768,
                model=create_model_from_config(self.model_name,max_tokens=cap))
            self.register_worker(name,agent,name)
        self.set_step_validator(self.validate_step)
        self.tool_context=bounded_input(facts)

    def _dispatch(self,refined_task,image_path=None):
        return [{'agent':i,'name':name,'subtask':{
            'EvidenceAuditAgent':'核对同次工具事实、案例、时相、来源范围，输出简短结构化证据核查记录。',
            'ReasoningAgent':'结合前序证据核查记录，输出完整中文影像复核报告。',
            'ReportReviewAgent':'对照原始事实和核查记录，修正前序报告中的越界内容，输出完整最终正文。'}[name]}
            for i,name in enumerate(self.route['roles'])]

    def validate_step(self,name,output):
        if name=='ReportReviewAgent' and '[REVIEW_BLOCKED]' in str(output):
            return ['reviewer_unresolved_evidence_conflict']
        if name!='EvidenceAuditAgent':return []
        try:value=json.loads(output)
        except (TypeError,ValueError):return ['evidence_audit_not_complete_JSON']
        if not isinstance(value,dict):return ['evidence_audit_not_object']
        scope=self._evidence_context or {};errors=[]
        for key in ('case_id','request_id'):
            if value.get(key)!=scope.get(key):errors.append('audit_'+key+'_mismatch')
        if value.get('decision') not in ('continue','needs_review','block'):errors.append('audit_invalid_decision')
        for key in ('checks','unknowns','issues','source_ids'):
            if not isinstance(value.get(key),list) or not all(isinstance(x,str) for x in value.get(key,[])):
                errors.append('audit_invalid_'+key)
        allowed={r['evidence_id'] for r in scope.get('evidence',{}).get('active',[])}
        if isinstance(value.get('source_ids'),list) and any(x not in allowed for x in value['source_ids']):
            errors.append('audit_unprovided_source')
        if value.get('decision')=='block':errors.append('audit_blocked_input')
        if not errors:
            from report_grounding import check_detection_numbers,check_observation_claims,check_known_input_claims
            notes='\n'.join(value['checks']+value['unknowns']+value['issues'])
            if check_detection_numbers(notes,self.facts):errors.append('audit_numeric_claim_inconsistent')
            if check_observation_claims(notes,self.facts):errors.append('audit_observation_claim_inconsistent')
            if check_known_input_claims(notes,self.facts):errors.append('audit_provided_condition_called_unknown')
        return errors

    def run_with_trace(self,task_content,image_path=None,**kwargs):
        task_content+='\n同次工具字段（优先于解释和一般知识）：\n'+json.dumps(self.tool_context,ensure_ascii=False,separators=(',',':'))
        return super().run_with_trace(task_content,image_path=image_path,**kwargs)

    def status(self,trace):
        return workflow_status(trace,self.route['mode'],self.route['reason'])
