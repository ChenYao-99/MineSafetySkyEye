"""Local regression of actual previous role outputs; no network or new AI scores."""
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from datetime import datetime
import copy,hashlib,json,os,sys
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT));os.environ['KAT_LOCAL_FACTS_ONLY']='1'
from report_workflow import ReportCoordinator
from evidence_integrity import evidence_context
from report_grounding import check_observation_claims,check_detection_numbers
import report_grounding

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def run(label):
    path=OUT/(label+'.json')
    if path.exists():raise RuntimeError('Preserve raw replay receipts; do not overwrite')
    cloud=json.loads((ROOT/'maintenance/phase5_20261014/cloud_acceptance.json').read_text('utf-8'))
    rows=[]
    for row in cloud['rows']:
        task=row['task_id'];body=row['response']
        det=json.loads((ROOT/('maintenance/phase4_20261013/cloud_inputs/'+task+'_detector.json')).read_text('utf-8'))
        scope=body['quality']['evidence_context']
        facts={'stats':det['stats'],'band_meta':det.get('band_meta'),'roi':det.get('roi'),
               'kind':'compare' if task=='M02' else 'single','case_id':scope['case_id']}
        coord=ReportCoordinator.__new__(ReportCoordinator)
        coord._evidence_context=scope;coord.facts=copy.deepcopy(facts)
        audit=body['agents'][0]['output']
        audit_errors=coord.validate_step('EvidenceAuditAgent',audit)
        final=body['final_answer']
        errors=check_detection_numbers(final,facts)+check_observation_claims(final,facts)
        if hasattr(report_grounding,'check_known_input_claims'):
            errors+=report_grounding.check_known_input_claims(final,facts)
        rows.append({'task_id':task,'old_audit_errors_detected':audit_errors,'old_final_errors_detected':errors,
                     'audit_notes_before_repair':audit,'final_before_repair':final})
    value={'client_work_date':'2026-10-14','at':datetime.now().astimezone().isoformat(),'label':label,
        'old_cloud_receipt_sha256':sha(ROOT/'maintenance/phase5_20261014/cloud_acceptance.json'),
        'runtime_hashes':{name:sha(ROOT/name) for name in ['report_workflow.py','report_grounding.py','evidence_reporting.py','backend.py']},
        'new_cloud_calls':0,'new_independent_tests':0,'previous_scores_rewritten':False,'rows':rows}
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n','utf-8')
    print(json.dumps({'label':label,'audit_errors':[len(r['old_audit_errors_detected']) for r in rows],
                      'final_errors':[len(r['old_final_errors_detected']) for r in rows],'new_cloud_calls':0},ensure_ascii=False))

if __name__=='__main__':run(sys.argv[1])
