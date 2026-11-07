"""Local replay of the recorded failures, not another model-performance study."""
from pathlib import Path
import hashlib
import json
import sys
from types import SimpleNamespace

ROOT=Path(r'D:\GeoMMAgent')
sys.path.insert(0,str(ROOT))
from evidence_integrity import bind_structured_report, completion_status, resolve_scoped_evidence

OUT=ROOT/'maintenance/phase3_20261013'
OLD=ROOT/'maintenance/phase2_20261013'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

data=read(OLD/'results.json');protocol=read(OLD/'protocol.json')
targets={t['id']:t['target'] for t in protocol['tasks']}
repairs=[]
for task,wrong in [('P1-02','8679'),('P1-06','48.67')]:
    original=next(r['report'] for r in data['rows'] if r['task_id']==task and r['arm']=='B_single_checked')
    fixed=bind_structured_report(original,targets[task])
    assert wrong not in fixed['conclusion'] and fixed['review_required']
    assert wrong in fixed['raw_generated_report']['conclusion']
    repairs.append({'task':task,'accepted_conclusion':fixed['conclusion'],
                    'interpretation_status':fixed['interpretation_status'],
                    'old_draft_preserved':True,'stale_number_removed_from_delivery':True})

gams=Path(r'D:\LMHLD\datasets\arid_accident_operation_20261004\ZA_GAMS2020S\pixel_reference_v1_wang2026')
andoom=Path(r'D:\LMHLD\datasets\tropical_work_pairs_local_quality_20261005\AU_ANDOOM_LOCAL_SLATS2324\work_reference_v1_SLATS\reference.json')
formal=read(gams/'reference.json');pending=read(gams/'reference_review_pending.json')
assert formal['case_id']==pending['case_id'] and formal['event_date']==pending['event_date']
assert formal['frozen_assets']==pending['frozen_assets']
accept=read(Path(r'D:\LMHLD\datasets\tropical_work_pairs_20261005\work_reference_acceptance.json'))['accepted']['AU_ANDOOM_LOCAL_SLATS2324']['references'][0]
assert accept['sha256']==sha(andoom)
records=[
    {'evidence_id':'GAMS-PENDING','scope':'case','case_id':formal['case_id'],'event_id':formal['event_date'],'content_sha256':sha(gams/'reference_review_pending.json')},
    {'evidence_id':'GAMS-FORMAL','scope':'case','case_id':formal['case_id'],'event_id':formal['event_date'],'supersedes':['GAMS-PENDING'],
     'accepts_sha256':sha(gams/'reference_review_pending.json'),'content_sha256':sha(gams/'reference.json')},
    {'evidence_id':'AND-REF','scope':'case','case_id':'AU_ANDOOM_LOCAL_SLATS2324','content_sha256':sha(andoom)},
    {'evidence_id':'AND-ACCEPT','scope':'case','case_id':'AU_ANDOOM_LOCAL_SLATS2324','supersedes':['AND-REF'],'accepts_sha256':accept['sha256']},
]
resolved=resolve_scoped_evidence(records)
assert len(resolved['by_case'])==2
assert {r['evidence_id'] for r in resolved['active']}=={'GAMS-FORMAL','AND-ACCEPT'}
crossed=dict(records[3]);crossed['supersedes']=['GAMS-PENDING']
try:
    resolve_scoped_evidence(records[:3]+[crossed])
except ValueError:
    cross_case_replacement_rejected=True
else:
    raise AssertionError('Cross-case replacement was accepted')
old_scope=next(r['report'] for r in data['rows'] if r['task_id']=='P1-03' and r['arm']=='C_three_role_checked')
scope_checked=bind_structured_report(old_scope,targets['P1-03'],evidence_records=records)
assert scope_checked['review_required'] and scope_checked['source_scope_errors']

latest={}
for line in (OLD/'provider_receipts.jsonl').read_text(encoding='utf-8').splitlines():
    r=json.loads(line);latest[r['key']]=r
truncated=[r for r in latest.values() if r['role']=='evidence']
assert len(truncated)==8
assert all(not completion_status(SimpleNamespace(choices=[SimpleNamespace(finish_reason=r['finish_reason'])]))['complete'] for r in truncated)
result={'passed':True,'scope':'offline regression replay of already observed failures; not a new AI benchmark',
        'numeric_repairs':repairs,'case_records':resolved,'cross_case_replacement_rejected':cross_case_replacement_rejected,
        'old_combined_case_explanation_withheld':scope_checked['interpretation_status'],
        'old_evidence_truncations_detected':len(truncated),'new_cloud_calls':0,'old_results_unchanged':True}
(OUT/'replay_validation.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps({'passed':True,'numeric_failures_replayed':2,'case_groups':2,'truncations_detected':8,'new_cloud_calls':0},ensure_ascii=False))
