"""Adapt the existing bounded acquisition code for a reserved, exact-date-only event."""
from pathlib import Path
import json,hashlib
ROOT=Path('D:/LMHLD');DATA=ROOT/'datasets/mine_benchmark_20261002'
OUT=DATA/'reserved_candidates/TR_COP2024'
context=json.loads((OUT/'context.json').read_text(encoding='utf-8'))
dates=context['source_requested_optical_dates'];case={**context,'time_interval':dates['pre']+'T00:00:00Z/'+dates['post']+'T23:59:59Z',
 'pre_through':dates['pre'],'post_from':dates['post'],'post_target':dates['post']}
plan={'cases':{'TR_COP2024':case},'scene_choice_rule':'EXACT external2025fieldpaper dates11Nov2023/24Feb2024, full coverage, highest processing baseline/sceneID; no response-based date fallback',
      'external_paper_sha256':context['boundary_source_pdf_sha256'],'mine_detector_predictions':0,'local_quality_review_status':'pending; all source-date originals retained, not silently switched for cloud/method outcomes'}
pp=OUT/'data_plan.json'
if pp.exists() and json.loads(pp.read_text(encoding='utf-8'))!=plan:raise ValueError('Exact-date plan changed')
if not pp.exists():pp.write_text(json.dumps(plan,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
base=ROOT/'validation/fetch_arid_incident_windows.py';text=base.read_text(encoding='utf-8')
replacements={
 "folder=OUT/eid":"folder=OUT",
 "OUT=ROOT/'datasets/arid_accident_operation_20261004'":"OUT=ROOT/'datasets/mine_benchmark_20261002/reserved_candidates/TR_COP2024'",
 "from mine_data_protocol import assert_development_cases,verify_assets":"from mine_data_protocol import assert_mine_group_role,verify_assets,registry,DATA as BENCHMARK_DATA",
 "assert_development_cases(['ZA_GAMS2020S','ZA_GAMS2023W'])":"assert_mine_group_role('TR_COP2024','reserved_final_test')",
 "else:eligible=[s for s in features if coverage(s)>=.999 and s['properties'].get('eo:cloud_cover',100)<20]":"else:\n            exact=c['source_requested_optical_dates']\n            groups={d:[s for s in features if coverage(s)>=.999 and s['properties']['datetime'][:10]==d] for d in exact.values()}\n            eligible=[max(group,key=lambda s:(str(s['properties'].get('s2:processing_baseline','')),s['id'])) for group in groups.values() if group]",
 "write(ledger,record)":"write(ledger,record)\n            value=registry();assets=value.setdefault('additional_frozen_assets',{})\n            for name,digest in record['images'][period]['assets_sha256'].items():\n                key=(folder/name).relative_to(BENCHMARK_DATA).as_posix()\n                if key in assets and assets[key]!=digest:raise ValueError('Frozen original changed')\n                assets[key]=digest\n            write(BENCHMARK_DATA/'mine_case_registry.json',value)",
 "Source-confirmed incident and anthropogenic-change raw pairs acquired; no predicted labels or safe-ground claim":"Exact fieldpaper-date raw pair acquired; mine stays reserved; detector predictions0"
}
for old,new in replacements.items():
 if text.count(old)!=1:raise ValueError('Unexpected base code structure:'+old)
 text=text.replace(old,new)
dest=ROOT/'validation/fetch_copler_reserved_original_pair.py'
if dest.exists() and dest.read_text(encoding='utf-8')!=text:raise ValueError('Generated download code changed')
dest.write_text(text,encoding='utf-8')
(OUT/'acquisition_code_provenance.json').write_text(json.dumps({'original_bounded_acquisition_code':str(base),'base_sha256':hashlib.sha256(base.read_bytes()).hexdigest(),'generated_code_sha256':hashlib.sha256(dest.read_bytes()).hexdigest(),'exact_dates_only':True,'raws_immediately_hash_frozen_and_upload_protected':True},indent=2)+'\n',encoding='utf-8')
print('Exact source dates frozen; downloads remain D: and no mining detector runs')
