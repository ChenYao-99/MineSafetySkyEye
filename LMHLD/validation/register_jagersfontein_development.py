"""Reserve a new development group and exact actual sensing dates before classifier use."""
from pathlib import Path
from datetime import datetime
import json,sys,hashlib
ROOT=Path('D:/LMHLD');sys.path.insert(0,str(ROOT))
from model_registry import file_sha256
from mine_data_protocol import registry,verify_assets,assert_mine_group_role
DATA=ROOT/'datasets/mine_benchmark_20261002';OUT=ROOT/'datasets/jagersfontein_development_20261005';OUT.mkdir(exist_ok=True)
SRC=ROOT/'datasets/maxclosure_sources_20261005/jagersfontein/charter_product_28498588.img'
v=registry();verify_assets(v)
eid='ZA_JAGS2022'
if eid in v['cases']:raise SystemExit('Group already registered;no role changes')
if any((c.get('mine_group') or c.get('mine_id'))=='ZA_JAGERSFONTEIN' for c in v['cases'].values()):raise ValueError('Mine already registered')
case={'case_id':eid,'mine_id':'ZA_JAGERSFONTEIN','mine_group':'ZA_JAGERSFONTEIN','role':'development','scope':'tailings_failure_land_deposit_extended','event_date':'2022-09-11',
 'quality_review_status':'pending','model_predictions_run':0,'independent_expert_or_field_audit':False,'use_all_times_with_same_mine_role':True,
 'source_authority':'SANSA official InternationalDisastersCharter product,13Sept2022Pleiades base with red flooded-area outline',
 'source_product_url':'https://disasterscharter.org/cos-api/api/file/public/article-image/28498588','source_product_sha256':file_sha256(SRC),
 'reference_geometry_scope':'mapped visible land deposit/affected area inside cropped reference viewport;notcomplete downstream extent ororiginalshapefile',
 'source_map_as_model_features':False,'source_map_capture_date':'2022-09-13',
 'bbox_wgs84':[25.399,-29.805,25.463,-29.748],'crs':'EPSG:32735',
 'source_requested_optical_dates':{'pre':'2022-09-06','post':'2022-09-16'},
 'date_basis':'5days before confirmed11Septembercollapse and3days after official13Septemberdatedsource;16SeptSAR/Sentinel officialmap also reportssamepostdate',
 'normal_safety_claim_allowed':False,'new_unseen_test_claim_allowed':False,'acquisition':'../jagersfontein_development_20261005/ZA_JAGS2022/acquisition.json'}
v['cases'][eid]=case;assert_mine_group_role(eid,'development',v)
(DATA/'mine_case_registry.json').write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
(OUT/'context.json').write_text(json.dumps({'registered_at':datetime.now().isoformat(timespec='seconds'),**case},ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
plan={'cases':{eid:{**case,'time_interval':'2022-09-06T00:00:00Z/2022-09-16T23:59:59Z','pre_through':'2022-09-06','post_from':'2022-09-16','post_target':'2022-09-16'}},
 'scene_choice_rule':'exact registered dates only;fullcoverage then highestprocessingbaseline/sceneID;no algorithm response or cloud-based date substitution','model_outputs_for_plan':0}
(OUT/'data_plan.json').write_text(json.dumps(plan,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
base=ROOT/'validation/fetch_arid_incident_windows.py';text=base.read_text(encoding='utf-8')
replacements={"OUT=ROOT/'datasets/arid_accident_operation_20261004'":"OUT=ROOT/'datasets/jagersfontein_development_20261005'",
 "assert_development_cases(['ZA_GAMS2020S','ZA_GAMS2023W'])":"assert_development_cases(['ZA_JAGS2022'])",
 "else:eligible=[s for s in features if coverage(s)>=.999 and s['properties'].get('eo:cloud_cover',100)<20]":"else:\n            groups={d:[s for s in features if coverage(s)>=.999 and s['properties']['datetime'][:10]==d] for d in ('2022-09-06','2022-09-16')}\n            eligible=[max(g,key=lambda s:(str(s['properties'].get('s2:processing_baseline','')),s['id'])) for g in groups.values() if g]",
 "Source-confirmed incident and anthropogenic-change raw pairs acquired; no predicted labels or safe-ground claim":"Exact-date independentofficial-map developmentcase originalS2windows acquired;sourceplotnotmodelinput;no safety claim"}
for old,new in replacements.items():
 if text.count(old)!=1:raise ValueError('Unexpected bounded downloader base')
 text=text.replace(old,new)
dest=ROOT/'validation/fetch_jagersfontein_original_pair.py';dest.write_text(text,encoding='utf-8')
(OUT/'acquisition_code_provenance.json').write_text(json.dumps({'base_code_sha256':file_sha256(base),'download_code_sha256':file_sha256(dest),'source_code_reused_without_old_cases_or_automatic_role_promotion':True},indent=2)+'\n',encoding='utf-8')
sys.stdout.reconfigure(encoding='utf-8');print('NewZA_JAGERSFONTEIN developmentgroup andexactdates locked; no classifieroutputs')
