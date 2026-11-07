"""Reserve a previously unused mine before any detector outputs or model fitting."""
from pathlib import Path
from datetime import datetime
import json,sys
ROOT=Path('D:/LMHLD');sys.path.insert(0,str(ROOT))
from model_registry import file_sha256
from mine_data_protocol import verify_assets,assert_mine_group_role
DATA=ROOT/'datasets/mine_benchmark_20261002';REGISTRY=DATA/'mine_case_registry.json'
OUT=DATA/'reserved_candidates/TR_COP2024';SRC=ROOT/'datasets/pixel_boundary_sources_20261004/copler2025_field_paper.pdf'
value=json.loads(REGISTRY.read_text(encoding='utf-8'));verify_assets(value)
case={'mine_id':'TR_COPLER','mine_group':'TR_COPLER','case_id':'TR_COP2024','event_date':'2024-02-13',
      'scope':'heap_leach_flow_failure_extended','role':'reserved_final_test','quality_review_status':'pending',
      'model_predictions_run':0,'use_all_times_with_same_mine_role':True,'independent_expert_or_field_audit':False,
      'external_published_field_campaign_support':True,'external_campaign_is_not_project_field_feedback':True,
      'boundary_source':'https://doi.org/10.1785/0320250007','boundary_source_pdf_sha256':file_sha256(SRC),
      'operator_event_source':'https://www.ssrmining.com/2024-copler-incident/',
      'boundary_source_kind':'published field-informed research figure; original surveyGIS not obtained',
      'bbox_wgs84':[38.485,39.405,38.57,39.474],'crs':'EPSG:32637',
      'source_requested_optical_dates':{'pre':'2023-11-11','post':'2024-02-24'},
      'source_base_map_date':'2021-11-19','source_base_map_is_not_postaccident_photo':True,
      'acquisition':'reserved_candidates/TR_COP2024/acquisition.json',
      'prior_exposure_audit':'no event/group in existing case registry; published inventory bbox query0; detector outputs0; ImageNet pretraining is not a field-test guarantee',
      'final_acceptance_requires':['source-date/source-polygon identity','independent held-ground registration','local visibility and actual producer calibration','historical mine exposure review','frozen candidate models passing development checks'],
      'new_test_model_selection_allowed':False}
if 'TR_COP2024' in value['cases']:raise SystemExit('Already reserved; no rewrite')
if any(c.get('mine_group')=='TR_COPLER' or c.get('mine_id')=='TR_COPLER' for c in value['cases'].values()):raise ValueError('Mine already registered')
value['cases']['TR_COP2024']=case;assert_mine_group_role('TR_COP2024','reserved_final_test',value)
OUT.mkdir(exist_ok=True);(OUT/'context.json').write_text(json.dumps({'registered_at':datetime.now().isoformat(timespec='seconds'),**case},ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
REGISTRY.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
sys.stdout.reconfigure(encoding='utf-8');print('Reserved TR_COP2024 heap-leach extension before any mine detector predictions')
