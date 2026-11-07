"""Freeze reviewed research benchmarks separately from model readiness and field certification."""
from pathlib import Path
from datetime import datetime
import json,sys
ROOT=Path('D:/LMHLD');sys.path.insert(0,str(ROOT))
from model_registry import file_sha256
from mine_data_protocol import registry,verify_assets,assert_mine_group_role
DATA=ROOT/'datasets/mine_benchmark_20261002'
FOLDERS={'CL0002':DATA/'quality_audit/CL0002/published_inventory_benchmark_v2_post_and_pair_20261005','KG0001':DATA/'quality_audit/KG0001/published_inventory_benchmark_v1_20261005'}
def write(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def run():
 v=registry();verify_assets(v)
 for eid,folder in FOLDERS.items():
  assert_mine_group_role(eid,'reserved_final_test',v)
  if v['cases'][eid]['model_predictions_run']!=0:raise ValueError('Reserved detector already observed')
  pending=json.loads((folder/'review_pending.json').read_text(encoding='utf-8'))
  for p,sha in pending['frozen_dependencies'].items():
   if file_sha256(p)!=sha:raise ValueError('Source dependency changed:'+p)
  for name,sha in pending['frozen_assets'].items():
   if file_sha256(folder/name)!=sha:raise ValueError('Benchmark derivative changed:'+name)
  if not pending['dataset_quality_numeric_gate_pass']:raise ValueError('Numeric data qualification failed')
  reference=folder/'reference.json'
  if not reference.exists():
   pending.update(reviewed_at=datetime.now().isoformat(timespec='seconds'),accepted_data_reference=True,manual_review_status='accepted_independent_published_reference_scored_support',
     visual_review={'reviewer':'projectagent;not independent fieldexpert','original_source_dates_and_method_reviewed':True,'original_post_polygon_geometry_and_native_pair_visibility_reviewed':True,'all30_held_ground_correspondences_viewed':True,'weather_or_edge_uncertainty_not_forced_to_background':True,'field_or_safety_certification':False})
   write(reference,pending)
  if eid=='CL0002':
   scopes={'post_observed_reference':{'mask':'labels_post_reference_interior.tif','input_modes':['post_only','pair_with_encoded_missingness'],'positive_pixels':2163,'annotation_background_pixels':243973},
           'paired_observed_reference':{'mask':'labels_paired_reference_interior.tif','input_modes':['paired_observed_only'],'positive_pixels':521,'annotation_background_pixels':242289}}
  else:scopes={'paired_observed_reference':{'mask':'labels_reference_interior.tif','input_modes':['post_only','pair_with_encoded_missingness','paired_observed_only'],'positive_pixels':1751,'annotation_background_pixels':242532}}
  contract={'schema_version':1,'case_id':eid,'role':'reserved_final_test','reference_kind':'independent_complete_published_inventory_with_scored_support','reference_path':str(reference),'reference_sha256':file_sha256(reference),
    'target_label_epoch':'post','target_definition':'mapped postevent footprint;not a per-pixel declaration of newly created movement since pre',
    'scopes':scopes,'allowed_metrics':['precision_on_scored_published_reference','recall_on_scored_published_reference','IoU_on_scored_published_reference'],
    'unknown_labels':255,'unknown_scored_as_background':False,'outside_mapped_event_is_geotechnically_safe':False,
    'source_original_field_survey':False,'unmasked_absolute_boundary_accuracy_allowed':False,'mine_wide_all_hazard_absence_allowed':False,
    'development_training_allowed':False,'detector_execution_requires_existing_frozen_final_gate':True,'input_support_scope_must_be_frozen_before_predictions':True}
  cp=folder/'usage_contract.json'
  if cp.exists() and json.loads(cp.read_text(encoding='utf-8'))!=contract:raise ValueError('Frozen benchmark use contract changed')
  if not cp.exists():write(cp,contract)
  c=v['cases'][eid];c.update(quality_review_status='accepted',data_reference_readiness='accepted_scoped_independent_published_inventory',
    model_evaluation_readiness='not_opened;waitingfrozenpassingmodelandfinalgate',benchmark_reference=str(reference),benchmark_reference_sha256=file_sha256(reference),benchmark_usage_contract=str(cp),
    benchmark_usage_contract_sha256=file_sha256(cp),benchmark_reference_scopes=list(scopes),complete_published_inventory_boundary_available=True,
    original_independent_professional_field_survey_received=False,independent_expert_or_field_audit=False,unresolved_quality=[],
    quality_acceptance_scope='scoped research benchmark agreement only;no absolute survey accuracy or safety certification',
    new_unseen_main_event_accepted=(eid=='CL0002'),event_boundary_completeness_is_publisher_interpretation=True)
  assets=v.setdefault('additional_frozen_assets',{})
  protected=[p for p in folder.iterdir() if p.is_file()]
  if eid=='CL0002':protected += [DATA/'quality_audit/CL0002/chronology_safe_preimage_v3_consistent_quality_20261005/pre_producer_BGRN_original_grid.tif']
  for p in protected:
   key=p.relative_to(DATA).as_posix();digest=file_sha256(p)
   if key in assets and assets[key]!=digest:raise ValueError('Existing protected dataset asset changed')
   assets[key]=digest
 write(DATA/'mine_case_registry.json',v);verify_assets(v)
 s=json.loads((DATA/'STATE.json').read_text(encoding='utf-8'));s.update(accepted_scoped_published_inventory_test_cases=['CL0002','KG0001'],
   final_candidate_quality_all_pending=False,accepted_new_main_pit_wall_benchmark_cases=['CL0002'],second_independent_main_test_still_missing=True,
   complete_independent_published_inventory_vectors_qualified=2,original_independent_professional_survey_boundaries_qualified=0,
   main_test_model_predictions_completed=False,new_final_test_predictions=0,running_download_script=None,
   next='Two independent published-reference benchmark data cases admitted without detector outputs;second main pitwall case/true survey proof still missing.Expand trainable cases and test a materially different representation;freeze input/score scopes before one final evaluation')
 write(DATA/'STATE.json',s)
 print('CLmain/KGextended benchmark data admitted;explicit scope and input-mode contracts;detector0;field certification0')
if __name__=='__main__':sys.stdout.reconfigure(encoding='utf-8');run()
