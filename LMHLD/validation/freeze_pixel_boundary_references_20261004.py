"""Freeze reviewed partial references with explicit admissible uses and mine isolation."""
from pathlib import Path
from datetime import datetime
import json,sys
ROOT=Path('D:/LMHLD');sys.path.insert(0,str(ROOT))
from model_registry import file_sha256
from mine_data_protocol import registry,verify_assets,assert_mine_group_role
DATA=ROOT/'datasets/mine_benchmark_20261002';CASE=DATA/'reserved_candidates/TR_COP2024';OUT=CASE/'pixel_reference_v1_field2025'
GAMS=ROOT/'datasets/arid_accident_operation_20261004/ZA_GAMS2020S/pixel_reference_v1_wang2026'
def write(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def run():
 value=registry();verify_assets(value);assert_mine_group_role('TR_COP2024','reserved_final_test',value)
 if value['cases']['TR_COP2024']['model_predictions_run']!=0:raise ValueError('Reserved mine already seen')
 pending=OUT/'reference_review_pending.json';v=json.loads(pending.read_text(encoding='utf-8'))
 for path,digest in v['frozen_dependencies'].items():
  if file_sha256(path)!=digest:raise ValueError('Frozen dependency changed')
 for name,digest in v['frozen_assets'].items():
  if file_sha256(OUT/name)!=digest:raise ValueError('Frozen partial reference changed')
 ref=OUT/'reference.json'
 if not ref.exists():
  v.update(reviewed_at=datetime.now().isoformat(timespec='seconds'),review_status='accepted_reserved_partial_field_informed_research_reference',accepted_for_reserved_known_core_reference=True,
    visual_review={'performed_by':'project agent;no project expert credential claimed','original2025Figure1_and_fieldcampaign_description_read':True,
      'all30_sourcepre_and_all30_nativepair_holdout_sites_viewed':True,'original_pre_post_partial_outline_overlay_viewed':True,
      'known_core_post_morphology_consistent_with_source_mapped_failure':True,'excluded_pre_shadow_support_not_relabeled':True,
      'post_SCL6_dark_or_wet_material_is_not_negative_or_safety_evidence':True,'full_boundary_completeness_not_accepted':True},
    target_label_epoch='post',pre_overlay_is_location_comparison_only=True,pre_epoch_accident_absence_asserted=False,
    actual_sensor_platform_from_scene_ids='Sentinel-2B on both exact dates;paper describes Sentinel-2A generically;not asserted to be identical processing products')
  write(ref,v)
 for eid,folder,role,positive in [('ZA_GAMS2020S',GAMS,'development',38),('TR_COP2024',OUT,'reserved_final_test',2978)]:
  contract={'schema_version':1,'case_id':eid,'role':role,'reference':str(folder/'reference.json'),'reference_sha256':file_sha256(folder/'reference.json'),
    'native_resolution_m':10,'positive_pixels':positive,'target_label_epoch':'post','pre_overlay_is_location_comparison_only':True,
    'labels':{'positive':1,'unknown':255,'background_not_defined':0},'background_pixels':0,
    'accepted_metric_scope':['recall_on_observed_conservative_positive_core'],
    'full_accident_IoU_allowed':False,'full_area_accuracy_allowed':False,'unknown_pixels_may_become_background':False,
    'training_allowed':role=='development','final_detector_execution_allowed_now':False,
    'original_survey_GIS':False,'qualification':'traceable registered partial research reference;not full independent professional ground truth'}
  path=folder/'usage_contract.json'
  if path.exists() and json.loads(path.read_text(encoding='utf-8'))!=contract:raise ValueError('Frozen use contract changed')
  if not path.exists():write(path,contract)
 c=value['cases']['TR_COP2024'];c.update(partial_pixel_reference=str(ref),partial_pixel_reference_sha256=file_sha256(ref),
   partial_research_reference_accepted=True,partial_research_positive_pixels=v['positive_pixels'],complete_independent_accident_GIS=False,
   source_pre_and_original_pair_geolocation_accepted=True,source_post_direct_registration_accepted=False,
   quality_review_status='partial_boundary_accepted_final_case_quality_pending',historical_supervised_mine_exposure_review='documented protocols audit pending;no observed detector outputs')
 # Known previews and all label/QA derivatives receive the same holdout guard as the raw images.
 protected=[CASE/'pre_original_preview.png',CASE/'post_original_preview.png',CASE/'native_pair_geolocation_v1/all30_native_pair_holdout_sites.png',CASE/'reference_registration_v2_field2025_map_prior/pre_all30_holdout_sites.png']
 protected += [OUT/name for name in ('labels_partial.tif','confidence_partial.tif','author_outline_geometry_only.tif','boundary_uncertainty_support.tif','published_outline_and_conservative_core_wgs84.geojson','original_pair_partial_reference_review.png','reference.json','usage_contract.json')]
 assets=value.setdefault('additional_frozen_assets',{})
 for p in protected:
  key=p.relative_to(DATA).as_posix();digest=file_sha256(p)
  if key in assets and assets[key]!=digest:raise ValueError('Frozen reserved derived asset changed')
  assets[key]=digest
 write(DATA/'mine_case_registry.json',value);verify_assets(value)
 print('Two scoped native10m references frozen;38 development /2978 reserved positives;no wholeIoU truth;no detector outputs')
if __name__=='__main__':sys.stdout.reconfigure(encoding='utf-8');run()
