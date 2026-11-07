"""Accept a new official-map-backed development core after original/holdout visual review."""
from pathlib import Path
from datetime import datetime
import json,sys
ROOT=Path('D:/LMHLD');sys.path.insert(0,str(ROOT))
from model_registry import file_sha256
from mine_data_protocol import registry,verify_assets,assert_development_cases
DATA=ROOT/'datasets/mine_benchmark_20261002';CASE=ROOT/'datasets/jagersfontein_development_20261005/ZA_JAGS2022';OUT=CASE/'pixel_reference_v1_official_partial'
def write(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def run():
 assert_development_cases(['ZA_JAGS2022']);verify_assets();v=json.loads((OUT/'reference_review_pending.json').read_text(encoding='utf-8'))
 for p,digest in v['frozen_dependencies'].items():
  if file_sha256(p)!=digest:raise ValueError('Frozen officialcore dependency changed')
 for name,digest in v['frozen_assets'].items():
  if file_sha256(OUT/name)!=digest:raise ValueError('Frozen officialcore changed')
 if not(OUT/'reference.json').exists():
  v.update(reviewed_at=datetime.now().isoformat(timespec='seconds'),accepted_for_development_positive_core_supervision=True,
    review_status='accepted_registered_partial_official_reference_development_core',
    visual_review={'reviewer':'projectagent;no fieldcredential claimed','officialfloodedarea legend/capture13Sept2022 read':True,'all30source_ground_holdouts_viewed':True,
      'source_and_original16Septcore_overlay_viewed':True,'knowncore_within_actual_landdeposit_on_originalRGB':True,'sourcecropedge_andcomplete_downstream_extent_not_promoted':True,
      'source_and_target_three_day_gap_preserved':True,'project_field_trial_or_expert_feedback':False})
  write(OUT/'reference.json',v)
 contract={'schema_version':1,'case_id':'ZA_JAGS2022','role':'development','reference':str(OUT/'reference.json'),'reference_sha256':file_sha256(OUT/'reference.json'),
  'native_resolution_m':10,'target_label_epoch':'post','positive_pixels':16681,'labels':{'positive':1,'unknown':255,'background_not_defined':0},'background_pixels':0,
  'training_allowed':True,'paired_loss_or_metrics_require_subset_support':'paired_positive_support.tif',
  'accepted_metric_scope':['recall_on_observed_conservative_positive_core'],'full_accident_IoU_allowed':False,'unknown_pixels_may_become_background':False,
  'original_survey_GIS':False,'qualification':'professionalofficialmappedpartial reference registered byproject;not originalshapefile/full downstreamboundary orprojectfieldcertificate'}
 write(OUT/'usage_contract.json',contract)
 r=registry();c=r['cases']['ZA_JAGS2022'];c.update(quality_review_status='accepted_for_development_partial_official_reference',partial_pixel_reference=str(OUT/'reference.json'),
  partial_pixel_reference_sha256=file_sha256(OUT/'reference.json'),partial_research_positive_pixels=16681,partial_research_reference_accepted=True,complete_independent_accident_GIS=False)
 write(DATA/'mine_case_registry.json',r)
 s=json.loads((DATA/'STATE.json').read_text(encoding='utf-8'));s.update(trainable_official_mapped_new_mine_groups=1,
   new_trainable_official_partial_core_pixels=16681,new_trainable_official_partial_core_reference=str(OUT/'reference.json'),new_original_jagersfontein_pair_complete=True,
   next='Trainable officialmapbacked ZA_JAGS2022 partialcore accepted;compare materially different frozenstructure representation with matched data controls on developmentmines;CLKG scoped independent test data accepted andremainunopened')
 if 'ZA_JAGS2022' not in s['accepted_development_reference_ids']:s['accepted_development_reference_ids'].append('ZA_JAGS2022')
 write(DATA/'STATE.json',s);print('16681new trainable postcore /15974pairedcore;zero invented background;developmentmine nottest;officialcropstillpartial')
if __name__=='__main__':sys.stdout.reconfigure(encoding='utf-8');run()
