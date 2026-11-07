"""Annual mine-work context, registered before imagery or detector predictions."""
from pathlib import Path
from datetime import datetime,timezone,timedelta
import copy,json,sys
ROOT=Path('D:/LMHLD');sys.path.insert(0,str(ROOT))
from model_registry import file_sha256
OUT=ROOT/'datasets/slats_operation_development_20261005';OUT.mkdir(exist_ok=True)
PARENT=ROOT/'datasets/source_backed_operation_change_20261005/full_published_geometry_window_v2/site_plan.json'

def run():
    old=json.loads(PARENT.read_text(encoding='utf-8'));amrun=copy.deepcopy(old['sites']['AU_AMRUN202309_FULL'])
    amrun.update(site_id='AU_AMRUN_SLATS2324',scope='annual source-backed expected mining-work context, not geotechnical safety')
    caval={'site_id':'AU_CAVAL_REGION_SLATS2324','mine_group':'AU_BMA_CAVAL_PEAK_COMPLEX',
           'role':'auxiliary_development_train','center_wgs84':[148.075,-22.08],
           'bbox_wgs84':[148.04,-22.13,148.11,-22.03],'crs':'EPSG:32755','commodity':'steelmaking coal region',
           'identity_source':'https://www.bhp.com/what-we-do/global-locations/australia/queensland/caval-ridge',
           'location_source':'https://www.coordinatorgeneral.qld.gov.au/projects/find-a-project/completed-projects/caval-ridge-mine',
           'coordinate_source':'approximate regional search window south of Moranbah;actual government mine-clearing geometry and imagery require review',
           'multiple_nearby_BMA_mines_not_claimed_independent_sites':True,'normal_or_stability_labels_available':False,
           'not_a_new_blind_mine':True,'scope':'annual expected mining-work context in vicinity;no exact mine boundary or slope safety assertion'}
    plan={'registered_at':datetime.now(timezone(timedelta(hours=8))).isoformat(),'sites':{'AU_AMRUN_SLATS2324':amrun,'AU_CAVAL_REGION_SLATS2324':caval},
          'scene_interval':'2023-07-01/2024-09-30','catalogue_intervals':['2023-07-01/2023-07-31','2024-09-01/2024-09-30'],
          'scene_cutoffs':{'pre_latest':'2023-07-31','post_earliest':'2024-09-01'},'max_native_window_dimension':1600,
          'acquisition_rule':'latest full-cover throughJul31_2023;earliest full-cover fromSep1_2024;EOcloud<20%;noimage/modelscoreselection',
          'source':'https://spatial-gis.information.qld.gov.au/arcgis/rest/services/Environment/SLATS/MapServer/255',
          'source_nominal_period':'August2023 toAugust2024;exact per-polygon image dates not available in service fields',
          'source_clearing_dates_not_assumed_to_be_September2023':True,
          'endpoint_scope':'wider than nominalannualperiod;additionalchange between endpoints remains unknown without provider geometry',
          'source_semantics':'government-classified Clearing-Mine and/or Regrowth-Mine supports expected work landcover,not absenceofinstability',
          'observational_concurrence_and_boundary_uncertainty_review_required':True,
          'classifier_predictions_at_registration':0,'pixel_labels_accepted':False,
          'training_sites':['AU_AMRUN_SLATS2324','AU_CAVAL_REGION_SLATS2324'],'normal_holdout_sites':[],
          'reserved_geography_and_Morenci_training_excluded':True,'new_failure_positive_cases':0,
          'no_new_independent_test_or_field_trial_claim':True,'parent_plan_sha256':file_sha256(PARENT)}
    p=OUT/'site_plan.json'
    if p.exists():
        saved=json.loads(p.read_text(encoding='utf-8'))
        if any(saved[k]!=v for k,v in plan.items() if k!='registered_at'):raise ValueError('Annual source-backed plan changed')
    else:p.write_text(json.dumps(plan,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print('2 annual expected-work development regions registered;sourceyearnot dailytruth;labels/predictions0')

if __name__=='__main__':run()
