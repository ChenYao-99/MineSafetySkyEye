"""A new light/grey field-investigated pitwall accident, before imagery predictions."""
from pathlib import Path
from datetime import datetime,timezone,timedelta
import json,sys
ROOT=Path('D:/LMHLD');sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'validation'))
from runtime_paths import configure_runtime_paths
configure_runtime_paths()
from rasterio.warp import transform_bounds
from model_registry import file_sha256
from mine_data_protocol import registry,verify_assets,assert_development_cases,DATA
OUT=ROOT/'datasets/madenkoy_development_20261005'

def load(p):return json.loads(p.read_text(encoding='utf-8'))
def write(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')

def run():
    verify_assets();bbox=list(transform_bounds('EPSG:32638','EPSG:4326',248000,4218000,252500,4222500))
    site={'site_id':'TR_MAD2016','mine_group':'TR_MADENKOY_SIIRT','role':'development',
          'scope':'dated openpitwallfailure;fieldpublication outline pending identity/geometric registration',
          'center_wgs84':[(bbox[0]+bbox[2])/2,(bbox[1]+bbox[3])/2],'bbox_wgs84':bbox,'crs':'EPSG:32638',
          'commodity':'copper with lightgrey carbonate/colluvium slope;not mineral inference from model',
          'coordinate_source':'regional search window from printed249000..252000/4218500..4222000 map inYavuz2023;UTM38 assumption only locates imagery;not accidentboundary',
          'coordinate_datum_not_yet_independently_verified':True,'not_a_new_blind_mine':True,
          'normal_or_stability_labels_available':False,'event_date':'2016-11-17',
          'event_time_local_reported':'20:30','same_day_images_not_accepted_as_post':True,
          'source_event_primary':'https://dergipark.org.tr/en/pub/ngumuh/article/1269412',
          'source_field_outline_primary':'https://doi.org/10.1016/j.enggeo.2018.01.021',
          'field_outline_source_anonymizes_mine_identity':True,
          'independent_cross_source_identity_alignment_required':True,'source_publication_fieldwork_not_project_trial':True}
    proposal={'registered_at':datetime.now(timezone(timedelta(hours=8))).isoformat(),'sites':{'TR_MAD2016':site},
              'source_nominal_period':'17November2016 accident;sourceaerialexactflightdatesunknownbutcaptionbracketsfailure',
              'scene_interval':'2016-10-01/2016-12-15','scene_cutoffs':{'pre_latest':'2016-11-16','post_earliest':'2016-11-18'},
              'max_native_window_dimension':800,'scene_cloud_limit':20,
              'acquisition_rule':'latest fullcover strictly before17Nov2016 and earliest fullcover strictly after17Nov2016;actualscenecloud<20%;no model or derivedlabelscore selection',
              'training_sites':['TR_MAD2016'],'normal_holdout_sites':[],
              'reserved_geography_and_same_mine_roles_excluded':True,'classifier_predictions_at_registration':0,
              'pixel_labels_accepted':False,
              'source_frozen_assets':{str(p):file_sha256(p) for p in [OUT/'yavuz2023_field_original.pdf',OUT/'carla2018_original_public_fig2.jpg',OUT/'carla2018_original_public_fig3.jpg']},
              'old_geological_landslide_boundaries_and_simulated_sections_not_accidentlabels':True,
              'original_professional_boundary_vector_received':False}
    path=OUT/'site_plan.json'
    if path.exists():
        old=load(path);assert all(old.get(k)==v for k,v in proposal.items() if k!='registered_at')
    else:write(path,proposal)
    inventory=registry()
    if 'TR_MAD2016' not in inventory['cases']:
        inventory['cases']['TR_MAD2016']={'mine_id':site['mine_group'],'mine_group':site['mine_group'],'event_date':'2016-11-17',
              'scope':'open_pit_wall_failure','role':'development','quality_review_status':'pending_independent_identity_registration_and_original_visibility',
              'model_predictions_run':0,'pixel_reference_accepted':False,'source_plan':str(path),
              'event_source':site['source_event_primary'],'field_boundary_source':site['source_field_outline_primary'],
              'field_boundary_source_anonymized_identity_needs_correspondence':True,
              'original_expert_GIS_received':False,'independent_expert_or_field_audit':False,'not_new_blind_test':True}
        write(DATA/'mine_case_registry.json',inventory)
    assert_development_cases(['TR_MAD2016'])
    print('New developmentaccident registered;identity andfieldboundary pending;rawGISabsent;frozenbbox',bbox,flush=True)

if __name__=='__main__':run()
