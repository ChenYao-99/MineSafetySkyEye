"""Keep source rechecks separate from new qualified accident ground truth."""
from pathlib import Path
from datetime import datetime,timezone,timedelta
import sys,json
ROOT=Path('D:/LMHLD');sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'validation'))
from runtime_paths import configure_runtime_paths
configure_runtime_paths()
from model_registry import file_sha256
from mine_data_protocol import DATA,registry,reserved_case_ids,verify_assets
from finalize_cause_terrain_round_20261005 import protected_runtime
OUT=ROOT/'datasets/independent_boundary_source_triage_20261006'

def load(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def write(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')

def run():
    verify_assets();inventory=registry();protected=protected_runtime()
    reserved={c:inventory['cases'][c]['model_predictions_run'] for c in sorted(reserved_case_ids(inventory))}
    assert len(reserved)==4 and not any(reserved.values()) and not (DATA/'final_evaluation_once.json').exists()
    acquisition=load(OUT/'acquisition.json')
    for row in acquisition['original_files']:assert file_sha256(row['path'])==row['sha256']
    pdf=load(OUT/'author_pdf_acquisition.json')
    for row in pdf['rows']:
        if row['acquired']:assert file_sha256(OUT/row['name'])==row['sha256']
    las=load(DATA/'development/LasCruces/label_reference.json')
    assert las['reference_figure'].startswith('Figure 3b') and las['reference_date']=='2019-01-25 helicopter SfM'
    assert las['target_date']=='2019-01-27 Sentinel-2' and las['date_difference_days']==2
    decisions=[
        {'source':'ESA Zenodo821031','URL':'https://zenodo.org/records/821031',
         'original_source_provider':'ESA author;modifiedCopernicus observations',
         'file_inventory':'two3channelNIRfalsecolourTIFFs,twoPNGs,twoPNGworldfiles;no original accidentpolygon GIS',
         'reference_before_after_dates':['2017-06-01','2017-06-24'],
         'new_native_BGRN_pair_or_strong_accident_labels_accepted':False,'images_not_downloaded_no_duplicate_source_count':True},
        {'source':'PLOS2025 DOI10.1371/journal.pone.0323650 Fig27','URL':'https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0323650',
         'original_author_figure_downloaded_and_MD5_verified':True,'event_date_from_article':'2023-09-16T15:12:00;anonymousEastChinaopenpit',
         'visual_review':'oblique minewall photograph with slope/crack/slide-body annotations;redcurve is shear outlet,not closed full accident map polygon',
         'original_source_mine_geolocation_unambiguous':False,'ground_georeference_or_independent_controls_available':False,
         'complete_source_runout_inventory_available':False,'new_native_case_or_strong_pixel_labels_accepted':False,
         'not_weakpresence_training_just_because_fieldimage_exists':True},
        {'source':'LasCruces2020 Zenodo4704955 authororiginal','URL':'https://zenodo.org/records/4704955',
         'new_document_copy_is_recheck_not_new_accident_or_new_independent_geometry':True,
         'event_date':'2019-01-23','post_author_photos_date':'2019-01-25','existing_Sentinel2_target_date':'2019-01-27',
         'pre_author_airphotos':'February2016','georeferencing':'2013ALS5GCPs outsidepit;pre2016DEM usedoneadditionalinpitGCP',
         'external_boundary_source':'author photointerpretation of pre/post orthophotos,Fig3;distinct from theoretical basal shear plane/simulatedsourcearea inFig4',
         'already_registered_label_dates_confirmed_not_changed':True,'original204helicopter_photos_or_originalDEMs_or_accidentGIS_acquired':False,
         'new_complete_original_survey_GIS_or_independent_currentpit_measurements':0},
        {'source':'Carla2018 university fullauthorPDF','URL':'https://flore.unifi.it/bitstream/2158/1131006/1/Carla%CC%80%20et%20al%20ENGINEERING%20GEOLOGY%202018.pdf',
         'fresh_indexed_address_checked_once':True,'HTTP_status':403,'full_author_PDF_acquired':False,'existing_public_figures_not_modified_or_upgraded':True}
    ]
    next_work='Seek publicly released original dated accident GIS or georeferencedauthorortho and exact-date independent operation polygons innewmaterials;do not redigitize obliquephotos or anonymousshearoutlets as completegroundtruth;keepCNNtraining boundeduntil better sourceevidence;no more sameRGBnormalization/changeonly/SIFTmargin/thresholdsweeps'
    receipt={'verified_at':datetime.now(timezone(timedelta(hours=8))).isoformat(timespec='seconds'),
        'status':'next independentboundary source round executed;one newunlocated fieldfigure rejected;existingLasCruces source/date reverified;no new strongpixeltruth accepted',
        'source_decisions':decisions,'new_original_author_fieldfigures':1,'new_qualified_full_accident_boundaries':0,
        'new_original_GIS_or_current_terrain_or_field_trials_or_expertfeedback':0,
        'all_existing_source_labels_and_date_gaps_unchanged':True,'new_native_pairs':0,'new_trained_models_or_prediction_calls':0,
        'reserved_predictions':reserved,'native_CNN_total':218,'trained_feature_diagnostics_total':119,
        'production_default_changed':False,'protected_runtime_artifacts':protected,'all_new_downloads_cache_and_temp_on_D':True,
        'new_cloud_LLM_or_contact_calls':0,'competition_materials_created':False,'next':next_work,
        'frozen_original_source_assets':{p.name:file_sha256(p) for p in OUT.iterdir() if p.is_file() and p.suffix in ('.tif','.pdf','.json') and p.name not in ('verification.json',)}}
    write(OUT/'verification.json',receipt)
    state=load(DATA/'STATE.json')
    assert state['completed_native_training_runs']==state['completed_native_evaluated_runs']==218
    assert state['completed_lightweight_feature_diagnostics']==119
    state.update(updated_at=receipt['verified_at'],phase='independent_boundary_source_recheck_complete_no_new_strongtruth_sensor_input_repair_verified',
        current_study=str(OUT/'source_triage_protocol.json'),latest_completed_independent_boundary_source_triage=str(OUT/'verification.json'),
        new_full_independent_boundary_this_source_round=0,new_native_CNN_fits_current_phase=0,
        next=next_work,next_local_work=next_work,default_model_changed=False,new_final_test_predictions=0,
        running_training_session=None,running_evaluation_session=None,running_download_session=None,running_training_script=None,running_download_script=None)
    write(DATA/'STATE.json',state)
    print('Independent source roundcomplete;newfieldfigure rejected as ungeoreferenced;existingLasCruces dates confirmed;0newstronglabels;0predictions;218CNN unchanged',flush=True)

if __name__=='__main__':run()
