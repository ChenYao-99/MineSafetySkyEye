"""Register new data roles before any candidate inference, grouping all Gamsberg incidents."""
from pathlib import Path
from datetime import datetime
import json,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from mine_data_protocol import registry,REGISTRY,verify_assets
from audit_auxiliary_geometry_split import reserved_envelopes
from model_registry import file_sha256
OUT=ROOT/'datasets/arid_accident_operation_20261004'


def write(p,v):p.write_bytes((json.dumps(v,ensure_ascii=False,indent=2)+'\n').encode('utf-8'))


def run():
    OUT.mkdir(exist_ok=True);path=OUT/'data_plan.json';value=registry();verify_assets(value)
    cases={
      'ZA_GAMS2020S':{'mine_group':'ZA_GAMSBERG','role':'development','scope':'open_pit_wall_failure',
        'event_date':'2020-11-17','event_source':'https://www.vedantalimited.com/Vedanta2021/zinc-international.html',
        'reference_capture_date':'2020-12-03','reference_scope':'public originally published SkySat reference; not field GIS',
        'bbox_wgs84':[18.925,-29.268,18.995,-29.205],'crs':'EPSG:32734',
        'time_interval':'2020-11-01T00:00:00Z/2020-12-12T23:59:59Z',
        'pre_through':'2020-11-16','post_from':'2020-11-18','post_target':'2020-12-03',
        'location_basis':'approximate mineral-locality anchor cross-checked against operator site plan; never accident boundary coordinates',
        'coordinate_source':'https://www.mindat.org/maps.php?id=53701'},
      'ZA_GAMS2023W':{'mine_group':'ZA_GAMSBERG','role':'development','scope':'open_pit_wall_failure',
        'event_date':'2023-08-03','event_source':'https://papers.acg.uwa.edu.au/d/2535_52_Pretorius/52_Pretorius.pdf',
        'reference_scope':'author reports2023 failure; Fig8 black polygon marks2022 failure and must not be relabelled2023 boundary',
        'bbox_wgs84':[18.925,-29.268,18.995,-29.205],'crs':'EPSG:32734',
        'time_interval':'2023-07-15T00:00:00Z/2023-08-25T23:59:59Z',
        'pre_through':'2023-08-02','post_from':'2023-08-04','post_target':'2023-08-05'},
      'FaxeDeposition2021':{'mine_group':'Faxe','role':'auxiliary_development','scope':'regulator_confirmed_anthropogenic_deposition_safety_unknown',
        'event_date_interval':['2021-06-22','2022-03-02'],'event_source':'https://app-rsjdxp-cms-prod-001.azurewebsites.net/media/wt3k122o/13-07-2022-thornfjeldsvej-0.pdf',
        'reference_scope':'regulator site inspectionMarch2,2022 and drone elevationsDecember2021 confirm placed material; original height breach and safety unknown',
        'bbox_wgs84':[12.090,55.235,12.140,55.275],'crs':'EPSG:32633',
        'time_interval':'2021-06-01T00:00:00Z/2022-06-30T23:59:59Z',
        'pre_through':'2021-06-21','post_from':'2022-03-02','post_target':'2022-06-15',
        'normal_safe_negative_admissible':False,'source_confirms_human_deposition_not_safe_ground':True}}
    excluded=reserved_envelopes()
    for row in cases.values():
        b=row['bbox_wgs84']
        if any(b[0]<=r[2] and b[2]>=r[0] and b[1]<=r[3] and b[3]>=r[1] for r in excluded.values()):raise ValueError('Protected mine region overlaps new data')
    p={'registered_at':datetime.now().isoformat(timespec='seconds'),'cases':cases,'source_manifest_sha256':file_sha256(OUT/'source_review/acquisition.json'),
       'scene_choice_rule':'fullAOI, scenecloud<20%; latest eligible before source-defined pre cutoff; post closest to source target among allowed dates; no model scores',
       'all_Gamsberg_dates_and_pits_share_role':'development','new_true_accident_mine_groups':1,
       'current_pixel_masks_accepted':False,'new_expert_project_feedback':0,'source_displacement_or_plan_colours_not_model_inputs':True,
       'unknown_or_risky_operational_changes_not_forced_normal':True,'reserved_regions_excluded':excluded}
    if path.exists():
        old=json.loads(path.read_text(encoding='utf-8'));p['registered_at']=old['registered_at']
        if old!=p:raise ValueError('Frozen incident data plan changed')
    else:write(path,p)
    for eid in ('ZA_GAMS2020S','ZA_GAMS2023W'):
        row=cases[eid]
        entry={'mine_id':row['mine_group'],'event_date':row['event_date'],'scope':row['scope'],'role':'development',
          'all_pits_dates_same_mine_group':True,'quality_review_status':'pending_original_and_reference_review',
          'event_source':row['event_source'],'data_plan':str(path),'model_predictions_run':0,
          'original_expert_GIS_received':False,'independent_expert_or_field_audit':False,
          'pixel_reference_accepted':False,'not_new_blind_test':True}
        if eid in value['cases']:
            if value['cases'][eid]['role']!='development' or value['cases'][eid]['mine_id']!='ZA_GAMSBERG':raise ValueError('Mine role changed')
        else:value['cases'][eid]=entry
    write(REGISTRY,value);print('Registered one new accident mine group, two source-confirmed incident dates; no pixel labels or predictions yet')


if __name__=='__main__':sys.stdout.reconfigure(encoding='utf-8');run()
