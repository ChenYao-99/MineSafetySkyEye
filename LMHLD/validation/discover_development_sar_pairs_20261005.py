"""Chronology/orbit-matched public GRD metadata, before radar/model selection."""
from pathlib import Path
from datetime import datetime,timedelta,timezone
import json,sys
ROOT=Path('D:/LMHLD');sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'validation'));sys.path.insert(0,str(ROOT/'cache/research_dependencies'))
from runtime_paths import configure_runtime_paths
configure_runtime_paths()
import requests,rasterio
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from rasterio.warp import transform_bounds
from shapely.geometry import shape,box
from model_registry import file_sha256
from mine_data_protocol import registry,verify_assets,reserved_case_ids
OUT=ROOT/'datasets/sar_development_pairs_20261005';CASES=['Amyntaio','LasCruces','Hpakant','CN_XJ2023']

def load(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def write(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')

def run():
    OUT.mkdir(exist_ok=True);verify_assets();inventory=registry();assert not set(CASES)&set(reserved_case_ids(inventory))
    optical=load(ROOT/'experiments/cause_terrain_round_20261005/round_plan.json')['all_sources'];definitions={}
    for case in CASES:
        p=Path(optical[case])/'post_BGRN_10m.tif'
        with rasterio.open(p) as src:bbox=list(transform_bounds(src.crs,'EPSG:4326',*src.bounds))
        definitions[case]={'event_date':inventory['cases'][case]['event_date'],'bbox_wgs84':bbox,'optical_grid_source_sha256':file_sha256(p)}
    proposal={'registered_at':datetime.now(timezone(timedelta(hours=8))).isoformat(),'cases':definitions,
              'collection':'sentinel-1-grd','date_window_days_before_after':30,
              'selection':'fullAOI coverage>=.999;IW;VVandVH;strictly before/after eventcalendar day;sametracksatdirection;minimumcombineddategap;deterministic ties',
              'same_calendar_day_not_assumed_post_event':True,'new_reserved_geography_or_model_outputs_used':False,
              'raw_GRD_detectedamplitude_not_phase_coherence_mm_deformation_or_terraincorrectedSAR':True,
              'required_before_model_use':['radiometriccalibrationmetadata','geometricterraincorrectionandindependentgroundQA','actualsampleunitsandqualityreview'],
              'RTC_requires_existingaccount_not_configured':True,'rawGRD_public_access_must_be_verified':True,
              'metadata_only_no_classifier_or_image_inference':True,'max_metadata_bytes_per_response':30*1024**2,
              'max_radar_window_download_MiB_per_scene':64,'all_storage_drive':'D'}
    path=OUT/'source_plan.json'
    if path.exists():
        old=load(path);assert all(old.get(k)==v for k,v in proposal.items() if k!='registered_at')
    else:write(path,proposal)
    session=requests.Session();session.mount('https://',HTTPAdapter(max_retries=Retry(total=3,backoff_factor=1,status_forcelist=(429,500,502,503,504))))
    selected={};pending={}
    for case,definition in definitions.items():
        folder=OUT/case;folder.mkdir(exist_ok=True);event=datetime.fromisoformat(definition['event_date']);catalogue=folder/'catalogue.json'
        if not catalogue.exists():
            query={'collections':'sentinel-1-grd','bbox':','.join(map(str,definition['bbox_wgs84'])),
                   'datetime':(event-timedelta(days=30)).strftime('%Y-%m-%dT00:00:00Z')+'/'+(event+timedelta(days=30)).strftime('%Y-%m-%dT23:59:59Z'),'limit':100}
            response=session.get('https://planetarycomputer.microsoft.com/api/stac/v1/search',params=query,timeout=(10,40));response.raise_for_status();assert len(response.content)<30*1024**2
            value=response.json()
            if any(l.get('rel')=='next' for l in value.get('links',[])):raise ValueError('Explicit pagination needed for '+case)
            write(catalogue,value)
        features=load(catalogue)['features'];target=box(*definition['bbox_wgs84']);groups={}
        for feature in features:
            p=feature['properties'];coverage=shape(feature['geometry']).intersection(target).area/target.area
            if coverage<.999 or p.get('sar:instrument_mode')!='IW' or not {'vv','vh'}<=set(feature['assets']):continue
            key=(p.get('sat:relative_orbit'),p.get('sat:orbit_state'))
            if None in key:continue
            groups.setdefault(key,[]).append(feature)
        choices=[]
        for key,group in groups.items():
            before=sorted([x for x in group if x['properties']['datetime'][:10]<definition['event_date']],key=lambda x:(x['properties']['datetime'],x['id']),reverse=True)
            after=sorted([x for x in group if x['properties']['datetime'][:10]>definition['event_date']],key=lambda x:(x['properties']['datetime'],x['id']))
            if not before or not after:continue
            a,b=before[0],after[0];days=(datetime.fromisoformat(b['properties']['datetime'].replace('Z','+00:00'))-datetime.fromisoformat(a['properties']['datetime'].replace('Z','+00:00'))).total_seconds()/86400
            choices.append((days,b['properties']['datetime'],a['properties']['datetime'],str(key),a,b,key))
        if not choices:pending[case]={'reason':'no full-cover chronologically matched IW dualpolarization sameorbit pair','training_admitted':False};continue
        _,_,_,_,before,after,key=min(choices,key=lambda x:x[:4])
        entry={'case_id':case,'orbit_relative':key[0],'orbit_direction':key[1],'event_date':definition['event_date'],
               'catalogue_sha256':file_sha256(catalogue),'source_plan_sha256':file_sha256(path),'native_images_downloaded':False,
               'model_input_admitted':False,'scene_ids':{'pre':before['id'],'post':after['id']},
               'dates':{'pre':before['properties']['datetime'],'post':after['properties']['datetime']},
               'original_selected_items':{'pre':before,'post':after},'signedurls_or_accountkeys_stored':False}
        p=folder/'pair_selection.json'
        if p.exists():assert load(p)==entry
        else:write(p,entry)
        selected[case]={'pair_selection':str(p),'sha256':file_sha256(p),'dates':entry['dates'],'orbit':key}
        print(case,'sameorbit',key,'dates',entry['dates'],flush=True)
    write(OUT/'discovery.json',{'selected':selected,'pending':pending,'native_radar_data_or_quality_admission_yet':False,'model_predictions':0})
    print('Selected radar metadata pairs',len(selected),'of4;not yetcalibratedoradmitted',flush=True)

if __name__=='__main__':run()
