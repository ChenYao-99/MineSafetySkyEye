"""Acquire a new pit-wall event candidate without any model prediction or acceptance claim."""
import sys,json,hashlib,urllib.parse
from pathlib import Path
from datetime import datetime
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import numpy as np
import rasterio
from rasterio.transform import Affine,array_bounds
from rasterio.warp import transform_bounds,reproject,Resampling
from rasterio.windows import from_bounds
from mine_data_protocol import DATA,REGISTRY,registry,verify_assets,sha

ID='US_BC2021'
OUT=DATA/'reserved_candidates'/ID
CONTEXT={'case_id':ID,'mine_id':'BinghamCanyon','event_date':'2021-05-31',
         'scope':'open_pit_wall_failure','wall':'south east','role':'reserved_final_test',
         'bbox_wgs84':[-112.185,40.488,-112.120,40.548],'crs':'EPSG:32612',
         'event_primary_source':'https://www.riotinto.com/en/news/releases/2021/rio-tinto-releases-second-quarter-production-results',
         'operator_original_report':'https://www.riotinto.com/-/media/content/documents/invest/financial-news-and-performance/production/2021/rt-second-quarter-operations-review-2021-pdf.pdf?rev=a273c45050f34ee09ef925147b1dcc5d',
         'boundary_available':False,'independent_expert_or_field_audit':False,
         'event_time_not_verified':True,'scene_selection':'strict prior/next calendar day, >=99.9% AOI coverage, scene-cloud<20%; nearest date, ID tie-break; no model-based selection',
         'final_acceptance_requires':['historical same-mine exposure audit','independent accident boundary','local visibility/date/grid and radiometry review']}


def write(path,value):path.write_bytes((json.dumps(value,ensure_ascii=False,indent=2)+'\n').encode('utf-8'))


def run():
    verify_assets();value=registry();OUT.mkdir(parents=True,exist_ok=True)
    context=OUT/'context.json'
    if context.exists() and json.loads(context.read_text(encoding='utf-8'))!=CONTEXT:raise ValueError('Candidate context changed')
    write(context,CONTEXT)
    if ID not in value['cases']:
        value['cases'][ID]={**CONTEXT,'quality_review_status':'pending','use_all_times_with_same_mine_role':True,
                          'model_predictions_run':0,'raw_source':'Copernicus S2L2A / anonymous public Planetary Computer',
                          'context':str(context.relative_to(DATA)).replace('\\','/'),'geometric_gate_pass':False,
                          'independent_expert_or_field_audit':False,'new_unseen_main_event_accepted':False,
                          'unresolved_quality':CONTEXT['final_acceptance_requires']}
        write(REGISTRY,value)
    if value['cases'][ID]['role']!='reserved_final_test' or value['cases'][ID]['model_predictions_run']!=0:
        raise ValueError('Candidate reserved role/prediction count changed')
    session=requests.Session()
    retry=Retry(total=2,backoff_factor=.8,status_forcelist=(429,500,502,503,504),allowed_methods=frozenset(('GET',)))
    session.mount('https://',HTTPAdapter(max_retries=retry))
    source=OUT/'operator_event_confirmation.json'
    if not source.exists():
        confirmation={'url':CONTEXT['event_primary_source'],'source_type':'mine operator original 2021Q2 release',
                      'verified_via_web_browser':True,'verified_at':'2026-10-04',
                      'short_excerpt':'On 31 May, an anticipated slope failure occurred in the south east wall of the Bingham Canyon pit at Kennecott.',
                      'event_date':'2021-05-31','not_a_boundary_or_geolocation_reference':True,
                      'original_PDF_download_status':'HTTP403; no local PDF claimed'}
        write(source,confirmation)
    catalogue=OUT/'catalogue.json'
    if not catalogue.exists():
        response=session.get('https://planetarycomputer.microsoft.com/api/stac/v1/search',params={
            'collections':'sentinel-2-l2a','bbox':','.join(map(str,CONTEXT['bbox_wgs84'])),
            'datetime':'2021-05-15T00:00:00Z/2021-06-15T23:59:59Z','limit':100},timeout=(15,30))
        response.raise_for_status();catalogue_data=response.json()
        if any(l.get('rel')=='next' for l in catalogue_data.get('links',[])):raise ValueError('Catalogue pagination requires review')
        write(catalogue,catalogue_data)
    features=json.loads(catalogue.read_text(encoding='utf-8'))['features']
    bounds=transform_bounds('EPSG:4326',CONTEXT['crs'],*CONTEXT['bbox_wgs84'])
    left,bottom=np.floor(np.asarray(bounds[:2])/10)*10;right,top=np.ceil(np.asarray(bounds[2:])/10)*10
    width,height=int(round((right-left)/10)),int(round((top-bottom)/10));affine=Affine(10,0,left,0,-10,top)
    if max(width,height)>800:raise ValueError('Unexpected native request size')
    def coverage(scene):
        asset=scene['assets']['B02'];props=scene['properties']
        b=array_bounds(*asset.get('proj:shape',props.get('proj:shape')),Affine(*asset.get('proj:transform',props.get('proj:transform'))[:6]))
        w=transform_bounds(CONTEXT['crs'],f"EPSG:{props['proj:epsg']}",left,bottom,right,top)
        return max(0,min(b[2],w[2])-max(b[0],w[0]))*max(0,min(b[3],w[3])-max(b[1],w[1]))/((w[2]-w[0])*(w[3]-w[1]))
    full=[s for s in features if coverage(s)>=.999 and s['properties'].get('eo:cloud_cover',100)<20]
    pre=sorted([s for s in full if s['properties']['datetime'][:10]<CONTEXT['event_date']],key=lambda s:(s['properties']['datetime'],s['id']),reverse=True)
    post=sorted([s for s in full if s['properties']['datetime'][:10]>CONTEXT['event_date']],key=lambda s:(s['properties']['datetime'],s['id']))
    if not pre or not post:raise ValueError('No strict complete pre/post candidate; keep pending')
    plan={'context_sha':sha(context),'catalogue_sha':sha(catalogue),'operator_confirmation_sha':sha(source),
          'scenes':{'pre':pre[0]['id'],'post':post[0]['id']},'role':'reserved_final_test','model_predictions':0}
    planpath=OUT/'acquisition_plan.json'
    if planpath.exists() and json.loads(planpath.read_text(encoding='utf-8'))!=plan:raise ValueError('Frozen acquisition plan changed')
    write(planpath,plan)
    path=OUT/'acquisition.json';record=json.loads(path.read_text(encoding='utf-8')) if path.exists() else {'case_id':ID,'images':{},'model_predictions':0,'role':'reserved_final_test'}
    for period,scene in (('pre',pre[0]),('post',post[0])):
        output=OUT/f'{period}_BGRN_10m.tif';sclpath=OUT/f'{period}_SCL_10m.tif'
        if period in record['images']:
            item=record['images'][period]
            if sha(output)!=item['sha256'] or sha(sclpath)!=item['scl_sha256']:raise ValueError('Candidate raw changed')
            continue
        bands=[];assets=[];scl=None
        for band in ('B02','B03','B04','B08','SCL'):
            href=scene['assets'][band]['href']
            response=session.get('https://planetarycomputer.microsoft.com/api/sas/v1/sign?href='+urllib.parse.quote(href,safe=''),timeout=(10,30));response.raise_for_status()
            with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN='EMPTY_DIR',GDAL_HTTP_VERSION='1.1',GDAL_HTTP_TIMEOUT='30',GDAL_HTTP_MAX_RETRY='2',CPL_VSIL_CURL_ALLOWED_EXTENSIONS='.tif'):
                with rasterio.open(response.json()['href']) as src:
                    window=from_bounds(*transform_bounds(CONTEXT['crs'],src.crs,left,bottom,right,top),transform=src.transform).round_offsets().round_lengths()
                    raw=src.read(1,window=window,boundless=True,fill_value=0)
                    dest=np.zeros((height,width),np.uint8 if band=='SCL' else np.uint16)
                    reproject(raw,dest,src_transform=src.window_transform(window),src_crs=src.crs,dst_transform=affine,
                              dst_crs=CONTEXT['crs'],src_nodata=0,dst_nodata=0,resampling=Resampling.nearest)
            if band=='SCL':scl=dest
            else:bands.append(dest)
            assets.append({'band':band,'unsigned_asset':href,'source_native_resolution_m':20 if band=='SCL' else 10})
            print(f'{period} {band}: {dest.shape}, prediction0',flush=True)
        profile={'driver':'GTiff','width':width,'height':height,'count':4,'dtype':'uint16','crs':CONTEXT['crs'],'transform':affine,'nodata':0,'compress':'deflate'}
        with rasterio.open(output,'w',**profile) as f:
            f.write(np.stack(bands));f.descriptions=('B02','B03','B04','B08')
            f.update_tags(acquisition_datetime=scene['properties']['datetime'],case_id=ID,role='reserved_final_test',radiometry_status='product_XML_review_required')
        profile.update(count=1,dtype='uint8')
        with rasterio.open(sclpath,'w',**profile) as f:f.write(scl,1);f.set_band_description(1,'SCL')
        record['images'][period]={'scene_id':scene['id'],'datetime':scene['properties']['datetime'],'shape':[height,width],
             'file':output.name,'sha256':sha(output),'scl_file':sclpath.name,'scl_sha256':sha(sclpath),
             'SCL_obscured_fraction':float(np.isin(scl,[0,1,3,8,9,10,11]).mean()),'source_assets':assets,'accepted_for_final_evaluation':False}
        write(path,record)
    value=registry()
    for item in record['images'].values():
        for field in ('file','scl_file'):
            raw=OUT/item[field];value.setdefault('additional_frozen_assets',{})[raw.relative_to(DATA).as_posix()]=sha(raw)
    value['cases'][ID].update(raw_pair_acquired=True,acquisition=str(path.relative_to(DATA)).replace('\\','/'))
    write(REGISTRY,value);verify_assets()
    print(json.dumps({'case':ID,'pair_acquired':True,'model_predictions':0,'boundary_available':False,'accepted_for_final_evaluation':False}))


if __name__=='__main__':sys.stdout.reconfigure(encoding='utf-8');run()
