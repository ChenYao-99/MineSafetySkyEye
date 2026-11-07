# -*- coding: utf-8 -*-
"""Upgrade historical Xinjing imagery; prior event exposure forbids a blind-test claim."""
import os
from pathlib import Path
import json
import sys
import hashlib
import urllib.parse
from datetime import datetime
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import numpy as np
import rasterio
from rasterio.warp import transform_bounds,reproject,Resampling
from rasterio.windows import from_bounds
from rasterio.transform import Affine,array_bounds
from mine_data_protocol import DATA,REGISTRY,registry,verify_assets

ID='CN_XJ2023'
OUT=DATA/'development/Xinjing'
SOURCE='https://www.mdpi.com/2072-4292/17/1/19'
CONTEXT={'case_id':ID,'event_date':'2023-02-22','scope':'open_pit_wall_failure',
    'bbox_wgs84':[105.62,37.95,105.69,38.00],'crs':'EPSG:32648',
    'location_evidence':'Chang et al2025 Fig1c graticule105.64/105.66/105.68E,37.96/37.98N',
    'source':SOURCE,'author_post_reference_date':'2023-02-24',
    'original_expert_GIS_available':False,'boundary_available':False,
    'selection':'nearest complete scene strictly before/after22Feb, tile-cloud<20%; dates fixed before native visibility review; no model outputs'}


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path,value):path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')


def run():
    verify_assets();OUT.mkdir(parents=True,exist_ok=True)
    ctx=OUT/'context.json'
    if ctx.exists() and json.loads(ctx.read_text(encoding='utf-8'))!=CONTEXT:raise ValueError('Candidate context changed')
    write(ctx,CONTEXT)
    value=registry()
    if ID not in value['cases']:
        historical=Path('D:/竞赛材料/第十八组实验报告_阿拉善露天矿坍塌事件案例验证.md')
        if not historical.exists():raise ValueError('Audit historical Xinjing exposure before registering acquisition')
        value['cases'][ID]={'mine_id':ID,'event_date':CONTEXT['event_date'],'scope':CONTEXT['scope'],
            'role':'development','quality_review_status':'pending','use_all_times_with_same_mine_role':True,
            'context':str(ctx.relative_to(DATA)).replace('\\','/'),'raw_source':'Copernicus S2L2A / anonymous Planetary Computer',
            'label_source':SOURCE,'label_quality':'original expert GIS unavailable; source figure not yet geographically transferred',
            'independent_expert_or_field_audit':False,'model_predictions_run':8,'native_model_predictions_run':0,
            'historical_event_model_exposure':True,'no_new_blind_test_claim':True,
            'historical_exposure_evidence':{'path':str(historical),'sha256':sha(historical),'date':'2026-09-30','observed_model_configurations_at_least':8},
            'unresolved_quality':['pre/post native visibility','geographic accident reference','original expert GIS unavailable'],
            'registered_at':datetime.now().isoformat(timespec='seconds')}
        write(REGISTRY,value)
    if value['cases'][ID]['role']!='development' or not value['cases'][ID].get('historical_event_model_exposure'):raise ValueError('Xinjing must stay historical development')
    session=requests.Session()
    retry=Retry(total=3,backoff_factor=.8,status_forcelist=(429,500,502,503,504),allowed_methods=frozenset(('GET',)))
    session.mount('https://',HTTPAdapter(max_retries=retry))
    catalogue=OUT/'catalogue.json'
    if catalogue.exists():features=json.loads(catalogue.read_text(encoding='utf-8'))['features']
    else:
        r=session.get('https://planetarycomputer.microsoft.com/api/stac/v1/search',params={
            'collections':'sentinel-2-l2a','bbox':','.join(map(str,CONTEXT['bbox_wgs84'])),
            'datetime':'2023-02-01T00:00:00Z/2023-03-10T23:59:59Z','limit':100},timeout=(15,45))
        r.raise_for_status();data=r.json()
        if any(l.get('rel')=='next' for l in data.get('links',[])):raise ValueError('Catalogue pagination requires explicit review')
        write(catalogue,data);features=data['features']
    bounds=transform_bounds('EPSG:4326',CONTEXT['crs'],*CONTEXT['bbox_wgs84'])
    left,bottom=np.floor(np.array(bounds[:2])/10)*10;right,top=np.ceil(np.array(bounds[2:])/10)*10
    width,height=int(round((right-left)/10)),int(round((top-bottom)/10))
    if max(width,height)>800:raise ValueError('Native request unexpectedly large')
    affine=Affine(10,0,left,0,-10,top)
    def coverage(f):
        a=f['assets']['B02'];b=array_bounds(*a.get('proj:shape',f['properties'].get('proj:shape')),Affine(*a.get('proj:transform',f['properties'].get('proj:transform'))[:6]))
        w=transform_bounds(CONTEXT['crs'],f"EPSG:{f['properties']['proj:epsg']}",left,bottom,right,top)
        return max(0,min(b[2],w[2])-max(b[0],w[0]))*max(0,min(b[3],w[3])-max(b[1],w[1]))/((w[2]-w[0])*(w[3]-w[1]))
    full=[f for f in features if coverage(f)>=.999 and f['properties'].get('eo:cloud_cover',100)<20]
    before=sorted([f for f in full if f['properties']['datetime'][:10]<CONTEXT['event_date']],key=lambda f:(f['properties']['datetime'],f['id']),reverse=True)
    after=sorted([f for f in full if f['properties']['datetime'][:10]>CONTEXT['event_date']],key=lambda f:(f['properties']['datetime'],f['id']))
    if not before or not after:raise ValueError('No complete strict pre/post scenes; keep pending without relaxing selection')
    planpath=OUT/'acquisition_plan.json'
    plan={'context_sha':sha(ctx),'catalogue_sha':sha(catalogue),'scenes':{'pre':before[0]['id'],'post':after[0]['id']},'model_predictions':0}
    if planpath.exists() and json.loads(planpath.read_text(encoding='utf-8'))!=plan:raise ValueError('Frozen acquisition scenes changed')
    write(planpath,plan)
    recordpath=OUT/'acquisition.json';record=json.loads(recordpath.read_text(encoding='utf-8')) if recordpath.exists() else {'case_id':ID,'role':'development','historical_event_model_exposure':True,'new_original_pair_predictions':0,'native_model_predictions_run':0,'images':{}}
    for when,scene in (('pre',before[0]),('post',after[0])):
        target=OUT/f'{when}_BGRN_10m.tif';sclpath=OUT/f'{when}_SCL_10m.tif'
        if when in record['images']:
            old=record['images'][when]
            if sha(target)!=old['sha256'] or sha(sclpath)!=old['scl_sha256']:raise ValueError('Candidate originals changed')
            continue
        channels=[];metadata=[];scl=None;scales=[];offsets=[]
        for band in ('B02','B03','B04','B08','SCL'):
            asset=scene['assets'][band];href=asset['href']
            r=session.get('https://planetarycomputer.microsoft.com/api/sas/v1/sign?href='+urllib.parse.quote(href,safe=''),timeout=(15,30));r.raise_for_status()
            with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN='EMPTY_DIR',GDAL_HTTP_VERSION='1.1',GDAL_HTTP_MAX_RETRY='2',GDAL_HTTP_TIMEOUT='30',CPL_VSIL_CURL_ALLOWED_EXTENSIONS='.tif'):
                with rasterio.open(r.json()['href']) as src:
                    w=from_bounds(*transform_bounds(CONTEXT['crs'],src.crs,left,bottom,right,top),transform=src.transform).round_offsets().round_lengths()
                    raw=src.read(1,window=w,boundless=True,fill_value=0);dest=np.zeros((height,width),np.uint8 if band=='SCL' else np.uint16)
                    reproject(raw,dest,src_transform=src.window_transform(w),src_crs=src.crs,dst_transform=affine,dst_crs=CONTEXT['crs'],src_nodata=0,dst_nodata=0,resampling=Resampling.nearest)
                    scale,offset=float(src.scales[0]),float(src.offsets[0])
            if band=='SCL':scl=dest
            else:channels.append(dest);scales.append(scale);offsets.append(offset)
            metadata.append({'band':band,'unsigned_asset':href,'original_scale':scale,'original_offset':offset,'raster_bands':asset.get('raster:bands')})
            print(f'{when} {band} {dest.shape}; no predictions',flush=True)
        profile=dict(driver='GTiff',width=width,height=height,count=4,dtype='uint16',crs=CONTEXT['crs'],transform=affine,nodata=0,compress='deflate')
        with rasterio.open(target,'w',**profile) as dst:
            dst.write(np.stack(channels));dst.scales=tuple(scales);dst.offsets=tuple(offsets)
            for i,b in enumerate(('B02','B03','B04','B08'),1):dst.set_band_description(i,b)
            dst.update_tags(scene_id=scene['id'],acquisition_datetime=scene['properties']['datetime'],role='development',channel_order='B,G,R,NIR')
        profile.update(count=1,dtype='uint8')
        with rasterio.open(sclpath,'w',**profile) as dst:dst.write(scl,1)
        record['images'][when]={'path':str(target),'sha256':sha(target),'scl_path':str(sclpath),'scl_sha256':sha(sclpath),'scene_id':scene['id'],'datetime':scene['properties']['datetime'],
            'tile_cloud_percentage':scene['properties'].get('eo:cloud_cover'),'shape':[4,height,width],'crs':CONTEXT['crs'],'transform':list(affine)[:6],'bands':metadata,
            'SCL_counts':{str(int(v)):int(n) for v,n in zip(*np.unique(scl,return_counts=True))},'optical_keep_fraction':float((~np.isin(scl,[0,1,8,9,10,11])).mean()),
            'quality_is_pending':True,'radiometric_calibration_confirmed':False,
            'calibration_note':'Original COG scale/offset retained. Identity metadata does not proveBOA offset zero; processing-baseline XML still needs audit before physical indices.',
            'reference_calendar_day_match':scene['properties']['datetime'][:10]==CONTEXT['author_post_reference_date'] if when=='post' else False}
        write(recordpath,record)
    value=registry();value['additional_frozen_assets']=value.get('additional_frozen_assets',{})
    for when,r in record['images'].items():
        for key,dk in (('path','sha256'),('scl_path','scl_sha256')):value['additional_frozen_assets'][str(Path(r[key]).relative_to(DATA)).replace('\\','/')]=r[dk]
    value['cases'][ID]['acquisition']=str(recordpath.relative_to(DATA)).replace('\\','/');write(REGISTRY,value)
    print('Historical Xinjing original imagery upgraded onD; new pair predictions0; event already observed, never a new blind test',flush=True)


if __name__=='__main__':sys.stdout.reconfigure(encoding='utf-8');run()
