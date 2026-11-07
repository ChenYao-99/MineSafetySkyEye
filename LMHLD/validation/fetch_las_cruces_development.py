# -*- coding: utf-8 -*-
"""Fetch small original windows for an explicitly registered development mine."""
import hashlib
import argparse
import json
from pathlib import Path
import sys
import urllib.parse
import requests
import numpy as np
import rasterio
from rasterio.warp import transform_bounds, reproject, Resampling
from rasterio.windows import from_bounds
from rasterio.transform import Affine, array_bounds
from mine_data_protocol import DATA, assert_development_cases


def fetch(case_id='LasCruces'):
    settings={'LasCruces':('las_cruces_context.json','EPSG:32629'),'Hpakant':('hpakant_context.json','EPSG:32647')}
    if case_id not in settings: raise ValueError('Only explicitly registered historical development mines allowed')
    assert_development_cases([case_id])
    context_name,crs=settings[case_id]
    context=json.loads((DATA/'downloads'/context_name).read_text(encoding='utf-8'))
    catalogue=json.loads((DATA/f'scene_catalogues/{case_id}_planetarycomputer.json').read_text(encoding='utf-8-sig'))
    bbox=transform_bounds('EPSG:4326',crs,*context['bbox_wgs84'])
    left,bottom=np.floor(np.array(bbox[:2])/10)*10
    right,top=np.ceil(np.array(bbox[2:])/10)*10
    width,height=int(round((right-left)/10)),int(round((top-bottom)/10))
    if width>800 or height>800:
        raise ValueError('Refuse unexpectedly large downloads')
    transform=Affine(10,0,left,0,-10,top)
    def coverage(feature):
        asset=feature['assets']['B02']
        bounds=array_bounds(*asset.get('proj:shape',feature['properties'].get('proj:shape')),
                            Affine(*asset.get('proj:transform',feature['properties'].get('proj:transform'))[:6]))
        wanted=transform_bounds(crs,f"EPSG:{feature['properties']['proj:epsg']}",left,bottom,right,top)
        intersection=max(0,min(bounds[2],wanted[2])-max(bounds[0],wanted[0]))*max(0,min(bounds[3],wanted[3])-max(bounds[1],wanted[1]))
        return intersection/((wanted[2]-wanted[0])*(wanted[3]-wanted[1]))
    output=DATA/f'development/{case_id}'; output.mkdir(parents=True,exist_ok=True)
    record_path=output/'acquisition.json'
    record=json.loads(record_path.read_text(encoding='utf-8')) if record_path.exists() else {
        'case_id':case_id,'role':'development','event_date':context['event_date'],
        'source':'Microsoft Planetary Computer / Copernicus Sentinel-2 L2A','channel_order':'B,G,R,NIR',
        'model_training_performed':False,'native_model_predictions_run':0,'images':{}}
    session=requests.Session()
    for when in ('pre','post'):
        chosen=[f for f in catalogue['features'] if f['properties']['datetime'][:10]==context[when+'_date'] and coverage(f)>=.999]
        if not chosen:
            raise ValueError('No full scene on the preregistered date; do not silently switch dates')
        scene=sorted(chosen,key=lambda f:(f['properties'].get('eo:cloud_cover',100),f['id']))[0]
        target=output/f'{when}_BGRN_10m.tif'; sclpath=output/f'{when}_SCL_10m.tif'
        if when in record['images']:
            old=record['images'][when]
            if hashlib.sha256(target.read_bytes()).hexdigest()!=old['sha256'] or hashlib.sha256(sclpath.read_bytes()).hexdigest()!=old['scl_sha256']:
                raise ValueError('Existing original data changed')
            continue
        channels=[]; band_meta=[]; scl=None
        for band in ('B02','B03','B04','B08','SCL'):
            asset=scene['assets'][band]; href=asset['href']
            endpoint='https://planetarycomputer.microsoft.com/api/sas/v1/sign?href='+urllib.parse.quote(href,safe='')
            response=session.get(endpoint,timeout=30); response.raise_for_status()
            signed=response.json()['href']
            with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN='EMPTY_DIR',GDAL_HTTP_VERSION='1.1',
                              GDAL_HTTP_MAX_RETRY='2',GDAL_HTTP_RETRY_DELAY='1',GDAL_HTTP_TIMEOUT='30',
                              CPL_VSIL_CURL_ALLOWED_EXTENSIONS='.tif'):
                with rasterio.open(signed) as src:
                    bounds=transform_bounds(crs,src.crs,left,bottom,right,top)
                    window=from_bounds(*bounds,transform=src.transform).round_offsets().round_lengths()
                    raw=src.read(1,window=window,boundless=True,fill_value=0)
                    dst=np.zeros((height,width),np.uint8 if band=='SCL' else np.uint16)
                    reproject(raw,dst,src_transform=src.window_transform(window),src_crs=src.crs,
                              dst_transform=transform,dst_crs=crs,src_nodata=0,dst_nodata=0,resampling=Resampling.nearest)
            if band=='SCL': scl=dst
            else: channels.append(dst)
            band_meta.append({'band':band,'unsigned_asset':href,'radiometry':asset.get('raster:bands'),
                              'native_crs':f"EPSG:{scene['properties']['proj:epsg']}",
                              'native_resolution':asset.get('gsd',20 if band=='SCL' else 10)})
            print(f'{when} {band}: native ROI {dst.shape}',flush=True)
        profile=dict(driver='GTiff',height=height,width=width,count=4,dtype='uint16',crs=crs,
                     transform=transform,nodata=0,compress='deflate')
        with rasterio.open(target,'w',**profile) as dst:
            dst.write(np.stack(channels))
            for index,band in enumerate(('B02','B03','B04','B08'),1): dst.set_band_description(index,band)
            dst.update_tags(scene_id=scene['id'],acquisition_datetime=scene['properties']['datetime'],
                            role='development',channel_order='B,G,R,NIR')
        profile.update(count=1,dtype='uint8')
        with rasterio.open(sclpath,'w',**profile) as dst: dst.write(scl,1)
        record['images'][when]={'path':str(target),'scene_id':scene['id'],'datetime':scene['properties']['datetime'],
            'tile_cloud_percentage':scene['properties'].get('eo:cloud_cover'),'bands':band_meta,
            'crs':crs,'transform':list(transform)[:6],'shape':[4,height,width],
            'sha256':hashlib.sha256(target.read_bytes()).hexdigest(),'scl_path':str(sclpath),
            'scl_sha256':hashlib.sha256(sclpath.read_bytes()).hexdigest(),
            'scl_classes':{str(int(v)):int(n) for v,n in zip(*np.unique(scl,return_counts=True))},
            'nodata_pixels':int((np.stack(channels)==0).any(0).sum())}
        record_path.write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print('Original development pair acquired; labels require independent source registration',flush=True)


if __name__=='__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    parser=argparse.ArgumentParser(); parser.add_argument('--case-id',choices=('LasCruces','Hpakant'),default='LasCruces')
    fetch(parser.parse_args().case_id)
