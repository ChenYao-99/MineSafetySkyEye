# -*- coding: utf-8 -*-
"""Small actualSWIR windows to audit winter observation; not a failure detector."""
from pathlib import Path
import json,sys,hashlib,urllib.parse
import requests
import numpy as np
import rasterio
from rasterio.warp import reproject,Resampling,transform_bounds
from rasterio.windows import from_bounds
from mine_data_protocol import DATA,assert_development_cases,verify_assets

FOLDER=DATA/'development/Xinjing';OUT=DATA/'radiometry_audit_20261004/Xinjing'


def run():
    assert_development_cases(['CN_XJ2023']);verify_assets();cal=json.loads((OUT.parent/'audit.json').read_text(encoding='utf-8'))['cases']['CN_XJ2023']
    acq=json.loads((FOLDER/'acquisition.json').read_text(encoding='utf-8'));cat=json.loads((FOLDER/'catalogue.json').read_text(encoding='utf-8'));rows={}
    for period,record in acq['images'].items():
        scene=next(f for f in cat['features'] if f['id']==record['scene_id']);href=scene['assets']['B11']['href'];dest=OUT/f'{period}_B11_native20m_on10m_grid.tif'
        with rasterio.open(FOLDER/f'{period}_BGRN_10m.tif') as target:profile=target.profile.copy();grid=target.transform;crs=target.crs;bounds=target.bounds;shape=(target.height,target.width);green=target.read(2).astype(np.float32);mask=target.dataset_mask()>0
        if not dest.exists():
            response=requests.get('https://planetarycomputer.microsoft.com/api/sas/v1/sign?href='+urllib.parse.quote(href,safe=''),timeout=(15,35));response.raise_for_status()
            with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN='EMPTY_DIR',GDAL_HTTP_TIMEOUT='35',GDAL_HTTP_MAX_RETRY='2',CPL_VSIL_CURL_ALLOWED_EXTENSIONS='.tif'):
                with rasterio.open(response.json()['href']) as source:
                    window=from_bounds(*transform_bounds(crs,source.crs,*bounds),transform=source.transform).round_offsets().round_lengths();raw=source.read(1,window=window,boundless=True,fill_value=0);swir=np.zeros(shape,np.uint16)
                    reproject(raw,swir,src_transform=source.window_transform(window),src_crs=source.crs,dst_transform=grid,dst_crs=crs,src_nodata=0,dst_nodata=0,resampling=Resampling.nearest)
            profile.update(count=1,dtype='uint16',compress='deflate')
            with rasterio.open(dest,'w',**profile) as f:f.write(swir,1);f.set_band_description(1,'B11');f.update_tags(native_resolution_m=20,grid_resolution_m=10,upsampling_adds_no_detail=True,unsigned_source=href)
        with rasterio.open(dest) as f:swir=f.read(1).astype(np.float32)
        c=cal[period];q=c['BOA_quantification'];offset=c['DN_offsets'][0];g=(green+offset)/q;s=(swir+offset)/q
        valid=mask&(swir>0);ndsi=(g-s)/np.maximum(g+s,1e-6);candidate=valid&(ndsi>.4)&(g>.1)
        profile.update(count=1,dtype='uint8',nodata=None,compress='deflate')
        with rasterio.open(OUT/f'{period}_snow_bright_obscuration_candidate.tif','w',**profile) as f:f.write(candidate.astype(np.uint8),1)
        rows[period]={'source_asset':href,'SWIR_sha':hashlib.sha256(dest.read_bytes()).hexdigest(),'native_SWIR_resolution_m':20,'analysis_grid_m':10,
            'native_detail_not_increased':True,'candidate_pixels':int(candidate.sum()),'valid_pixels':int(valid.sum()),'candidate_fraction':float(candidate[valid].mean()),
            'rule':'project conservative observation flag: calibratedNDSI>.4 and green>.1, NOT expert snow truth or landslide label',
            'model_predictions_used':False,'no_old_labels_changed':True}
        print(period,'snow/bright-obscuration candidate fraction',rows[period]['candidate_fraction'],flush=True)
    (OUT/'winter_observation_audit.json').write_bytes((json.dumps(rows,ensure_ascii=False,indent=2)+'\n').encode('utf-8'));verify_assets()


if __name__=='__main__':sys.stdout.reconfigure(encoding='utf-8');run()
