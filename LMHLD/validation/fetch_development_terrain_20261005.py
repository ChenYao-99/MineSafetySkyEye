"""Native public DSM windows for explicitly admitted development sources only."""
from pathlib import Path
from datetime import datetime,timezone,timedelta
import json,sys,urllib.parse
ROOT=Path('D:/LMHLD');sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'validation'))
from runtime_paths import configure_runtime_paths
configure_runtime_paths()
import requests,numpy as np,rasterio
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from rasterio.windows import from_bounds,Window
from rasterio.warp import transform_bounds,reproject,Resampling
from rasterio.transform import Affine
from model_registry import file_sha256
from mine_data_protocol import verify_assets
PLAN=ROOT/'experiments/cause_terrain_round_20261005/round_plan.json'
OUT=ROOT/'datasets/historical_terrain_development_20261005';OUT.mkdir(exist_ok=True)

def load(p):return json.loads(p.read_text(encoding='utf-8'))
def write(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def run():
    verify_assets();plan=load(PLAN);session=requests.Session();ledger=OUT/'acquisition.json'
    session.mount('https://',HTTPAdapter(max_retries=Retry(total=3,backoff_factor=2,
        status_forcelist=(429,500,502,503,504),allowed_methods=frozenset(('GET',)),respect_retry_after_header=True)))
    record=load(ledger) if ledger.exists() else {'round_plan_sha256':file_sha256(PLAN),'cases':{},'reserved_inputs':0,'model_predictions':0}
    if record['round_plan_sha256']!=file_sha256(PLAN):raise ValueError('Terrain source plan changed')
    for case,folder in plan['all_sources'].items():
        folder=Path(folder);source_image=folder/'post_BGRN_10m.tif'
        with rasterio.open(source_image) as src:
            crs=src.crs;left,bottom,right,top=src.bounds;optical_grid={'crs':str(crs),'transform':list(src.transform)[:6],'shape':[src.height,src.width]}
        # Margin supports derivative statistics; no undeclared broad-world download.
        bounds=(left-300,bottom-300,right+300,top+300)
        bbox=transform_bounds(crs,'EPSG:4326',*bounds)
        dest=OUT/case;dest.mkdir(exist_ok=True)
        if case in record['cases']:
            row=record['cases'][case]
            for name,digest in row['assets_sha256'].items():assert file_sha256(dest/name)==digest
            continue
        cat=dest/'catalogue.json'
        if not cat.exists():
            response=session.get('https://planetarycomputer.microsoft.com/api/stac/v1/search',params={
                'collections':'cop-dem-glo-30','bbox':','.join(map(str,bbox)),'limit':100},timeout=(10,35))
            response.raise_for_status();value=response.json()
            if any(l.get('rel')=='next' for l in value.get('links',[])):raise ValueError('Terrain catalogue needs explicit pagination')
            write(cat,value)
        scenes=sorted(load(cat)['features'],key=lambda x:x['id'])
        if not scenes:raise ValueError('No public historical DSM found')
        dst_left,dst_bottom=np.floor(np.asarray(bounds[:2])/30)*30
        dst_right,dst_top=np.ceil(np.asarray(bounds[2:])/30)*30
        width,height=int(round((dst_right-dst_left)/30)),int(round((dst_top-dst_bottom)/30))
        if max(width,height)>800:raise ValueError('Unexpected terrain crop')
        target=Affine(30,0,dst_left,0,-30,dst_top);result=np.full((height,width),np.nan,np.float32);native=[]
        for i,scene in enumerate(scenes):
            assets=scene['assets'];asset=assets.get('data')
            if not asset:raise ValueError('Original DSM asset is missing')
            href=asset['href'];response=session.get('https://planetarycomputer.microsoft.com/api/sas/v1/sign',params={'href':href},timeout=(10,25))
            response.raise_for_status();signed=response.json()['href']
            with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN='EMPTY_DIR',GDAL_HTTP_VERSION='1.1',GDAL_HTTP_TIMEOUT='30',GDAL_HTTP_MAX_RETRY='2',CPL_VSIL_CURL_ALLOWED_EXTENSIONS='.tif'):
                with rasterio.open(signed) as src:
                    requested=from_bounds(*transform_bounds(crs,src.crs,*bounds),transform=src.transform)
                    x0=max(0,int(np.floor(requested.col_off))-2);y0=max(0,int(np.floor(requested.row_off))-2)
                    x1=min(src.width,int(np.ceil(requested.col_off+requested.width))+2);y1=min(src.height,int(np.ceil(requested.row_off+requested.height))+2)
                    if x1<=x0 or y1<=y0:continue
                    win=Window(x0,y0,x1-x0,y1-y0);raw=src.read(1,window=win);native_transform=src.window_transform(win)
                    name=f'original_native_DSM_{i:02d}.tif';profile=src.profile.copy();profile.update(width=raw.shape[1],height=raw.shape[0],transform=native_transform,compress='deflate')
                    with rasterio.open(dest/name,'w',**profile) as out:out.write(raw,1);out.update_tags(source_asset=href,source_native_spacing='approximately30m,geographic angular grid',not_current_mine_survey='true')
                    observation=src.read_masks(1,window=win)>0;values=raw.astype(np.float32);values[~observation]=np.nan
                    part=np.full((height,width),np.nan,np.float32)
                    reproject(values,part,src_transform=native_transform,src_crs=src.crs,dst_transform=target,dst_crs=crs,
                              src_nodata=np.nan,dst_nodata=np.nan,resampling=Resampling.bilinear)
                    keep=np.isfinite(part)&~np.isfinite(result);result[keep]=part[keep]
                    native.append({'scene_id':scene['id'],'asset_url':href,'native_original_file':name,
                                   'native_window_shape':list(raw.shape),'native_transform':list(native_transform)[:6],
                                   'native_crs':str(src.crs),'native_source_resolution':list(src.res),
                                   'catalogue_temporal_metadata':{k:scene.get('properties',{}).get(k) for k in ('datetime','start_datetime','end_datetime')},
                                   'native_samples_unchanged':True})
        if not native or not np.isfinite(result).any():raise ValueError('No usable original terrain crop')
        path=dest/'historical_DSM_projected_30m.tif'
        with rasterio.open(path,'w',driver='GTiff',width=width,height=height,count=1,dtype='float32',crs=crs,transform=target,nodata=np.nan,compress='deflate') as out:
            out.write(result,1);out.update_tags(source='CopernicusGLO30,historical2011-2015 DSM including vegetation and infrastructure',analysis_resolution_m='30',current_mine_topography_verified='false')
        row={'source_optical_image_sha256':file_sha256(source_image),'source_optical_grid':optical_grid,'catalogue_sha256':file_sha256(cat),
             'sources':native,'projected_30m_shape':[height,width],'projected_visible_fraction':float(np.isfinite(result).mean()),
             'not_current_ground_survey_or_deformation':True,'not10m_native_terrain':True,
             'assets_sha256':{p.name:file_sha256(p) for p in dest.iterdir() if p.is_file() and p.suffix=='.tif'}}
        record['cases'][case]=row;write(ledger,record)
        print(case,'DSM30m',result.shape,'originaltiles',len(native),'coverage',round(row['projected_visible_fraction'],4),'no_model_predictions',flush=True)
    record['completed_at']=datetime.now(timezone(timedelta(hours=8))).isoformat();write(ledger,record)

if __name__=='__main__':run()
