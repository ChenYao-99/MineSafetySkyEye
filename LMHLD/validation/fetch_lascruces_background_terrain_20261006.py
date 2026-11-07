"""One fixed larger terrain window to support SAR background controls."""
from pathlib import Path
from datetime import datetime,timezone,timedelta
import sys,json,os
ROOT=Path('D:/LMHLD');sys.path.insert(0,str(ROOT))
from runtime_paths import configure_runtime_paths
configure_runtime_paths()
os.environ['PROJ_USER_WRITABLE_DIRECTORY']=str(ROOT/'cache/runtime/proj');os.environ['PROJ_NETWORK']='OFF'
import requests,numpy as np,rasterio
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from rasterio.windows import from_bounds,Window
from rasterio.warp import transform_bounds,reproject,Resampling
from rasterio.transform import Affine
from model_registry import file_sha256
DATA=ROOT/'datasets/sar_development_pairs_20261005'
OUT=DATA/'LasCruces/terrain_background_v2'

def load(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def write(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')

def run():
    OUT.mkdir(exist_ok=True)
    optical=Path(load(ROOT/'experiments/cause_terrain_round_20261005/round_plan.json')['all_sources']['LasCruces'])/'post_BGRN_10m.tif'
    with rasterio.open(optical) as src:crs=src.crs;left,bottom,right,top=src.bounds
    bounds=[left-900,bottom-900,right+900,top+900]
    protocol={'registered_at':datetime.now(timezone(timedelta(hours=8))).isoformat(),'code_sha256':file_sha256(__file__),
       'case':'LasCruces','margin_outside_original_optical_AOI_m':900,'bounds_projected':bounds,'projected_CRS':str(crs),
       'source_optical_sha256':file_sha256(optical),'purpose':'more groundbackground correspondences;original alignment gates unchanged',
       'new_groundtruth_or_currentpit_height_or_new_SAR_dates':False,'source_GRD_window_not_expanded':True,
       'max_native_window_dimension':800,'historical_DSM_dates':'2011-2015;not current mine survey','all_storage_D':True}
    pp=OUT/'acquisition_protocol.json'
    if pp.exists():assert all(load(pp).get(k)==v for k,v in protocol.items() if k!='registered_at')
    else:write(pp,protocol)
    receipt=OUT/'acquisition.json'
    if receipt.exists():
        record=load(receipt)
        for name,digest in record['assets_sha256'].items():assert file_sha256(OUT/name)==digest
        print('Expanded terrain already verified;no new source download',flush=True);return
    session=requests.Session();session.mount('https://',HTTPAdapter(max_retries=Retry(total=3,backoff_factor=1,status_forcelist=(429,500,502,503,504))))
    cat=OUT/'catalogue.json'
    if not cat.exists():
        response=session.get('https://planetarycomputer.microsoft.com/api/stac/v1/search',params={'collections':'cop-dem-glo-30','bbox':','.join(map(str,transform_bounds(crs,'EPSG:4326',*bounds))),'limit':100},timeout=(10,35))
        response.raise_for_status();value=response.json()
        if any(x.get('rel')=='next' for x in value.get('links',[])):raise ValueError('Unregistered catalogue pagination')
        write(cat,value)
    tiles=sorted(load(cat)['features'],key=lambda s:s['id'])
    if not tiles:raise ValueError('No source historical DSM')
    dl,db=np.floor(np.asarray(bounds[:2])/30)*30;dr,dt=np.ceil(np.asarray(bounds[2:])/30)*30
    w,h=int(round((dr-dl)/30)),int(round((dt-db)/30));transform=Affine(30,0,dl,0,-30,dt)
    if max(w,h)>800:raise ValueError('Declared DSM footprint budget exceeded')
    terrain=np.full((h,w),np.nan,np.float32);sources=[]
    for i,scene in enumerate(tiles):
        url=scene['assets']['data']['href'];r=session.get('https://planetarycomputer.microsoft.com/api/sas/v1/sign',params={'href':url},timeout=(10,25));r.raise_for_status()
        with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN='EMPTY_DIR',GDAL_HTTP_TIMEOUT='30',GDAL_HTTP_MAX_RETRY='2',CPL_VSIL_CURL_ALLOWED_EXTENSIONS='.tif'):
            with rasterio.open(r.json()['href']) as src:
                requested=from_bounds(*transform_bounds(crs,src.crs,*bounds),transform=src.transform)
                x0=max(0,int(np.floor(requested.col_off))-2);y0=max(0,int(np.floor(requested.row_off))-2)
                x1=min(src.width,int(np.ceil(requested.col_off+requested.width))+2);y1=min(src.height,int(np.ceil(requested.row_off+requested.height))+2)
                if x1<=x0 or y1<=y0:continue
                window=Window(x0,y0,x1-x0,y1-y0)
                if max(window.width,window.height)>800:raise ValueError('Native DSM crop exceeded budget')
                raw=src.read(1,window=window);t=src.window_transform(window);profile=src.profile.copy();profile.update(width=raw.shape[1],height=raw.shape[0],transform=t,compress='deflate')
                name=f'original_native_DSM_{i:02d}.tif'
                with rasterio.open(OUT/name,'w',**profile) as dst:dst.write(raw,1);dst.update_tags(source_asset=url,source_samples_unchanged='true',not_currentpit_height='true')
                values=raw.astype(np.float32);values[src.read_masks(1,window=window)==0]=np.nan
                part=np.full(terrain.shape,np.nan,np.float32);reproject(values,part,src_transform=t,src_crs=src.crs,dst_transform=transform,dst_crs=crs,src_nodata=np.nan,dst_nodata=np.nan,resampling=Resampling.bilinear)
                good=np.isfinite(part)&~np.isfinite(terrain);terrain[good]=part[good]
                sources.append({'scene':scene['id'],'unsigned_original_url':url,'native_samples_file':name,'shape':list(raw.shape),'native_spacing':list(src.res)})
    if not sources or np.isfinite(terrain).mean()<.99:raise ValueError('Insufficient expanded terrain coverage')
    profile={'driver':'GTiff','width':w,'height':h,'count':1,'dtype':'float32','crs':crs,'transform':transform,'nodata':-9999,'compress':'deflate'}
    orthometric=terrain.copy();orthometric[~np.isfinite(orthometric)]=-9999
    with rasterio.open(OUT/'historical_DSM_projected_30m_EGM2008.tif','w',**profile) as dst:dst.write(orthometric,1);dst.update_tags(historical_2011_2015='true',height_reference='EGM2008 orthometric')
    geoid=DATA/'LasCruces/original_EGM2008_geoid_native_window.tif'
    offset=np.full(terrain.shape,np.nan,np.float32)
    with rasterio.open(geoid) as src:reproject(src.read(1),offset,src_transform=src.transform,src_crs=src.crs,dst_transform=transform,dst_crs=crs,resampling=Resampling.bilinear)
    valid=np.isfinite(terrain)
    if not np.isfinite(offset[valid]).all():raise ValueError('Original geoid lacks expanded support')
    ellipsoid=terrain+offset;ellipsoid[~valid]=-9999
    with rasterio.open(OUT/'historical_DSM_30m_WGS84_ellipsoid.tif','w',**profile) as dst:dst.write(ellipsoid,1);dst.update_tags(height_reference='WGS84 ellipsoid;H_EGM2008+N_original',historical_2011_2015_not_currentpit='true',geoid_sha256=file_sha256(geoid))
    write(receipt,{'protocol_sha256':file_sha256(pp),'sources':sources,'projected_shape':[h,w],'terrain_coverage':float(valid.mean()),
       'original_geoid_sha256':file_sha256(geoid),'catalogue_sha256':file_sha256(cat),'new_terrain_observation_or_model_admission':False,
       'assets_sha256':{p.name:file_sha256(p) for p in OUT.glob('*.tif')}})
    print('LasCruces fixed900m background margin',h,w,'historic terrain;original GRD sources unchanged',flush=True)

if __name__=='__main__':run()
