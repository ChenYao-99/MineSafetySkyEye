"""Fixed local observation screening, retaining actual catalogue cloud values."""
from pathlib import Path
from datetime import datetime, timezone, timedelta
import json, sys, urllib.parse
ROOT=Path('D:/LMHLD');sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'validation'))
from runtime_paths import configure_runtime_paths
configure_runtime_paths()
import requests, numpy as np, rasterio
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from rasterio.warp import transform_bounds,reproject,Resampling
from rasterio.transform import Affine,array_bounds
from rasterio.windows import from_bounds
from model_registry import file_sha256
from fetch_diverse_operation_pairs import run as fetch
SOURCE=ROOT/'datasets/tropical_work_pairs_20261005'
OUT=ROOT/'datasets/tropical_work_pairs_local_quality_20261005'
CASES=['AU_ANDOOM_SLATS2324','AU_BAUXITEHILLS_SLATS2324']

def load(p):return json.loads(p.read_text(encoding='utf-8'))
def write(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')

def run():
    OUT.mkdir(exist_ok=True);old=load(SOURCE/'site_plan.json');sites={}
    for case in CASES:
        new=case.replace('_SLATS','_LOCAL_SLATS');sites[new]={**old['sites'][case],'site_id':new,'old_acquisition_case':case}
        folder=OUT/new;folder.mkdir(exist_ok=True);target=folder/'official_mining_change_subset.geojson'
        if not target.exists():target.write_bytes((SOURCE/case/target.name).read_bytes())
    proposal={**old,'sites':sites,'registered_at':datetime.now(timezone(timedelta(hours=8))).isoformat(),
              'parent_site_plan_sha256':file_sha256(SOURCE/'site_plan.json'),
              'scene_cloud_limit':100,'training_sites':list(sites),'normal_holdout_sites':[],
              'local_SCL_support_minimum':.65,'local_SCL_nodata_maximum':.01,'joint_local_support_minimum':.6,
              'fixed_quality_rescue':'latest qualifyingJuly2023 pre;earliest qualifyingSeptember2024 post;nativeSCLlandclasses2,4,5;no model or worklabelsize selection',
              'acquisition_rule':'locally quality-selected fullcover pair;actual coarse scene cloud values preserved;maxcloud100 with localSCL gates;model predictions0'}
    path=OUT/'site_plan.json'
    if path.exists():
        saved=load(path);assert all(saved[k]==v for k,v in proposal.items() if k!='registered_at')
    else:write(path,proposal)
    session=requests.Session();session.mount('https://',HTTPAdapter(max_retries=Retry(total=3,backoff_factor=1,status_forcelist=(429,500,502,503,504))))
    def sign(href):
        response=session.get('https://planetarycomputer.microsoft.com/api/sas/v1/sign',params={'href':href},timeout=(10,40));response.raise_for_status();return response.json()['href']
    for case,site in sites.items():
        folder=OUT/case;original=folder/'original_catalogue.json'
        if not original.exists():original.write_bytes((SOURCE/site['old_acquisition_case']/'catalogue.json').read_bytes())
        catalogue=load(original);bounds=transform_bounds('EPSG:4326',site['crs'],*site['bbox_wgs84'])
        left,bottom=np.floor(np.asarray(bounds[:2])/10)*10;right,top=np.ceil(np.asarray(bounds[2:])/10)*10
        width,height=int(round((right-left)/10)),int(round((top-bottom)/10));affine=Affine(10,0,left,0,-10,top)
        def coverage(scene):
            a=scene['assets']['B02'];p=scene['properties'];b=array_bounds(*a.get('proj:shape',p.get('proj:shape')),Affine(*a.get('proj:transform',p.get('proj:transform'))[:6]));w=transform_bounds(site['crs'],f"EPSG:{p['proj:epsg']}",left,bottom,right,top)
            return max(0,min(b[2],w[2])-max(b[0],w[0]))*max(0,min(b[3],w[3])-max(b[1],w[1]))/((w[2]-w[0])*(w[3]-w[1]))
        eligible=[s for s in catalogue['features'] if coverage(s)>=.999 and s['properties'].get('eo:cloud_cover',100)<100]
        before=sorted([s for s in eligible if s['properties']['datetime'][:10]<='2023-07-31'],key=lambda s:(s['properties']['datetime'],s['id']),reverse=True)
        after=sorted([s for s in eligible if s['properties']['datetime'][:10]>='2024-09-01'],key=lambda s:(s['properties']['datetime'],s['id']))
        checks=[];cache=folder/'quality_candidates';cache.mkdir(exist_ok=True)
        def local(scene):
            target=cache/(scene['id']+'_SCL_10m.tif')
            if not target.exists():
                href=scene['assets']['SCL']['href']
                with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN='EMPTY_DIR',GDAL_HTTP_VERSION='1.1',GDAL_HTTP_TIMEOUT='30',GDAL_HTTP_MAX_RETRY='2',CPL_VSIL_CURL_ALLOWED_EXTENSIONS='.tif'):
                    with rasterio.open(sign(href)) as src:
                        window=from_bounds(*transform_bounds(site['crs'],src.crs,left,bottom,right,top),transform=src.transform).round_offsets().round_lengths()
                        raw=src.read(1,window=window,boundless=True,fill_value=0);dest=np.zeros((height,width),np.uint8)
                        reproject(raw,dest,src_transform=src.window_transform(window),src_crs=src.crs,dst_transform=affine,dst_crs=site['crs'],src_nodata=0,dst_nodata=0,resampling=Resampling.nearest)
                with rasterio.open(target,'w',driver='GTiff',height=height,width=width,count=1,dtype='uint8',crs=site['crs'],transform=affine,nodata=0,compress='deflate') as dst:dst.write(dest,1)
            with rasterio.open(target) as src:scl=src.read(1)
            support=np.isin(scl,[2,4,5]);nodata=float((scl==0).mean())
            receipt={'scene':scene['id'],'actual_catalogue_scene_cloud':scene['properties'].get('eo:cloud_cover'),
                     'native_local_land_support':float(support.mean()),'local_nodata':nodata,'SCL_sha256':file_sha256(target)}
            checks.append(receipt);print(case,scene['id'],'local',round(float(support.mean()),3),'nodata',round(nodata,3),flush=True)
            return support,float(support.mean())>=.65 and nodata<=.01
        first=pre_support=None
        for scene in before:
            support,good=local(scene)
            if good:first,pre_support=scene,support;break
        if first is None:raise ValueError('No fixed-quality pre image for '+case)
        second=None;joint=None
        for scene in after:
            support,good=local(scene);joint=float((support&pre_support).mean())
            if good and joint>=.6:second=scene;break
        if second is None:raise ValueError('No fixed-quality post image for '+case)
        selected={'type':'FeatureCollection','features':[first,second],'links':[],
                  'local_quality_selection_only':True,'original_catalogue_sha256':file_sha256(original)}
        catpath=folder/'catalogue.json'
        if catpath.exists():assert load(catpath)==selected
        else:write(catpath,selected)
        receipt={'pre':first['id'],'post':second['id'],'checks':checks,'joint_local_support':joint,
                 'quality_policy_sha256':file_sha256(path),'original_catalogue_sha256':file_sha256(original),
                 'provider_values_geometry_and_scene_dates_unmodified':True,'labels_or_model_predictions_used':False}
        checkpath=folder/'local_quality_selection.json'
        if checkpath.exists():assert load(checkpath)==receipt
        else:write(checkpath,receipt)
        fetch(out=OUT,plan={**load(path),'sites':{case:site}})
    print('Local quality rescue acquired;all originalcloud values and existinglabelgates preserved',flush=True)

if __name__=='__main__':sys.stdout.reconfigure(encoding='utf-8');run()
