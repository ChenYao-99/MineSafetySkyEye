"""Quality-only bounded local SCL screen; scene-average cloud is not mine visibility."""
from pathlib import Path
from datetime import datetime
import json,sys
import numpy as np
import rasterio,requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from rasterio.transform import Affine,array_bounds
from rasterio.warp import transform_bounds,reproject,Resampling
from rasterio.windows import from_bounds
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from model_registry import file_sha256
import fetch_arid_incident_windows as fetcher
OUT=ROOT/'datasets/arid_accident_operation_20261004';SITE=OUT/'UK_NEWPORT2023';Q=SITE/'local_quality_screen'


def write(p,v):p.write_bytes((json.dumps(v,ensure_ascii=False,indent=2)+'\n').encode('utf-8'))


def run():
    p=json.loads((OUT/'newport_operation_plan.json').read_text(encoding='utf-8'));c=p['cases']['UK_NEWPORT2023'];Q.mkdir(exist_ok=True)
    contract={'registered_at':None,'original_plan_sha256':file_sha256(OUT/'newport_operation_plan.json'),
       'reason':'all eligible dates after independentNov27 inspection have whole-scene cloud>20; localAOI unverified',
       'rule':'localAOI SCL clear fraction>=0.98 excluding0,1,3,8,9,10,11; inspect at most6 nearest scenes each side, preferred nativeUTM31; reference/mine visibility still visually reviewed',
       'no_model_or_answer_scores_used':True,'keep_dates_and_source_roles_unchanged':True,'clear_fraction_min':.98,
       'SCL_is_observation_flag_not_ground_truth':True,'no_pixel_labels_created':True}
    cp=Q/'protocol.json'
    if cp.exists():
        old=json.loads(cp.read_text(encoding='utf-8'));contract['registered_at']=old['registered_at']
        if old!=contract:raise ValueError('Local quality protocol changed')
    else:contract['registered_at']=datetime.now().isoformat(timespec='seconds');write(cp,contract)
    catalogue=json.loads((SITE/'catalogue.json').read_text(encoding='utf-8'))['features'];crs=c['crs'];b=transform_bounds('EPSG:4326',crs,*c['bbox_wgs84'])
    left,bottom=np.floor(np.asarray(b[:2])/10)*10;right,top=np.ceil(np.asarray(b[2:])/10)*10;shape=(int((top-bottom)/10),int((right-left)/10));grid=Affine(10,0,left,0,-10,top)
    def coverage(s):
        a=s['assets']['B02'];prop=s['properties'];ab=array_bounds(*a.get('proj:shape',prop.get('proj:shape')),Affine(*a.get('proj:transform',prop.get('proj:transform'))[:6]));mb=transform_bounds(crs,f"EPSG:{prop['proj:epsg']}",left,bottom,right,top)
        return max(0,min(ab[2],mb[2])-max(ab[0],mb[0]))*max(0,min(ab[3],mb[3])-max(ab[1],mb[1]))/((mb[2]-mb[0])*(mb[3]-mb[1]))
    eligible=[s for s in catalogue if coverage(s)>=.999 and s['properties']['proj:epsg']==32631]
    # One processing version per acquisition. Selection ignores observed colours/change/model output.
    unique={}
    for s in sorted(eligible,key=lambda s:s['id']):unique[s['properties']['datetime'][:10]]=s
    before=sorted([s for d,s in unique.items() if d<=c['pre_through']],key=lambda s:s['properties']['datetime'],reverse=True)[:6]
    after=sorted([s for d,s in unique.items() if d>=c['post_from']],key=lambda s:(abs((datetime.fromisoformat(s['properties']['datetime'].replace('Z','+00:00')).date()-datetime.fromisoformat(c['post_target']).date()).days),s['id']))[:6]
    session=requests.Session();session.mount('https://',HTTPAdapter(max_retries=Retry(total=2,backoff_factor=1,status_forcelist=(429,500,502,503,504))))
    rows=[]
    for group,scenes in (('pre',before),('post',after)):
        for s in scenes:
            path=Q/(s['id']+'_SCL.tif');rp=path.with_suffix('.json')
            if rp.exists():
                row=json.loads(rp.read_text(encoding='utf-8'))
                if file_sha256(path)!=row['sha256']:raise ValueError('Local screen changed')
            else:
                href=s['assets']['SCL']['href'];r=session.get('https://planetarycomputer.microsoft.com/api/sas/v1/sign',params={'href':href},timeout=(10,30));r.raise_for_status();dest=np.zeros(shape,np.uint8)
                with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN='EMPTY_DIR',GDAL_HTTP_TIMEOUT='30',GDAL_HTTP_MAX_RETRY='2',CPL_VSIL_CURL_ALLOWED_EXTENSIONS='.tif'):
                    with rasterio.open(r.json()['href']) as f:
                        win=from_bounds(*transform_bounds(crs,f.crs,left,bottom,right,top),transform=f.transform).round_offsets().round_lengths();raw=f.read(1,window=win,boundless=True,fill_value=0)
                        reproject(raw,dest,src_transform=f.window_transform(win),src_crs=f.crs,dst_transform=grid,dst_crs=crs,src_nodata=0,dst_nodata=0,resampling=Resampling.nearest)
                with rasterio.open(path,'w',driver='GTiff',width=shape[1],height=shape[0],count=1,dtype='uint8',crs=crs,transform=grid,nodata=0,compress='deflate') as f:f.write(dest,1)
                clear=float((~np.isin(dest,[0,1,3,8,9,10,11])).mean());row={'scene_id':s['id'],'datetime':s['properties']['datetime'],'group':group,
                  'whole_scene_cloud':s['properties'].get('eo:cloud_cover'),'local_clear_fraction':clear,'passes_local_gate':clear>=.98,'sha256':file_sha256(path)};write(rp,row)
            rows.append(row);print(group,row['datetime'][:10],row['local_clear_fraction'],row['passes_local_gate'],flush=True)
            if row['passes_local_gate']:break
    accepted=[r['scene_id'] for r in rows if r['passes_local_gate']]
    result={'protocol_sha256':file_sha256(cp),'rows':rows,'accepted_scene_ids':accepted,'model_predictions':0};write(Q/'results.json',result)
    if not any(r['group']=='pre' and r['passes_local_gate'] for r in rows) or not any(r['group']=='post' and r['passes_local_gate'] for r in rows):raise ValueError('No local98%-clear pair; no silent gate reduction')
    p={**p,'original_plan_sha256':file_sha256(OUT/'newport_operation_plan.json'),'registered_local_quality_screen':{'path':str(Q/'results.json'),'sha256':file_sha256(Q/'results.json')},
       'scene_choice_rule':'registered local98%-clearSCL screen; same source/date cutoffs; all no model scores'}
    path=OUT/'newport_local_quality_plan.json';write(path,p);fetcher.PLAN=path;fetcher.run()


if __name__=='__main__':sys.stdout.reconfigure(encoding='utf-8');run()
