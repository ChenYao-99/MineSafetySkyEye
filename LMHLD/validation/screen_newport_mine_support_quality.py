"""Independently located mine-centred quality screen, preserving rejected wider-area checks."""
from pathlib import Path
from datetime import datetime
import json,sys
import numpy as np
import rasterio,requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from rasterio.transform import Affine,array_bounds
from rasterio.warp import transform_bounds,reproject,Resampling,transform
from rasterio.windows import from_bounds
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from model_registry import file_sha256
import fetch_arid_incident_windows as fetcher
OUT=ROOT/'datasets/arid_accident_operation_20261004';SITE=OUT/'UK_NEWPORT2023';Q=SITE/'mine_support_quality_v2'


def write(p,v):p.write_bytes((json.dumps(v,ensure_ascii=False,indent=2)+'\n').encode('utf-8'))


def run():
    base=json.loads((OUT/'newport_operation_plan.json').read_text(encoding='utf-8'));c=base['cases']['UK_NEWPORT2023'];Q.mkdir(exist_ok=True)
    # Geological-survey anchor and fixed350m radius, never a colour-/prediction-selected clear patch.
    east,north=transform('EPSG:27700',c['crs'],[552500],[233100]);ex,ny=east[0],north[0]
    parent=transform_bounds('EPSG:4326',c['crs'],*c['bbox_wgs84']);left,bottom=np.floor(np.asarray(parent[:2])/10)*10;right,top=np.ceil(np.asarray(parent[2:])/10)*10
    shape=(int((top-bottom)/10),int((right-left)/10));grid=Affine(10,0,left,0,-10,top)
    yy,xx=np.indices(shape);coordsx=left+(xx+.5)*10;coordsy=top-(yy+.5)*10;support=(abs(coordsx-ex)<=350)&(abs(coordsy-ny)<=350)
    protocol={'registered_at':None,'source_plan_sha256':file_sha256(OUT/'newport_operation_plan.json'),
      'support_rule':'fixed square350m each direction around independently published BGS TL525331 quarry point; not an accident or working-phase boundary',
      'support_center_native':[ex,ny],'minimum_clear_fraction':.98,'whole_widerAOI_metrics_retained':True,
      'date_cutoffs_and_roles_unchanged':True,'selection_order':'whole-scene quality then distance to source date; capped6 acquisitions each side; no model/answer scores',
      'support_never_used_as_normal_pixel_truth':True,'reason':'widerAOI includes unrelated settlement/fields; prior98% widerAOI gate failed; check independent actual mine support with same98% criterion'}
    cp=Q/'protocol.json'
    if cp.exists():
        old=json.loads(cp.read_text(encoding='utf-8'));protocol['registered_at']=old['registered_at']
        if old!=protocol:raise ValueError('Mine-support quality protocol changed')
    else:protocol['registered_at']=datetime.now().isoformat(timespec='seconds');write(cp,protocol)
    scenes=json.loads((SITE/'catalogue.json').read_text(encoding='utf-8'))['features'];unique={}
    for s in sorted(scenes,key=lambda s:s['id']):
        if s['properties']['proj:epsg']==32631:unique[s['properties']['datetime'][:10]]=s
    before=[s for d,s in unique.items() if d<=c['pre_through']];after=[s for d,s in unique.items() if d>=c['post_from']]
    before.sort(key=lambda s:(s['properties'].get('eo:cloud_cover',100),-datetime.fromisoformat(s['properties']['datetime'].replace('Z','+00:00')).timestamp()))
    after.sort(key=lambda s:(s['properties'].get('eo:cloud_cover',100),abs((datetime.fromisoformat(s['properties']['datetime'].replace('Z','+00:00')).date()-datetime.fromisoformat(c['post_target']).date()).days)))
    session=requests.Session();session.mount('https://',HTTPAdapter(max_retries=Retry(total=2,backoff_factor=1,status_forcelist=(429,500,502,503,504))))
    rows=[]
    for group,choices in (('pre',before[:6]),('post',after[:6])):
        for s in choices:
            prior=SITE/'local_quality_screen'/(s['id']+'_SCL.tif');destpath=Q/(s['id']+'_SCL.tif');recordpath=Q/(s['id']+'.json')
            if recordpath.exists():
                row=json.loads(recordpath.read_text(encoding='utf-8'))
                if file_sha256(row['scl_path'])!=row['scl_sha256']:raise ValueError('Quality source changed')
            else:
                if prior.exists():
                    with rasterio.open(prior) as f:dest=f.read(1)
                    sclpath=prior
                else:
                    href=s['assets']['SCL']['href'];r=session.get('https://planetarycomputer.microsoft.com/api/sas/v1/sign',params={'href':href},timeout=(10,30));r.raise_for_status();dest=np.zeros(shape,np.uint8)
                    with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN='EMPTY_DIR',GDAL_HTTP_TIMEOUT='30',GDAL_HTTP_MAX_RETRY='2',CPL_VSIL_CURL_ALLOWED_EXTENSIONS='.tif'):
                        with rasterio.open(r.json()['href']) as f:
                            win=from_bounds(*transform_bounds(c['crs'],f.crs,left,bottom,right,top),transform=f.transform).round_offsets().round_lengths();raw=f.read(1,window=win,boundless=True,fill_value=0)
                            reproject(raw,dest,src_transform=f.window_transform(win),src_crs=f.crs,dst_transform=grid,dst_crs=c['crs'],src_nodata=0,dst_nodata=0,resampling=Resampling.nearest)
                    with rasterio.open(destpath,'w',driver='GTiff',width=shape[1],height=shape[0],count=1,dtype='uint8',crs=c['crs'],transform=grid,nodata=0,compress='deflate') as f:f.write(dest,1)
                    sclpath=destpath
                visible=~np.isin(dest,[0,1,3,8,9,10,11]);fraction=float(visible[support].mean())
                row={'scene_id':s['id'],'datetime':s['properties']['datetime'],'group':group,'scene_cloud':s['properties'].get('eo:cloud_cover'),
                  'mine_support_clear_fraction':fraction,'widerAOI_clear_fraction':float(visible.mean()),'passes_local_gate':fraction>=.98,
                  'scl_path':str(sclpath),'scl_sha256':file_sha256(sclpath)};write(recordpath,row)
            rows.append(row);print(group,row['datetime'][:10],row['mine_support_clear_fraction'],row['passes_local_gate'],flush=True)
            if row['passes_local_gate']:break
    result={'protocol_sha256':file_sha256(cp),'rows':rows,'accepted_scene_ids':[r['scene_id'] for r in rows if r['passes_local_gate']],'model_predictions':0};write(Q/'results.json',result)
    if not all(any(r['group']==g and r['passes_local_gate'] for r in rows) for g in ('pre','post')):raise ValueError('No mine-centred98%-clear pair, keep pending without lowering criterion')
    p={**base,'registered_local_quality_screen':{'path':str(Q/'results.json'),'sha256':file_sha256(Q/'results.json')},
       'scene_choice_rule':'fixed geological-survey-centred98%-clear support; originaldates; independently documented planned works; no model output'}
    path=OUT/'newport_mine_support_plan.json';write(path,p);fetcher.PLAN=path;fetcher.run()


if __name__=='__main__':sys.stdout.reconfigure(encoding='utf-8');run()
