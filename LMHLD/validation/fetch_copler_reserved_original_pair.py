"""Bounded actual Sentinel-2 windows on frozen new-incident/operation dates."""
from pathlib import Path
from datetime import datetime
import json,sys,urllib.parse
import numpy as np
import requests
import rasterio
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from rasterio.warp import transform_bounds,reproject,Resampling
from rasterio.transform import Affine,array_bounds
from rasterio.windows import from_bounds
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from runtime_paths import configure_runtime_paths
configure_runtime_paths()
from model_registry import file_sha256
from radiometry import parse_xml
from mine_data_protocol import assert_mine_group_role,verify_assets,registry,DATA as BENCHMARK_DATA
OUT=ROOT/'datasets/mine_benchmark_20261002/reserved_candidates/TR_COP2024';PLAN=OUT/'data_plan.json'


def write(p,v):p.write_bytes((json.dumps(v,ensure_ascii=False,indent=2)+'\n').encode('utf-8'))


def run():
    p=json.loads(PLAN.read_text(encoding='utf-8'));assert_mine_group_role('TR_COP2024','reserved_final_test');verify_assets()
    session=requests.Session();session.mount('https://',HTTPAdapter(max_retries=Retry(total=3,backoff_factor=1,status_forcelist=(429,500,502,503,504),allowed_methods=frozenset(('GET',)))))
    for eid,c in p['cases'].items():
        folder=OUT;folder.mkdir(exist_ok=True);cat=folder/'catalogue.json'
        if not cat.exists():
            features=[];url='https://planetarycomputer.microsoft.com/api/stac/v1/search'
            params={'collections':'sentinel-2-l2a','bbox':','.join(map(str,c['bbox_wgs84'])),'datetime':c['time_interval'],'limit':100};pages=[]
            for _ in range(12):
                r=session.get(url,params=params,timeout=(10,30));r.raise_for_status();page=r.json();features.extend(page['features']);pages.append(page)
                nexts=[l for l in page.get('links',[]) if l.get('rel')=='next']
                if not nexts:break
                if nexts[0].get('method','GET')!='GET':raise ValueError('Unsupported STAC pagination method')
                url=nexts[0]['href'];params=None
            else:raise ValueError('Catalogue page budget exceeded')
            write(cat,{'features':features,'pages':pages,'source':'public STAC collection','model_scores_used':False})
        features=json.loads(cat.read_text(encoding='utf-8'))['features'];crs=c['crs'];bounds=transform_bounds('EPSG:4326',crs,*c['bbox_wgs84'])
        left,bottom=np.floor(np.asarray(bounds[:2])/10)*10;right,top=np.ceil(np.asarray(bounds[2:])/10)*10
        shape=(int(round((top-bottom)/10)),int(round((right-left)/10)));grid=Affine(10,0,left,0,-10,top)
        if max(shape)>800:raise ValueError('New window exceeds800 native pixels')
        def coverage(scene):
            asset=scene['assets']['B02'];prop=scene['properties'];ab=array_bounds(*asset.get('proj:shape',prop.get('proj:shape')),Affine(*asset.get('proj:transform',prop.get('proj:transform'))[:6]))
            mb=transform_bounds(crs,f"EPSG:{prop['proj:epsg']}",left,bottom,right,top)
            return max(0,min(ab[2],mb[2])-max(ab[0],mb[0]))*max(0,min(ab[3],mb[3])-max(ab[1],mb[1]))/((mb[2]-mb[0])*(mb[3]-mb[1]))
        if p.get('registered_local_quality_screen'):
            screen=p['registered_local_quality_screen'];quality_path=Path(screen['path'])
            if file_sha256(quality_path)!=screen['sha256']:raise ValueError('Registered local quality screen changed')
            passed=json.loads(quality_path.read_text(encoding='utf-8'))['accepted_scene_ids']
            eligible=[s for s in features if coverage(s)>=.999 and s['id'] in passed]
        else:
            exact=c['source_requested_optical_dates']
            groups={d:[s for s in features if coverage(s)>=.999 and s['properties']['datetime'][:10]==d] for d in exact.values()}
            eligible=[max(group,key=lambda s:(str(s['properties'].get('s2:processing_baseline','')),s['id'])) for group in groups.values() if group]
        before=sorted([s for s in eligible if s['properties']['datetime'][:10]<=c['pre_through']],key=lambda s:(s['properties']['datetime'],s['id']),reverse=True)
        post=[s for s in eligible if s['properties']['datetime'][:10]>=c['post_from']]
        post.sort(key=lambda s:(abs((datetime.fromisoformat(s['properties']['datetime'].replace('Z','+00:00')).date()-datetime.fromisoformat(c['post_target']).date()).days),s['properties']['datetime'],s['id']))
        if not before or not post:raise ValueError(f'{eid}: no qualifying pair; no model-based date fallback')
        selection={'data_plan_sha256':file_sha256(PLAN),'catalogue_sha256':file_sha256(cat),'pre':before[0]['id'],'post':post[0]['id'],'rule':p['scene_choice_rule']}
        sp=folder/'acquisition_plan.json'
        if sp.exists() and json.loads(sp.read_text(encoding='utf-8'))!=selection:raise ValueError('Frozen pair selection changed')
        write(sp,selection);ledger=folder/'acquisition.json';record=json.loads(ledger.read_text(encoding='utf-8')) if ledger.exists() else {'case_id':eid,'mine_group':c['mine_group'],'role':c['role'],'images':{},'model_predictions':0,'pixel_labels_accepted':False}
        def signed(href):
            r=session.get('https://planetarycomputer.microsoft.com/api/sas/v1/sign',params={'href':href},timeout=(10,30));r.raise_for_status();return r.json()['href']
        for period,scene in (('pre',before[0]),('post',post[0])):
            if period in record['images']:
                for name,digest in record['images'][period]['assets_sha256'].items():
                    if file_sha256(folder/name)!=digest:raise ValueError('Existing raw acquisition changed')
                continue
            xml=folder/f'{period}_product.xml';r=session.get(signed(scene['assets']['product-metadata']['href']),timeout=(10,30));r.raise_for_status();cal=parse_xml(r.content);xml.write_bytes(r.content)
            bands=[];sources=[];scl=None
            for name in ('B02','B03','B04','B08','SCL'):
                href=scene['assets'][name]['href']
                with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN='EMPTY_DIR',GDAL_HTTP_VERSION='1.1',GDAL_HTTP_TIMEOUT='30',GDAL_HTTP_MAX_RETRY='2',CPL_VSIL_CURL_ALLOWED_EXTENSIONS='.tif'):
                    with rasterio.open(signed(href)) as f:
                        win=from_bounds(*transform_bounds(crs,f.crs,left,bottom,right,top),transform=f.transform).round_offsets().round_lengths();raw=f.read(1,window=win,boundless=True,fill_value=0);dest=np.zeros(shape,np.uint8 if name=='SCL' else np.uint16)
                        reproject(raw,dest,src_transform=f.window_transform(win),src_crs=f.crs,dst_transform=grid,dst_crs=crs,src_nodata=0,dst_nodata=0,resampling=Resampling.nearest)
                if name=='SCL':scl=dest
                else:bands.append(dest)
                sources.append({'band':name,'unsigned_href':href,'native_resolution_m':20 if name=='SCL' else 10})
                print(eid,period,name,shape,'model0',flush=True)
            target=folder/f'{period}_BGRN_10m.tif';quality=folder/f'{period}_SCL_10m.tif'
            profile={'driver':'GTiff','height':shape[0],'width':shape[1],'count':4,'dtype':'uint16','crs':crs,'transform':grid,'nodata':0,'compress':'deflate'}
            with rasterio.open(target,'w',**profile) as f:
                f.write(np.stack(bands));f.descriptions=('B02','B03','B04','B08');f.scales=tuple(cal['scales']);f.offsets=tuple(cal['reflectance_offsets'])
                f.update_tags(case_id=eid,mine_group=c['mine_group'],role=c['role'],acquisition_datetime=scene['properties']['datetime'],raw_DN_values='preserved',radiometry_source_xml_sha=file_sha256(xml))
            with rasterio.open(quality,'w',**{**profile,'count':1,'dtype':'uint8'}) as f:f.write(scl,1);f.set_band_description(1,'SCL')
            record['images'][period]={'scene_id':scene['id'],'datetime':scene['properties']['datetime'],'shape':list(shape),
               'assets_sha256':{a.name:file_sha256(a) for a in (target,quality,xml)},'calibration':cal,'source_assets':sources,
               'SCL_obscured_fraction':float(np.isin(scl,[0,1,3,8,9,10,11]).mean()),'local_visibility_review_pending':True}
            write(ledger,record)
            value=registry();assets=value.setdefault('additional_frozen_assets',{})
            for name,digest in record['images'][period]['assets_sha256'].items():
                key=(folder/name).relative_to(BENCHMARK_DATA).as_posix()
                if key in assets and assets[key]!=digest:raise ValueError('Frozen original changed')
                assets[key]=digest
            write(BENCHMARK_DATA/'mine_case_registry.json',value)
    print('Exact fieldpaper-date raw pair acquired; mine stays reserved; detector predictions0')


if __name__=='__main__':sys.stdout.reconfigure(encoding='utf-8');run()
