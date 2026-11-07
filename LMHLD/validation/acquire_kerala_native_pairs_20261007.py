"""Native Sentinel-2 windows for source-boundary qualification;no CNN inference."""
from pathlib import Path
from datetime import datetime,timezone
from urllib.parse import urlsplit
import json,hashlib,sys
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'validation')]
from runtime_paths import configure_runtime_paths
configure_runtime_paths()
import requests,numpy as np,rasterio
from rasterio.warp import transform_bounds,reproject,Resampling
from rasterio.windows import from_bounds
from radiometry import parse_xml
OUT=ROOT/'datasets/accident_boundary_pursuit_20261007/Kerala2018'


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path,value):Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')


def main():
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--plan');parser.add_argument('--output');parser.add_argument('--post-audit-auxiliary',action='store_true');args=parser.parse_args()
    plan_path=Path(args.plan).resolve() if args.plan else OUT/'native_pair_plan.json'
    output=Path(args.output).resolve() if args.output else OUT
    allowed=ROOT/'datasets/post_audit_development_20261008' if args.post_audit_auxiliary else OUT
    assert plan_path.is_relative_to(allowed) and output.is_relative_to(allowed)
    output.mkdir(parents=True,exist_ok=True);plan=json.loads(plan_path.read_text(encoding='utf-8'))
    if args.post_audit_auxiliary:
        assert all(case['role']=='development_auxiliary_change_context_only' for case in plan['cases'].values())
        protected=json.loads((ROOT/'datasets/input_reserve_registry.json').read_text(encoding='utf-8'))['cases']
        main=json.loads((ROOT/'datasets/mine_benchmark_20261002/mine_case_registry.json').read_text(encoding='utf-8'))['cases']
        for case in plan['cases'].values():
            box=case['bbox_wgs84']
            for entry in list(protected.values())+[entry for entry in main.values() if entry.get('role')=='reserved_final_test']:
                other=entry.get('protected_geographic_extent_wgs84')
                assert not other or not (min(box[2],other[2])>max(box[0],other[0]) and min(box[3],other[3])>max(box[1],other[1]))
    session=requests.Session();session.trust_env=False
    from requests.adapters import HTTPAdapter
    from urllib3.util.retry import Retry
    session.mount('https://',HTTPAdapter(max_retries=Retry(total=3,backoff_factor=2,respect_retry_after_header=True,status_forcelist=[429,500,502,503,504],allowed_methods=['GET'])))
    tokens={}
    def signed(href):
        # The official SDK caches one public token per account/container instead
        # of making one signing request per image band. Never persist the token.
        parsed=urlsplit(href);account=parsed.hostname.split('.')[0];container=parsed.path.strip('/').split('/')[0]
        assert parsed.hostname.endswith('.blob.core.windows.net') and not parsed.query
        key=(account,container);token=tokens.get(key)
        if token is None or datetime.fromisoformat(token['msft:expiry'].replace('Z','+00:00'))<=datetime.now(timezone.utc):
            r=session.get(f'https://planetarycomputer.microsoft.com/api/sas/v1/token/{account}/{container}',timeout=(10,30));r.raise_for_status();token=r.json();tokens[key]=token
        return href+'?'+token['token']
    for key,case in plan['cases'].items():
        folder=output/key;folder.mkdir(exist_ok=True);cat=folder/'catalogue.json'
        if not cat.exists():
            features=[];url='https://planetarycomputer.microsoft.com/api/stac/v1/search'
            params={'collections':'sentinel-2-l2a','bbox':','.join(map(str,case['bbox_wgs84'])),
                'datetime':case['query_interval'],'query':json.dumps({'eo:cloud_cover':{'lt':20}}),'limit':100}
            for _ in range(8):
                response=session.get(url,params=params,timeout=(10,40));response.raise_for_status();j=response.json();features.extend(j['features'])
                links=[x for x in j.get('links',[]) if x.get('rel')=='next']
                if not links:break
                assert links[0].get('method','GET')=='GET';url=links[0]['href'];params=None
            else:raise ValueError('Catalogue page budget exceeded')
            write(cat,{'source':'MicrosoftPlanetaryComputer/ESA sentinel-2-l2a','features':features,'model_scores_used':False})
        features=json.loads(cat.read_text(encoding='utf-8'))['features']
        def cover(scene):
            b=scene.get('bbox',[]);a=case['bbox_wgs84']
            return len(b)==4 and b[0]<=a[0] and b[1]<=a[1] and b[2]>=a[2] and b[3]>=a[3]
        eligible=[s for s in features if cover(s) and s['properties'].get('eo:cloud_cover',100)<20]
        before=sorted([s for s in eligible if s['properties']['datetime'][:10]<=case['pre_through']],key=lambda s:(s['properties']['datetime'],s['id']),reverse=True)
        post=sorted([s for s in eligible if s['properties']['datetime'][:10]>=case['post_from']],
            key=lambda s:(abs((datetime.fromisoformat(s['properties']['datetime'].replace('Z','+00:00')).date()-datetime.fromisoformat(case['post_target']).date()).days),s['properties']['datetime'],s['id']))
        if case.get('fixed_pre_scene_id'):before=[s for s in before if s['id']==case['fixed_pre_scene_id']]
        if case.get('fixed_post_scene_id'):post=[s for s in post if s['id']==case['fixed_post_scene_id']]
        if not before or not post:
            write(folder/'not_acquired.json',{'reason':'No original scenes satisfying frozen dates,coverage and cloud metadata','new_model_predictions':0});continue
        selection={'plan_sha256':sha(plan_path),'catalogue_sha256':sha(cat),'pre':before[0]['id'],'post':post[0]['id'],'selection_rule':plan['selection_rule']}
        selected_path=folder/'acquisition_plan.json'
        if selected_path.exists():assert json.loads(selected_path.read_text(encoding='utf-8'))==selection
        else:write(selected_path,selection)
        ledger=folder/'acquisition.json';record=json.loads(ledger.read_text(encoding='utf-8')) if ledger.exists() else {'case_id':key,'role':case['role'],'source_event_ids':case['source_event_ids'],'images':{},'multispectral_resampled':False,'model_predictions':0,'pixel_labels_accepted':False}
        native_grid=None;native_shape=None;native_crs=None
        for epoch,scene in [('pre',before[0]),('post',post[0])]:
            target=folder/(epoch+'_BGRN_native_10m.tif')
            if epoch in record['images']:
                for filename,digest in record['images'][epoch]['assets_sha256'].items():assert sha(folder/filename)==digest
                with rasterio.open(target) as src:native_grid=src.transform;native_shape=src.shape;native_crs=src.crs
                continue
            xml=folder/(epoch+'_product.xml');response=session.get(signed(scene['assets']['product-metadata']['href']),timeout=(10,40));response.raise_for_status();cal=parse_xml(response.content);xml.write_bytes(response.content)
            bands=[];sources=[];scl=None;quality_path=folder/(epoch+'_SCL_on_10m_grid.tif')
            for name in ['B02','B03','B04','B08','SCL']:
                href=scene['assets'][name]['href']
                with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN='EMPTY_DIR',GDAL_HTTP_VERSION='1.1',GDAL_HTTP_TIMEOUT='35',GDAL_HTTP_MAX_RETRY='2',CPL_VSIL_CURL_ALLOWED_EXTENSIONS='.tif'):
                    with rasterio.open(signed(href)) as src:
                        if native_grid is None:
                            bounds=transform_bounds('EPSG:4326',src.crs,*case['bbox_wgs84'])
                            window=from_bounds(*bounds,transform=src.transform).round_offsets().round_lengths()
                            native_grid=src.window_transform(window);native_shape=(int(window.height),int(window.width));native_crs=src.crs
                            assert max(native_shape)<=plan['max_pixels_per_side'] and src.res==(10.0,10.0)
                        else:
                            assert src.crs==native_crs,'No cross-CRS warping allowed for these native windows'
                            bounds=rasterio.transform.array_bounds(*native_shape,native_grid)
                            window=from_bounds(*bounds,transform=src.transform).round_offsets().round_lengths()
                        raw=src.read(1,window=window);source_grid=src.window_transform(window)
                        assert raw.size and window.col_off>=0 and window.row_off>=0 and window.col_off+window.width<=src.width and window.row_off+window.height<=src.height
                        if name=='SCL':
                            original_scl=folder/(epoch+'_SCL_native_20m.tif')
                            with rasterio.open(original_scl,'w',driver='GTiff',height=raw.shape[0],width=raw.shape[1],count=1,dtype='uint8',crs=src.crs,transform=source_grid,nodata=0,compress='deflate') as dst:dst.write(raw.astype(np.uint8),1)
                            scl=np.zeros(native_shape,np.uint8)
                            reproject(raw,scl,src_transform=source_grid,src_crs=src.crs,dst_transform=native_grid,dst_crs=native_crs,src_nodata=0,dst_nodata=0,resampling=Resampling.nearest)
                        else:
                            assert raw.shape==native_shape and source_grid.almost_equals(native_grid,precision=1e-7) and src.res==(10.0,10.0)
                            bands.append(raw)
                sources.append({'band':name,'unsigned_href':href,'native_resolution_m':20 if name=='SCL' else 10})
                print(key,epoch,name,native_shape,'noModel',flush=True)
            profile={'driver':'GTiff','height':native_shape[0],'width':native_shape[1],'count':4,'dtype':'uint16','crs':native_crs,'transform':native_grid,'nodata':0,'compress':'deflate'}
            with rasterio.open(target,'w',**profile) as dst:
                dst.write(np.stack(bands));dst.descriptions=('B02','B03','B04','B08');dst.scales=tuple(cal['scales']);dst.offsets=tuple(cal['reflectance_offsets'])
                dst.update_tags(case_id=key,role=case['role'],classification_use_allowed='false',acquisition_datetime=scene['properties']['datetime'],raw_DN_preserved='true',multispectral_resampled='false')
            with rasterio.open(quality_path,'w',**{**profile,'count':1,'dtype':'uint8'}) as dst:dst.write(scl,1);dst.set_band_description(1,'SCL_nearest_from_native20m')
            record['images'][epoch]={'scene_id':scene['id'],'datetime':scene['properties']['datetime'],'native_shape':list(native_shape),
                'assets_sha256':{p.name:sha(p) for p in [target,quality_path,original_scl,xml]},'calibration':cal,'source_assets':sources,
                'SCL_resampled_only_for_quality_masks':True,'multispectral_original_native_grid_preserved':True,
                'local_obscured_fraction':float(np.isin(scl,[0,1,3,8,9,10,11]).mean()),'visual_boundary_and_date_qualification_pending':True}
            write(ledger,record)
            if not args.post_audit_auxiliary:
                registry_path=ROOT/'datasets/input_reserve_registry.json';registry=json.loads(registry_path.read_text(encoding='utf-8'))
                for digest in record['images'][epoch]['assets_sha256'].values():
                    if digest not in registry['cases'][key]['input_sha256_values']:registry['cases'][key]['input_sha256_values'].append(digest)
                write(registry_path,registry)
    print('Native source windows acquired where frozen source gates allow;no classifier or accepted accident labels',flush=True)


if __name__=='__main__':sys.stdout.reconfigure(encoding='utf-8');main()
