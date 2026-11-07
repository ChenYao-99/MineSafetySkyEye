"""Unannotated winter coal-mine pairs for constructed-surface review, not new accidents."""
from pathlib import Path
import sys,json,urllib.parse
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import numpy as np
import rasterio
from rasterio.warp import transform_bounds,reproject,Resampling
from rasterio.transform import Affine,array_bounds
from rasterio.windows import from_bounds
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from model_registry import file_sha256
from radiometry import parse_xml
OUT=ROOT/'datasets/arid_operation_pairs_20261004'
SOURCE='https://www.mdpi.com/2072-4292/18/18/3094'
SITES={'ZhundongSouth':(89.2331,44.8189),'ZhundongMine1':(89.1742,44.9486)}


def write(path,value):path.write_bytes((json.dumps(value,ensure_ascii=False,indent=2)+'\n').encode('utf-8'))


def run():
    OUT.mkdir(exist_ok=True);session=requests.Session()
    session.mount('https://',HTTPAdapter(max_retries=Retry(total=2,backoff_factor=.8,status_forcelist=(429,500,502,503,504),allowed_methods=frozenset(('GET',)))))
    for site,(lon,lat) in SITES.items():
        folder=OUT/site;folder.mkdir(exist_ok=True)
        bbox=[lon-.025,lat-.018,lon+.025,lat+.018];crs='EPSG:32645'
        context={'site_id':site,'coordinate_source':SOURCE,'source_locator':'Table2 published mine coordinates',
                 'center_wgs84':[lon,lat],'bbox_wgs84':bbox,'role':'unannotated_auxiliary_development',
                 'scope':'winter constructed-surface appearance review','accident_event_asserted':False,
                 'normal_or_stability_labels_available':False,'independent_expert_feedback':False,
                 'historical_Zhundong_region_exposure_possible':True,'not_a_new_blind_mine':True}
        ctx=folder/'context.json'
        if ctx.exists() and json.loads(ctx.read_text(encoding='utf-8'))!=context:raise ValueError('Frozen site context changed')
        write(ctx,context)
        cat=folder/'catalogue.json'
        if not cat.exists():
            response=session.get('https://planetarycomputer.microsoft.com/api/stac/v1/search',params={
                'collections':'sentinel-2-l2a','bbox':','.join(map(str,bbox)),
                'datetime':'2023-02-01T00:00:00Z/2023-03-10T23:59:59Z','limit':100},timeout=(10,30))
            response.raise_for_status();value=response.json()
            if any(l.get('rel')=='next' for l in value.get('links',[])):raise ValueError('Catalogue pagination requires explicit handling')
            write(cat,value)
        features=json.loads(cat.read_text(encoding='utf-8'))['features']
        bounds=transform_bounds('EPSG:4326',crs,*bbox)
        left,bottom=np.floor(np.asarray(bounds[:2])/10)*10;right,top=np.ceil(np.asarray(bounds[2:])/10)*10
        width,height=int(round((right-left)/10)),int(round((top-bottom)/10));affine=Affine(10,0,left,0,-10,top)
        if max(width,height)>800:raise ValueError('Unexpected native window size')
        def coverage(scene):
            asset=scene['assets']['B02'];p=scene['properties']
            b=array_bounds(*asset.get('proj:shape',p.get('proj:shape')),Affine(*asset.get('proj:transform',p.get('proj:transform'))[:6]))
            w=transform_bounds(crs,f"EPSG:{p['proj:epsg']}",left,bottom,right,top)
            return max(0,min(b[2],w[2])-max(b[0],w[0]))*max(0,min(b[3],w[3])-max(b[1],w[1]))/((w[2]-w[0])*(w[3]-w[1]))
        eligible=[s for s in features if coverage(s)>=.999 and s['properties'].get('eo:cloud_cover',100)<20]
        before=sorted([s for s in eligible if s['properties']['datetime'][:10]<='2023-02-15'],key=lambda s:(s['properties']['datetime'],s['id']),reverse=True)
        after=sorted([s for s in eligible if s['properties']['datetime'][:10]>='2023-02-23'],key=lambda s:(s['properties']['datetime'],s['id']))
        if not before or not after:raise ValueError('No qualifying winter pair; do not choose scenes from model scores')
        plan={'context_sha':file_sha256(ctx),'catalogue_sha':file_sha256(cat),'scenes':{'pre':before[0]['id'],'post':after[0]['id']},
              'rule':'latest full-cover scene no later thanFeb15 and earliest no earlier thanFeb23; scenecloud<20%; model predictions0'}
        planpath=folder/'acquisition_plan.json'
        if planpath.exists() and json.loads(planpath.read_text(encoding='utf-8'))!=plan:raise ValueError('Frozen scene plan changed')
        write(planpath,plan)
        ledger=folder/'acquisition.json';record=json.loads(ledger.read_text(encoding='utf-8')) if ledger.exists() else {'site_id':site,'role':context['role'],'images':{},'model_predictions':0,'annotation_count':0}
        for period,scene in (('pre',before[0]),('post',after[0])):
            target=folder/f'{period}_BGRN_10m.tif';sclpath=folder/f'{period}_SCL_10m.tif';xml=folder/f'{period}_product.xml'
            if period in record['images']:
                old=record['images'][period]
                if file_sha256(target)!=old['sha256'] or file_sha256(sclpath)!=old['scl_sha256']:raise ValueError('Saved pair changed')
                continue
            def signed(href):
                response=session.get('https://planetarycomputer.microsoft.com/api/sas/v1/sign?href='+urllib.parse.quote(href,safe=''),timeout=(10,30))
                response.raise_for_status();return response.json()['href']
            if not xml.exists():
                response=session.get(signed(scene['assets']['product-metadata']['href']),timeout=(10,30));response.raise_for_status()
                if len(response.content)>5*1024*1024:raise ValueError('Oversized product XML')
                parse_xml(response.content);xml.write_bytes(response.content)
            calibration=parse_xml(xml.read_bytes());bands=[];assets=[];scl=None
            for band in ('B02','B03','B04','B08','SCL'):
                href=scene['assets'][band]['href']
                with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN='EMPTY_DIR',GDAL_HTTP_VERSION='1.1',GDAL_HTTP_TIMEOUT='30',GDAL_HTTP_MAX_RETRY='2',CPL_VSIL_CURL_ALLOWED_EXTENSIONS='.tif'):
                    with rasterio.open(signed(href)) as src:
                        window=from_bounds(*transform_bounds(crs,src.crs,left,bottom,right,top),transform=src.transform).round_offsets().round_lengths()
                        raw=src.read(1,window=window,boundless=True,fill_value=0);dest=np.zeros((height,width),np.uint8 if band=='SCL' else np.uint16)
                        reproject(raw,dest,src_transform=src.window_transform(window),src_crs=src.crs,dst_transform=affine,dst_crs=crs,
                                  src_nodata=0,dst_nodata=0,resampling=Resampling.nearest)
                if band=='SCL':scl=dest
                else:bands.append(dest)
                assets.append({'band':band,'unsigned_asset':href,'source_native_resolution_m':20 if band=='SCL' else 10})
                print(site,period,band,dest.shape,'prediction0',flush=True)
            profile={'driver':'GTiff','width':width,'height':height,'count':4,'dtype':'uint16','crs':crs,'transform':affine,'nodata':0,'compress':'deflate'}
            with rasterio.open(target,'w',**profile) as f:
                f.write(np.stack(bands));f.descriptions=('B02','B03','B04','B08');f.scales=tuple(calibration['scales']);f.offsets=tuple(calibration['reflectance_offsets'])
                f.update_tags(site_id=site,role=context['role'],acquisition_datetime=scene['properties']['datetime'],radiometry_source_xml_sha=file_sha256(xml),raw_DN_samples='preserved')
            profile.update(count=1,dtype='uint8')
            with rasterio.open(sclpath,'w',**profile) as f:f.write(scl,1);f.set_band_description(1,'SCL')
            record['images'][period]={'scene_id':scene['id'],'datetime':scene['properties']['datetime'],'shape':[height,width],
                 'sha256':file_sha256(target),'scl_sha256':file_sha256(sclpath),'product_XML_sha256':file_sha256(xml),'calibration':calibration,
                 'source_assets':assets,'SCL_obscured_fraction':float(np.isin(scl,[0,1,3,8,9,10,11]).mean()),
                 'constructed_surfaces_and_local_visibility_review_pending':True}
            write(ledger,record)
    print('Two unannotated winter mining pairs acquired; no normal/failure labels or model predictions')


if __name__=='__main__':sys.stdout.reconfigure(encoding='utf-8');run()
