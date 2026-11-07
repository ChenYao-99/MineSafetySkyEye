"""Official SCL on both exact original acquisition dates, no model or label changes."""
from pathlib import Path
from datetime import datetime
import json,sys,urllib.parse
import numpy as np
import rasterio
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from rasterio.warp import transform_bounds,reproject,Resampling
from rasterio.windows import from_bounds
from rasterio.transform import Affine,array_bounds
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from runtime_paths import configure_runtime_paths
configure_runtime_paths()
from model_registry import file_sha256
from radiometry import parse_xml
DATA=ROOT/'datasets/mine_benchmark_20261002'
RAW=DATA/'raw_sources/dlr/s1s2_landslide_reference_data/original_scenes/BR0001'
OUT=DATA/'development/Brumadinho/exact_date_quality_20261004'


def write(p,v):p.write_bytes((json.dumps(v,ensure_ascii=False,indent=2)+'\n').encode('utf-8'))


def run():
    OUT.mkdir(exist_ok=True);session=requests.Session()
    session.mount('https://',HTTPAdapter(max_retries=Retry(total=3,backoff_factor=1,status_forcelist=(429,500,502,503,504),allowed_methods=frozenset(('GET',)))))
    maskpath=RAW/'BR0001__None__None__MASK.tif'
    with rasterio.open(maskpath) as f:profile=f.profile.copy();native_crs=f.crs;grid=f.transform;shape=f.shape;truth=f.read(1)>0;bounds=f.bounds
    bbox=transform_bounds(native_crs,'EPSG:4326',*bounds)
    path=OUT/'manifest.json';p=json.loads(path.read_text(encoding='utf-8')) if path.exists() else {'created_at':datetime.now().isoformat(timespec='seconds'),'mask_sha':file_sha256(maskpath),'images':{},'new_model_predictions':0}
    def signed(href):
        r=session.get('https://planetarycomputer.microsoft.com/api/sas/v1/sign',params={'href':href},timeout=(10,30));r.raise_for_status();return r.json()['href']
    for period,date in (('pre','2019-01-07'),('post','2019-02-01')):
        if period in p['images']:
            for name,digest in p['images'][period]['assets'].items():
                if file_sha256(OUT/name)!=digest:raise ValueError('Frozen exact-date quality changed')
            continue
        cat=OUT/(period+'_catalogue.json')
        if not cat.exists():
            r=session.get('https://planetarycomputer.microsoft.com/api/stac/v1/search',params={'collections':'sentinel-2-l2a','bbox':','.join(map(str,bbox)),'datetime':date+'T00:00:00Z/'+date+'T23:59:59Z','limit':100},timeout=(10,30));r.raise_for_status();write(cat,r.json())
        catalogue=json.loads(cat.read_text(encoding='utf-8'))
        if any(x.get('rel')=='next' for x in catalogue.get('links',[])):raise ValueError('Unresolved exact-date catalogue pagination')
        def coverage(scene):
            asset=scene['assets']['SCL'];prop=scene['properties'];ab=array_bounds(*asset.get('proj:shape',prop.get('proj:shape')),Affine(*asset.get('proj:transform',prop.get('proj:transform'))[:6]))
            mb=transform_bounds(native_crs,f"EPSG:{prop['proj:epsg']}",*bounds)
            return max(0,min(ab[2],mb[2])-max(ab[0],mb[0]))*max(0,min(ab[3],mb[3])-max(ab[1],mb[1]))/((mb[2]-mb[0])*(mb[3]-mb[1]))
        eligible=[s for s in catalogue['features'] if coverage(s)>=.999]
        if not eligible:raise ValueError('No complete exact-date coverage')
        scene=sorted(eligible,key=lambda s:s['id'])[0]
        xml=OUT/(period+'_product.xml');r=session.get(signed(scene['assets']['product-metadata']['href']),timeout=(10,30));r.raise_for_status();cal=parse_xml(r.content);xml.write_bytes(r.content)
        dest=np.zeros(shape,np.uint8);href=scene['assets']['SCL']['href']
        with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN='EMPTY_DIR',GDAL_HTTP_VERSION='1.1',GDAL_HTTP_TIMEOUT='30',GDAL_HTTP_MAX_RETRY='2',CPL_VSIL_CURL_ALLOWED_EXTENSIONS='.tif'):
            with rasterio.open(signed(href)) as f:
                window=from_bounds(*transform_bounds(native_crs,f.crs,*bounds),transform=f.transform).round_offsets().round_lengths();src=f.read(1,window=window,boundless=True,fill_value=0)
                reproject(src,dest,src_transform=f.window_transform(window),src_crs=f.crs,dst_transform=grid,dst_crs=native_crs,src_nodata=0,dst_nodata=0,resampling=Resampling.nearest)
        sclpath=OUT/(period+'_SCL_original_grid.tif')
        with rasterio.open(sclpath,'w',**{**profile,'count':1,'dtype':'uint8','nodata':0,'compress':'deflate'}) as f:f.write(dest,1);f.set_band_description(1,'SCL')
        p['images'][period]={'scene_id':scene['id'],'datetime':scene['properties']['datetime'],'unsigned_asset':href,
          'asset_scope':'actual same-date quality product, processing version may differ from published originalDN','calibration':cal,
          'assets':{xml.name:file_sha256(xml),sclpath.name:file_sha256(sclpath)},'original_DN_unchanged':True,
          'foreground_weather_clear_fraction':float((~np.isin(dest[truth],[0,1,3,8,9,10,11])).mean()),
          'SCL_classes_not_expert_visibility_truth':True}
        write(path,p);print(period,date,p['images'][period]['foreground_weather_clear_fraction'],flush=True)
    print('Both exact original-date quality sources archived; model predictions0')


if __name__=='__main__':sys.stdout.reconfigure(encoding='utf-8');run()
