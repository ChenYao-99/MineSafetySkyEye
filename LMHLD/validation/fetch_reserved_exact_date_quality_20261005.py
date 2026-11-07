"""Fetch same-capture quality/producer metadata without applying modern coefficients to legacy DN."""
from pathlib import Path
from datetime import datetime
import json,sys
ROOT=Path('D:/LMHLD');sys.path.insert(0,str(ROOT))
from runtime_paths import configure_runtime_paths
configure_runtime_paths()
import numpy as np,rasterio,requests
from rasterio.transform import Affine,array_bounds
from rasterio.warp import transform_bounds,reproject,Resampling
from rasterio.windows import from_bounds
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from radiometry import parse_xml
from model_registry import file_sha256
from mine_data_protocol import registry,verify_assets,assert_mine_group_role
DATA=ROOT/'datasets/mine_benchmark_20261002'
RAW=DATA/'raw_sources/dlr/s1s2_landslide_reference_data/original_scenes'
def write(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def run():
 value=registry();verify_assets(value)
 session=requests.Session();session.mount('https://',HTTPAdapter(max_retries=Retry(total=3,backoff_factor=1,status_forcelist=(429,500,502,503,504))))
 for eid in ('CL0002','KG0001'):
  assert_mine_group_role(eid,'reserved_final_test',value)
  if value['cases'][eid]['model_predictions_run']!=0:raise ValueError('Reserved event already observed by detector')
  audit=json.loads((DATA/f'quality_audit/{eid}/audit.json').read_text(encoding='utf-8'))
  out=DATA/f'quality_audit/{eid}/exact_date_quality_v1_20261005';out.mkdir(exist_ok=True)
  with rasterio.open(RAW/eid/f'{eid}__None__None__MASK.tif') as f:profile=f.profile.copy();crs=f.crs;grid=f.transform;shape=f.shape;bounds=f.bounds
  bbox=transform_bounds(crs,'EPSG:4326',*bounds);manifest=out/'manifest.json'
  m=json.loads(manifest.read_text(encoding='utf-8')) if manifest.exists() else {'created_at':datetime.now().isoformat(timespec='seconds'),'case_id':eid,'original_audit_sha256':file_sha256(DATA/f'quality_audit/{eid}/audit.json'),'images':{},'mine_detector_predictions':0,'original_DN_unchanged':True}
  def sign(href):
   r=session.get('https://planetarycomputer.microsoft.com/api/sas/v1/sign',params={'href':href},timeout=(10,30));r.raise_for_status();return r.json()['href']
  for period in ('pre','post'):
   if period in m['images']:
    for name,digest in m['images'][period]['assets_sha256'].items():
     if file_sha256(out/name)!=digest:raise ValueError('Frozen exact-date quality changed')
    continue
   date=audit[period+'_date'];cat=out/f'{period}_catalogue.json'
   if not cat.exists():
    r=session.get('https://planetarycomputer.microsoft.com/api/stac/v1/search',params={'collections':'sentinel-2-l2a','bbox':','.join(map(str,bbox)),'datetime':date+'T00:00:00Z/'+date+'T23:59:59Z','limit':100},timeout=(10,30));r.raise_for_status();write(cat,r.json())
   catalogue=json.loads(cat.read_text(encoding='utf-8'))
   if any(l.get('rel')=='next' for l in catalogue.get('links',[])):raise ValueError('Unresolved exact-date catalogue pagination')
   def covers(s):
    asset=s['assets']['SCL'];prop=s['properties'];a=array_bounds(*asset.get('proj:shape',prop.get('proj:shape')),Affine(*asset.get('proj:transform',prop.get('proj:transform'))[:6]));b=transform_bounds(crs,f"EPSG:{prop['proj:epsg']}",*bounds)
    return a[0]<=b[0] and a[1]<=b[1] and a[2]>=b[2] and a[3]>=b[3]
   scenes=[s for s in catalogue['features'] if covers(s)]
   if not scenes:raise ValueError('No covering scene on original date')
   scene=max(scenes,key=lambda s:(str(s['properties'].get('s2:processing_baseline','')),s['id']))
   xml=out/f'{period}_modern_product.xml';r=session.get(sign(scene['assets']['product-metadata']['href']),timeout=(10,30));r.raise_for_status();cal=parse_xml(r.content);xml.write_bytes(r.content)
   dest=np.zeros(shape,np.uint8);href=scene['assets']['SCL']['href']
   with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN='EMPTY_DIR',GDAL_HTTP_VERSION='1.1',GDAL_HTTP_TIMEOUT='30',GDAL_HTTP_MAX_RETRY='2',CPL_VSIL_CURL_ALLOWED_EXTENSIONS='.tif'):
    with rasterio.open(sign(href)) as f:
     window=from_bounds(*transform_bounds(crs,f.crs,*bounds),transform=f.transform).round_offsets().round_lengths();src=f.read(1,window=window,boundless=True,fill_value=0)
     reproject(src,dest,src_transform=f.window_transform(window),src_crs=f.crs,dst_transform=grid,dst_crs=crs,src_nodata=0,dst_nodata=0,resampling=Resampling.nearest)
   quality=out/f'{period}_SCL_original_grid.tif'
   with rasterio.open(quality,'w',**{**profile,'count':1,'dtype':'uint8','nodata':0,'compress':'deflate'}) as f:f.write(dest,1);f.set_band_description(1,'SCL')
   m['images'][period]={'scene_id':scene['id'],'datetime':scene['properties']['datetime'],'unsigned_source_SCL':href,'processing_baseline':cal['processing_baseline'],
    'quality_applicability':'same acquisition date;modern processing may differ from original published pixels;provider version disagreement preserved',
    'modern_coefficients_not_applied_to_published_original_DN':True,'modern_product_calibration':cal,
    'assets_sha256':{p.name:file_sha256(p) for p in (xml,quality)},'SCL_not_expert_visibility_truth':True}
   write(manifest,m);print(eid,period,date,scene['id'],'qualityonly detector0',flush=True)
 print('All4 exact-date quality assets retrieved; original pixels intact; no detector predictions')
if __name__=='__main__':sys.stdout.reconfigure(encoding='utf-8');run()
