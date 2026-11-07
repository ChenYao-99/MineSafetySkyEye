"""Resolve tile sensing times and byte compare publisher originals to actual producer bands."""
from pathlib import Path
from datetime import datetime
import json,sys,xml.etree.ElementTree as ET
ROOT=Path('D:/LMHLD');sys.path.insert(0,str(ROOT))
from runtime_paths import configure_runtime_paths
configure_runtime_paths()
import numpy as np,rasterio,requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from rasterio.windows import from_bounds
from rasterio.warp import transform_bounds,reproject,Resampling
from model_registry import file_sha256
from mine_data_protocol import registry,verify_assets
DATA=ROOT/'datasets/mine_benchmark_20261002';RAW=DATA/'raw_sources/dlr/s1s2_landslide_reference_data/original_scenes'
def write(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def run():
 verify_assets();session=requests.Session()
 session.mount('https://',HTTPAdapter(max_retries=Retry(total=4,backoff_factor=2,status_forcelist=(429,500,502,503,504),allowed_methods=frozenset(('GET',)))))
 def sign(href):
  r=session.get('https://planetarycomputer.microsoft.com/api/sas/v1/sign',params={'href':href},timeout=(10,30));r.raise_for_status();return r.json()['href']
 for eid in ('CL0002','KG0001'):
  if registry()['cases'][eid]['model_predictions_run']!=0:raise ValueError('Reserved detector already run')
  out=DATA/f'quality_audit/{eid}/exact_date_quality_v1_20261005';m=json.loads((out/'manifest.json').read_text(encoding='utf-8'));result={'created_at':datetime.now().isoformat(timespec='seconds'),'case_id':eid,'quality_manifest_sha256':file_sha256(out/'manifest.json'),'images':{},'mine_detector_predictions':0,'source_published_DN_unchanged':True}
  dest=out/'original_DN_and_tile_time_verification.json'
  if dest.exists():
   prior=json.loads(dest.read_text(encoding='utf-8'))
   for p,digest in prior['frozen_assets'].items():
    if file_sha256(out/p)!=digest:raise ValueError('Frozen source verification changed')
   print(eid,'frozen DN/time verification reused');continue
  for period in ('pre','post'):
   scene=next(s for s in json.loads((out/f'{period}_catalogue.json').read_text(encoding='utf-8'))['features'] if s['id']==m['images'][period]['scene_id'])
   tl=out/f'{period}_granule.xml';r=session.get(sign(scene['assets']['granule-metadata']['href']),timeout=(10,30));r.raise_for_status();tl.write_bytes(r.content)
   tags={}
   for n in ET.fromstring(r.content).iter():
    if n.tag.rsplit('}',1)[-1] in ('SENSING_TIME','TILE_ID','DATASTRIP_ID') and n.text:tags[n.tag.rsplit('}',1)[-1]]=n.text.strip()
   row={'scene_id':scene['id'],'product_datatake_start_time':scene['properties']['datetime'],'tile_metadata_fields':tags,'bands':{},'calibration_from_actual_matching_source':m['images'][period]['modern_product_calibration']}
   bands=[]
   for band in ('B02','B03','B04','B08'):
    path=RAW/eid/f"{eid}__S2L2A__{m['images'][period]['datetime'][:10]}__{band}.tif"
    with rasterio.open(path) as f:original=f.read(1);profile=f.profile.copy();crs=f.crs;grid=f.transform;bounds=f.bounds;valid=f.read_masks(1)>0
    array=np.zeros(original.shape,np.uint16);href=scene['assets'][band]['href']
    with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN='EMPTY_DIR',GDAL_HTTP_VERSION='1.1',GDAL_HTTP_TIMEOUT='30',GDAL_HTTP_MAX_RETRY='2',CPL_VSIL_CURL_ALLOWED_EXTENSIONS='.tif'):
     with rasterio.open(sign(href)) as f:
      win=from_bounds(*transform_bounds(crs,f.crs,*bounds),transform=f.transform).round_offsets().round_lengths();src=f.read(1,window=win,boundless=True,fill_value=0)
      reproject(src,array,src_transform=f.window_transform(win),src_crs=f.crs,dst_transform=grid,dst_crs=crs,src_nodata=0,dst_nodata=0,resampling=Resampling.nearest)
    common=valid & (original>0) & (array>0);delta=array.astype(np.int32)-original.astype(np.int32)
    row['bands'][band]={'original_published_sha256':file_sha256(path),'unsigned_current_source':href,'identical_all_original_grid_pixels':bool(np.array_equal(original,array)),
     'common_valid_pixels':int(common.sum()),'median_DN_difference':float(np.median(delta[common])),'p99_absolute_DN_difference':float(np.percentile(abs(delta[common]),99))}
    bands.append(array);print(eid,period,band,'exact_original_match',row['bands'][band]['identical_all_original_grid_pixels'],flush=True)
   target=out/f'{period}_producer_BGRN_original_grid.tif';cal=row['calibration_from_actual_matching_source']
   with rasterio.open(target,'w',**{**profile,'count':4,'dtype':'uint16','nodata':0,'compress':'deflate'}) as f:
    f.write(np.stack(bands));f.descriptions=('B02','B03','B04','B08');f.scales=tuple(cal['scales']);f.offsets=tuple(cal['reflectance_offsets'])
    f.update_tags(case_id=eid,role='reserved_final_test',acquisition_datetime=tags.get('SENSING_TIME',''),source_product_start_time=scene['properties']['datetime'],radiometry_source_xml_sha=file_sha256(out/f'{period}_modern_product.xml'),source='actual producer pixels;publisher originals preserved')
   row['current_producer_image_sha256']=file_sha256(target);row['original_all4bands_byte_match']=all(x['identical_all_original_grid_pixels'] for x in row['bands'].values())
   result['images'][period]=row
  result['frozen_assets']={p.name:file_sha256(p) for p in out.iterdir() if p.name.endswith('_granule.xml') or p.name.endswith('_producer_BGRN_original_grid.tif')}
  write(dest,result);print(eid,'tile_sensing_times',{p:r['tile_metadata_fields'].get('SENSING_TIME') for p,r in result['images'].items()},flush=True)
 print('Original producer metadata/time and16band comparisons finished;no detector outputs')
if __name__=='__main__':sys.stdout.reconfigure(encoding='utf-8');run()
