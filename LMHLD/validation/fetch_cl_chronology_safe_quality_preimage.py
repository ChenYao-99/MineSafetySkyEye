"""Quality-only pre-event fallback; original post/labels and all old assets remain unchanged."""
from pathlib import Path
from datetime import datetime
import json,sys
ROOT=Path('D:/LMHLD');sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'cache/research_dependencies'))
from runtime_paths import configure_runtime_paths
configure_runtime_paths()
import numpy as np,rasterio,requests
from rasterio.transform import Affine,array_bounds
from rasterio.warp import transform_bounds,reproject,Resampling,transform_geom
from rasterio.features import rasterize
from rasterio.windows import from_bounds
from shapely.geometry import shape,mapping
from shapely.ops import unary_union
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from radiometry import parse_xml
from model_registry import file_sha256
from mine_data_protocol import verify_assets,registry
DATA=ROOT/'datasets/mine_benchmark_20261002';Q=DATA/'quality_audit/CL0002';OUT=Q/'chronology_safe_preimage_v1_20261005'
def write(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def run():
 verify_assets()
 if registry()['cases']['CL0002']['model_predictions_run']!=0:raise ValueError('Reserved detector already run')
 OUT.mkdir(exist_ok=True)
 if (OUT/'selection.json').exists():print('Frozen preimage selection exists;no reselection');return
 p={'case_id':'CL0002','role':'reserved_final_test','interval':'2021-10-20T00:00:00Z/2021-11-07T23:59:59Z',
  'max_date_strictly_before_last_unchanged_primary_2021_11_08_1355Z':True,
  'selection':'closest fully covering pre-event scene with>=500 positive reference interiors and>=1000 annotation background jointly clear with unchanged originalpost all-provider QC; processingbaseline/ID breaks ties',
  'SCL_excluded':[0,1,3,7,8,9,10,11],'boundary_m':30,'original_post_or_published_geometry_changed':False,'model_scores_used':False,
  'initial_original_pair_failed_quality_record_sha256':file_sha256(Q/'published_inventory_benchmark_v1_20261005/review_pending.json')}
 plan=OUT/'protocol.json'
 if plan.exists() and json.loads(plan.read_text(encoding='utf-8'))!=p:raise ValueError('Preimage quality protocol changed')
 if not plan.exists():write(plan,p)
 post=Q/'exact_date_quality_v1_20261005/post_producer_BGRN_original_grid.tif'
 with rasterio.open(post) as f:profile=f.profile.copy();crs=f.crs;grid=f.transform;size=f.shape;bounds=f.bounds;post_valid=f.dataset_mask()>0
 geo=json.loads((Q/'published_boundary_wgs84.geojson').read_text(encoding='utf-8'));body=unary_union([shape(transform_geom('EPSG:4326',crs,f['geometry'])) for f in geo['features']])
 inner=rasterize([(mapping(body.buffer(-30)),1)],out_shape=size,transform=grid,dtype='uint8')>0
 outer=rasterize([(mapping(body.buffer(30)),1)],out_shape=size,transform=grid,dtype='uint8')>0
 clearpost=post_valid.copy()
 for source in [Q/'exact_date_quality_v1_20261005/post_SCL_original_grid.tif',*sorted(Q.glob('official_post_SCL*.tif'))]:
  with rasterio.open(source) as f:clearpost &= ~np.isin(f.read(1),p['SCL_excluded'])
 session=requests.Session();session.mount('https://',HTTPAdapter(max_retries=Retry(total=4,backoff_factor=2,status_forcelist=(429,500,502,503,504))))
 def sign(href):
  r=session.get('https://planetarycomputer.microsoft.com/api/sas/v1/sign',params={'href':href},timeout=(10,30));r.raise_for_status();return r.json()['href']
 def band(s,name):
  array=np.zeros(size,np.uint8 if name=='SCL' else np.uint16)
  with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN='EMPTY_DIR',GDAL_HTTP_VERSION='1.1',GDAL_HTTP_TIMEOUT='30',GDAL_HTTP_MAX_RETRY='2',CPL_VSIL_CURL_ALLOWED_EXTENSIONS='.tif'):
   with rasterio.open(sign(s['assets'][name]['href'])) as f:
    w=from_bounds(*transform_bounds(crs,f.crs,*bounds),transform=f.transform).round_offsets().round_lengths();x=f.read(1,window=w,boundless=True,fill_value=0)
    reproject(x,array,src_transform=f.window_transform(w),src_crs=f.crs,dst_transform=grid,dst_crs=crs,src_nodata=0,dst_nodata=0,resampling=Resampling.nearest)
  return array
 cat=OUT/'catalogue.json'
 if not cat.exists():
  r=session.get('https://planetarycomputer.microsoft.com/api/stac/v1/search',params={'collections':'sentinel-2-l2a','bbox':','.join(map(str,transform_bounds(crs,'EPSG:4326',*bounds))),'datetime':p['interval'],'limit':100},timeout=(10,30));r.raise_for_status();write(cat,r.json())
 catalogue=json.loads(cat.read_text(encoding='utf-8'))
 if any(l.get('rel')=='next' for l in catalogue.get('links',[])):raise ValueError('Unresolved quality catalogue pagination')
 def covers(s):
  a=s['assets']['B02'];prop=s['properties'];ab=array_bounds(*a.get('proj:shape',prop.get('proj:shape')),Affine(*a.get('proj:transform',prop.get('proj:transform'))[:6]));b=transform_bounds(crs,f"EPSG:{prop['proj:epsg']}",*bounds)
  return ab[0]<=b[0] and ab[1]<=b[1] and ab[2]>=b[2] and ab[3]>=b[3]
 scenes=sorted([s for s in catalogue['features'] if covers(s)],key=lambda s:(s['properties']['datetime'],str(s['properties'].get('s2:processing_baseline','')),s['id']),reverse=True)
 checks=[];chosen=None
 write(OUT/'quality_checks.json',{'plan_sha256':file_sha256(plan),'catalogue_sha256':file_sha256(cat),'covering_scene_count':len(scenes),'checks':checks,'model_predictions':0})
 for scene in scenes:
  path=OUT/(scene['id']+'_SCL.tif')
  if path.exists():
   with rasterio.open(path) as f:scl=f.read(1)
  else:
   scl=band(scene,'SCL')
   with rasterio.open(path,'w',**{**profile,'count':1,'dtype':'uint8','nodata':0}) as f:f.write(scl,1)
  support=clearpost & ~np.isin(scl,p['SCL_excluded']);positive=int((support & inner).sum());negative=int((support & ~outer).sum())
  row={'scene_id':scene['id'],'product_start_time':scene['properties']['datetime'],'scored_positive_support':positive,'scored_annotation_background_support':negative,'SCL_sha256':file_sha256(path),'eligible':positive>=500 and negative>=1000};checks.append(row)
  write(OUT/'quality_checks.json',{'plan_sha256':file_sha256(plan),'catalogue_sha256':file_sha256(cat),'checks':checks,'model_predictions':0})
  print(scene['id'],positive,negative,'qualityonly',flush=True)
  if row['eligible']:chosen=scene;chosen_scl=scl;break
 if chosen is None:
  write(OUT/'selection.json',{'accepted':False,'quality_checks_sha256':file_sha256(OUT/'quality_checks.json'),'reason':'No physically qualified exact-date preimage in frozen event window','model_predictions':0});return
 row={'accepted':True,'case_id':'CL0002','role':'reserved_final_test','scene_id':chosen['id'],'product_start_time':chosen['properties']['datetime'],'quality_checks_sha256':file_sha256(OUT/'quality_checks.json'),'source_plan_sha256':file_sha256(plan)}
 for key,file in [('product-metadata','pre_product.xml'),('granule-metadata','pre_granule.xml')]:
  r=session.get(sign(chosen['assets'][key]['href']),timeout=(10,30));r.raise_for_status();(OUT/file).write_bytes(r.content)
 cal=parse_xml((OUT/'pre_product.xml').read_bytes());tl=ET.fromstring((OUT/'pre_granule.xml').read_bytes());sense=next(n.text.strip() for n in tl.iter() if n.tag.rsplit('}',1)[-1]=='SENSING_TIME')
 if datetime.fromisoformat(sense.replace('Z','+00:00'))>=datetime.fromisoformat('2021-11-08T13:55:00+00:00'):raise ValueError('Actual tile sensing too late for pre-event bracket')
 arrays=[band(chosen,b) for b in ('B02','B03','B04','B08')]
 image=OUT/'pre_producer_BGRN_original_grid.tif';quality=OUT/'pre_SCL_original_grid.tif'
 with rasterio.open(image,'w',**profile) as f:
  f.write(np.stack(arrays));f.scales=tuple(cal['scales']);f.offsets=tuple(cal['reflectance_offsets']);f.descriptions=('B02','B03','B04','B08');f.update_tags(case_id='CL0002',role='reserved_final_test',acquisition_datetime=sense,source='actual chronology-safe producer;old originals intact')
 with rasterio.open(quality,'w',**{**profile,'count':1,'dtype':'uint8','nodata':0}) as f:f.write(chosen_scl,1);f.set_band_description(1,'SCL')
 row.update(tile_sensing_time=sense,calibration=cal,assets_sha256={a.name:file_sha256(a) for a in (image,quality,OUT/'pre_product.xml',OUT/'pre_granule.xml')},original_pre_post_or_labels_unchanged=True,model_predictions=0)
 write(OUT/'selection.json',row);print('Accepted chronology-safe preimage source',sense,'no detector0',flush=True)
if __name__=='__main__':sys.stdout.reconfigure(encoding='utf-8');run()
