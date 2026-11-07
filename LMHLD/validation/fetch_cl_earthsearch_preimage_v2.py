"""Fixed-date-quality fallback from a second open catalogue, keeping original post/labels.

Use latest-processing quality per capture; use documented legacy zero-offset COG
measurements for the same capture. Never guess PB>=4 COG offset application.
"""
from pathlib import Path
from datetime import datetime
import json,sys,requests,xml.etree.ElementTree as ET
ROOT=Path('D:/LMHLD');sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'cache/research_dependencies'))
from runtime_paths import configure_runtime_paths
configure_runtime_paths()
import numpy as np,rasterio
from rasterio.transform import Affine,array_bounds
from rasterio.warp import transform_bounds,reproject,Resampling,transform_geom
from rasterio.windows import from_bounds
from rasterio.features import rasterize
from shapely.geometry import shape,mapping
from shapely.ops import unary_union
from model_registry import file_sha256
from mine_data_protocol import registry,verify_assets
DATA=ROOT/'datasets/mine_benchmark_20261002';Q=DATA/'quality_audit/CL0002';OUT=Q/'chronology_safe_preimage_v2_earthsearch_20261005'
CAT=Q/'earthsearch_pre_window_catalogue_20261005.json'
def write(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def run():
 verify_assets()
 if registry()['cases']['CL0002']['model_predictions_run']!=0:raise ValueError('Reserved detector already run')
 OUT.mkdir(exist_ok=True)
 if (OUT/'selection.json').exists():print('Frozen secondary-catalogue selection exists');return
 p={'case_id':'CL0002','role':'reserved_final_test','catalogue_sha256':file_sha256(CAT),'event_pre_window':'2021-10-20/2021-11-07',
  'source_first_catalogue_returned_zero_scenes':True,'selection':'nearest fully covering capture before verified failure bracket;highestPB quality per capture;>=500 jointly observable erodedpositive and>=1000 annotationbackground',
  'SCL_excluded':[0,1,3,7,8,9,10,11],'boundary_m':30,'originalpost_multi_version_mask_preserved':True,
  'quality_algorithm_basis':'official Copernicus quality documentation explains newer processing reduces false cloud identification over bright surfaces;latestPB chosen uniformly,not by detector performance',
  'radiometry_rule':'prefer same-capture PB<4 zero-offset COG with frozen provider scale/offset and original product/granule metadata;PB>=4 COG offset ambiguity not silently corrected',
  'source_original_pixels_and_published_geometry_unchanged':True,'detector_predictions':0}
 pp=OUT/'protocol.json'
 if pp.exists() and json.loads(pp.read_text(encoding='utf-8'))!=p:raise ValueError('Secondary-catalogue quality plan changed')
 if not pp.exists():write(pp,p)
 post=Q/'exact_date_quality_v1_20261005/post_producer_BGRN_original_grid.tif'
 with rasterio.open(post) as f:profile=f.profile.copy();crs=f.crs;grid=f.transform;size=f.shape;bounds=f.bounds;clearpost=f.dataset_mask()>0
 for source in [Q/'exact_date_quality_v1_20261005/post_SCL_original_grid.tif',*sorted(Q.glob('official_post_SCL*.tif'))]:
  with rasterio.open(source) as f:clearpost &= ~np.isin(f.read(1),p['SCL_excluded'])
 geo=json.loads((Q/'published_boundary_wgs84.geojson').read_text(encoding='utf-8'));body=unary_union([shape(transform_geom('EPSG:4326',crs,x['geometry'])) for x in geo['features']])
 def m(g):return rasterize([(mapping(g),1)],out_shape=size,transform=grid,dtype='uint8')>0
 inner=m(body.buffer(-30));outer=m(body.buffer(30));features=json.loads(CAT.read_text(encoding='utf-8'))['features']
 def covers(s):
  a=s['assets']['blue'];b=transform_bounds(crs,f"EPSG:{s['properties']['proj:epsg']}",*bounds);ab=array_bounds(*a['proj:shape'],Affine(*a['proj:transform'][:6]));return ab[0]<=b[0] and ab[1]<=b[1] and ab[2]>=b[2] and ab[3]>=b[3]
 def read(s,key):
  # Serial small-range requests avoid partially returned merged TIFF tile ranges.
  # A public cache-busting query on retries changes no underlying source asset.
  for attempt in range(3):
   dest=np.zeros(size,np.uint8 if key=='scl' else np.uint16)
   href=s['assets'][key]['href'];request_href=href if attempt==0 else href+'?geoqa_retry='+str(attempt)
   try:
    with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN='EMPTY_DIR',GDAL_HTTP_TIMEOUT='30',GDAL_HTTP_MAX_RETRY='2',GDAL_HTTP_MULTIRANGE='SERIAL',VSI_CACHE='FALSE',CPL_VSIL_CURL_CHUNK_SIZE='16384',CPL_VSIL_CURL_ALLOWED_EXTENSIONS='.tif'):
     with rasterio.open(request_href) as f:
      w=from_bounds(*transform_bounds(crs,f.crs,*bounds),transform=f.transform).round_offsets().round_lengths();a=f.read(1,window=w,boundless=True,fill_value=0)
      reproject(a,dest,src_transform=f.window_transform(w),src_crs=f.crs,dst_transform=grid,dst_crs=crs,src_nodata=0,dst_nodata=0,resampling=Resampling.nearest)
    return dest
   except rasterio.errors.RasterioIOError:
    if attempt==2:
     if key=='scl':raise
     from verified_cog_window import read_native_cog_window
     asset=s['assets'][key];source_grid=Affine(*asset['proj:transform'][:6]);source_crs=f"EPSG:{s['properties']['proj:epsg']}"
     win=from_bounds(*transform_bounds(crs,source_crs,*bounds),transform=source_grid).round_offsets().round_lengths()
     raw,evidence=read_native_cog_window(asset['href'],int(win.row_off),int(win.col_off),int(win.height),int(win.width),ROOT/'cache/verified_tiff_ranges_20261005',asset['proj:shape'],asset['proj:transform'])
     reproject(raw,dest,src_transform=source_grid*Affine.translation(win.col_off,win.row_off),src_crs=source_crs,dst_transform=grid,dst_crs=crs,src_nodata=0,dst_nodata=0,resampling=Resampling.nearest)
     write(OUT/(s['id']+'_'+key+'_verified_transport.json'),evidence)
     return dest
 groups={}
 for s in features:
  if covers(s):groups.setdefault(s['properties']['datetime'][:10],[]).append(s)
 checks=[];chosen=None
 for date in sorted(groups,reverse=True):
  variants=groups[date];quality_scene=max(variants,key=lambda s:(float(s['properties']['s2:processing_baseline']),s['id']))
  legacy=[s for s in variants if float(s['properties']['s2:processing_baseline'])<4 and all(abs(s['assets'][key]['raster:bands'][0].get('offset',0))<1e-12 for key in ('blue','green','red','nir'))]
  if not legacy:checks.append({'date':date,'eligible':False,'reason':'No unambiguous zero-offset original COG measurements'});continue
  measure=max(legacy,key=lambda s:s['id']);scl=read(quality_scene,'scl');clear=clearpost & ~np.isin(scl,p['SCL_excluded']);pos=int((clear&inner).sum());neg=int((clear&~outer).sum())
  sp=OUT/(date+'_latest_SCL.tif')
  with rasterio.open(sp,'w',**{**profile,'count':1,'dtype':'uint8','nodata':0}) as f:f.write(scl,1)
  row={'date':date,'quality_scene':quality_scene['id'],'quality_processing_baseline':quality_scene['properties']['s2:processing_baseline'],'measurement_scene':measure['id'],'measurement_processing_baseline':measure['properties']['s2:processing_baseline'],'positive_support':pos,'annotation_background_support':neg,'eligible':pos>=500 and neg>=1000};checks.append(row)
  write(OUT/'quality_checks.json',{'protocol_sha256':file_sha256(pp),'checks':checks,'detector_predictions':0});print(date,pos,neg,'latestQC/legacyDN;model0',flush=True)
  if row['eligible']:chosen=measure;chosen_quality=quality_scene;chosen_scl=scl;break
 if chosen is None:
  write(OUT/'selection.json',{'accepted':False,'checks':checks,'detector_predictions':0});return
 original_tile=chosen['assets']['granule_metadata']['href'];r=requests.get(original_tile,timeout=(10,30));r.raise_for_status();(OUT/'pre_granule.xml').write_bytes(r.content)
 root=ET.fromstring(r.content);sense=next(n.text.strip() for n in root.iter() if n.tag.rsplit('}',1)[-1]=='SENSING_TIME')
 if datetime.fromisoformat(sense.replace('Z','+00:00'))>=datetime.fromisoformat('2021-11-08T13:55:00+00:00'):raise ValueError('Actual tile too late for pre event bracket')
 bands=[read(chosen,key) for key in ('blue','green','red','nir')];scales=[chosen['assets'][k]['raster:bands'][0]['scale'] for k in ('blue','green','red','nir')]
 image=OUT/'pre_producer_BGRN_original_grid.tif';qc=OUT/'pre_SCL_original_grid.tif'
 with rasterio.open(image,'w',**profile) as f:
  f.write(np.stack(bands));f.scales=tuple(scales);f.offsets=(0,0,0,0);f.descriptions=('B02','B03','B04','B08');f.update_tags(case_id='CL0002',role='reserved_final_test',acquisition_datetime=sense,radiometry_source='zero-offset PB<4 provider rasterband metadata;pixel values unaltered')
 with rasterio.open(qc,'w',**{**profile,'count':1,'dtype':'uint8','nodata':0}) as f:f.write(chosen_scl,1);f.set_band_description(1,'SCL')
 write(OUT/'measurement_scene.json',chosen);write(OUT/'quality_scene.json',chosen_quality)
 v={'accepted':True,'case_id':'CL0002','role':'reserved_final_test','tile_sensing_time':sense,'measurement_scene_id':chosen['id'],'measurement_processing_baseline':chosen['properties']['s2:processing_baseline'],
  'quality_scene_id':chosen_quality['id'],'quality_processing_baseline':chosen_quality['properties']['s2:processing_baseline'],'same_capture_different_processing_quality_explicit':True,
  'radiometry':{'scales':scales,'effective_offsets':[0,0,0,0],'provider_metadata_verified':True,'full_original_product_XML_downloaded':False,'physical_indices_not_asserted_from_ambiguous_PB>=4_COG':True},
  'selection_protocol_sha256':file_sha256(pp),'quality_checks_sha256':file_sha256(OUT/'quality_checks.json'),'assets_sha256':{p.name:file_sha256(p) for p in (image,qc,OUT/'pre_granule.xml',OUT/'measurement_scene.json',OUT/'quality_scene.json')},
  'original_post_vector_and_reference_mask_unchanged':True,'new_test_group':False,'new_detector_predictions':0}
 write(OUT/'selection.json',v);print('Chronology-safe accepted pre source',sense,'all originalpost/labels intact',flush=True)
if __name__=='__main__':sys.stdout.reconfigure(encoding='utf-8');run()
