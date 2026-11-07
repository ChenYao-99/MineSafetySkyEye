"""Distinguish post-event reference observability from paired-change input support.

The published label is a post-event footprint, not a declaration that every pixel
changed since pre. Post-only segmentation and pair-with-missingness models can use
post-observed labels. Strict paired-change metrics have their own smaller support.
"""
from pathlib import Path
from datetime import datetime
import json,sys
ROOT=Path('D:/LMHLD');sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'cache/research_dependencies'));sys.path.insert(0,str(ROOT/'validation'))
from runtime_paths import configure_runtime_paths
configure_runtime_paths()
import cv2,numpy as np,rasterio
from rasterio.features import rasterize
from rasterio.warp import transform_geom
from shapely.geometry import shape,mapping
from shapely.ops import unary_union
from PIL import Image,ImageDraw
from model_registry import file_sha256
from mine_data_protocol import registry,verify_assets
from qualify_published_inventory_benchmarks_20261005 import ground,EXCLUSIONS
from prepare_winter_constructed_normals import display
DATA=ROOT/'datasets/mine_benchmark_20261002';Q=DATA/'quality_audit/CL0002';PRE=Q/'chronology_safe_preimage_v3_consistent_quality_20261005';POST=Q/'exact_date_quality_v1_20261005'
OUT=Q/'published_inventory_benchmark_v2_post_and_pair_20261005'
def write(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def run():
 verify_assets()
 if registry()['cases']['CL0002']['model_predictions_run']!=0:raise ValueError('Reserved detector already run')
 selection=json.loads((PRE/'selection.json').read_text(encoding='utf-8'))
 if not selection['accepted']:raise ValueError('No accepted chronological source preimage')
 OUT.mkdir(exist_ok=True)
 if (OUT/'review_pending.json').exists():print('Frozen scoped review already exists');return
 protocol={'case_id':'CL0002','role':'reserved_final_test','target_label_epoch':'post','target_definition':'independent published post-event mapped footprint;not change/newfailure at every pixel since pre',
  'metric_support':{'post_event_segmentation':'post-observed erodedforeground/dilatedoutside annotationbackground','paired_change':'same fixed reference interior where both pre/post observed'},
  'input_model_admission':['post_only','pair_with_encoded_missingness'],'paired_change_only_models_must_use_paired_support':True,
  'primary_quality_policy':'consistent latest same-capture processing quality chosen byPB not classifier performance;all original masks retained as sensitivity',
  'official_quality_basis':'https://sentiwiki.copernicus.eu/web/s2-processing','older_classifier_falsecloud_on_bright_targets_documentation':'https://sentiwiki.copernicus.eu/__attachments/1673423/OMPC.CS.DQR.002.12-2022%20-%20MSI%20L2A%20DQR%20January%202023%20-%2057.0.pdf',
  'boundary_m':30,'SCL_invalid':[0,1,3,7,8,9,10,11],'positive_pixels_min':500,'annotation_background_pixels_min':1000,
  'registration_gate':{'unique_min':90,'fit_min':30,'heldout':30,'unshifted_median_px_max':1,'unshifted_p90_px_max':2,'within2px_min':.9},
  'detector_outputs_used_for_data_or_support_selection':False,'original_published_pixels_polygon_or_case_group_changed':False,'absolute_field_or_safety_accuracy_claim_allowed':False}
 write(OUT/'protocol.json',protocol)
 prepath=PRE/'pre_producer_BGRN_original_grid.tif';postpath=POST/'post_producer_BGRN_original_grid.tif'
 pre,pv=display(prepath);post,av=display(postpath)
 with rasterio.open(postpath) as f:profile=f.profile.copy();crs=f.crs;grid=f.transform;size=f.shape
 with rasterio.open(prepath) as f:
  if f.crs!=crs or f.transform!=grid or f.shape!=size:raise ValueError('Actual pair grids differ')
 with rasterio.open(PRE/'pre_SCL_original_grid.tif') as f:preclear=pv & ~np.isin(f.read(1),protocol['SCL_invalid'])
 with rasterio.open(Q/'official_post_SCL_sequence1.tif') as f:postclear=av & ~np.isin(f.read(1),protocol['SCL_invalid'])
 paired=preclear & postclear
 geo=json.loads((Q/'published_boundary_wgs84.geojson').read_text(encoding='utf-8'));body=unary_union([shape(transform_geom('EPSG:4326',crs,x['geometry'])) for x in geo['features']])
 def mask(g):return rasterize([(mapping(g),1)],out_shape=size,transform=grid,dtype='uint8')>0
 foot=mask(body);inside=mask(body.buffer(-30));outside=~mask(body.buffer(30))
 afterlabels=np.full(size,255,np.uint8);afterlabels[inside & postclear]=1;afterlabels[outside & postclear]=0
 pairlabels=np.full(size,255,np.uint8);pairlabels[inside & paired]=1;pairlabels[outside & paired]=0
 ga=ground(pre,post,paired,foot,EXCLUSIONS['CL0002'],OUT);write(OUT/'ground_registration.json',ga)
 times=json.loads((POST/'original_DN_and_tile_time_verification.json').read_text(encoding='utf-8'))['images'];posttime=times['post']['tile_metadata_fields']['SENSING_TIME'];pretime=selection['tile_sensing_time']
 chronological=datetime.fromisoformat(pretime.replace('Z','+00:00'))<datetime.fromisoformat('2021-11-08T13:55:00+00:00') and datetime.fromisoformat(posttime.replace('Z','+00:00'))>datetime.fromisoformat('2021-11-09T14:42:00+00:00')
 for name,a in [('labels_post_reference_interior.tif',afterlabels),('labels_paired_reference_interior.tif',pairlabels),('post_observable_support.tif',postclear.astype(np.uint8)),('paired_observable_support.tif',paired.astype(np.uint8))]:
  with rasterio.open(OUT/name,'w',**{**profile,'count':1,'dtype':'uint8','nodata':255 if name.startswith('labels') else None}) as f:
   f.write(a,1);f.update_tags(case_id='CL0002',role='reserved_final_test',target_epoch='post',reference_kind='independent_published_postevent_inventory',field_safety_truth='false')
 canvas=Image.new('RGB',(size[1]*2,size[0]+45),'white');draw=ImageDraw.Draw(canvas)
 for i,(name,img,support) in enumerate([('pre',pre,preclear),('post',post,postclear)]):
  x=img.copy();x[~support]=(.65*x[~support]+.35*np.array([150,150,150])).astype(np.uint8);contours,_=cv2.findContours(foot.astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE);cv2.drawContours(x,contours,-1,(255,90,0),1)
  canvas.paste(Image.fromarray(x),(i*size[1],45));draw.text((i*size[1]+4,4),name+' native10m; grey=own-date quality unknown',fill='black')
 draw.text((4,24),'orange=unaltered published postevent footprint;pre overlay is location only;paired support separately frozen',fill='black');canvas.save(OUT/'native_post_reference_and_pair_quality_review.png')
 scopes={}
 for name,a in [('post_event_segmentation',afterlabels),('paired_change',pairlabels)]:scopes[name]={'positive_pixels':int((a==1).sum()),'annotation_background_pixels':int((a==0).sum()),'unknown_pixels':int((a==255).sum()),'mask_name':'labels_post_reference_interior.tif' if name.startswith('post') else 'labels_paired_reference_interior.tif'}
 ready=bool(chronological and ga['gate_pass'] and all(s['positive_pixels']>=500 and s['annotation_background_pixels']>=1000 for s in scopes.values()))
 deps=[PRE/'selection.json',PRE/'protocol.json',PRE/'pre_granule.xml',PRE/'measurement_scene.json',PRE/'quality_scene.json',prepath,PRE/'pre_SCL_original_grid.tif',postpath,POST/'original_DN_and_tile_time_verification.json',POST/'post_modern_product.xml',POST/'post_granule.xml',Q/'official_post_SCL_sequence1.tif',Q/'official_cloud_audit_sequence1.json',Q/'published_boundary_wgs84.geojson',Q/'published_inventory_benchmark_v1_20261005/review_pending.json',DATA/'quality_audit/public_source_snapshots_20261005.txt',ROOT/'validation/qualify_cl_postreference_and_pair_support_v2.py']
 v={'created_at':datetime.now().isoformat(timespec='seconds'),'case_id':'CL0002','role':'reserved_final_test','scope':'open_pit_wall_failure','source':'https://zenodo.org/records/17007637',
  'full_original_published_vector_available':True,'source_original_independent_professional_field_survey':False,'original_published_post_pixels_and_geometry_unchanged':True,
  'original_preimage_preserved_alternative_only_for_observation_quality':True,'pre_sensing_time':pretime,'post_sensing_time':posttime,'chronology_gate_pass':chronological,
  'primary_quality_processing_baseline_both_dates':'05.00','legacy_allversion_intersection_as_separate_sensitivity':str(Q/'published_inventory_benchmark_v1_20261005/review_pending.json'),
  'original_producer_radiometry_verified_post':True,'pre_radiometry_provenance':'unambiguous zero-offset legacy PB<4 provider rasterband metadata, not ambiguous PB>=4 COGradiometry;actual samples preserved',
  'reference_metric_scopes':scopes,'post_mapped_foreground_observed_fraction':float(postclear[foot].mean()),'paired_mapped_foreground_observed_fraction':float(paired[foot].mean()),
  'registration':{k:x for k,x in ga.items() if k not in ('points','fit_indices','heldout_indices','unshifted_holdout_error_px','diagnostic_affine')},
  'dataset_quality_numeric_gate_pass':ready,'data_acceptance_depends_on_model_performance':False,'accepted_data_reference':False,'manual_review_status':'pending native source geometry/date/visibility and30heldpoint review',
  'not_full_unmasked_world_IoU_or_safety_validation':True,'mine_detector_predictions':0,
  'frozen_dependencies':{str(p):file_sha256(p) for p in deps},'frozen_assets':{p.name:file_sha256(p) for p in OUT.iterdir() if p.is_file()}}
 write(OUT/'review_pending.json',v);print(json.dumps({k:v[k] for k in ('dataset_quality_numeric_gate_pass','reference_metric_scopes','post_mapped_foreground_observed_fraction','paired_mapped_foreground_observed_fraction','registration')},ensure_ascii=False))
if __name__=='__main__':sys.stdout.reconfigure(encoding='utf-8');run()
