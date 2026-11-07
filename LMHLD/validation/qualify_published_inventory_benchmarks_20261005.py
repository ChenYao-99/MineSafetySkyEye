"""Qualify independent published-reference data before detector evaluation.

Original polygons/pixels stay frozen. Annotation background is not geological safety.
Only predefined common observation and uncertain-edge masking enter scored support.
"""
from pathlib import Path
from datetime import datetime
import json,sys
ROOT=Path('D:/LMHLD');sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'cache/research_dependencies'))
from runtime_paths import configure_runtime_paths
configure_runtime_paths()
import cv2,numpy as np,rasterio
from rasterio.features import rasterize
from rasterio.warp import transform_geom
from shapely.geometry import shape,mapping
from shapely.ops import unary_union
from PIL import Image,ImageDraw
from model_registry import file_sha256
from mine_data_protocol import registry,verify_assets,assert_mine_group_role
from prepare_winter_constructed_normals import display
DATA=ROOT/'datasets/mine_benchmark_20261002';RAW=DATA/'raw_sources/dlr/s1s2_landslide_reference_data/original_scenes'
PLAN=ROOT/'validation/maxclosure_execution_plan_20261005.json'
EXCLUSIONS={'CL0002':[(150,20,445,405),(185,385,315,499)],'KG0001':[]}
def write(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def ground(pre,post,observable,footprint,extra,out):
 mask=observable.copy();mask[cv2.dilate(footprint.astype(np.uint8),np.ones((21,21),np.uint8))>0]=False
 for x0,y0,x1,y1 in extra:mask[y0:y1,x0:x1]=False
 s=cv2.SIFT_create(nfeatures=12000,nOctaveLayers=5,contrastThreshold=.005,edgeThreshold=12,sigma=1.);cl=cv2.createCLAHE(2,(8,8))
 sk,sd=s.detectAndCompute(cl.apply(cv2.cvtColor(pre,cv2.COLOR_RGB2GRAY)),mask.astype(np.uint8)*255)
 tk,td=s.detectAndCompute(cl.apply(cv2.cvtColor(post,cv2.COLOR_RGB2GRAY)),mask.astype(np.uint8)*255);points=[]
 if sd is not None and td is not None:
  matches=[]
  for pair in cv2.BFMatcher().knnMatch(sd,td,k=2):
   if len(pair)==2 and pair[0].distance<.85*pair[1].distance:matches.append(pair[0])
  for m in sorted(matches,key=lambda m:m.distance):
   a=np.array(sk[m.queryIdx].pt);b=np.array(tk[m.trainIdx].pt)
   if np.linalg.norm(a-b)>8:continue
   if any(np.linalg.norm(a-q['pre_xy'])<3 or np.linalg.norm(b-q['post_xy'])<3 for q in points):continue
   points.append({'pre_xy':a.tolist(),'post_xy':b.tolist(),'descriptor_distance':float(m.distance)})
 result={'unique_ground_sites':len(points),'points':points,'heldout_allocated_before_any_fit':True,'images_warped':False,'gate_pass':False}
 if len(points)>=90:
  a=np.array([p['pre_xy'] for p in points],np.float32);b=np.array([p['post_xy'] for p in points],np.float32);cells={}
  for i,p in enumerate(a):cells.setdefault((int(p[0]//80),int(p[1]//80)),[]).append(i)
  rng=np.random.default_rng(51005);keys=sorted(cells);rng.shuffle(keys)
  for indices in cells.values():rng.shuffle(indices)
  order=[]
  while any(cells.values()):
   for key in keys:
    if cells[key]:order.append(cells[key].pop())
  held=np.array(order[:30]);fit=np.array([i for i in order[30:] if np.min(np.linalg.norm(a[i]-a[held],axis=1))>=4],int)
  cv2.setRNGSeed(51005);matrix,inliers=cv2.estimateAffinePartial2D(a[fit],b[fit],method=cv2.RANSAC,ransacReprojThreshold=2,maxIters=10000,confidence=.999)
  error=np.linalg.norm(a[held]-b[held],axis=1)
  result.update(heldout_indices=held.tolist(),fit_indices=fit.tolist(),fit_inliers=int(inliers.sum()) if inliers is not None else 0,
   unshifted_holdout_error_px=error.tolist(),unshifted_holdout_median_px=float(np.median(error)),unshifted_holdout_p90_px=float(np.percentile(error,90)),unshifted_holdout_within2px=float((error<=2).mean()),diagnostic_affine=matrix.tolist() if matrix is not None else None)
  result['gate_pass']=bool(result['fit_inliers']>=30 and np.median(error)<=1 and np.percentile(error,90)<=2 and (error<=2).mean()>=.9)
  panel=Image.new('RGB',(900,720),'white');draw=ImageDraw.Draw(panel)
  for i,idx in enumerate(held):
   x=(i%5)*180;y=(i//5)*120;draw.text((x+4,y+2),str(i+1),fill='black')
   for j,(im,p) in enumerate(((pre,a[idx]),(post,b[idx]))):
    ix,iy=np.rint(p).astype(int);tile=Image.fromarray(im).crop((ix-16,iy-16,ix+17,iy+17)).resize((80,80),Image.Resampling.NEAREST)
    d=ImageDraw.Draw(tile);d.line((35,40,45,40),fill='red');d.line((40,35,40,45),fill='red');panel.paste(tile,(x+j*88+2,y+20))
  panel.save(out/'all30_ground_holdout_sites.png')
 return result
def run():
 value=registry();verify_assets(value);plan=json.loads(PLAN.read_text(encoding='utf-8'));policy=plan['public_inventory_data_protocol'];rows=[]
 for eid in policy['cases']:
  assert_mine_group_role(eid,'reserved_final_test',value)
  if value['cases'][eid]['model_predictions_run']!=0:raise ValueError('Reserved detector already observed')
  q=DATA/f'quality_audit/{eid}';quality=q/'exact_date_quality_v1_20261005';out=q/'published_inventory_benchmark_v1_20261005';out.mkdir(exist_ok=True)
  if (out/'review_pending.json').exists():print(eid,'frozen pending review reused');continue
  verification=quality/'original_DN_and_tile_time_verification.json';v=json.loads(verification.read_text(encoding='utf-8'))
  if not all(r['original_all4bands_byte_match'] for r in v['images'].values()):raise ValueError('Physical original source calibration unverified')
  audit=json.loads((q/'audit.json').read_text(encoding='utf-8'));geo=json.loads((q/'published_boundary_wgs84.geojson').read_text(encoding='utf-8'))
  protocol={'case_id':eid,'execution_plan_sha256':file_sha256(PLAN),'boundary_m':policy['boundary_uncertainty_m'],'provider_observation_classes_excluded':policy['SCL_obscured_or_unclassified_classes'],
    'additional_known_mine_working_exclusion_rectangles_for_ground_registration_only':EXCLUSIONS[eid],
    'target':'publisher postevent mapped footprint;no assertion all positive pixels are new since preimage',
    'outside_mapped_event':'publisher annotation background, not a geotechnical safety certificate','use_all_same_date_quality_versions':True,
    'detector_outputs_used_for_roi_mask_or_quality_selection':False,'field_survey_claim_allowed':False,'geolocation_holdout_seed':51005}
  write(out/'protocol.json',protocol)
  with rasterio.open(quality/'post_producer_BGRN_original_grid.tif') as f:profile=f.profile.copy();crs=f.crs;grid=f.transform;size=f.shape
  polygons=[shape(transform_geom('EPSG:4326',crs,f['geometry'])) for f in geo['features']];body=unary_union(polygons)
  if body.is_empty or not body.is_valid:raise ValueError('Invalid original published polygon')
  def mask(g):return rasterize([(mapping(g),1)],out_shape=size,transform=grid,fill=0,dtype='uint8')>0
  foot=mask(body);inner=mask(body.buffer(-policy['boundary_uncertainty_m']));outer=mask(body.buffer(policy['boundary_uncertainty_m']))
  with rasterio.open(RAW/eid/f'{eid}__None__None__MASK.tif') as f:
   original=f.read(1)
   if not set(np.unique(original)).issubset({0,1}) or not np.array_equal(original>0,foot):raise ValueError('Original external mask/vector mismatch')
  images={};observed=np.ones(size,bool);details={};dependencies=[PLAN,q/'audit.json',q/'published_boundary_wgs84.geojson',verification,quality/'manifest.json',DATA/'quality_audit/public_source_snapshots_20261005.txt']
  for period in ('pre','post'):
   imagepath=quality/f'{period}_producer_BGRN_original_grid.tif';images[period],valid=display(imagepath);observed &= valid;dependencies.append(imagepath)
   sources=[quality/f'{period}_SCL_original_grid.tif']
   if period=='post':sources += sorted(q.glob('official_post_SCL*.tif'))
   details[period]=[]
   for p in sources:
    with rasterio.open(p) as f:
     if f.crs!=crs or f.transform!=grid or f.shape!=size:raise ValueError('Quality variant grid mismatch')
     scl=f.read(1)
    clear=~np.isin(scl,policy['SCL_obscured_or_unclassified_classes']);observed &= clear
    details[period].append({'path':str(p),'sha256':file_sha256(p),'published_foreground_clear_fraction':float(clear[foot].mean()),'whole_frame_clear_fraction':float(clear.mean())});dependencies.append(p)
   dependencies += [quality/f'{period}_modern_product.xml',quality/f'{period}_granule.xml']
  before=v['images']['pre']['tile_metadata_fields']['SENSING_TIME'];after=v['images']['post']['tile_metadata_fields']['SENSING_TIME']
  if eid=='CL0002':
   chronological=before<'2021-11-08T13:55:00Z' and after>'2021-11-09T14:42:00Z'
   time_scope={'failure_bracket_from_primary_author':['2021-11-08T13:55:00Z','2021-11-09T14:42:00Z'],'source_date_8vs9_ambiguity_does_not_change_bracket':True,'pre_existing_smaller_slide_not_claimed_absent':True}
   dependencies += [ROOT/'datasets/maxclosure_sources_20261005/cl_before_reference_exact.jpg',ROOT/'datasets/maxclosure_sources_20261005/cl_after_reference_exact.jpg']
  else:
   chronological=before[:10]=='2020-09-07' and after[:10]=='2020-09-17';time_scope={'event_date_confirmed_by_government_road_ministry':'2020-09-14','strict_pit_wall_claim_allowed':False}
  labels=np.full(size,255,np.uint8);labels[inner & observed]=1;labels[(~outer) & observed]=0;score=labels!=255
  ga=ground(images['pre'],images['post'],observed,foot,EXCLUSIONS[eid],out);write(out/'ground_registration.json',ga)
  enough=(labels==1).sum()>=policy['observed_scored_positive_pixels_min'] and (labels==0).sum()>=policy['observed_scored_annotation_background_pixels_min']
  qualified=bool(chronological and enough and ga['gate_pass'])
  for name,array,desc in [('labels_reference_interior.tif',labels,'1=published mapped foreground;0=annotation background;255=unknown'),('observable_common_support.tif',observed.astype(np.uint8),'Conservative exact-date all-provider weather support')]:
   with rasterio.open(out/name,'w',**{**profile,'count':1,'dtype':'uint8','nodata':255 if name.startswith('labels') else None}) as f:
    f.write(array,1);f.set_band_description(1,desc);f.update_tags(case_id=eid,role='reserved_final_test',reference_kind='complete_external_published_inventory_with_scored_interior',absolute_survey_accuracy='unverified')
  qa=Image.new('RGB',(size[1]*2,size[0]+40),'white');draw=ImageDraw.Draw(qa)
  for i,p in enumerate(('pre','post')):
   im=images[p].copy();im[labels==255]=(.7*im[labels==255]+.3*np.array([160,160,160])).astype(np.uint8)
   contours,_=cv2.findContours(foot.astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE);cv2.drawContours(im,contours,-1,(255,100,0),1)
   qa.paste(Image.fromarray(im),(i*size[1],40));draw.text((i*size[1]+5,4),p+' original10m; orange=original published boundary',fill='black')
  draw.text((5,22),'grey=unscored weather/edge; geometry agreement not independent absolute survey accuracy',fill='black');qa.save(out/'original_pair_published_inventory_review.png')
  dependencies += [RAW/eid/f'{eid}__None__None__MASK.tif',ROOT/'validation/qualify_published_inventory_benchmarks_20261005.py']
  result={'created_at':datetime.now().isoformat(timespec='seconds'),'case_id':eid,'role':'reserved_final_test','scope':value['cases'][eid]['scope'],
   'source':'https://zenodo.org/records/17007637','source_method':'independent DLR/GFZ authors visually delineated inventory using pre/post Sentinel imagery;not project labels or field surveys',
   'full_original_published_vector_available':True,'source_original_independent_professional_field_survey':False,'original_publisher_mask_vs_vector_IoU':1,'comparison_is_not_model_or_absolute_accuracy':True,
   'native_resolution_m':10,'source_original_pixels_match_producer_all16bands':True,'tile_sensing_times':{'pre':before,'post':after},'source_chronology':time_scope,'chronology_gate_pass':chronological,
   'boundary_uncertainty_m':policy['boundary_uncertainty_m'],'uncertainty_is_advisory_not_calibrated95percent_error':True,'published_foreground_pixels':int(foot.sum()),
   'scored_positive_pixels':int((labels==1).sum()),'scored_annotation_background_pixels':int((labels==0).sum()),'unknown_pixels':int((labels==255).sum()),
   'published_foreground_retained_fraction':float(((labels==1)&foot).sum()/foot.sum()),'common_published_foreground_observation_fraction':float(observed[foot].mean()),'scored_fraction_of_frame':float(score.mean()),'quality_provider_versions':details,
   'registration':{k:x for k,x in ga.items() if k not in ('points','heldout_indices','fit_indices','unshifted_holdout_error_px','diagnostic_affine')},
   'dataset_quality_numeric_gate_pass':qualified,'manual_review_status':'pending original-source/native-visibility/30heldground visual review','accepted_data_reference':False,
   'allowed_metric_scope':policy['allowed_metrics'],'not_full_unmasked_world_IoU_or_safety_validation':True,
   'data_acceptance_depends_on_model_performance':False,'frozen_model_evaluation_still_separate':True,'mine_detector_predictions':0,
   'frozen_dependencies':{str(p):file_sha256(p) for p in dependencies},'frozen_assets':{p.name:file_sha256(p) for p in out.iterdir() if p.is_file()}}
  write(out/'review_pending.json',result);rows.append(result);print(json.dumps({k:result[k] for k in ('case_id','scored_positive_pixels','scored_annotation_background_pixels','unknown_pixels','common_published_foreground_observation_fraction','dataset_quality_numeric_gate_pass','registration')},ensure_ascii=False),flush=True)
if __name__=='__main__':sys.stdout.reconfigure(encoding='utf-8');run()
