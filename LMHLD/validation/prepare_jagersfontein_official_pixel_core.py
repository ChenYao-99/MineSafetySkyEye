"""Create post-observed accident cores inside official partial map, never a synthetic complete boundary."""
from pathlib import Path
from datetime import datetime
import json,sys
ROOT=Path('D:/LMHLD');sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'validation'))
from runtime_paths import configure_runtime_paths
configure_runtime_paths()
import cv2,numpy as np,rasterio
from PIL import Image,ImageDraw
from rasterio.features import shapes
from rasterio.warp import transform_geom
from model_registry import file_sha256
from mine_data_protocol import assert_development_cases,verify_assets
from prepare_winter_constructed_normals import display
CASE=ROOT/'datasets/jagersfontein_development_20261005/ZA_JAGS2022';REG=CASE/'official_reference_registration_v2_actual_viewport';OUT=CASE/'pixel_reference_v1_official_partial'
SRC=ROOT/'datasets/maxclosure_sources_20261005/jagersfontein/charter_product_28498588.img'
def write(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def run():
 assert_development_cases(['ZA_JAGS2022']);verify_assets();OUT.mkdir(exist_ok=True)
 if (OUT/'reference.json').exists():raise ValueError('Reference already frozen')
 reg=json.loads((REG/'registration.json').read_text(encoding='utf-8'));proto=json.loads((REG/'protocol.json').read_text(encoding='utf-8'))
 if not reg['registration_gate_pass']:raise ValueError('Officialmap registration not accepted')
 matrix=np.asarray(reg['sourcecrop_to_native_affine']);sourcebody=cv2.imdecode(np.fromfile(REG/'source_partial_mapped_interior.png',np.uint8),0)>0
 with rasterio.open(CASE/'post_BGRN_10m.tif') as f:profile=f.profile.copy();size=f.shape;grid=f.transform;crs=f.crs;postvalid=f.dataset_mask()>0
 h,w=size;foot=cv2.warpAffine(sourcebody.astype(np.uint8),matrix,(w,h),flags=cv2.INTER_NEAREST)>0
 hull=np.zeros(sourcebody.shape,np.uint8);cv2.fillPoly(hull,[np.rint(reg['source_fit_ground_convex_hull']).astype(np.int32)],1)
 hull_native=cv2.warpAffine(hull,matrix,(w,h),flags=cv2.INTER_NEAREST)>0
 # Geometry closure at the viewport is an extraction operation;all source edges stay unknown.
 domain=np.ones(sourcebody.shape,np.uint8);domain[[0,-1]]=0;domain[:,[0,-1]]=0
 source_mpp=10*float(np.min(reg['affine_singular_values']));inside_domain=(cv2.distanceTransform(domain,cv2.DIST_L2,cv2.DIST_MASK_PRECISE)*source_mpp>=120)
 domain_native=cv2.warpAffine(inside_domain.astype(np.uint8),matrix,(w,h),flags=cv2.INTER_NEAREST)>0
 reference_interior=cv2.distanceTransform(foot.astype(np.uint8),cv2.DIST_L2,cv2.DIST_MASK_PRECISE)>=6
 with rasterio.open(CASE/'post_SCL_10m.tif') as f:post_scl=f.read(1);postclear=postvalid & ~np.isin(post_scl,[0,1,3,7,8,9,10,11])
 with rasterio.open(CASE/'pre_SCL_10m.tif') as f:pre_scl=f.read(1);preclear=~np.isin(pre_scl,[0,1,3,7,8,9,10,11])
 positive=reference_interior & domain_native & hull_native & postclear
 labels=np.full(size,255,np.uint8);labels[positive]=1;confidence=np.zeros(size,np.float32);confidence[positive]=.6
 if positive.sum()<500:raise ValueError('Officialmapped observed core too small for new trainable data goal')
 for name,array,dtype,nodata,desc in [('labels_partial.tif',labels,'uint8',255,'POST mapped landdepositcore1;unknown255;no background'),('confidence_partial.tif',confidence,'float32',None,'Referenceweight.6 uncalibrated;unknown0'),('mapped_partial_footprint_geometry_only.tif',foot.astype(np.uint8),'uint8',None,'Partial sourceviewportgeometry;notcompleteextents'),('paired_positive_support.tif',(positive&preclear).astype(np.uint8),'uint8',None,'Subset ofpostcore with comparable pre observations')]:
  with rasterio.open(OUT/name,'w',**{**profile,'count':1,'dtype':dtype,'nodata':nodata}) as f:
   f.write(array,1);f.set_band_description(1,desc);f.update_tags(case_id='ZA_JAGS2022',mine_group='ZA_JAGERSFONTEIN',role='development',target_epoch='post',reference_kind='registered_partial_official_floodmap_core',source_original_GIS='false')
 feats=[]
 for geometry,value in shapes(positive.astype(np.uint8),mask=positive,transform=grid):feats.append({'type':'Feature','properties':{'case_id':'ZA_JAGS2022','role':'development','kind':'observed_officialmapped_core','event_date':'2022-09-11','source_date':'2022-09-13','target_date':'2022-09-16','native_resolution_m':10,'full_boundary':False},'geometry':transform_geom(crs,'EPSG:4326',geometry)})
 write(OUT/'observed_official_partial_core_wgs84.geojson',{'type':'FeatureCollection','features':feats})
 sourcepic=np.asarray(Image.open(SRC).convert('RGB'));x0,y0,x1,y1=proto['cropped_source_frame_xyxy'];sourcepic=sourcepic[y0:y1,x0:x1]
 warped=cv2.warpAffine(sourcepic,matrix,(w,h));review=Image.new('RGB',(w*2,h+50),'white');draw=ImageDraw.Draw(review)
 for i,(title,image) in enumerate([('official13Septmap registered reference',warped),('original16Sept10m;postlabels only',display(CASE/'post_BGRN_10m.tif')[0])]):
  x=image.copy();x[positive]=(.65*x[positive]+.35*np.array([255,30,0])).astype(np.uint8)
  review.paste(Image.fromarray(x),(i*w,50));draw.text((i*w+5,5),title,fill='black')
 draw.text((5,25),'red=known accidentinterior;60m edge/120m viewport buffer;all remaining pixelsUNKNOWN',fill='black');review.save(OUT/'source_and_original_post_core_review.png')
 deps=[SRC,REG/'protocol.json',REG/'registration.json',REG/'source_partial_mapped_interior.png',CASE/'acquisition.json',CASE.parent/'context.json',ROOT/'validation/prepare_jagersfontein_official_pixel_core.py']
 deps += [CASE/f'{period}_{suffix}' for period in ('pre','post') for suffix in ('BGRN_10m.tif','SCL_10m.tif','product.xml')]
 v={'created_at':datetime.now().isoformat(timespec='seconds'),'case_id':'ZA_JAGS2022','mine_group':'ZA_JAGERSFONTEIN','role':'development','scope':'tailings_failure_land_deposit_extended','event_date':'2022-09-11',
  'source_authority':'SANSA/InternationalDisastersCharter officialFloodedarea map;Pleiades13Sept2022','source_url':'https://disasterscharter.org/cos-api/api/file/public/article-image/28498588',
  'source_map_is_not_model_input':True,'source_original_survey_GIS':False,'target_label_epoch':'post','target_date':'2022-09-16','source_date':'2022-09-13','source_target_capture_gap_days':3,
  'source_map_viewport_clipped_not_complete_downstream_boundary':True,'viewport_edge_is_not_physical_failure_boundary':True,'source_shape_or_unknowns_not_completed_from_classifier_or_difference_images':True,
  'registration':{k:reg[k] for k in ('unique_ground_sites','fit_inliers','holdout_median_px','holdout_p90_px','holdout_within3px_fraction')},
  'new_expert_feedback_or_project_field_trial':False,'source_edge_buffer_m':60,'source_viewport_buffer_m':120,'fit_ground_hull_extrapolation_excluded':True,
  'positive_pixels':int(positive.sum()),'positive_pixel_area_m2':int(positive.sum()*100),'unknown_pixels':int((labels==255).sum()),'background_pixels':0,'paired_observed_positive_pixels':int((positive&preclear).sum()),
  'reference_confidence':.6,'confidence_is_calibrated_probability':False,'paired_visibility_not_required_for_postonly_footprint_label':True,
  'post_source_core_weather_clear_fraction':float(postclear[reference_interior & domain_native & hull_native].mean()),
  'accepted_for_development_positive_core_supervision':False,'accepted_for_full_accident_IoU_or_area_accuracy':False,'new_blind_test_mine':False,'new_mine_detector_predictions':0,
  'review_status':'pending originalprofessionalmap/source-date/coreonoriginal16Sept/30sourceholdpoint visual review;not field certification',
  'frozen_dependencies':{str(p):file_sha256(p) for p in deps},'frozen_assets':{p.name:file_sha256(p) for p in OUT.iterdir() if p.is_file()}}
 write(OUT/'reference_review_pending.json',v);print(json.dumps({k:v[k] for k in ('positive_pixels','unknown_pixels','background_pixels','paired_observed_positive_pixels','post_source_core_weather_clear_fraction','registration')},ensure_ascii=False))
if __name__=='__main__':sys.stdout.reconfigure(encoding='utf-8');run()
