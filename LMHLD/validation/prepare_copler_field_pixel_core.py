"""Conservative held-out reference from the field-informed published outline.

Coordinate chain: external figure -> accepted originalpre frame -> independently
validated identical native grid. The failed direct source-to-post fit stays failed.
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
from shapely.geometry import Polygon,mapping
from PIL import Image,ImageDraw
from model_registry import file_sha256
from mine_data_protocol import registry,verify_assets,assert_mine_group_role
from prepare_winter_constructed_normals import display
CASE=ROOT/'datasets/mine_benchmark_20261002/reserved_candidates/TR_COP2024'
REG=CASE/'reference_registration_v2_field2025_map_prior'
PAIR=CASE/'native_pair_geolocation_v1';OUT=CASE/'pixel_reference_v1_field2025'
SOURCE=ROOT/'datasets/pixel_boundary_sources_20261004';BUFFER=80.
def write(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def run():
 value=registry();verify_assets(value);assert_mine_group_role('TR_COP2024','reserved_final_test',value)
 if value['cases']['TR_COP2024']['model_predictions_run']!=0:raise ValueError('Mine already observed by detector')
 OUT.mkdir(exist_ok=True)
 if (OUT/'reference.json').exists():raise ValueError('Reference already frozen')
 reg=json.loads((REG/'registration.json').read_text(encoding='utf-8'));pre=next(r for r in reg['rows'] if r['period']=='pre');post=next(r for r in reg['rows'] if r['period']=='post')
 pair=json.loads((PAIR/'audit.json').read_text(encoding='utf-8'))
 if not pre['registration_gate_pass'] or not pair['common_grid_gate_pass']:raise ValueError('Both coordinate-chain gates must pass')
 with rasterio.open(CASE/'pre_BGRN_10m.tif') as f:profile=f.profile.copy();grid=f.transform;crs=f.crs;shape=f.shape
 with rasterio.open(CASE/'post_BGRN_10m.tif') as f:
  if f.crs!=crs or f.transform!=grid or f.shape!=shape:raise ValueError('Native grids are not identical')
 xy=np.asarray(pre['mapped_polygon_native_xy']);polygon=Polygon([grid*(x+.5,y+.5) for x,y in xy])
 if not polygon.is_valid or polygon.area<=0:raise ValueError('Source outline invalid')
 core=polygon.buffer(-BUFFER);outer=polygon.buffer(BUFFER)
 if core.is_empty:raise ValueError('Uncertainty leaves no known interior')
 def mask(g):return rasterize([(mapping(g),1)],out_shape=shape,transform=grid,fill=0,dtype='uint8')>0
 footprint=mask(polygon);proposed=mask(core);broad=mask(outer);common=np.ones(shape,bool);quality={}
 acquisition=json.loads((CASE/'acquisition.json').read_text(encoding='utf-8'))
 for period in ('pre','post'):
  with rasterio.open(CASE/f'{period}_BGRN_10m.tif') as f:
   valid=f.dataset_mask()>0;cal=acquisition['images'][period]['calibration']
   if list(f.scales)!=cal['scales'] or list(f.offsets)!=cal['reflectance_offsets']:raise ValueError('Actual producer calibration mismatch')
  with rasterio.open(CASE/f'{period}_SCL_10m.tif') as f:
   if f.crs!=crs or f.transform!=grid or f.shape!=shape:raise ValueError('Quality grid mismatch')
   scl=f.read(1)
  observed=valid & ~np.isin(scl,[0,1,3,8,9,10,11]);common &= observed
  quality[period]={'source_datetime':acquisition['images'][period]['datetime'],'core_weather_clear_fraction':float(observed[proposed].mean()),'core_SCL_counts':{str(k):int(v) for k,v in zip(*np.unique(scl[proposed],return_counts=True))},'SCL_bright_ground_disagreements_not_overridden':True,'SCL_is_not_expert_visibility_truth':True}
 positive=proposed & common;unknown_support=broad & ~positive
 labels=np.full(shape,255,np.uint8);labels[positive]=1;confidence=np.zeros(shape,np.float32);confidence[positive]=.6
 if positive.sum()<20:raise ValueError('Too few jointly observed source-backed positives')
 for name,data,dtype,nodata,desc in [('labels_partial.tif',labels,'uint8',255,'1=field-informed published accident core;255=unknown;no0background'),('confidence_partial.tif',confidence,'float32',None,'0.6 uncalibrated reference weight;unknown0'),('author_outline_geometry_only.tif',footprint.astype(np.uint8),'uint8',None,'Published geometry only;outside0doesnotmeanstable'),('boundary_uncertainty_support.tif',unknown_support.astype(np.uint8),'uint8',None,'80m boundary and obscured interior unknown')]:
  with rasterio.open(OUT/name,'w',**{**profile,'count':1,'dtype':dtype,'nodata':nodata}) as f:
   f.write(data,1);f.set_band_description(1,desc);f.update_tags(case_id='TR_COP2024',mine_group='TR_COPLER',role='reserved_final_test',source_original_survey_GIS='false',reference_kind='field_informed_published_partial_research_outline')
 features=[]
 for kind,g in [('author_mapped_footprint',polygon),('conservative_positive_core',core),('uncertainty_outer_support',outer)]:
  features.append({'type':'Feature','properties':{'case_id':'TR_COP2024','mine_group':'TR_COPLER','role':'reserved_final_test','scope':'heap_leach_flow_failure_extended','kind':kind,'event_date':'2024-02-13','source_DOI':'10.1785/0320250007','buffer_m':BUFFER,'native_resolution_m':10,'original_survey_GIS':False},'geometry':transform_geom(crs,'EPSG:4326',mapping(g))})
 write(OUT/'published_outline_and_conservative_core_wgs84.geojson',{'type':'FeatureCollection','features':features})
 box=(max(0,int(xy[:,0].min()-30)),max(0,int(xy[:,1].min()-30)),min(shape[1],int(xy[:,0].max()+30)),min(shape[0],int(xy[:,1].max()+30)))
 width=(box[2]-box[0])*3;height=(box[3]-box[1])*3;qa=Image.new('RGB',(width*2,height+55),'white');draw=ImageDraw.Draw(qa)
 for i,period in enumerate(('pre','post')):
  im,_=display(CASE/f'{period}_BGRN_10m.tif');im[unknown_support]=(.75*im[unknown_support]+.25*np.array([255,210,0])).astype(np.uint8);im[positive]=(.55*im[positive]+.45*np.array([255,30,30])).astype(np.uint8)
  cv2.polylines(im,[np.rint(xy).astype(np.int32)],True,(255,130,0),1);tile=Image.fromarray(im).crop(box).resize((width,height),Image.Resampling.NEAREST);qa.paste(tile,(i*width,55))
  draw.text((i*width+5,5),period+' original10m; display3x',fill='black')
 draw.text((5,27),'orange=author outline; red=jointly observed core; yellow=UNKNOWN; all remaining pixels UNKNOWN',fill='black')
 qa.save(OUT/'original_pair_partial_reference_review.png')
 deps=[SOURCE/'copler2025_field_paper.pdf',SOURCE/'copler_author_figure1.png',CASE/'context.json',CASE/'acquisition.json',CASE/'source_digitisation_provenance.json',REG/'registration.json',REG/'protocol.json',PAIR/'audit.json',PAIR/'protocol.json',ROOT/'validation/prepare_copler_field_pixel_core.py']
 deps += [CASE/f'{period}_{suffix}' for period in ('pre','post') for suffix in ('BGRN_10m.tif','SCL_10m.tif','product.xml')]
 v={'created_at':datetime.now().isoformat(timespec='seconds'),'case_id':'TR_COP2024','mine_group':'TR_COPLER','role':'reserved_final_test','scope':'heap_leach_flow_failure_extended','event_date':'2024-02-13',
  'source_DOI':'10.1785/0320250007','source_figure':'Figure1 white2024failure outline; p5 two-day field campaign and exact optical dates','source_base_image_date':'2021-11-19','source_base_image_is_not_postevent_image':True,
  'external_published_field_campaign_support':True,'external_field_campaign_date':'2024-02-14, two days','project_expert_feedback_or_field_trial':False,'source_original_survey_GIS':False,
  'source_high_resolution_post_interpretation_image_capture_date_not_given_in_obtained_main_figure':True,'event_identity_date_verified':True,
  'coordinate_chain':{'source_to_original_pre_registration_gate_pass':True,'source_pre_heldout_sites':30,'source_pre_median_px':pre['heldout_median_px'],'source_pre_p90_px':pre['heldout_p90_px'],'source_pre_fraction_within3px':pre['heldout_within3px_fraction'],'source_pre_polygon_inside_ground_control_hull':pre['polygon_within_fit_ground_convex_hull'],
    'original_pre_post_common_grid_gate_pass':True,'original_pair_holdout_sites':30,'native_pair_unshifted_median_px':pair['unshifted_holdout_median_px'],'native_pair_unshifted_p90_px':pair['unshifted_holdout_p90_px'],
    'source_post_direct_registration_gate_pass':post['registration_gate_pass'],'failed_direct_post_fit_not_used_for_labels':True,'images_or_labels_shifted_or_warped':False},
  'label_derivation':'accepted external sourcepre affine to existing world grid; preserved identical original post grid verified on separate heldground; known core limited to common observation',
  'source_lineage':'2025researchers visited site14Feb2024 and delineated failure with high-resolution optics; project digitises published Figure1, has not obtained their original survey vectors; later2026paper shares an author and is not an independent ground survey',
  'uncertainty_buffer_m':BUFFER,'uncertainty_basis':'80m conservative support exceeds sourcepre heldp90~29m + common-grid heldp90~11m + white-stroke and10m-grid discretisation;notcalibrated95percent absolute survey accuracy',
  'author_mapped_area_m2':float(polygon.area),'core_vector_area_m2':float(core.area),'research_field_report_approx_total_area_m2':963000,'reported_area_agreement_not_used_to_rescale_or_score_geometry':True,
  'positive_pixels':int(positive.sum()),'positive_pixel_area_m2':int(positive.sum()*100),'unknown_pixels':int((labels==255).sum()),'background_pixels':0,'reference_confidence':.6,'confidence_is_calibrated_probability':False,'quality':quality,
  'accepted_for_development_positive_core_supervision':False,'accepted_for_reserved_known_core_reference':False,'accepted_for_full_accident_IoU_or_area_accuracy':False,'full_boundary_or_all_clear_ground_asserted':False,
  'new_mine_detector_predictions':0,'new_model_selection_from_test':False,'review_status':'pending visual review; complete final-test admission still separate',
  'frozen_dependencies':{str(p):file_sha256(p) for p in deps},'frozen_assets':{p.name:file_sha256(p) for p in OUT.iterdir() if p.is_file()}}
 write(OUT/'reference_review_pending.json',v);print(json.dumps({k:v[k] for k in ('author_mapped_area_m2','positive_pixels','unknown_pixels','background_pixels','quality')},ensure_ascii=False))
if __name__=='__main__':sys.stdout.reconfigure(encoding='utf-8');run()
