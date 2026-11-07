"""Native10m partial accident reference from an externally published 2020 outline.

Only a conservative, observed interior may be positive. No source screenshot is a
model input; outside/core-edge/dust/runout/other pits remain unknown, never 'safe'.
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
from mine_data_protocol import assert_development_cases,verify_assets
from prepare_winter_constructed_normals import display
CASE=ROOT/'datasets/arid_accident_operation_20261004/ZA_GAMS2020S'
REG=CASE/'reference_registration_v5_wang2026_verified_axes'
SOURCE=ROOT/'datasets/pixel_boundary_sources_20261004'
OUT=CASE/'pixel_reference_v1_wang2026'
BUFFER_M=60.
def write(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')

def run():
    assert_development_cases(['ZA_GAMS2020S']);verify_assets()
    OUT.mkdir(exist_ok=True)
    if (OUT/'reference.json').exists():raise ValueError('Reference already frozen; do not regenerate')
    reg=json.loads((REG/'registration.json').read_text(encoding='utf-8'))
    if not reg['both_original_dates_gate_pass']:raise ValueError('Both original-date registrations must pass')
    rows={r['period']:r for r in reg['rows']}
    prepoly=np.asarray(rows['pre']['mapped_polygon_native_xy']);postpoly=np.asarray(rows['post']['mapped_polygon_native_xy'])
    discrepancy=np.linalg.norm(prepoly-postpoly,axis=1)
    if np.max(discrepancy)>2:raise ValueError('Same published outline projects inconsistently between original dates')
    with rasterio.open(CASE/'post_BGRN_10m.tif') as f:
        profile=f.profile.copy();shape=f.shape;grid=f.transform;crs=f.crs
    with rasterio.open(CASE/'pre_BGRN_10m.tif') as f:
        if f.crs!=crs or f.transform!=grid or f.shape!=shape:raise ValueError('Original pair grid mismatch')
    polygon=Polygon([grid*(x+.5,y+.5) for x,y in postpoly])
    if not polygon.is_valid or polygon.area<=0:raise ValueError('Published outline is invalid')
    core=polygon.buffer(-BUFFER_M);outer=polygon.buffer(BUFFER_M)
    if core.is_empty:raise ValueError('Uncertainty leaves no defensible positive interior')
    footprint=rasterize([(mapping(polygon),1)],out_shape=shape,transform=grid,fill=0,dtype='uint8')>0
    candidate_core=rasterize([(mapping(core),1)],out_shape=shape,transform=grid,fill=0,dtype='uint8')>0
    broad=rasterize([(mapping(outer),1)],out_shape=shape,transform=grid,fill=0,dtype='uint8')>0
    weather=np.ones(shape,bool);quality={}
    for period in ('pre','post'):
        with rasterio.open(CASE/f'{period}_BGRN_10m.tif') as f:valid=f.dataset_mask()>0
        with rasterio.open(CASE/f'{period}_SCL_10m.tif') as f:
            if f.crs!=crs or f.transform!=grid or f.shape!=shape:raise ValueError('Quality grid mismatch')
            scl=f.read(1)
        observed=valid & ~np.isin(scl,[0,1,3,8,9,10,11])
        weather &= observed
        quality[period]={'candidate_core_clear_fraction':float(observed[candidate_core].mean()),'SCL_class_counts':{str(k):int(v) for k,v in zip(*np.unique(scl[candidate_core],return_counts=True))},'SCL_not_expert_visibility_truth':True}
    positive=candidate_core & weather;uncertain=broad & ~positive
    if positive.sum()<20:raise ValueError('Too little observed conservative core')
    labels=np.full(shape,255,np.uint8);labels[positive]=1
    confidence=np.zeros(shape,np.float32);confidence[positive]=.6
    outputs={'labels_partial.tif':(labels,'uint8',255,'1=conservative published accident core;255=unknown;no0labels'),
             'confidence_partial.tif':(confidence,'float32',None,'0.6 partial research reference;0 outside known core'),
             'author_outline_geometry_only.tif':(footprint.astype(np.uint8),'uint8',None,'Source geometry only;outside0doesnotmeanabsence'),
             'boundary_uncertainty_support.tif':(uncertain.astype(np.uint8),'uint8',None,'Uncertainty support;never ground truth/background')}
    for name,(array,dtype,nodata,desc) in outputs.items():
        with rasterio.open(OUT/name,'w',**{**profile,'count':1,'dtype':dtype,'nodata':nodata}) as f:
            f.write(array,1);f.set_band_description(1,desc)
            f.update_tags(case_id='ZA_GAMS2020S',mine_group='ZA_GAMSBERG',role='development',reference_kind='partial_external_published_research_outline',survey_GIS='false',full_accident_boundary='false')
    features=[]
    for kind,geometry in (('author_mapped_footprint',polygon),('conservative_positive_core',core),('uncertainty_outer_support',outer)):
        g=transform_geom(crs,'EPSG:4326',mapping(geometry))
        features.append({'type':'Feature','properties':{'kind':kind,'case_id':'ZA_GAMS2020S','event_date':'2020-11-17','role':'development','source_figure':'Wang2026S13/S14','source_DOI':'10.1016/j.jag.2026.105217','independent_survey_GIS':False,'buffer_m':BUFFER_M,'native_resolution_m':10},'geometry':g})
    write(OUT/'published_outline_and_conservative_core_wgs84.geojson',{'type':'FeatureCollection','features':features})
    xs=postpoly[:,0];ys=postpoly[:,1];box=(max(0,int(xs.min()-35)),max(0,int(ys.min()-35)),min(shape[1],int(xs.max()+35)),min(shape[0],int(ys.max()+35)))
    qa=Image.new('RGB',((box[2]-box[0])*5*2,(box[3]-box[1])*5+55),'white');draw=ImageDraw.Draw(qa)
    for i,period in enumerate(('pre','post')):
        im,_=display(CASE/f'{period}_BGRN_10m.tif');overlay=im.copy()
        overlay[uncertain]=(.72*overlay[uncertain]+.28*np.array([255,215,20])).astype(np.uint8)
        overlay[positive]=(.55*overlay[positive]+.45*np.array([255,30,30])).astype(np.uint8)
        cv2.polylines(overlay,[np.rint(postpoly).astype(np.int32)],True,(255,130,0),1)
        tile=Image.fromarray(overlay).crop(box).resize(((box[2]-box[0])*5,(box[3]-box[1])*5),Image.Resampling.NEAREST)
        left=i*tile.width;qa.paste(tile,(left,55));draw.text((left+5,7),period+' original10m; display5x',fill='black')
    draw.text((5,27),'orange=author outline; red=known interior; yellow=unknown uncertainty; all remaining pixels UNKNOWN',fill='black')
    qa.save(OUT/'original_pair_partial_reference_review.png')
    dependencies=[REG/'registration.json',REG/'protocol.json',ROOT/'validation/register_gamsberg_published_boundary_v5.py',SOURCE/'wang2026_supplement.docx',SOURCE/'wang_supplement_original_media/image13.png',SOURCE/'wang_supplement_original_media/image14.png',SOURCE/'gamsberg_original_graticule_verification.json']
    dependencies += [CASE/f'{period}_{suffix}' for period in ('pre','post') for suffix in ('BGRN_10m.tif','SCL_10m.tif','product.xml')]
    ref=CASE.parent/'source_review'
    dependencies += [ref/'gamsberg_operator_2021.html',ref/'dated_archived_gamsberg_article.html',ref/'dated_archive_revision.json',ref/'gamsberg_skysat_full_20201203.jpg',CASE/'presence_reference_v1/presence_reference.json',CASE/'presence_reference_v1/pixel_labels_unknown.tif']
    value={'created_at':datetime.now().isoformat(timespec='seconds'),'case_id':'ZA_GAMS2020S','mine_group':'ZA_GAMSBERG','role':'development','event_date':'2020-11-17',
        'source_DOI':'10.1016/j.jag.2026.105217','source_figure':'S13 author white failure outline;S14 explicitly dated2020-11-17',
        'source_original_survey_GIS':False,'source_google_earth_base_capture_date':None,'annotation_event_identity_verified':True,
        'dated_temporal_corroboration':{'SkySat_reference':'2020-12-03','original_post_S2':'2020-12-04','capture_date_basis':'author explicitly dated archived revision;JPEG EXIF editor date not used','used_as_model_features':False},
        'source_lineage':'2026authors interpret publicly documented2020accident and Planet/AGU imagery; separate paper is not an independent original ground survey; S13/S14 replicas not multiple truths',
        'source_outline_not_expanded_from_S2_change_or_detector':True,'scope':'partial author-mapped accident footprint only; completeness of source/runout/secondary failure not asserted',
        'registration':{period:{k:rows[period][k] for k in ('unique_ground_sites','fit_inliers','heldout_median_px','heldout_p90_px','heldout_within3px_fraction','polygon_within_fit_ground_convex_hull')} for period in ('pre','post')},
        'registration_holdout_source':'30 distinct surrounding-ground sites per date, held before anyfit; not independent surveyed checkpoints',
        'post_pre_polygon_discrepancy_native_px_max':float(discrepancy.max()),'uncertainty_buffer_m':BUFFER_M,
        'uncertainty_basis':'conservative60m exceeds held-ground p90~15m + printed white stroke halfwidth~24m +10m-grid halfdiagonal7.1m; not a calibrated95percent absolute geodetic confidence interval',
        'author_mapped_area_m2':float(polygon.area),'conservative_core_vector_area_m2':float(core.area),'positive_pixels':int(positive.sum()),'unknown_pixels':int((labels==255).sum()),'background_pixels':0,'positive_pixel_area_m2':int(positive.sum()*100),
        'reference_confidence':.6,'reference_confidence_is_calibrated_probability':False,'quality':quality,
        'accepted_for_development_positive_core_supervision':False,'accepted_for_known_core_recall_evaluation':False,
        'accepted_for_full_accident_IoU_or_area_accuracy':False,'independent_expert_review':False,'project_field_feedback':0,
        'old_weak_presence_targets_unchanged':True,'new_accident_detector_predictions':0,'new_blind_test_mine':False,
        'review_status':'pending explicit visual review of original-source geometry, all60 held points and original pair; no model output',
        'frozen_dependencies':{str(p):file_sha256(p) for p in dependencies},'frozen_assets':{p.name:file_sha256(p) for p in OUT.iterdir() if p.is_file()}}
    write(OUT/'reference_review_pending.json',value)
    print(json.dumps({k:value[k] for k in ('author_mapped_area_m2','positive_pixels','unknown_pixels','background_pixels','post_pre_polygon_discrepancy_native_px_max','quality')},ensure_ascii=False))

if __name__=='__main__':sys.stdout.reconfigure(encoding='utf-8');run()
