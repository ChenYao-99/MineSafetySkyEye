"""Register source-published pit-wall geometry using held-out surrounding terrain."""
from pathlib import Path
from datetime import datetime
import sys,json
ROOT=Path('D:/LMHLD');sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'validation'));sys.path.insert(0,str(ROOT/'cache/research_dependencies'))
from runtime_paths import configure_runtime_paths
configure_runtime_paths()
import cv2,numpy as np,rasterio,torch
from PIL import Image,ImageDraw
from kornia.feature import LoFTR
from rasterio.warp import transform as coordinates
from model_registry import file_sha256
from mine_data_protocol import assert_development_cases,verify_assets
from prepare_winter_constructed_normals import display
from register_gamsberg_published_boundary_v5 import padded,sheet,weights,WEIGHT
CASE=ROOT/'datasets/carmen_development_20261005/PH_CARM2020'
SOURCE=ROOT/'datasets/pixel_boundary_sources_20261004/wang_supplement_original_media/image17.png'
OUT=CASE/'reference_registration_v1_wang2026'
# Author's white outline centreline, read from original S17 before target fitting.
POLYGON=np.array([[449,185],[459,190],[463,200],[462,209],[458,218],[452,229],
                  [450,240],[445,247],[438,254],[429,263],[420,270],[412,272],
                  [407,269],[405,261],[409,248],[411,237],[408,226],[405,214],
                  [408,203],[417,195],[431,189],[442,186]],np.float64)
AXES={'longitude':{'source_x':[117.,328.,541.,752.],'degrees':[123.7,123.72,123.74,123.76]},
      'latitude':{'source_y':[213.,436.],'degrees':[10.34,10.32]}}
def write(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def prior(points,frame):
    lon=np.polyval(np.polyfit(AXES['longitude']['source_x'],AXES['longitude']['degrees'],1),points[:,0])
    lat=np.polyval(np.polyfit(AXES['latitude']['source_y'],AXES['latitude']['degrees'],1),points[:,1])
    x,y=coordinates('EPSG:4326',frame.crs,lon.tolist(),lat.tolist());inverse=~frame.transform
    return np.array([inverse*(a,b) for a,b in zip(x,y)])-.5
def run():
    assert_development_cases(['PH_CARM2020']);verify_assets();OUT.mkdir(exist_ok=True)
    p={'created_at':datetime.now().isoformat(timespec='seconds'),'case_id':'PH_CARM2020','role':'development',
       'source_sha256':file_sha256(SOURCE),'source_figure':'Wang2026S17 author white failure outline;S18 explicitly21Dec2020',
       'source_google_earth_base_capture_date':None,'source_original_survey_GIS':False,
       'source_polygon':POLYGON.tolist(),'printed_axes':AXES,'implementation_sha256':file_sha256(__file__),
       'matcher':'fixed LoFTR outdoor;not mine detector','matcher_weights_sha256':file_sha256(WEIGHT),
       'native_images_and_labels_resized_or_warped':False,'matcher_only_multiple8_zero_padding':True,
       'source_mask':'maprect31:837,31:590;all main workingsx110:790,y155:580 excluded;white annotations excluded',
       'target_mask':'same printed-prior working rectangle excluded;strict observed land only',
       'SCL_excluded':[0,1,3,6,7,8,9,10,11],'matcher_confidence_min':.5,'geographic_prior_tolerance_px':15,
       'duplicate_radius_source_px':5,'duplicate_radius_native_px':2,'heldout_sites':30,'spatial_seed':51005,
       'fit_holdout_exclusion_px':4,'holdouts_allocated_before_any_fit':True,
       'gates':{'unique_sites_min':60,'fit_inliers_min':30,'median_px_max':1.5,'p90_px_max':3,'fraction_within3px_min':.9},
       'all_polygon_vertices_inside_fit_ground_hull_required':True,'labels_or_predictions_enter_matcher':False,
       'reserved_predictions':0,'production_changed':False,'preimage_is_not_a_safety_negative':True}
    pp=OUT/'protocol.json'
    if pp.exists():
        old=json.loads(pp.read_text(encoding='utf-8'))
        if any(old[k]!=v for k,v in p.items() if k!='created_at'):raise ValueError('Frozen source registration changed')
        if (OUT/'registration.json').exists():print('Frozen registration reused;no refit');return
    else:write(pp,p)
    source=np.asarray(Image.open(SOURCE).convert('RGB'));mask=np.zeros(source.shape[:2],bool);mask[31:590,31:837]=True
    mask[155:580,110:790]=False;mask[source.min(2)>230]=False
    original=Image.fromarray(source.copy());d=ImageDraw.Draw(original);d.line([tuple(v) for v in np.vstack([POLYGON,POLYGON[0]])],fill='red',width=1)
    original.save(OUT/'original_author_outline_digitisation.png')
    model=LoFTR(pretrained=None).eval();model.load_state_dict(weights());model.cuda();torch.set_num_threads(4)
    x,m=padded(source,mask);rows=[]
    for period in ('pre','post'):
        target,valid=display(CASE/(period+'_BGRN_10m.tif'))
        with rasterio.open(CASE/(period+'_SCL_10m.tif')) as f:valid &= ~np.isin(f.read(1),p['SCL_excluded'])
        with rasterio.open(CASE/(period+'_BGRN_10m.tif')) as frame:
            workings=prior(np.array([[110,155],[790,155],[790,580],[110,580]]),frame)
            tmask=valid.astype(np.uint8);cv2.fillPoly(tmask,[np.rint(workings).astype(np.int32)],0)
        y,n=padded(target,tmask>0)
        with torch.inference_mode():output=model({'image0':x,'image1':y,'mask0':m,'mask1':n})
        a=output['keypoints0'].cpu().numpy();b=output['keypoints1'].cpu().numpy();confidence=output['confidence'].cpu().numpy()
        with rasterio.open(CASE/(period+'_BGRN_10m.tif')) as frame:geo=prior(a,frame)
        unique=[]
        for i in np.argsort(-confidence):
            u,v=a[i],b[i];sx,sy=np.rint(u).astype(int);tx,ty=np.rint(v).astype(int)
            if confidence[i]<.5 or not(0<=sx<mask.shape[1] and 0<=sy<mask.shape[0] and mask[sy,sx]):continue
            if not(0<=tx<tmask.shape[1] and 0<=ty<tmask.shape[0] and tmask[ty,tx]>0):continue
            if np.linalg.norm(v-geo[i])>15:continue
            if any(np.linalg.norm(u-q['source_xy'])<5 or np.linalg.norm(v-q['target_native_xy'])<2 for q in unique):continue
            unique.append({'source_xy':u.tolist(),'target_native_xy':v.tolist(),'confidence':float(confidence[i])})
        r={'period':period,'target_sha256':file_sha256(CASE/(period+'_BGRN_10m.tif')),'raw_matcher_pairs':len(a),
           'unique_ground_sites':len(unique),'points':unique,'registration_gate_pass':False}
        if len(unique)>=60:
            aa=np.array([q['source_xy'] for q in unique],np.float32);bb=np.array([q['target_native_xy'] for q in unique],np.float32)
            cells={}
            for i,point in enumerate(aa):cells.setdefault((int(point[0]//100),int(point[1]//100)),[]).append(i)
            rng=np.random.default_rng(51005);keys=sorted(cells);rng.shuffle(keys)
            for bucket in cells.values():rng.shuffle(bucket)
            order=[]
            while any(cells.values()):
                for cell in keys:
                    if cells[cell]:order.append(cells[cell].pop())
            held=np.array(order[:30]);fit=np.array([i for i in order[30:] if np.min(np.linalg.norm(bb[i]-bb[held],axis=1))>=4],int)
            r.update(heldout_indices=held.tolist(),fit_indices=fit.tolist())
            if len(fit)>=30:
                cv2.setRNGSeed(51005);matrix,inliers=cv2.estimateAffine2D(aa[fit],bb[fit],method=cv2.RANSAC,ransacReprojThreshold=2,maxIters=10000,confidence=.999)
                if matrix is not None:
                    error=np.linalg.norm(aa[held]@matrix[:,:2].T+matrix[:,2]-bb[held],axis=1)
                    singular=np.linalg.svd(matrix[:,:2],compute_uv=False);hull=cv2.convexHull(aa[fit][inliers.ravel()>0])
                    inside=all(cv2.pointPolygonTest(hull,tuple(point.astype(float)),False)>=0 for point in POLYGON)
                    passed=bool(inliers.sum()>=30 and np.median(error)<=1.5 and np.percentile(error,90)<=3 and (error<=3).mean()>=.9
                                and .6<singular.min() and singular.max()<1.5 and singular.max()/singular.min()<1.2 and inside)
                    mapped=POLYGON@matrix[:,:2].T+matrix[:,2]
                    r.update(source_to_native_affine=matrix.tolist(),mapped_polygon_native_xy=mapped.tolist(),fit_inliers=int(inliers.sum()),
                        heldout_median_px=float(np.median(error)),heldout_p90_px=float(np.percentile(error,90)),
                        heldout_within3px_fraction=float((error<=3).mean()),heldout_errors_px=error.tolist(),
                        affine_singular_values=singular.tolist(),polygon_within_fit_ground_convex_hull=inside,registration_gate_pass=passed)
                    review=target.copy();cv2.polylines(review,[np.rint(mapped).astype(np.int32)],True,(255,70,30),1)
                    Image.fromarray(review).save(OUT/(period+'_original_outline_review.png'))
                    sheet(source,target,aa[held],bb[held],OUT/(period+'_all30_ground_holdouts.png'))
        rows.append(r);print(json.dumps({k:v for k,v in r.items() if k not in ('points','heldout_indices','fit_indices','heldout_errors_px','source_to_native_affine','mapped_polygon_native_xy')}),flush=True)
        del output,y,n;torch.cuda.empty_cache()
    result={'created_at':datetime.now().isoformat(timespec='seconds'),'protocol_sha256':file_sha256(pp),'source_sha256':file_sha256(SOURCE),
            'rows':rows,'both_original_dates_gate_pass':all(r['registration_gate_pass'] for r in rows),
            'mine_detector_predictions':0,'source_outline_is_original_professional_survey':False,'label_promoted':False}
    write(OUT/'registration.json',result)
if __name__=='__main__':sys.stdout.reconfigure(encoding='utf-8');run()
