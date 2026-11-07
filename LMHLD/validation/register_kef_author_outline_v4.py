"""Original author 2022 NW pit-wall outline, registered on held-out terrain only."""
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
from register_gamsberg_published_boundary_v5 import padded,sheet,weights,WEIGHT
CASE=ROOT/'datasets/kef_essnoun_development_20261005/DZ_KEF2022'
SOURCE=ROOT/'datasets/pixel_boundary_sources_20261004/wang_supplement_original_media/image29.png'
OUT=CASE/'reference_registration_v4_mutual_sift'
POLYGON=np.array([[269,233],[274,229],[281,225],[289,222],[297,221],[304,221],[309,224],
                  [308,230],[303,235],[299,240],[292,243],[283,244],[275,243],[270,240],[268,236]],np.float64)
AXES={'longitude':{'source_x':[163.,300.,437.,574.,711.],
    'degrees':[7+56/60+31/3600,7+57/60+50/3600,7+59/60+10/3600,8+29/3600,8+1/60+48/3600]},
      'latitude':{'source_y':[143.,264.,385.,506.],
    'degrees':[34+43/60+12/3600,34+42/60,34+40/60+48/3600,34+39/60+36/3600]}}
WORKINGS=np.array([[220,185],[400,185],[545,212],[540,368],[425,387],[223,375]],np.float32)
def write(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def prior(points,frame):
    lon=np.polyval(np.polyfit(AXES['longitude']['source_x'],AXES['longitude']['degrees'],1),points[:,0])
    lat=np.polyval(np.polyfit(AXES['latitude']['source_y'],AXES['latitude']['degrees'],1),points[:,1])
    x,y=coordinates('EPSG:4326',frame.crs,lon.tolist(),lat.tolist());inverse=~frame.transform
    return np.array([inverse*(a,b) for a,b in zip(x,y)])-.5
def target_display(path,quality):
    with rasterio.open(path) as f:a=f.read()[:3].transpose(1,2,0).astype(np.float32);valid=f.dataset_mask()>0
    valid &= ~np.isin(quality,[0,1,3,6,7,8,9,10,11])
    for band in range(3):
        low,high=np.percentile(a[:,:,band][valid],[2,98]);a[:,:,band]=np.clip((a[:,:,band]-float(low))/max(float(high-low),1e-6),0,1)
    a[~valid]=0;return (a[:,:,::-1]*255).astype(np.uint8),valid
def run():
    assert_development_cases(['DZ_KEF2022']);verify_assets();OUT.mkdir(exist_ok=True)
    p={'created_at':datetime.now().isoformat(timespec='seconds'),'case_id':'DZ_KEF2022','role':'development',
       'source_figure':'Wang2026S29 white NW outline;S30 explicitly7Sep2022;not whole2007pitfailure',
       'source_sha256':file_sha256(SOURCE),'source_base_capture_date':None,'polygon_original_source_xy':POLYGON.tolist(),
       'printed_graticule':AXES,'primary_field_event_corroboration':'https://link.springer.com/article/10.1007/s10706-024-03027-5',
       'public_publisher_previews_not_label_or_model_input':True,'published_field_campaign_is_not_project_field_feedback':True,
       'source_workings_excluded':WORKINGS.tolist(),'source_legend_scalebar_white_and_coloured_annotations_excluded':True,
       'target_quality_classes_excluded':[0,1,3,6,7,8,9,10,11],'matcher':'SIFT localdescriptors;mutualnearest ratio.8;not mine detector','LoFTR_prior_failures_preserved':True,
       'matcher_weight_sha256':file_sha256(WEIGHT),'descriptor_quality_min':.2,'mutual_ratio_max':.8,'geographic_prior_tolerance_native_px':15,
       'duplicate_radius_source_px':5,'duplicate_radius_native_px':2,'heldout_before_fit':30,'holdout_seed':52105,
       'fit_near_holdouts_excluded_native_px':4,'holdout_allocation_before_any_fit':True,
       'gates':{'unique_sites_min':90,'fit_inliers_min':30,'holdout_median_max_px':1.5,'holdout_p90_max_px':3,'fraction_within3px_min':.9},
       'source_polygon_within_ground_fit_hull_required':True,'raw_images_or_labels_resized_or_shifted':False,
       'padded_for_matcher_only':True,'source_implementation_sha256':file_sha256(__file__),'accident_detector_predictions':0,
       'original_independent_survey_GIS':False,'unknown_outside_conservative_positive_core':True,'uncertainty_core_buffer_m_planned':80}
    pp=OUT/'protocol.json'
    if pp.exists():
        old=json.loads(pp.read_text(encoding='utf-8'))
        if any(old[k]!=v for k,v in p.items() if k!='created_at'):raise ValueError('Frozen source quality method changed')
        if (OUT/'registration.json').exists():print('Frozen source registration reused;notrefitted');return
    else:write(pp,p)
    source=np.asarray(Image.open(SOURCE).convert('RGB'));mask=np.zeros(source.shape[:2],np.uint8);mask[37:540,37:829]=1
    cv2.fillPoly(mask,[np.rint(WORKINGS).astype(np.int32)],0);mask[37:230,705:]=0;mask[486:,700:]=0
    r,g,b=[source[:,:,i].astype(float) for i in range(3)]
    annotated=((r>190)&(g>190)&(b<110))|((r>180)&(g<90)&(b<120))|((r>150)&(b>150)&(g<100))|((b>160)&(r<110)&(g<130))|(source.min(2)>235)
    mask[cv2.dilate(annotated.astype(np.uint8),np.ones((5,5),np.uint8))>0]=0
    review=Image.fromarray(source.copy());d=ImageDraw.Draw(review);d.line([tuple(v) for v in np.vstack([POLYGON,POLYGON[0]])],fill='orange',width=1);review.save(OUT/'original_author_outline_digitisation.png')
    sift=cv2.SIFT_create(nfeatures=30000,contrastThreshold=.008);clahe=cv2.createCLAHE(2,(8,8));source_gray=clahe.apply(cv2.cvtColor(source,cv2.COLOR_RGB2GRAY));source_keypoints,source_descriptors=sift.detectAndCompute(source_gray,mask*255);rows=[]
    for period in ('pre','post'):
        path=CASE/(period+'_BGRN_10m.tif')
        with rasterio.open(CASE/(period+'_SCL_10m.tif')) as f:quality=f.read(1)
        target,valid=target_display(path,quality);tmask=valid.astype(np.uint8)
        with rasterio.open(path) as frame:mapped_workings=prior(WORKINGS,frame)
        excluded=np.zeros(tmask.shape,np.uint8);cv2.fillPoly(excluded,[np.rint(mapped_workings).astype(np.int32)],1)
        tmask[cv2.dilate(excluded,np.ones((11,11),np.uint8))>0]=0
        target_gray=clahe.apply(cv2.cvtColor(target,cv2.COLOR_RGB2GRAY));target_keypoints,target_descriptors=sift.detectAndCompute(target_gray,tmask*255)
        pairs=[]
        if source_descriptors is not None and target_descriptors is not None:
            matcher=cv2.BFMatcher();forward=matcher.knnMatch(source_descriptors,target_descriptors,k=2);reverse=matcher.knnMatch(target_descriptors,source_descriptors,k=2)
            backward={a.queryIdx:a.trainIdx for match in reverse if len(match)==2 for a,b in [match] if a.distance<.8*b.distance}
            pairs=[(a,b) for match in forward if len(match)==2 for a,b in [match] if a.distance<.8*b.distance and backward.get(a.trainIdx)==a.queryIdx]
        aa=np.array([source_keypoints[a.queryIdx].pt for a,b in pairs],np.float32).reshape(-1,2);bb=np.array([target_keypoints[a.trainIdx].pt for a,b in pairs],np.float32).reshape(-1,2)
        cc=np.array([1-a.distance/max(b.distance,1e-6) for a,b in pairs],np.float32)
        with rasterio.open(path) as frame:expected=prior(aa,frame)
        unique=[]
        for i in np.argsort(-cc):
            a,b=aa[i],bb[i];sx,sy=np.rint(a).astype(int);tx,ty=np.rint(b).astype(int)
            if cc[i]<.2 or not(0<=sx<mask.shape[1] and 0<=sy<mask.shape[0] and mask[sy,sx]):continue
            if not(0<=tx<tmask.shape[1] and 0<=ty<tmask.shape[0] and tmask[ty,tx]):continue
            if np.linalg.norm(b-expected[i])>15:continue
            if any(np.linalg.norm(a-q['source_xy'])<5 or np.linalg.norm(b-q['target_native_xy'])<2 for q in unique):continue
            unique.append({'source_xy':a.tolist(),'target_native_xy':b.tolist(),'confidence':float(cc[i])})
        row={'period':period,'raw_matcher_pairs':len(aa),'unique_ground_sites':len(unique),'points':unique,'target_sha256':file_sha256(path),'registration_gate_pass':False}
        if len(unique)>=90:
            a=np.array([q['source_xy'] for q in unique],np.float32);b=np.array([q['target_native_xy'] for q in unique],np.float32)
            cells={}
            for i,point in enumerate(a):cells.setdefault((int(point[0]//100),int(point[1]//100)),[]).append(i)
            rng=np.random.default_rng(52105);keys=sorted(cells);rng.shuffle(keys)
            for bucket in cells.values():rng.shuffle(bucket)
            order=[]
            while any(cells.values()):
                for key in keys:
                    if cells[key]:order.append(cells[key].pop())
            held=np.array(order[:30]);fit=np.array([i for i in order[30:] if np.min(np.linalg.norm(b[i]-b[held],axis=1))>=4],int)
            row.update(heldout_indices=held.tolist(),fit_indices=fit.tolist())
            if len(fit)>=30:
                cv2.setRNGSeed(52105);matrix,inliers=cv2.estimateAffine2D(a[fit],b[fit],method=cv2.RANSAC,ransacReprojThreshold=2,maxIters=10000,confidence=.999)
                if matrix is not None:
                    error=np.linalg.norm(a[held]@matrix[:,:2].T+matrix[:,2]-b[held],axis=1);singular=np.linalg.svd(matrix[:,:2],compute_uv=False)
                    hull=cv2.convexHull(a[fit][inliers.ravel()>0]);inside=all(cv2.pointPolygonTest(hull,tuple(v.astype(float)),False)>=0 for v in POLYGON)
                    passed=bool(inliers.sum()>=30 and np.median(error)<=1.5 and np.percentile(error,90)<=3 and (error<=3).mean()>=.9 and .8<singular.min() and singular.max()<2.5 and singular.max()/singular.min()<1.5 and inside)
                    mapped=POLYGON@matrix[:,:2].T+matrix[:,2]
                    row.update(source_to_native_affine=matrix.tolist(),mapped_polygon_native_xy=mapped.tolist(),fit_inliers=int(inliers.sum()),
                        heldout_errors_px=error.tolist(),heldout_median_px=float(np.median(error)),heldout_p90_px=float(np.percentile(error,90)),
                        heldout_within3px_fraction=float((error<=3).mean()),affine_singular_values=singular.tolist(),polygon_within_fit_ground_convex_hull=inside,registration_gate_pass=passed)
                    show=target.copy();cv2.polylines(show,[np.rint(mapped).astype(np.int32)],True,(255,70,30),1);Image.fromarray(show).save(OUT/(period+'_original_outline_review.png'))
                    sheet(source,target,a[held],b[held],OUT/(period+'_all30_ground_holdouts.png'))
        rows.append(row);print(json.dumps({k:v for k,v in row.items() if k not in ('points','heldout_indices','fit_indices','heldout_errors_px','source_to_native_affine','mapped_polygon_native_xy')}),flush=True)
        del target_gray,target_descriptors
    write(OUT/'registration.json',{'created_at':datetime.now().isoformat(timespec='seconds'),'protocol_sha256':file_sha256(pp),'rows':rows,
            'both_original_dates_gate_pass':all(r['registration_gate_pass'] for r in rows),'detector_predictions':0,'pixel_reference_admitted':False})
if __name__=='__main__':sys.stdout.reconfigure(encoding='utf-8');run()
