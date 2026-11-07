"""Register dated reference using ground outside failure/dust; hold points out before fitting."""
from pathlib import Path
from datetime import datetime
import json,sys
import cv2
import numpy as np
import rasterio
from PIL import Image
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from model_registry import file_sha256
from prepare_winter_constructed_normals import display
DATA=ROOT/'datasets/arid_accident_operation_20261004';SRC=DATA/'source_review/gamsberg_skysat_full_20201203.jpg'
OUT=DATA/'ZA_GAMS2020S/reference_registration_v1';CASE=DATA/'ZA_GAMS2020S'


def write(p,v):p.write_bytes((json.dumps(v,ensure_ascii=False,indent=2)+'\n').encode('utf-8'))


def run():
    OUT.mkdir(exist_ok=True);raw=np.asarray(Image.open(SRC).convert('RGB'));target,_=display(CASE/'post_BGRN_10m.tif')
    # Failure, possible dust, and neighbouring uncertain disturbed faces excluded from fit/validation.
    source_ok=np.ones(raw.shape[:2],np.uint8)*255;source_ok[120:1360,460:1880]=0
    source_ok[:210,420:1720]=0
    cv2.setRNGSeed(7104);sift=cv2.SIFT_create(nfeatures=8000,contrastThreshold=.006)
    clahe=cv2.createCLAHE(2,(8,8));ground=[]
    # Resampling here serves registration only. Candidate inputs remain real10m arrays.
    for target_scale in (2,4):
        enlarged=cv2.resize(target,None,fx=target_scale,fy=target_scale,interpolation=cv2.INTER_CUBIC)
        tgray=clahe.apply(cv2.cvtColor(enlarged,cv2.COLOR_RGB2GRAY));tk,td=sift.detectAndCompute(tgray,None)
        for source_scale in (.035,.045,.055,.07,.09,.12,.16):
            image=cv2.resize(raw,None,fx=source_scale,fy=source_scale,interpolation=cv2.INTER_AREA)
            mask=cv2.resize(source_ok,(image.shape[1],image.shape[0]),interpolation=cv2.INTER_NEAREST)
            sk,sd=sift.detectAndCompute(clahe.apply(cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)),mask)
            if sd is None or td is None:continue
            matches=[m for pair in cv2.BFMatcher().knnMatch(sd,td,k=2) if len(pair)==2 for m,n in [pair] if m.distance<.78*n.distance]
            if len(matches)<8:continue
            a=np.array([sk[m.queryIdx].pt for m in matches],np.float32)/source_scale
            b=np.array([tk[m.trainIdx].pt for m in matches],np.float32)/target_scale
            matrix,inliers=cv2.estimateAffinePartial2D(a,b,method=cv2.RANSAC,ransacReprojThreshold=2,maxIters=5000)
            if matrix is None:continue
            scale=float(np.linalg.norm(matrix[:,0]))
            if not .01<scale<.08:continue
            for i,m in enumerate(matches):
                if inliers[i,0]:ground.append({'source_xy':a[i].tolist(),'target_native_xy':b[i].tolist(),'descriptor_distance':float(m.distance)})
    unique=[]
    for point in sorted(ground,key=lambda p:p['descriptor_distance']):
        a=np.array(point['source_xy']);b=np.array(point['target_native_xy'])
        if any(np.linalg.norm(a-np.array(q['source_xy']))<40 or np.linalg.norm(b-np.array(q['target_native_xy']))<2 for q in unique):continue
        unique.append(point)
    p={'created_at':datetime.now().isoformat(timespec='seconds'),'source_sha256':file_sha256(SRC),
       'target_original_sha256':file_sha256(CASE/'post_BGRN_10m.tif'),'source_capture_date':'2020-12-03','target_date':'2020-12-04',
       'source_only_reference_not_model_input':True,'raw_or_training_images_spatially_resized':False,
       'registration_display_resampling_creates_no_new_observations':True,'failure_and_possible_dust_excluded_from_control_points':True,
       'unique_native_ground_points':len(unique),'points':unique,'new_expert_GIS':False,'registration_gate_pass':False}
    if len(unique)>=30:
        a=np.array([r['source_xy'] for r in unique],np.float32);b=np.array([r['target_native_xy'] for r in unique],np.float32)
        # Spatial ordering chooses fixed geographically spread validation points, not small-residual cherry-picking.
        order=np.argsort(np.arctan2(b[:,1]-np.median(b[:,1]),b[:,0]-np.median(b[:,0])))
        held=order[::5];train=np.setdiff1d(np.arange(len(a)),held)
        matrix,inliers=cv2.estimateAffinePartial2D(a[train],b[train],method=cv2.RANSAC,ransacReprojThreshold=2,maxIters=5000)
        if matrix is not None:
            predicted=a[held]@matrix[:,:2].T+matrix[:,2];res=np.linalg.norm(predicted-b[held],axis=1)
            p.update(source_to_native_affine=matrix.tolist(),fit_ground_points=int(inliers.sum()),spatial_holdout_indices=held.tolist(),
                holdout_errors_px=res.tolist(),holdout_median_px=float(np.median(res)),holdout_p90_px=float(np.percentile(res,90)),
                holdout_fraction_within3px=float((res<=3).mean()),
                registration_gate_pass=bool(inliers.sum()>=20 and len(held)>=6 and np.median(res)<=2 and (res<=3).mean()>=.9))
    write(OUT/'registration.json',p)
    print(json.dumps({k:v for k,v in p.items() if k not in ('points','source_to_native_affine')},ensure_ascii=False))


if __name__=='__main__':sys.stdout.reconfigure(encoding='utf-8');run()
