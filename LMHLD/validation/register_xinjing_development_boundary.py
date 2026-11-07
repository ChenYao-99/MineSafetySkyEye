# -*- coding: utf-8 -*-
"""Geographic-prior/stable-ground registration of an already-observed mine figure."""
from pathlib import Path
import json,sys,hashlib
import numpy as np
import cv2
import rasterio
from rasterio.warp import transform as reproject_points
from mine_data_protocol import DATA,assert_development_cases,verify_assets
from register_las_cruces_boundary import rgb

FOLDER=DATA/'development/Xinjing';OUT=FOLDER/'source_review'
CROP=(860,25,2427,1731)


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,v):p.write_bytes((json.dumps(v,ensure_ascii=False,indent=2)+'\n').encode('utf-8'))


def run():
    assert_development_cases(['CN_XJ2023']);verify_assets()
    path=OUT/'mainmap_registration.json'
    if path.exists():
        old=json.loads(path.read_text(encoding='utf-8'))
        if old['registration_gate_pass']:print('Reuse frozen Xinjing map registration');return
    sourcepath=OUT/'chang2025_figure1_full.png';full=cv2.imdecode(np.fromfile(sourcepath,np.uint8),1)
    x0,y0,x1,y1=CROP;source=full[y0:y1,x0:x1].copy();cv2.imencode('.png',source)[1].tofile(str(OUT/'chang2025_mainmap.png'))
    b,g,r=[source[:,:,i].astype(float) for i in range(3)]
    red=((r>150)&(r>1.8*g)&(r>1.8*b)).astype(np.uint8)
    closed=cv2.morphologyEx(red,cv2.MORPH_CLOSE,np.ones((5,5),np.uint8))
    contours,_=cv2.findContours(closed,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    candidates=[c for c in contours if cv2.contourArea(c)>3000]
    if len(candidates)!=1:raise ValueError('Author red accident outline is not unique/closed; retain pending')
    polygon=cv2.approxPolyDP(candidates[0],2.,True).reshape(-1,2);body=np.zeros(red.shape,np.uint8);cv2.fillPoly(body,[polygon],1)
    cv2.imencode('.png',body*255)[1].tofile(str(OUT/'chang2025_source_footprint.png'))
    mask=(~cv2.dilate(body,np.ones((101,101),np.uint8)).astype(bool)).astype(np.uint8)*255
    mask[:45]=0;mask[-65:]=0;mask[:,:30]=0;mask[:,-110:]=0
    # Remove figure labels, arrows, scale, and coloured borders from ground sites.
    mask[0:270,0:200]=0;mask[350:900,550:1230]=0;mask[910:1290,240:650]=0;mask[1110:1540,680:1450]=0
    coloured=((r>1.6*g)&(r>1.6*b))|((r>180)&(g>150)&(b<90))
    mask[cv2.dilate(coloured.astype(np.uint8),np.ones((7,7),np.uint8))>0]=0
    target=rgb(FOLDER/'post_BGRN_10m.tif');sift=cv2.SIFT_create(nfeatures=10000,contrastThreshold=.006,edgeThreshold=14)
    tk,td=sift.detectAndCompute(cv2.cvtColor(target,cv2.COLOR_BGR2GRAY),None);sites=[]
    for scale in (.25,.5):
        ss=cv2.resize(source,None,fx=scale,fy=scale,interpolation=cv2.INTER_AREA);mm=cv2.resize(mask,(ss.shape[1],ss.shape[0]),interpolation=cv2.INTER_NEAREST)
        sk,sd=sift.detectAndCompute(cv2.cvtColor(ss,cv2.COLOR_BGR2GRAY),mm)
        for a,bm in cv2.BFMatcher().knnMatch(sd,td,k=2):
            if a.distance<.8*bm.distance:sites.append((a.distance/bm.distance,(np.array(sk[a.queryIdx].pt)+.5)/scale-.5,np.array(tk[a.trainIdx].pt)))
    sites.sort(key=lambda v:v[0]);a=np.array([v[1] for v in sites]);b=np.array([v[2] for v in sites])
    lon=105.64+(a[:,0]+x0-876)*.02/737;lat=37.98-(a[:,1]+y0-637)*.02/903
    with rasterio.open(FOLDER/'post_BGRN_10m.tif') as f:
        east,north=reproject_points('EPSG:4326',f.crs,lon.tolist(),lat.tolist());inv=~f.transform;prior=np.array([inv*(x,y) for x,y in zip(east,north)])-.5
    keep=np.linalg.norm(prior-b,axis=1)<25;a,b=a[keep],b[keep]
    unique=[]
    for i in range(len(a)):
        if any(np.linalg.norm(a[i]-a[j])<8 or np.linalg.norm(b[i]-b[j])<2 for j in unique):continue
        unique.append(i)
    a,b=a[unique].astype(np.float32),b[unique].astype(np.float32)
    if len(a)<40:
        write(OUT/'mainmap_registration_pending.json',{'unique_ground_sites':len(a),'reason':'need >=40 unique stable sites','detector_predictions_used':False});raise ValueError('Insufficient uniquely located stable sites')
    rng=np.random.default_rng(9404);order=rng.permutation(len(a));held=order[:20];fit=np.array([i for i in order[20:] if np.min(np.linalg.norm(a[held]-a[i],axis=1))>=10])
    cv2.setRNGSeed(9404);matrix,inliers=cv2.estimateAffine2D(a[fit],b[fit],method=cv2.RANSAC,ransacReprojThreshold=2,maxIters=10000,confidence=.999)
    if matrix is None:raise ValueError('Stable-ground affine fit failed')
    residual=np.linalg.norm(a[held]@matrix[:,:2].T+matrix[:,2]-b[held],axis=1)
    passed=bool(inliers.sum()>=20 and (residual<=3).mean()>=.8 and np.median(residual)<=1.5)
    result={'source':'Chang etal2025 Figure1c GF2 post2023-02-24','source_sha':sha(sourcepath),'target':'Sentinel2 post2023-02-23, date1day differs',
        'source_crop_xyxy':list(CROP),'polygon_source_xy':polygon.tolist(),'source_to_target_affine':matrix.tolist(),
        'unique_ground_sites':len(a),'fit_sites':len(fit),'fit_inliers':int(inliers.sum()),'heldout_sites':20,'heldout_median_px':float(np.median(residual)),
        'heldout_within3px':float((residual<=3).mean()),'registration_gate_pass':passed,'printed_coords_prior':'250m match-search only; graticule/scale-bar discrepancy not forced as final calibration',
        'fit_source_points':a[fit].tolist(),'fit_target_points':b[fit].tolist(),'held_source_points':a[held].tolist(),'held_target_points':b[held].tolist(),
        'detector_predictions_used':False,'new_blind_case':False,'independent_expert_review':False,
        'publication_date_conflicts':'Zhang2024 section2 preFeb16 vs Figure5aFeb18; Chang2025 preNov13. PostreferencesFeb23/24. Never claim matching pre dates.',
        'second_publication_independence_limit':'Both show630/520m dimensions and may share images/geometry; not two independent expertGIS truths'}
    write(path,result)
    if not passed:raise ValueError('Held-out registration gate failed; boundary stays pending')
    warped=cv2.warpAffine(source,matrix,(target.shape[1],target.shape[0]));valid=warped.sum(2)>0;review=target.copy();review[valid]=(.5*target[valid]+.5*warped[valid]).astype(np.uint8)
    cv2.imencode('.png',review)[1].tofile(str(OUT/'mainmap_registration_review.png'))
    print({k:result[k] for k in ('unique_ground_sites','fit_inliers','heldout_median_px','heldout_within3px','registration_gate_pass')},flush=True)


if __name__=='__main__':sys.stdout.reconfigure(encoding='utf-8');run()
