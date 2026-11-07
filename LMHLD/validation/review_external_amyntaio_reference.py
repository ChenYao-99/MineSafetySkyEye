# -*- coding: utf-8 -*-
"""Independent-author image-change corroboration, never a new expert boundary."""
from pathlib import Path
import json,hashlib,sys
import cv2
import numpy as np
import rasterio
from mine_data_protocol import DATA,assert_development_cases,verify_assets
from register_las_cruces_boundary import rgb

OUT=DATA/'review_v2_20261003';CASE=DATA/'development/Amyntaio'


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,v):p.write_bytes((json.dumps(v,ensure_ascii=False,indent=2)+'\n').encode('utf-8'))


def run():
    assert_development_cases(['Amyntaio']);verify_assets();out=OUT/'boundary_audit';out.mkdir(exist_ok=True)
    path=out/'independent_author_registration.json'
    if path.exists():
        old=json.loads(path.read_text(encoding='utf-8'))
        if old.get('registration_gate_pass'):print('Reusing frozen independent-author registration');return
    source_path=OUT/'sources/figure11_xref51.jpeg';source=cv2.imdecode(np.fromfile(source_path,np.uint8),1)
    target=rgb(CASE/'post_BGRN_10m.tif')
    red=(source[:,:,2]>190)&(source[:,:,1]<90)&(source[:,:,0]<90)
    smask=(~cv2.dilate(red.astype(np.uint8),np.ones((41,41),np.uint8)).astype(bool)).astype(np.uint8)*255
    smask[:12]=0;smask[-12:]=0;smask[:,:12]=0;smask[:,-12:]=0
    with rasterio.open(CASE/'published_debris_reference.tif') as f:body=f.read(1)>0
    tmask=(~cv2.dilate(body.astype(np.uint8),np.ones((31,31),np.uint8)).astype(bool)).astype(np.uint8)*255
    sift=cv2.SIFT_create(nfeatures=10000,contrastThreshold=.008,edgeThreshold=14)
    sk,sd=sift.detectAndCompute(cv2.cvtColor(source,cv2.COLOR_BGR2GRAY),smask)
    tk,td=sift.detectAndCompute(cv2.cvtColor(target,cv2.COLOR_BGR2GRAY),tmask)
    matches=sorted([a for a,b in cv2.BFMatcher().knnMatch(sd,td,k=2) if a.distance<.70*b.distance],key=lambda m:m.distance)
    aa=[];bb=[]
    for m in matches:
        a=np.array(sk[m.queryIdx].pt);b=np.array(tk[m.trainIdx].pt)
        if any(np.linalg.norm(a-p)<4 or np.linalg.norm(b-q)<3 for p,q in zip(aa,bb)):continue
        aa.append(a);bb.append(b)
    a,b=np.array(aa,np.float32),np.array(bb,np.float32)
    if len(a)<40:raise ValueError('Insufficient unique stable-ground sites; keep external reference qualitative')
    blocks=np.floor(a/48).astype(np.int64);held=((blocks[:,0]*73856093)^(blocks[:,1]*19349663))%5==0
    hi=np.flatnonzero(held);fit=np.array([i for i in np.flatnonzero(~held) if len(hi) and np.min(np.linalg.norm(a[hi]-a[i],axis=1))>=8])
    if len(hi)<20 or len(fit)<20:raise ValueError('Spatial holdout support insufficient; do not loosen after seeing residuals')
    cv2.setRNGSeed(9310);matrix,inliers=cv2.estimateAffine2D(a[fit],b[fit],method=cv2.RANSAC,ransacReprojThreshold=2.,maxIters=10000,confidence=.999)
    if matrix is None:raise ValueError('Independent source failed stable-ground registration')
    residual=np.linalg.norm(a[hi]@matrix[:,:2].T+matrix[:,2]-b[hi],axis=1)
    passed=bool(inliers.sum()>=20 and (residual<=3).mean()>=.8 and np.median(residual)<=1.5)
    record={'source':'Karagianni/Lazos/Chatzipetros2018 Figure11, DOI10.5194/isprs-archives-XLII-3-W4-269-2018',
        'source_sha':sha(source_path),'target_sha':sha(CASE/'post_BGRN_10m.tif'),'source_to_target_affine':matrix.tolist(),
        'unique_ground_matches':len(a),'fit_sites':len(fit),'fit_inliers':int(inliers.sum()),'spatial_holdout_sites':len(hi),
        'holdout_within3px':float((residual<=3).mean()),'holdout_median_px':float(np.median(residual)),
        'spatial_split':'48source-pixel blocks, deterministic hashmod5,8px source gap; remove orientation duplicates first',
        'source_points':a.tolist(),'target_points':b.tolist(),'heldout_indices':hi.tolist(),'registration_gate_pass':passed,
        'project_detector_outputs_used':False,'new_independent_expert_feedback':False,
        'reference_type':'author ROI-constrained thresholded image-change support; not complete accident boundary or expert GIS'}
    write(path,record)
    if not passed:raise ValueError('External registration failed held-out ground-site gate')
    support=cv2.warpAffine(red.astype(np.uint8),matrix,(target.shape[1],target.shape[0]),flags=cv2.INTER_NEAREST)>0
    with rasterio.open(CASE/'partial_labels.tif') as f:y=f.read(1);profile=f.profile.copy()
    profile.update(nodata=None,dtype='uint8',compress='deflate')
    with rasterio.open(out/'amyntaio_external_change_support.tif','w',**profile) as f:f.write(support.astype(np.uint8),1)
    v={'support_pixels':int(support.sum()),'original_positive_core_support_fraction':float(support[y==1].mean()),
        'support_outside_original_footprint_pixels':int((support&~body).sum()),'not_boundary_IoU_accuracy':True,
        'no_old_labels_modified':True,'not_used_to_filter_evaluation_negatives':True,'model_outputs_used':False,
        'ESA_SAR_reference':'sources/esa_sar_source.json; qualitative cross-sensor check only, no fabricated georeferencing'}
    write(out/'cross_source_corroboration.json',v)
    display=target.copy();display[support]=(.5*display[support]+.5*np.array([30,50,240])).astype(np.uint8)
    cv2.imencode('.png',display)[1].tofile(str(out/'external_support_review.png'))
    print(json.dumps({**v,'registration_holdout':len(hi),'median_px':float(np.median(residual))}),flush=True)


if __name__=='__main__':sys.stdout.reconfigure(encoding='utf-8');run()
