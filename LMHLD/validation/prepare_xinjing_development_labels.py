# -*- coding: utf-8 -*-
"""New reference version for historical Xinjing, calibrated originals remain separate."""
from pathlib import Path
import json,sys,hashlib
import numpy as np
import cv2
import rasterio
from mine_data_protocol import DATA,REGISTRY,registry,assert_development_cases,verify_assets
from register_las_cruces_boundary import rgb

FOLDER=DATA/'development/Xinjing';OUT=FOLDER/'reference_v1_20261004'
NORMALS=[('eastern_haul_bench',[[414,258],[425,267],[417,289],[404,307],[398,301],[409,281]]),
 ('southern_haul_road',[[392,418],[416,418],[445,439],[442,448],[420,431],[393,427]]),
 ('western_haul_access',[[243,338],[253,341],[264,353],[258,359],[250,350],[241,346]]),
 ('eastern_dump_access',[[493,193],[509,197],[514,211],[509,223],[499,216],[493,202]])]


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,v):p.write_bytes((json.dumps(v,ensure_ascii=False,indent=2)+'\n').encode('utf-8'))


def run():
    assert_development_cases(['CN_XJ2023']);verify_assets();OUT.mkdir(exist_ok=True);path=OUT/'label_reference.json'
    if path.exists():
        p=json.loads(path.read_text(encoding='utf-8'))
        for name,digest in p['frozen_assets'].items():
            if sha(OUT/name)!=digest:raise ValueError('Frozen Xinjing reference changed')
        print('Reusing Xinjing development reference');return
    registration=json.loads((FOLDER/'source_review/mainmap_registration.json').read_text(encoding='utf-8'))
    if not registration['registration_gate_pass']:raise ValueError('Boundary registration pending')
    matrix=np.array(registration['source_to_target_affine']);points=np.array(registration['polygon_source_xy'])@matrix[:,:2].T+matrix[:,2]
    cal=DATA/'radiometry_audit_20261004/Xinjing'
    with rasterio.open(FOLDER/'post_BGRN_10m.tif') as f:shape=(f.height,f.width);profile=f.profile.copy();valid=f.dataset_mask()>0
    with rasterio.open(FOLDER/'post_SCL_10m.tif') as f:post_scl=f.read(1)
    with rasterio.open(FOLDER/'pre_SCL_10m.tif') as f:pre_scl=f.read(1)
    with rasterio.open(cal/'pre_snow_bright_obscuration_candidate.tif') as f:pre_weather=f.read(1)>0
    with rasterio.open(cal/'post_snow_bright_obscuration_candidate.tif') as f:post_weather=f.read(1)>0
    truth=np.zeros(shape,np.uint8);cv2.fillPoly(truth,[np.round(points).astype(np.int32)],1)
    post_observed=valid&~np.isin(post_scl,[0,1,8,9,10,11])&~post_weather
    positive=(cv2.erode(truth,np.ones((11,11),np.uint8))>0)&post_observed
    forbidden=cv2.dilate(truth,np.ones((21,21),np.uint8))>0
    normal=np.zeros(shape,bool);regions=[]
    pair_visible=post_observed&~np.isin(pre_scl,[0,1,3,6,8,9,10,11])&~pre_weather
    for name,polygon in NORMALS:
        m=np.zeros(shape,np.uint8);cv2.fillPoly(m,[np.array(polygon,np.int32)],1);m=cv2.erode(m,np.ones((3,3),np.uint8))>0
        accepted=m&pair_visible&~forbidden
        if accepted.sum()<20:raise ValueError('Constructed normal region insufficient; hold pending')
        normal|=accepted;regions.append({'id':name,'polygon_pixel_xy':polygon,'accepted_pixels':int(accepted.sum()),'review_type':'project original-pair visual weakreference,not expert or geotechnical safety'})
    y=np.full(shape,255,np.uint8);y[positive]=1;y[normal]=0;w=np.zeros(shape,np.float32);w[y==1]=.6;w[y==0]=.25
    if positive.sum()<100 or normal.sum()<100:raise ValueError('Insufficient visible conservative support')
    # No-data/brightness candidates affect observation availability, never create
    # a negative label over the remaining mine or automatically remove benchmark errors.
    pre_available=~np.isin(pre_scl,[0,1,8,9,10,11])&~pre_weather
    files=[]
    for name,a,dtype,nodata in (('published_footprint.tif',truth,'uint8',None),('labels.tif',y,'uint8',255),('confidence.tif',w,'float32',None),('normal_mask.tif',normal.astype(np.uint8),'uint8',None),('pre_available.tif',pre_available.astype(np.uint8),'uint8',None)):
        pr=profile.copy();pr.update(count=1,dtype=dtype,nodata=nodata,compress='deflate')
        with rasterio.open(OUT/name,'w',**pr) as f:f.write(a,1)
        files.append(name)
    # Original pair stable-ground alignment, never an inferred displacement.
    pre,post=rgb(FOLDER/'pre_BGRN_10m.tif'),rgb(FOLDER/'post_BGRN_10m.tif');mask=(pair_visible&~forbidden).astype(np.uint8)*255
    sift=cv2.SIFT_create(nfeatures=7000,contrastThreshold=.015);ak,ad=sift.detectAndCompute(cv2.cvtColor(pre,cv2.COLOR_BGR2GRAY),mask);bk,bd=sift.detectAndCompute(cv2.cvtColor(post,cv2.COLOR_BGR2GRAY),mask)
    matches=sorted([a for a,b in cv2.BFMatcher().knnMatch(ad,bd,k=2) if a.distance<.7*b.distance],key=lambda a:a.distance)
    a=[];b=[]
    for m in matches:
        sa=np.array(ak[m.queryIdx].pt);tb=np.array(bk[m.trainIdx].pt)
        if any(np.linalg.norm(sa-x)<3 or np.linalg.norm(tb-z)<3 for x,z in zip(a,b)):continue
        a.append(sa);b.append(tb)
    a,b=np.array(a,np.float32),np.array(b,np.float32);cv2.setRNGSeed(9406);affine,inliers=cv2.estimateAffinePartial2D(a,b,method=cv2.RANSAC,ransacReprojThreshold=2.)
    if affine is None:raise ValueError('Pair stable-ground alignment pending')
    use=inliers.ravel()>0;residual=np.linalg.norm(a[use]-b[use],axis=1);pair_pass=bool(use.sum()>=30 and np.percentile(residual,90)<2.)
    alignment={'unique_matches':len(a),'inliers':int(use.sum()),'raw_grid_median_px':float(np.median(residual)),'raw_grid_p90_px':float(np.percentile(residual,90)),
        'raw_pair_gate_pass':pair_pass,'raw_images_warped':False,'not_landslide_displacement':True,'project_model_predictions':0}
    write(OUT/'pair_alignment.json',alignment);files.append('pair_alignment.json')
    if not pair_pass:raise ValueError('Original pair grid did not pass; labels stay unaccepted')
    review=(post*.4).astype(np.uint8);review[y==1]=(.65*post[y==1]+.35*np.array([40,230,40])).astype(np.uint8);review[y==0]=(.65*post[y==0]+.35*np.array([230,150,30])).astype(np.uint8)
    cv2.polylines(review,[np.round(points).astype(np.int32)],True,(30,60,240),1);cv2.imencode('.png',review)[1].tofile(str(OUT/'reference_review.png'))
    for image in (pre,post):
        for r in regions:cv2.polylines(image,[np.array(r['polygon_pixel_xy'],np.int32)],True,(40,220,240),1)
    cv2.imencode('.png',np.hstack([pre,post]))[1].tofile(str(OUT/'normal_pair_review.png'))
    reference={'case_id':'CN_XJ2023','role':'development','historical_event_exposure':True,'not_new_blind_test':True,
        'positive_pixels':int(positive.sum()),'weak_normal_pixels':int(normal.sum()),'unknown_pixels':int((y==255).sum()),'full_footprint_m2':int(truth.sum())*100,
        'positive_boundary_unknown_buffer_m':50,'normal_affected_buffer_m':100,'source_date':'2023-02-24GF2','target_date':'2023-02-23Sentinel2',
        'author_original_expert_GIS_received':False,'independent_project_expert_review':False,'quality':'registered authorfigure core + project weak constructed normals, development-only',
        'two_source_independence_limit':registration['second_publication_independence_limit'],'publication_date_conflicts':registration['publication_date_conflicts'],
        'normal_regions':regions,'radiometry_audit':'../../../radiometry_audit_20261004/audit.json','true_S2_BOA_DN_offset':-1000,
        'no_model_outputs_for_labels':True,'calibrated_derivatives_not_originals_modified':True,'pair_alignment':alignment,
        'frozen_assets':{n:sha(OUT/n) for n in files},'registration_sha':sha(FOLDER/'source_review/mainmap_registration.json')}
    write(path,reference);value=registry();c=value['cases']['CN_XJ2023'];c.update(quality_review_status='accepted_for_development_reference_only',label_reference=str(path.relative_to(DATA)).replace('\\','/'),
        label_quality=reference['quality'],unresolved_quality=['original expertGIS unavailable','1-day target/reference gap and source date conflicts','normal surfaces project-reviewed weak,not stability proof'])
    write(REGISTRY,value);verify_assets();print({k:reference[k] for k in ('positive_pixels','weak_normal_pixels','full_footprint_m2','unknown_pixels')});print(alignment)


if __name__=='__main__':sys.stdout.reconfigure(encoding='utf-8');run()
