# -*- coding: utf-8 -*-
"""Transfer the authors accident footprint to real imagery with explicit unknowns."""
import hashlib
import json
import sys
import cv2
import numpy as np
import rasterio
from rasterio.warp import transform_geom
from mine_data_protocol import DATA, REGISTRY, registry, assert_development_cases
from register_las_cruces_boundary import rgb

ROOT=DATA/'development/LasCruces'
OLD=DATA.parents[1]/'experiments/mining_validation_20261001'
# Project visual review of the original pre/post pair only, never model predictions.
# Farms/roads/plant and a small apparently unchanged eastern pit area are weak
# event-background references. No claim that these surfaces are geotechnically safe.
NEGATIVE_BOXES_NATIVE=[[45,20,120,80],[130,20,195,85],[230,20,290,75],
                       [135,350,185,395],[285,345,345,410],[276,177,310,252]]
INVALID_SCL=(0,1,8,9,10,11)


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def run():
    assert_development_cases(['LasCruces'])
    reference_path=ROOT/'label_reference.json'
    if reference_path.exists():
        saved=json.loads(reference_path.read_text(encoding='utf-8'))
        for name,digest in saved['frozen_assets'].items():
            if sha(ROOT/name)!=digest: raise ValueError('Frozen native reference changed')
        print('Reusing frozen Las Cruces development reference'); return
    registration=json.loads((ROOT/'source_review/registration_diagnostic.json').read_text(encoding='utf-8'))
    if not registration['registration_gate_pass']: raise ValueError('Registration review not passed')
    matrix=np.array(registration['source_to_target_affine'])
    source_mask=cv2.imdecode(np.fromfile(OLD/'cases/las_cruces_201901_reference.png',np.uint8),0)
    contours,_=cv2.findContours((source_mask>0).astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    source_polygon=cv2.approxPolyDP(max(contours,key=cv2.contourArea),1.,True).reshape(-1,2).astype(float)
    pixels=source_polygon@matrix[:,:2].T+matrix[:,2]
    with rasterio.open(ROOT/'post_BGRN_10m.tif') as src:
        raw=src.read(); profile=src.profile.copy(); transform,crs=src.transform,src.crs
        valid=(raw>0).all(0)
    with rasterio.open(ROOT/'post_SCL_10m.tif') as src: scl=src.read(1)
    known_optical=valid&~np.isin(scl,INVALID_SCL)
    shape=valid.shape; truth=np.zeros(shape,np.uint8)
    cv2.fillPoly(truth,[np.round(pixels).astype(np.int32)],1)
    positive=cv2.erode(truth,np.ones((11,11),np.uint8))>0
    negative=np.zeros(shape,np.uint8)
    for x0,y0,x1,y1 in NEGATIVE_BOXES_NATIVE: negative[y0:y1,x0:x1]=1
    negative=cv2.erode(negative,np.ones((5,5),np.uint8))>0
    negative&=~cv2.dilate(truth,np.ones((15,15),np.uint8)).astype(bool)
    labels=np.full(shape,255,np.uint8)
    labels[positive&known_optical]=1; labels[negative&known_optical]=0
    weights=np.zeros(shape,np.float32); weights[labels==1]=.6; weights[labels==0]=.25
    for name,array,dtype,nodata in (('published_accident_reference.tif',truth,'uint8',None),
                                    ('partial_labels.tif',labels,'uint8',255),
                                    ('label_confidence.tif',weights,'float32',None)):
        meta=profile.copy(); meta.update(count=1,dtype=dtype,nodata=nodata,compress='deflate')
        with rasterio.open(ROOT/name,'w',**meta) as dst: dst.write(array,1)
    coords=[list(transform*(float(x)+.5,float(y)+.5)) for x,y in pixels]; coords.append(coords[0])
    geometry=transform_geom(crs,'EPSG:4326',{'type':'Polygon','coordinates':[coords]})
    feature={'type':'FeatureCollection','features':[{'type':'Feature','properties':{
        'mine':'LasCruces','event':'2019-01-23','label_quality':'digitized published external footprint; not original expert GIS'},'geometry':geometry}]}
    (ROOT/'accident_reference_wgs84.geojson').write_text(json.dumps(feature,indent=2)+'\n',encoding='utf-8')
    # Stable-ground pre/post alignment is reviewed outside the author footprint.
    pre,post=rgb(ROOT/'pre_BGRN_10m.tif'),rgb(ROOT/'post_BGRN_10m.tif')
    mask=(~cv2.dilate(truth,np.ones((31,31),np.uint8)).astype(bool)&known_optical).astype(np.uint8)*255
    with rasterio.open(ROOT/'pre_SCL_10m.tif') as src: mask[np.isin(src.read(1),INVALID_SCL)]=0
    sift=cv2.SIFT_create(nfeatures=5000,contrastThreshold=.02)
    ak,ad=sift.detectAndCompute(cv2.cvtColor(pre,cv2.COLOR_BGR2GRAY),mask)
    bk,bd=sift.detectAndCompute(cv2.cvtColor(post,cv2.COLOR_BGR2GRAY),mask)
    matches=[a for a,b in cv2.BFMatcher().knnMatch(ad,bd,k=2) if a.distance<.7*b.distance]
    a=np.float32([ak[m.queryIdx].pt for m in matches]); b=np.float32([bk[m.trainIdx].pt for m in matches])
    cv2.setRNGSeed(7302)
    affine,inliers=cv2.estimateAffinePartial2D(a,b,method=cv2.RANSAC,ransacReprojThreshold=2.)
    if affine is None: raise ValueError('No stable ground period alignment')
    use=inliers.ravel()>0; residual=np.linalg.norm(a[use]-b[use],axis=1)
    alignment={'pre':'2019-01-07','post':'2019-01-27','stable_matches':len(matches),'inliers':int(use.sum()),
        'common_grid_median_stable_residual_pixels':float(np.median(residual)),
        'common_grid_p90_stable_residual_pixels':float(np.percentile(residual,90)),
        'diagnostic_affine':affine.tolist(),'raw_pair_gate_pass':bool(use.sum()>=30 and np.percentile(residual,90)<2),
        'raw_images_warped':False,'not_a_landslide_displacement_measurement':True,'model_predictions_run':0}
    (ROOT/'period_alignment.json').write_text(json.dumps(alignment,indent=2)+'\n',encoding='utf-8')
    review=(post*.35).astype(np.uint8)
    review[labels==1]=(.6*post[labels==1]+.4*np.array([30,210,30])).astype(np.uint8)
    review[labels==0]=(.6*post[labels==0]+.4*np.array([210,160,30])).astype(np.uint8)
    cv2.polylines(review,[np.round(pixels).astype(np.int32)],True,(30,40,240),1)
    cv2.imencode('.png',review)[1].tofile(str(ROOT/'label_review.png'))
    cv2.imencode('.png',np.hstack([pre,post]))[1].tofile(str(ROOT/'source_review/pre_post_RGB.png'))
    names=['pre_BGRN_10m.tif','post_BGRN_10m.tif','pre_SCL_10m.tif','post_SCL_10m.tif',
           'partial_labels.tif','label_confidence.tif','published_accident_reference.tif',
           'accident_reference_wgs84.geojson','source_review/registration_diagnostic.json','period_alignment.json']
    record={'mine_id':'LasCruces','role':'development','event':'2019-01-23',
        'boundary_source':'https://zenodo.org/records/4704955',
        'reference_figure':'Figure 3b external red boundary; research-author photointerpretation',
        'reference_date':'2019-01-25 helicopter SfM','target_date':'2019-01-27 Sentinel-2',
        'date_difference_days':2,'source_polygon_pixels':source_polygon.tolist(),'source_crop_xyxy':[30,388,503,714],
        'native_weak_negative_boxes_xyxy':NEGATIVE_BOXES_NATIVE,
        'label_schema':{'0':'project-reviewed weak stable event-background','1':'published-accident conservative interior','255':'unknown / uncertain boundary / optical exclusion'},
        'positive_uncertainty_buffer_m':50,'label_quality':'georeferenced published figure reference; not original expert masks',
        'positive_pixels':int((labels==1).sum()),'negative_pixels':int((labels==0).sum()),'unknown_pixels':int((labels==255).sum()),
        'digitized_raster_area_m2':int(truth.sum())*100,'optical_invalid_classes':list(INVALID_SCL),
        'SCL_7_unclassified_is_not_confirmed_visible':True,'SCL_3_shadow_retained':True,
        'full_reference_post_SCL_counts':{str(int(v)):int(n) for v,n in zip(*np.unique(scl[truth>0],return_counts=True))},
        'cloud_mask_restricts_label_support_not_claimed_to_improve_accuracy':True,
        'registration_holdout_consistency':{k:registration[k] for k in ('heldout_matches','heldout_within_3px_fraction','heldout_median_residual_target_pixels')},
        'field_or_independent_expert_review':False,'quality_review_status':'pending_project_visual_check',
        'model_predictions_during_labeling':0,'frozen_assets':{n:sha(ROOT/n) for n in names},
        'old_figure_reference_path':str(OLD/'cases/las_cruces_201901_reference.png'),
        'old_figure_reference_sha256':sha(OLD/'cases/las_cruces_201901_reference.png')}
    reference_path.write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:record[k] for k in ('positive_pixels','negative_pixels','unknown_pixels','digitized_raster_area_m2','full_reference_post_SCL_counts')}),flush=True)
    print(json.dumps(alignment),flush=True)


if __name__=='__main__':
    sys.stdout.reconfigure(encoding='utf-8'); run()
