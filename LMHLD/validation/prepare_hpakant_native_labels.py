# -*- coding: utf-8 -*-
"""Conservative source-zone labels on originals; runout/water and edges stay unknown."""
import hashlib
import json
import sys
import cv2
import numpy as np
import rasterio
from rasterio.warp import transform_geom
from mine_data_protocol import DATA,REGISTRY,registry,assert_development_cases,verify_assets
from register_las_cruces_boundary import rgb

ROOT=DATA/'development/Hpakant'
OLD=DATA.parents[1]/'experiments/positive_expansion_20261001'
NEGATIVE_BOXES=[[20,20,80,90],[275,30,345,95],[305,365,365,420],
                [20,350,70,410],[285,230,330,270],[235,140,270,185]]


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def run():
    assert_development_cases(['Hpakant']); path=ROOT/'label_reference.json'
    if path.exists():
        ref=json.loads(path.read_text(encoding='utf-8'))
        for name,digest in ref['frozen_assets'].items():
            if sha(ROOT/name)!=digest: raise ValueError('Frozen Hpakant original/label changed')
        print('Reusing frozen Hpakant source reference'); return
    registration=json.loads((ROOT/'source_review/registration_diagnostic.json').read_text(encoding='utf-8'))
    if not registration['registration_gate_pass']: raise ValueError('Native registration not accepted')
    source=json.loads((OLD/'positive_cases.json').read_text(encoding='utf-8'))['hpakant']
    matrix=np.array(registration['source_to_target_affine'])
    with rasterio.open(ROOT/'post_BGRN_10m.tif') as frame:
        raw=frame.read(); profile=frame.profile.copy(); crs,transform=frame.crs,frame.transform
    with rasterio.open(ROOT/'post_SCL_10m.tif') as frame: scl=frame.read(1)
    valid=(raw>0).all(0)&~np.isin(scl,[0,1,8,9,10,11]); shape=valid.shape
    footprint=np.zeros(shape,np.uint8); runout=np.zeros(shape,np.uint8); polygons=[]
    for ring in source['reference_polygons_xy']:
        points=np.array(ring)@matrix[:,:2].T+matrix[:,2]
        cv2.fillPoly(footprint,[np.round(points).astype(np.int32)],1); polygons.append(points)
    points=np.array(source['runout_polygon_xy'])@matrix[:,:2].T+matrix[:,2]
    cv2.fillPoly(runout,[np.round(points).astype(np.int32)],1)
    positive=cv2.erode(footprint,np.ones((11,11),np.uint8))>0
    positive&=valid&(scl!=6)
    negative=np.zeros(shape,np.uint8)
    for x0,y0,x1,y1 in NEGATIVE_BOXES: negative[y0:y1,x0:x1]=1
    negative=cv2.erode(negative,np.ones((5,5),np.uint8))>0
    negative&=valid&~cv2.dilate(runout,np.ones((15,15),np.uint8)).astype(bool)
    y=np.full(shape,255,np.uint8); y[positive]=1; y[negative]=0
    w=np.zeros(shape,np.float32); w[y==1]=.6; w[y==0]=.25
    if (y==1).sum()<64: raise ValueError('Too little conservative source support at10m; do not shrink uncertainty to make training possible')
    for name,array,dtype,nodata in (('published_source_reference.tif',footprint,'uint8',None),('published_runout_unknown.tif',runout,'uint8',None),
                                    ('partial_labels.tif',y,'uint8',255),('label_confidence.tif',w,'float32',None)):
        meta=profile.copy(); meta.update(count=1,dtype=dtype,nodata=nodata,compress='deflate')
        with rasterio.open(ROOT/name,'w',**meta) as dst: dst.write(array,1)
    features=[]
    for p in polygons:
        coords=[list(transform*(float(x)+.5,float(y)+.5)) for x,y in p]; coords.append(coords[0])
        geo=transform_geom(crs,'EPSG:4326',{'type':'Polygon','coordinates':[coords]})
        features.append({'type':'Feature','properties':{'mine':'Wai Khar, Hpakant','event':'2020-07-02',
            'semantics':'published collapsed source zone, runout water excluded','expert_GIS':False},'geometry':geo})
    (ROOT/'source_reference_wgs84.geojson').write_text(json.dumps({'type':'FeatureCollection','features':features},indent=2)+'\n',encoding='utf-8')
    pre,post=rgb(ROOT/'pre_BGRN_10m.tif'),rgb(ROOT/'post_BGRN_10m.tif')
    mask=(~cv2.dilate(runout,np.ones((31,31),np.uint8)).astype(bool)&valid).astype(np.uint8)*255
    with rasterio.open(ROOT/'pre_SCL_10m.tif') as src: mask[np.isin(src.read(1),[0,1,8,9,10,11])]=0
    sift=cv2.SIFT_create(nfeatures=6000,contrastThreshold=.02)
    ak,ad=sift.detectAndCompute(cv2.cvtColor(pre,cv2.COLOR_BGR2GRAY),mask); bk,bd=sift.detectAndCompute(cv2.cvtColor(post,cv2.COLOR_BGR2GRAY),mask)
    matches=[a for a,b in cv2.BFMatcher().knnMatch(ad,bd,k=2) if a.distance<.7*b.distance]
    a=np.float32([ak[m.queryIdx].pt for m in matches]); b=np.float32([bk[m.trainIdx].pt for m in matches]); cv2.setRNGSeed(7303)
    affine,inside=cv2.estimateAffinePartial2D(a,b,method=cv2.RANSAC,ransacReprojThreshold=2)
    if affine is None: raise ValueError('Stable two-period correspondence failed')
    use=inside.ravel()>0; residual=np.linalg.norm(a[use]-b[use],axis=1)
    alignment={'pre':'2020-05-10','post':'2020-08-03','inliers':int(use.sum()),'matches':len(matches),
        'common_grid_median_stable_residual_pixels':float(np.median(residual)),
        'common_grid_p90_stable_residual_pixels':float(np.percentile(residual,90)),
        'diagnostic_affine':affine.tolist(),'raw_pair_gate_pass':bool(use.sum()>=30 and np.percentile(residual,90)<2),
        'raw_images_warped':False,'not_a_landslide_displacement_measurement':True,'model_predictions_run':0}
    (ROOT/'period_alignment.json').write_text(json.dumps(alignment,indent=2)+'\n',encoding='utf-8')
    review=(post*.35).astype(np.uint8)
    review[y==1]=(.6*post[y==1]+.4*np.array([30,210,30])).astype(np.uint8)
    review[y==0]=(.6*post[y==0]+.4*np.array([210,160,30])).astype(np.uint8)
    for p in polygons: cv2.polylines(review,[np.round(p).astype(np.int32)],True,(30,40,240),1)
    cv2.imencode('.png',review)[1].tofile(str(ROOT/'label_review.png'))
    names=['pre_BGRN_10m.tif','post_BGRN_10m.tif','pre_SCL_10m.tif','post_SCL_10m.tif','partial_labels.tif','label_confidence.tif',
           'published_source_reference.tif','published_runout_unknown.tif','source_reference_wgs84.geojson',
           'source_figure1.jpg','source_review/registration_diagnostic.json','period_alignment.json']
    ref={'mine_id':'Hpakant','role':'development','event':'2020-07-02','boundary_source':source['source'],
        'reference_figure':'Lin et al. Figure7D BLUE source-zone outline, via Figure1 inset and geographic main panel',
        'reference_date':'2020-08-03 Planet; actual nativepost same calendar day different sensor/time',
        'target_date':'2020-08-03 Sentinel2','pre_date_note':'nativeMay10 vs authorPlanetMay11, pre labels unknown',
        'target_semantics':'collapsed source zones only; other runout and water unknown',
        'label_schema':{'0':'project-reviewed weak stable event-background','1':'published-source conservative interior','255':'unknown / uncertainty / water'},
        'positive_uncertainty_buffer_m':50,'source_polygon_pixels':source['reference_polygons_xy'],
        'native_weak_negative_boxes_xyxy':NEGATIVE_BOXES,'positive_pixels':int((y==1).sum()),'negative_pixels':int((y==0).sum()),
        'unknown_pixels':int((y==255).sum()),'digitized_source_area_m2':int(footprint.sum())*100,
        'label_quality':'georeferenced published figure reference; not original expert masks',
        'field_or_independent_expert_review':False,'quality_review_status':'pending_project_visual_check',
        'multiple_reference_stages_accumulate_uncertainty':True,'shared_author_panel_prior_is_not_independent_boundary_validation':True,
        'model_predictions_during_labeling':0,'frozen_assets':{n:sha(ROOT/n) for n in names}}
    path.write_text(json.dumps(ref,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:ref[k] for k in ('positive_pixels','negative_pixels','unknown_pixels','digitized_source_area_m2')}),flush=True)
    print(json.dumps(alignment),flush=True)


if __name__=='__main__':
    sys.stdout.reconfigure(encoding='utf-8'); run()
