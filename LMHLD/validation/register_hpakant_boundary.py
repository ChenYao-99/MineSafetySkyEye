# -*- coding: utf-8 -*-
"""Register published Hpakant source zones via the primary geographic context figure."""
import hashlib
import json
import sys
import cv2
import numpy as np
import rasterio
from rasterio.warp import transform as warp_coordinates
from mine_data_protocol import DATA,assert_development_cases
from register_las_cruces_boundary import rgb

ROOT=DATA/'development/Hpakant'
OLD=DATA.parents[1]/'experiments/positive_expansion_20261001'
CONTEXT_CROP=[183,59,1653,1618]


def affine_matches(source,target,source_mask,target_mask,name,prior=None,aspect=1.,ratio=.80,local_search=False):
    sift=cv2.SIFT_create(nfeatures=10000,nOctaveLayers=5,contrastThreshold=.005,edgeThreshold=12,sigma=1.)
    clahe=cv2.createCLAHE(clipLimit=2,tileGridSize=(8,8))
    tg=clahe.apply(cv2.cvtColor(target,cv2.COLOR_BGR2GRAY)); tk,td=sift.detectAndCompute(tg,target_mask)
    rows=[]
    for scale in (1.,.5,.25):
        image=cv2.resize(source,None,fx=scale*aspect,fy=scale,interpolation=cv2.INTER_AREA)
        mask=cv2.resize(source_mask,(image.shape[1],image.shape[0]),interpolation=cv2.INTER_NEAREST)
        sg=clahe.apply(cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)); sk,sd=sift.detectAndCompute(sg,mask)
        if sd is None or td is None: continue
        size=np.array([image.shape[1]/source.shape[1],image.shape[0]/source.shape[0]])
        before=len(rows)
        if local_search:
            sp=np.float32([(np.array(k.pt)+.5)/size-.5 for k in sk]); expected=prior(sp)
            tp=np.float32([k.pt for k in tk])
            for i,location in enumerate(expected):
                near=np.flatnonzero(np.sum((tp-location)**2,axis=1)<15**2)
                if len(near)<2: continue
                distances=np.sum((td[near]-sd[i])**2,axis=1); order=np.argsort(distances)[:2]
                if distances[order[0]]<ratio**2*distances[order[1]]:
                    j=near[order[0]]
                    rows.append((float(np.sqrt(distances[order[0]]/max(distances[order[1]],1e-6))),sp[i],tp[j]))
        else:
            matches=[(a,b) for a,b in cv2.BFMatcher().knnMatch(sd,td,k=2) if a.distance<ratio*b.distance]
            for m,n in matches:
                rows.append((m.distance/max(n.distance,1e-6),(np.array(sk[m.queryIdx].pt)+.5)/size-.5,np.array(tk[m.trainIdx].pt)))
        print(f'{name}: scale{scale} candidates {len(rows)-before}',flush=True)
    rows.sort(key=lambda r:r[0]); a=np.float32([r[1] for r in rows]); b=np.float32([r[2] for r in rows])
    if prior is not None:
        keep=np.linalg.norm(prior(a)-b,axis=1)<15; a,b=a[keep],b[keep]
    chosen=[]
    for i,(sa,tb) in enumerate(zip(a,b)):
        if any(np.linalg.norm(sa-a[j])<4 or np.linalg.norm(tb-b[j])<2 for j in chosen): continue
        chosen.append(i)
    a,b=a[chosen],b[chosen]
    if len(a)<40: raise ValueError(f'{name}: only {len(a)} distinct stable sites; reference not accepted')
    rng=np.random.default_rng(1704); order=rng.permutation(len(a)); split=min(int(.7*len(a)),len(a)-20)
    fit,held=order[:split],order[split:]; cv2.setRNGSeed(1704)
    matrix,inside=cv2.estimateAffine2D(a[fit],b[fit],method=cv2.RANSAC,ransacReprojThreshold=2,maxIters=10000,confidence=.999)
    if matrix is None: raise ValueError(f'{name}: no affine')
    use=inside.ravel()>0; residual=np.linalg.norm(a@matrix[:,:2].T+matrix[:,2]-b,axis=1); hr=residual[held]
    passed=bool(use.sum()>=20 and (hr<=3).mean()>=.8 and np.median(hr)<=1.5 and np.linalg.det(matrix[:,:2])>0)
    value={'name':name,'distinct_stable_matches':len(a),'fitting_matches':len(fit),'inliers':int(use.sum()),
        'heldout_matches':len(held),'heldout_within_3px_fraction':float((hr<=3).mean()),
        'heldout_median_target_pixels':float(np.median(hr)),'inlier_RMSE_target_pixels':float(np.sqrt(np.mean(residual[fit][use]**2))),
        'source_to_target_affine':matrix.tolist(),'gate_pass':passed,'source_points':a[fit][use].tolist(),
        'target_points':b[fit][use].tolist(),'heldout_source_points':a[held].tolist(),'heldout_target_points':b[held].tolist(),
        'not_absolute_geological_boundary_accuracy':True,'model_predictions_run':0,
        'matching_display_aspect':aspect,'descriptor_ratio':ratio,'coarse_prior_local_descriptor_search':local_search}
    out=ROOT/'source_review'; out.mkdir(exist_ok=True)
    from datetime import datetime
    trials=out/'trials'; trials.mkdir(exist_ok=True)
    (trials/(name+'_'+datetime.now().strftime('%Y%m%dT%H%M%S')+'.json')).write_text(json.dumps(value,indent=2)+'\n',encoding='utf-8')
    (out/(name+'.json')).write_text(json.dumps(value,indent=2)+'\n',encoding='utf-8')
    warped=cv2.warpAffine(source,matrix,(target.shape[1],target.shape[0])); overlap=warped.sum(-1)>0
    review=target.copy(); review[overlap]=(.5*target[overlap]+.5*warped[overlap]).astype(np.uint8)
    cv2.imencode('.png',review)[1].tofile(str(out/(name+'_review.png')))
    print(json.dumps({k:v for k,v in value.items() if 'points' not in k}),flush=True)
    if not passed: raise ValueError(f'{name}: held-out consistency failed; retain pending')
    return matrix,value


def run():
    assert_development_cases(['Hpakant'])
    if (ROOT/'label_reference.json').exists():
        ref=json.loads((ROOT/'label_reference.json').read_text(encoding='utf-8'))
        for name,digest in ref['frozen_assets'].items():
            if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=digest: raise ValueError('Frozen reference changed')
        print('Reusing frozen Hpakant reference registration'); return
    full=cv2.imdecode(np.fromfile(ROOT/'source_figure1.jpg',np.uint8),1)
    x0,y0,x1,y1=CONTEXT_CROP; context=full[y0:y1,x0:x1].copy()
    cmask=np.ones(context.shape[:2],np.uint8)*255
    cmask[:12]=0; cmask[-25:]=0; cmask[:,:12]=0; cmask[:,-12:]=0
    cmask[1000:,:800]=0; cmask[1400:]=0; cmask[:180,:110]=0
    b,g,r=context.transpose(2,0,1).astype(np.int16)
    annotations=((r>165)&(g>140)&(b<100))|((g>180)&(g-r>70)&(g-b>70))|((r>150)&(r-g>60)&(r-b>60))
    cmask[cv2.dilate(annotations.astype(np.uint8),np.ones((9,9),np.uint8))>0]=0
    same_date_context_mask=cmask.copy()
    cmask[575:870,250:640]=0  # complete affected footprint, not a matching control
    target=rgb(ROOT/'post_BGRN_10m.tif')
    with rasterio.open(ROOT/'post_BGRN_10m.tif') as src:
        inverse=~src.transform; crs=src.crs
    with rasterio.open(ROOT/'post_SCL_10m.tif') as src: quality=src.read(1)
    target_mask=(~np.isin(quality,[0,1,8,9,10,11])).astype(np.uint8)*255
    def printed_prior(points):
        lon=96.25+(points[:,0]-211)*(1/60)/371
        lat=25.5-(points[:,1]-173)*(1/60)/372
        east,north=warp_coordinates('EPSG:4326',crs,lon.tolist(),lat.tolist())
        return np.float32([inverse*(x,y) for x,y in zip(east,north)])-.5
    first,ga=affine_matches(context,target,cmask,target_mask,'context_to_native',printed_prior)
    source=cv2.imdecode(np.fromfile(OLD/'cases/hpakant_post_20200803.png',np.uint8),1)
    collapsed=cv2.imdecode(np.fromfile(OLD/'cases/hpakant_collapsed_reference.png',np.uint8),0)>0
    affected=collapsed|(cv2.imdecode(np.fromfile(OLD/'cases/hpakant_runout_unknown.png',np.uint8),0)>0)
    # These are two presentations of the same dated author reference. Natural
    # texture inside the affected body is valid for panel-to-panel alignment;
    # drawn answer lines/markers remain excluded. Cross-sensor registration above
    # still uses only terrain outside the complete affected footprint.
    borders=cv2.morphologyEx(affected.astype(np.uint8),cv2.MORPH_GRADIENT,np.ones((15,15),np.uint8))
    borders|=cv2.morphologyEx(collapsed.astype(np.uint8),cv2.MORPH_GRADIENT,np.ones((15,15),np.uint8))
    smask=(borders==0).astype(np.uint8)*255
    smask[:12]=0; smask[-20:]=0; smask[:,:12]=0; smask[:,-12:]=0
    smask[:100,:100]=0; smask[600:,700:]=0
    # The source mask excludes the affected body. For same-date reference-to-context
    # matching, retain stable margins around it in the target; the broad exclusion
    # used for context->satellite would otherwise remove the entire zoom panel.
    # Author Video1/2 symbols and the main-map red footprint place the zoom in
    # this generous context rectangle. Restrict the *descriptor search* before
    # matching, rather than searching unrelated quarries across the large map.
    # Rectangle placement is from source figures, never from model predictions.
    context_zoom_mask=np.zeros_like(same_date_context_mask)
    context_zoom_mask[540:920,220:700]=same_date_context_mask[540:920,220:700]
    # Main map plots equal lon/lat increments while the zoom is stretched to its
    # panel. Compensate their approx10% presentation aspect for descriptor matching;
    # coordinates are always returned in the unmodified original figure frame.
    # The same figure includes a larger natural-colour inset; use that panel as
    # an intermediate reference to avoid relying on tiny repeated quarry texture.
    inset=full[1086:1618,183:959].copy()
    ib,ig,ir=inset.transpose(2,0,1).astype(np.int16)
    red=((ir>150)&(ir-ig>60)&(ir-ib>60)).astype(np.uint8)
    closed=cv2.morphologyEx(red,cv2.MORPH_CLOSE,np.ones((5,5),np.uint8))
    contours,_=cv2.findContours(closed,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    body=np.zeros(inset.shape[:2],np.uint8)
    boundary=max(contours,key=cv2.contourArea)
    if cv2.contourArea(boundary)<.2*body.size: raise ValueError('Inset affected outline not recovered')
    cv2.drawContours(body,[boundary],-1,1,-1)
    imask=(cv2.morphologyEx(body,cv2.MORPH_GRADIENT,np.ones((15,15),np.uint8))==0).astype(np.uint8)*255
    imask[cv2.dilate(red,np.ones((9,9),np.uint8))>0]=0
    imask[:12]=0; imask[-30:]=0; imask[:,:12]=0; imask[:,-12:]=0; imask[440:,470:]=0
    green=((ig>180)&(ig-ir>70)&(ig-ib>70)).astype(np.uint8)
    imask[cv2.dilate(green,np.ones((9,9),np.uint8))>0]=0
    # Both panels explicitly reproduce the same author's red runout footprint.
    # Its bounding box supplies a coarse graphic-placement prior (not a final
    # georeference). Fit and held-out residuals still use unannotated photo texture.
    # This is NOT independent geological boundary validation: the reference panels
    # share an outline, and the cross-sensor geographic check above is separate.
    cb,cg,cr=context.transpose(2,0,1).astype(np.int16)
    cmred=((cr>150)&(cr-cg>60)&(cr-cb>60)).astype(np.uint8)
    cmred[:600]=0; cmred[820:]=0; cmred[:,:280]=0; cmred[:,630:]=0
    cc,_=cv2.findContours(cv2.morphologyEx(cmred,cv2.MORPH_CLOSE,np.ones((5,5),np.uint8)),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    context_bounds=cv2.boundingRect(max(cc,key=cv2.contourArea)); inset_bounds=cv2.boundingRect(boundary)
    print(f'Author reference bounding boxes: inset={inset_bounds}, context={context_bounds}',flush=True)
    def box_prior(start,end):
        sx,sy,sw,sh=start; tx,ty,tw,th=end
        def project(points):
            return (points-np.array([sx,sy]))*np.array([tw/sw,th/sh])+np.array([tx,ty])
        return project
    mid,gc=affine_matches(inset,context,imask,context_zoom_mask,'inset_to_context',box_prior(inset_bounds,context_bounds),ratio=.65,local_search=True)
    ac,_=cv2.findContours(affected.astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    source_bounds=cv2.boundingRect(max(ac,key=cv2.contourArea))
    second,gb=affine_matches(source,inset,smask,imask,'source_zone_to_inset',box_prior(source_bounds,inset_bounds),ratio=.65,local_search=True)
    fa=np.vstack([first,[0,0,1]]); fm=np.vstack([mid,[0,0,1]]); fb=np.vstack([second,[0,0,1]]); combined=(fa@fm@fb)[:2]
    record={'source':'Lin et al. 2021 Figure7D source-zone blue dashed outline, Planet2020-08-03',
        'context':'Figure1A geographic graticule, same Planet date; inset excluded',
        'target':'Original Sentinel-2 BGRN2020-08-03, different sensor and acquisition time',
        'context_crop_xyxy':CONTEXT_CROP,'source_to_target_affine':combined.tolist(),
        'context_native_holdout':{k:v for k,v in ga.items() if 'points' not in k},
        'source_context_holdout':{k:v for k,v in gb.items() if 'points' not in k},
        'inset_context_holdout':{k:v for k,v in gc.items() if 'points' not in k},
        'registration_gate_pass':True,'three_stage_errors_accumulate':True,
        'same_reference_panel_coarse_prior':'shared author-runout bounding boxes for descriptor screening only; same-photo unannotated texture fits',
        'reference_panels_do_not_independently_validate_boundary':True,
        'not_absolute_geological_boundary_accuracy':True,'model_predictions_run':0}
    (ROOT/'source_review/registration_diagnostic.json').write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8')
    warped=cv2.warpAffine(source,combined,(target.shape[1],target.shape[0])); mask=warped.sum(-1)>0
    review=target.copy(); review[mask]=(.5*target[mask]+.5*warped[mask]).astype(np.uint8)
    cv2.imencode('.png',review)[1].tofile(str(ROOT/'source_review/combined_registration_review.png'))
    print('Two-stage published reference registration complete; no model used',flush=True)


if __name__=='__main__':
    sys.stdout.reconfigure(encoding='utf-8'); run()
