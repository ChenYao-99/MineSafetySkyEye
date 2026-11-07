# -*- coding: utf-8 -*-
"""Register a figure reference with stable ground, without any model prediction."""
import json
import argparse
import hashlib
from pathlib import Path
import sys
import cv2
import numpy as np
import rasterio
from rasterio.warp import transform as warp_coordinates
from mine_data_protocol import DATA, assert_development_cases

ROOT=DATA/'development/LasCruces'
OLD=DATA.parents[1]/'experiments/mining_validation_20261001'


def rgb(path):
    with rasterio.open(path) as src: x=src.read()[:3].transpose(1,2,0).astype(np.float32)
    for c in range(3):
        lo,hi=np.percentile(x[:,:,c],[2,98]); x[:,:,c]=np.clip((x[:,:,c]-lo)/max(hi-lo,1),0,1)*255
    return np.ascontiguousarray(x.astype(np.uint8))


def run(source_scale=1., multi_scale=False):
    assert_development_cases(['LasCruces'])
    frozen=ROOT/'label_reference.json'
    if frozen.exists():
        reference=json.loads(frozen.read_text(encoding='utf-8'))
        path=ROOT/'source_review/registration_diagnostic.json'
        if hashlib.sha256(path.read_bytes()).hexdigest()!=reference['frozen_assets']['source_review/registration_diagnostic.json']:
            raise ValueError('Frozen registration changed')
        print('Reusing frozen development registration; do not rerun after labels are frozen')
        return
    out=ROOT/'source_review'; out.mkdir(exist_ok=True)
    source=cv2.imdecode(np.fromfile(OLD/'cases/las_cruces_201901_figure_input.png',np.uint8),1)
    body=cv2.imdecode(np.fromfile(OLD/'cases/las_cruces_201901_reference.png',np.uint8),0)>0
    target=rgb(ROOT/'post_BGRN_10m.tif')
    cv2.imencode('.png',target)[1].tofile(str(out/'post_RGB.png'))
    mask=(~cv2.dilate(body.astype(np.uint8),np.ones((25,25),np.uint8)).astype(bool)).astype(np.uint8)*255
    mask[:12]=0; mask[-15:]=0; mask[:,:12]=0; mask[:,-12:]=0
    mask[:80,420:]=0; mask[285:,315:]=0
    # Exclude the published outline and changing body; annotations are never input labels.
    sift=cv2.SIFT_create(nfeatures=8000,nOctaveLayers=5,contrastThreshold=.005,edgeThreshold=12,sigma=1.)
    clahe=cv2.createCLAHE(clipLimit=2,tileGridSize=(8,8))
    tg=clahe.apply(cv2.cvtColor(target,cv2.COLOR_BGR2GRAY)); tk,td=sift.detectAndCompute(tg,None)
    correspondences=[]
    for scale in ((1.,.5) if multi_scale else (source_scale,)):
        matching=cv2.resize(source,None,fx=scale,fy=scale,interpolation=cv2.INTER_AREA)
        matching_mask=cv2.resize(mask,(matching.shape[1],matching.shape[0]),interpolation=cv2.INTER_NEAREST)
        sg=clahe.apply(cv2.cvtColor(matching,cv2.COLOR_BGR2GRAY)); sk,sd=sift.detectAndCompute(sg,matching_mask)
        candidates=cv2.BFMatcher().knnMatch(sd,td,k=2)
        good=[(m,n) for m,n in candidates if m.distance<.85*n.distance]
        scale_xy=np.array([matching.shape[1]/source.shape[1],matching.shape[0]/source.shape[0]],np.float32)
        for m,n in good:
            correspondences.append((m.distance/max(n.distance,1e-6),(np.array(sk[m.queryIdx].pt)+.5)/scale_xy-.5,np.array(tk[m.trainIdx].pt)))
        print(f'Scale {scale:g}: keypoints {len(sk)}/{len(tk)}, ratio matches {len(good)}',flush=True)
    correspondences.sort(key=lambda row:row[0])
    a=np.float32([row[1] for row in correspondences]); b=np.float32([row[2] for row in correspondences])
    # An independently printed geographic prior rejects lookalike fields elsewhere
    # in the ROI. It is not fitted from the held-out correspondences or model output.
    # Approximate latitude scale assumes square geographic map pixels; 150m tolerance
    # explicitly allows figure layout/rotation uncertainty. It is not the final fit.
    lon=-6.1+(a[:,0]-115)*(1/60)/337
    metres_per_source_pixel=111320*np.cos(np.deg2rad(37.5))*(1/60)/337
    lat=37.5-(a[:,1]-267)*metres_per_source_pixel/111000
    with rasterio.open(ROOT/'post_BGRN_10m.tif') as frame:
        east,north=warp_coordinates('EPSG:4326',frame.crs,lon.tolist(),lat.tolist())
        inverse=~frame.transform
        prior=np.array([inverse*(x,y) for x,y in zip(east,north)],np.float32)-.5
    keep=np.linalg.norm(prior-b,axis=1)<15
    a,b=a[keep],b[keep]
    # Multiple SIFT orientations at one point are not independent validation sites.
    unique=[]
    for i,(sa,tb) in enumerate(zip(a,b)):
        if any(np.linalg.norm(sa-a[j])<4 or np.linalg.norm(tb-b[j])<2 for j in unique): continue
        unique.append(i)
    a,b=a[unique],b[unique]
    print(f'Unique printed-coordinate-compatible ground matches {len(a)}',flush=True)
    if len(a)<20:
        (out/'registration_failed.json').write_text(json.dumps({'reason':'insufficient ground matches','n':len(a)},indent=2)+'\n',encoding='utf-8')
        raise ValueError('Insufficient stable matches; do not fabricate geographic reference')
    # Keep at least 20 held-out sites and 20 fitting candidates when available.
    # Allocation uses only the candidate count, never residuals or model outcomes.
    rng=np.random.default_rng(1703); order=rng.permutation(len(a)); split=min(int(.7*len(a)),len(a)-20) if len(a)>=40 else int(.7*len(a))
    fitting,heldout=order[:split],order[split:]
    cv2.setRNGSeed(1703)
    matrix,inliers=cv2.estimateAffine2D(a[fitting],b[fitting],method=cv2.RANSAC,ransacReprojThreshold=2,maxIters=10000,confidence=.999)
    if matrix is None: raise ValueError('No affine registration')
    use=inliers.ravel()>0
    residual=np.linalg.norm(a@matrix[:,:2].T+matrix[:,2]-b,axis=1)
    hr=residual[heldout]
    passes=bool(len(heldout)>=20 and use.sum()>=20 and (hr<=3).mean()>=.8 and np.median(hr)<=1.5)
    result={'source':'Lopez-Vinielles et al. Figure 3b, 2019-01-25 helicopter orthoimage',
        'target':'original Sentinel-2 2019-01-27','source_to_target_affine':matrix.tolist(),
        'candidate_matches':len(correspondences),'unique_coordinate_compatible_matches':len(a),
        'prior':'Figure 3 axes + approximately square map pixels, 150m search tolerance, not a fitted prior',
        'fitting_matches':len(fitting),'inliers':int(use.sum()),
        'inlier_RMSE_target_pixels':float(np.sqrt(np.mean(residual[fitting][use]**2))),
        'heldout_matches':len(heldout),'heldout_within_3px_fraction':float((hr<=3).mean()),
        'heldout_median_residual_target_pixels':float(np.median(hr)),
        'registration_gate_pass':passes,'source_points':a[fitting][use].tolist(),'target_points':b[fitting][use].tolist(),
        'heldout_source_points':a[heldout].tolist(),'heldout_target_points':b[heldout].tolist(),
        'source_matching_scale':[1.,.5] if multi_scale else [source_scale],
        'holdout_allocation':'fixed RNG1703; >=20 held out and >=20 fitting candidates if total>=40',
        'model_predictions_run':0,'not_independent_absolute_geological_boundary_accuracy':True}
    trial='multi' if multi_scale else f'{source_scale:g}'
    (out/f'registration_trial_scale{trial}.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    (out/'registration_diagnostic.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    warped=cv2.warpAffine(source,matrix,(target.shape[1],target.shape[0]))
    overlap=warped.sum(-1)>0; review=target.copy()
    review[overlap]=(.5*target[overlap]+.5*warped[overlap]).astype(np.uint8)
    cv2.imencode('.png',review)[1].tofile(str(out/'registration_review.png'))
    print(json.dumps({k:v for k,v in result.items() if 'points' not in k}),flush=True)
    if not passes: raise ValueError('Registration not accepted; retain original data with labels pending')


if __name__=='__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    parser=argparse.ArgumentParser(); parser.add_argument('--source-scale',type=float,default=1.,choices=(1.,.5))
    parser.add_argument('--multi-scale',action='store_true'); args=parser.parse_args()
    run(args.source_scale,args.multi_scale)
