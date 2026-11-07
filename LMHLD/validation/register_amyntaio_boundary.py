# -*- coding: utf-8 -*-
"""Register a published development boundary figure using stable ground texture only."""
import json
from pathlib import Path
import sys
import cv2
import numpy as np
import rasterio

ROOT=Path(__file__).resolve().parents[1]/"datasets/mine_benchmark_20261002/development/Amyntaio"


def preview(bands):
    # Source figure is NIR/R/G as display RGB; cv2 expects display BGR.
    out=np.stack([bands[1],bands[2],bands[3]],axis=-1).astype(np.float32)
    good=np.isfinite(out).all(-1)&(out.sum(-1)>0)
    for c in range(3):
        lo,hi=np.percentile(out[:,:,c][good],[2,98])
        out[:,:,c]=np.clip((out[:,:,c]-lo)/max(hi-lo,1),0,1)*255
    out[~good]=0
    return out.astype(np.uint8)


def run():
    source=cv2.imdecode(np.fromfile(ROOT/"source_review/figure_xref4.png",np.uint8),1)
    with rasterio.open(ROOT/"post_BGRN_10m.tif") as src:
        target=preview(src.read())
    cv2.imencode(".png",target)[1].tofile(str(ROOT/"source_review/post_falsecolour_for_registration.png"))
    source_mask=np.ones(source.shape[:2],np.uint8)*255
    source_mask[:15]=0; source_mask[-24:]=0; source_mask[:,:13]=0; source_mask[:,-13:]=0
    # Do not use the changing slide, author outline, cracks, date or map annotations
    # as matching evidence. Fit only stable farms/roads outside the affected body.
    source_mask[55:250,135:295]=0
    source_mask[15:40,360:]=0
    source_mask[185:220,:135]=0
    source_mask[270:,335:]=0
    sift=cv2.SIFT_create(nfeatures=6000,contrastThreshold=.015)
    clahe=cv2.createCLAHE(clipLimit=2,tileGridSize=(8,8))
    sg=clahe.apply(cv2.cvtColor(source,cv2.COLOR_BGR2GRAY))
    tg=clahe.apply(cv2.cvtColor(target,cv2.COLOR_BGR2GRAY))
    sk,sd=sift.detectAndCompute(sg,source_mask)
    tk,td=sift.detectAndCompute(tg,None)
    candidates=cv2.BFMatcher().knnMatch(sd,td,k=2)
    good=[m for m,n in candidates if m.distance<.75*n.distance]
    print(f"Source/target keypoints {len(sk)}/{len(tk)}; ratio matches {len(good)}",flush=True)
    if len(good)<12:
        raise ValueError("Insufficient stable ground matches; do not invent georeferencing")
    a=np.float32([sk[m.queryIdx].pt for m in good]); b=np.float32([tk[m.trainIdx].pt for m in good])
    rng=np.random.default_rng(1701)
    order=rng.permutation(len(good)); split=int(.8*len(good))
    fitting,heldout=order[:split],order[split:]
    cv2.setRNGSeed(1701)
    transform,inliers=cv2.estimateAffine2D(a[fitting],b[fitting],method=cv2.RANSAC,ransacReprojThreshold=3,maxIters=10000,confidence=.999)
    if transform is None:
        raise ValueError("Ground registration failed")
    use=inliers.ravel()>0
    residual=np.linalg.norm(a@transform[:,:2].T+transform[:,2]-b,axis=1)
    hr=residual[heldout]
    passes=len(heldout)>=20 and float((hr<=3).mean())>=.8 and float(np.median(hr))<=1.5
    result={"source":"JISDM 2019 Vassilakis et al., Figure 1, RapidEye 2017-06-20",
        "target":"actual Sentinel-2 BGRN 2017-06-28; false-colour preview only for registration",
        "candidate_matches":len(good),"fitting_matches":len(fitting),"inliers":int(use.sum()),"inlier_ratio":float(use.mean()),
        "inlier_RMSE_target_pixels":float(np.sqrt(np.mean(residual[fitting][use]**2))),
        "heldout_matches":len(heldout),"heldout_within_3px_fraction":float((hr<=3).mean()),
        "heldout_median_residual_target_pixels":float(np.median(hr)),"registration_gate_pass":passes,
        "source_to_target_affine":transform.tolist(),"source_points":a[fitting][use].tolist(),"target_points":b[fitting][use].tolist(),
        "heldout_source_points":a[heldout].tolist(),"heldout_target_points":b[heldout].tolist(),
        "validation":"held-out correspondence consistency; not independent absolute ground accuracy",
        "no_claim_of_independent_geological_boundary_accuracy":True}
    (ROOT/"source_review/registration_diagnostic.json").write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8")
    warped=cv2.warpAffine(source,transform,(target.shape[1],target.shape[0]))
    overlap=(warped.sum(-1)>0)
    review=target.copy(); review[overlap]=(.5*target[overlap]+.5*warped[overlap]).astype(np.uint8)
    cv2.imencode(".png",review)[1].tofile(str(ROOT/"source_review/registration_review.png"))
    print(json.dumps({k:v for k,v in result.items() if k not in ("source_points","target_points")}),flush=True)
    if not passes:
        raise ValueError("Registration did not meet held-out ground-match consistency gates")


if __name__=="__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    run()
