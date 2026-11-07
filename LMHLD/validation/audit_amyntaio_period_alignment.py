# -*- coding: utf-8 -*-
"""Check real pre/post imagery on stable ground; no displacement claim or detector."""
import json
from pathlib import Path
import sys
import cv2
import numpy as np
import rasterio

ROOT=Path(__file__).resolve().parents[1]/"datasets/mine_benchmark_20261002/development/Amyntaio"


def rgb(path):
    with rasterio.open(path) as src:
        x=src.read()
    image=np.stack([x[0],x[1],x[2]],-1).astype(np.float32)
    valid=image.sum(-1)>0
    for c in range(3):
        lo,hi=np.percentile(image[:,:,c][valid],[2,98]); image[:,:,c]=np.clip((image[:,:,c]-lo)/max(hi-lo,1),0,1)*255
    return image.astype(np.uint8)


def audit():
    pre,post=rgb(ROOT/"pre_BGRN_10m.tif"),rgb(ROOT/"post_BGRN_10m.tif")
    with rasterio.open(ROOT/"published_debris_reference.tif") as src:
        body=src.read(1)>0
    mask=(~cv2.dilate(body.astype(np.uint8),np.ones((31,31),np.uint8)).astype(bool)).astype(np.uint8)*255
    sift=cv2.SIFT_create(nfeatures=5000,contrastThreshold=.02)
    ak,ad=sift.detectAndCompute(cv2.cvtColor(pre,cv2.COLOR_BGR2GRAY),mask)
    bk,bd=sift.detectAndCompute(cv2.cvtColor(post,cv2.COLOR_BGR2GRAY),mask)
    raw=cv2.BFMatcher().knnMatch(ad,bd,k=2)
    good=[a for a,b in raw if a.distance<.7*b.distance]
    a=np.float32([ak[m.queryIdx].pt for m in good]); b=np.float32([bk[m.trainIdx].pt for m in good])
    cv2.setRNGSeed(7301)
    matrix,inliers=cv2.estimateAffinePartial2D(a,b,method=cv2.RANSAC,ransacReprojThreshold=2.0)
    if matrix is None:
        raise ValueError("Stable-ground period registration not recoverable")
    use=inliers.ravel()>0
    grid_residual=np.linalg.norm(a[use]-b[use],axis=1)
    fit_residual=np.linalg.norm(a[use]@matrix[:,:2].T+matrix[:,2]-b[use],axis=1)
    value={"pre":"2017-06-01","post":"2017-06-28","stable_matches":len(good),"inliers":int(use.sum()),
        "common_grid_median_stable_residual_pixels":float(np.median(grid_residual)),
        "common_grid_p90_stable_residual_pixels":float(np.percentile(grid_residual,90)),
        "fit_median_residual_pixels":float(np.median(fit_residual)),"diagnostic_affine":matrix.tolist(),
        "raw_images_warped":False,"not_a_landslide_displacement_measurement":True,
        "raw_pair_gate_pass":bool(int(use.sum())>=30 and np.percentile(grid_residual,90)<2.),
        "model_predictions_run":0}
    (ROOT/"period_alignment.json").write_text(json.dumps(value,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(value),flush=True)
    if not value["raw_pair_gate_pass"]:
        raise ValueError("Real period pair needs additional spatial alignment")


if __name__=="__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    audit()
