# -*- coding: utf-8 -*-
"""Digitize a published debris outline onto genuine imagery; retain unknown areas."""
import hashlib
import json
from pathlib import Path
import sys
import cv2
import numpy as np
import rasterio
from rasterio.warp import transform_geom

ROOT=Path(__file__).resolve().parents[1]/"datasets/mine_benchmark_20261002/development/Amyntaio"
# Digitized along the BLACK DASHED debris boundary in the authors' Figure 1.
# Yellow fractures outside this outline are not used as negative supervision.
POLYGON=[[154,151],[167,130],[173,122],[182,115],[191,111],[202,102],
         [211,101],[216,96],[225,96],[230,93],[237,106],[240,127],
         [244,140],[249,146],[251,162],[257,176],[261,190],[269,201],
         [268,204],[244,212],[224,214],[204,207],[189,197],[179,190],
         [171,180],[164,166],[153,159]]
# Conservative visually stable farmlands, facilities and normal pit interiors.
# These are project-reviewed weak negatives, not independent expert measurements.
NEGATIVE_BOXES=[[30,55,110,120],[305,70,345,105],[320,125,375,177],
                [375,55,430,100],[345,180,390,205]]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare():
    record_path=ROOT/"label_reference.json"
    if record_path.exists():
        record=json.loads(record_path.read_text(encoding="utf-8"))
        for name,digest in record["frozen_assets"].items():
            if sha(ROOT/name)!=digest:
                raise ValueError("Frozen development reference changed")
        print("Reusing frozen development reference")
        return
    registration=json.loads((ROOT/"source_review/registration_diagnostic.json").read_text(encoding="utf-8"))
    if not registration.get("registration_gate_pass"):
        raise ValueError("No accepted held-out correspondence registration")
    matrix=np.asarray(registration["source_to_target_affine"],np.float64)
    pixels=np.asarray(POLYGON,np.float64)@matrix[:,:2].T+matrix[:,2]
    with rasterio.open(ROOT/"post_BGRN_10m.tif") as image:
        shape=(image.height,image.width)
        raw=image.read()
        transform,crs=image.transform,image.crs
        profile=image.profile.copy()
        valid=image.dataset_mask()>0
    truth=np.zeros(shape,np.uint8)
    cv2.fillPoly(truth,[np.round(pixels).astype(np.int32)],1)
    positive=cv2.erode(truth,np.ones((11,11),np.uint8))>0
    negative=np.zeros(shape,np.uint8)
    for x0,y0,x1,y1 in NEGATIVE_BOXES:
        box=np.asarray([[x0,y0],[x1,y0],[x1,y1],[x0,y1]],np.float64)
        target=box@matrix[:,:2].T+matrix[:,2]
        cv2.fillPoly(negative,[np.round(target).astype(np.int32)],1)
    negative=cv2.erode(negative,np.ones((5,5),np.uint8))>0
    negative &= ~cv2.dilate(truth,np.ones((15,15),np.uint8)).astype(bool)
    labels=np.full(shape,255,np.uint8)
    labels[positive&valid]=1; labels[negative&valid]=0
    confidence=np.zeros(shape,np.float32)
    confidence[labels==1]=.6
    confidence[labels==0]=.25
    # Preserve the full published footprint for descriptive checking; ignore band
    # and unknown areas belong to training/evaluation policy, not a new truth.
    for name,array,dtype,nodata in (("published_debris_reference.tif",truth,"uint8",None),
                                    ("partial_labels.tif",labels,"uint8",255),
                                    ("label_confidence.tif",confidence,"float32",None)):
        meta=profile.copy(); meta.update(count=1,dtype=dtype,nodata=nodata,compress="deflate")
        with rasterio.open(ROOT/name,"w",**meta) as dst:
            dst.write(array,1)
    points=[list(transform*(float(x)+.5,float(y)+.5)) for x,y in pixels]
    points.append(points[0])
    geometry={"type":"Polygon","coordinates":[points]}
    geo=transform_geom(crs,"EPSG:4326",geometry)
    fc={"type":"FeatureCollection","features":[{"type":"Feature","properties":{
         "mine":"Amyntaio","event":"2017-06-10","reference":"JISDM 2019 Figure 1 debris outline",
         "label_quality":"digitized published reference, not original expert GIS"},"geometry":geo}]}
    (ROOT/"debris_reference_wgs84.geojson").write_text(json.dumps(fc,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    rgb=np.stack([raw[2],raw[1],raw[0]],-1).astype(np.float32)
    for c in range(3):
        lo,hi=np.percentile(rgb[:,:,c][valid],[2,98]); rgb[:,:,c]=np.clip((rgb[:,:,c]-lo)/max(hi-lo,1),0,1)*255
    bgr=rgb[:,:,::-1].astype(np.uint8)
    review=(.35*bgr).astype(np.uint8)
    review[labels==1]=(.6*bgr[labels==1]+.4*np.array([30,210,30])).astype(np.uint8)
    review[labels==0]=(.6*bgr[labels==0]+.4*np.array([210,160,30])).astype(np.uint8)
    cv2.polylines(review,[np.round(pixels).astype(np.int32)],True,(30,40,240),1)
    cv2.imencode(".png",review)[1].tofile(str(ROOT/"label_review.png"))
    area=float(truth.sum())*100
    names=["pre_BGRN_10m.tif","post_BGRN_10m.tif","partial_labels.tif","label_confidence.tif",
           "published_debris_reference.tif","debris_reference_wgs84.geojson","source_review/figure_xref4.png",
           "source_review/registration_diagnostic.json"]
    record={"mine_id":"Amyntaio","role":"development","event":"2017-06-10",
        "boundary_source":"https://www.fig.net/resources/proceedings/2019/04_JISDM2019/65.pdf",
        "reference_figure":"Figure 1, PDF page 2; black dashed debris extent, not yellow crack envelope",
        "reference_date":"2017-06-20 RapidEye","target_date":"2017-06-28 Sentinel-2",
        "source_polygon_pixels":POLYGON,"stable_negative_boxes_source_pixels":NEGATIVE_BOXES,
        "label_schema":{"0":"project-reviewed weak stable background","1":"published-debris conservative interior","255":"unknown / boundary uncertainty"},
        "positive_uncertainty_buffer_m":50,"label_quality":"georeferenced published figure reference; not original expert masks",
        "registration_holdout_consistency":{k:registration[k] for k in ("heldout_matches","heldout_within_3px_fraction","heldout_median_residual_target_pixels")},
        "registration_residual_is_not_geological_boundary_accuracy":True,
        "figure_reported_area_m2":3190000,"digitized_raster_area_m2":area,
        "positive_pixels":int((labels==1).sum()),"negative_pixels":int((labels==0).sum()),"unknown_pixels":int((labels==255).sum()),
        "figure_date_difference_days":8,"date_limit":"core debris footprint assumption requires visual review; edge held unknown",
        "field_or_independent_expert_review":False,"quality_review_status":"pending",
        "data_redistribution":"author figure/underlying RapidEye rights retained; derived reference for project research, no open-original-dataset claim",
        "model_predictions_during_labeling":0,"frozen_assets":{n:sha(ROOT/n) for n in names}}
    if not .7*3190000<area<1.3*3190000:
        raise ValueError(f"Digitized footprint is inconsistent with published 3.19km2 ({area/1e6:.2f}); review outline instead of fitting area")
    record_path.write_text(json.dumps(record,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({k:record[k] for k in ("positive_pixels","negative_pixels","unknown_pixels","digitized_raster_area_m2","quality_review_status")}),flush=True)


if __name__=="__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    prepare()
