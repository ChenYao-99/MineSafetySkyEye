# -*- coding: utf-8 -*-
"""Obtain official SCL quality pixels on a published mask's native grid, no detector."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import rasterio
from rasterio.windows import from_bounds
from rasterio.transform import Affine,array_bounds
from rasterio.warp import reproject,transform_bounds,Resampling

ROOT=Path(__file__).resolve().parents[1]/"datasets/mine_benchmark_20261002"
DATA=ROOT/"raw_sources/dlr/s1s2_landslide_reference_data/original_scenes"
INVALID_OPTICAL_CLASSES={0,1,8,9,10,11}


def fetch(eid,sequence="0",local_scl=None):
    audit=json.loads((ROOT/f"quality_audit/{eid}/audit.json").read_text(encoding="utf-8"))
    catalogue=json.loads((ROOT/f"scene_catalogues/{eid}_earthsearch.json").read_text(encoding="utf-8-sig"))
    date=audit["post_date"]
    candidates=[f for f in catalogue["features"] if f["properties"]["datetime"][:10]==date and "scl" in f["assets"] and f["id"].endswith(f"_{sequence}_L2A")]
    if not candidates:
        raise ValueError("No official SCL scene on the published post-event date")
    mask_path=DATA/eid/f"{eid}__None__None__MASK.tif"
    with rasterio.open(mask_path) as mask:
        def coverage(scene):
            asset=scene["assets"]["scl"]
            src_crs=f"EPSG:{scene['properties']['proj:epsg']}"
            mb=transform_bounds(mask.crs,src_crs,*mask.bounds)
            footprint=array_bounds(*asset["proj:shape"],Affine(*asset["proj:transform"][:6]))
            intersection=max(0,min(mb[2],footprint[2])-max(mb[0],footprint[0]))*max(0,min(mb[3],footprint[3])-max(mb[1],footprint[1]))
            return intersection/max((mb[2]-mb[0])*(mb[3]-mb[1]),1)
        # Maximize image footprint coverage before processing-version tie-break;
        # two neighbouring MGRS tiles can both intersect the event center.
        scene=sorted(candidates,key=lambda f:(-coverage(f),f["id"]))[0]
        url=scene["assets"]["scl"]["href"]
        if not url.startswith("https://"):
            raise ValueError("SCL asset must use HTTPS")
        truth=mask.read(1)>0
        profile=mask.profile.copy()
        profile.update(dtype="uint8",count=1,nodata=0,compress="deflate")
        dst=np.zeros((mask.height,mask.width),np.uint8)
        with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR",GDAL_HTTP_VERSION="1.1",
                          GDAL_HTTP_TIMEOUT="25",GDAL_HTTP_MAX_RETRY="2",GDAL_HTTP_RETRY_DELAY="1",
                          CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tif"):
            source_path=url
            if local_scl:
                source_path=str(Path(local_scl).resolve())
                if not Path(source_path).is_relative_to(ROOT.resolve()):
                    raise ValueError("Cached SCL must remain in the D: dataset directory")
            with rasterio.open(source_path) as src:
                asset=scene["assets"]["scl"]
                if (src.shape!=tuple(asset["proj:shape"]) or
                    not src.transform.almost_equals(Affine(*asset["proj:transform"][:6])) or
                    src.crs.to_epsg()!=scene["properties"]["proj:epsg"]):
                    raise ValueError("SCL metadata does not match the declared source scene")
                bounds=transform_bounds(mask.crs,src.crs,*mask.bounds)
                window=from_bounds(*bounds,transform=src.transform).round_offsets().round_lengths()
                source=src.read(1,window=window,boundless=True,fill_value=0)
                reproject(source,dst,src_transform=src.window_transform(window),src_crs=src.crs,
                          dst_transform=mask.transform,dst_crs=mask.crs,src_nodata=0,dst_nodata=0,
                          resampling=Resampling.nearest)
                original_resolution=list(src.res)
    out=ROOT/"quality_audit"/eid
    suffix="" if sequence=="0" else f"_sequence{sequence}"
    scl_path=out/f"official_post_SCL{suffix}.tif"
    valid_path=out/f"post_optical_observation_valid{suffix}.tif"
    with rasterio.open(scl_path,"w",**profile) as target:
        target.write(dst,1)
    optical_valid=~np.isin(dst,list(INVALID_OPTICAL_CLASSES))
    valid_profile=profile.copy(); valid_profile.update(nodata=None)
    with rasterio.open(valid_path,"w",**valid_profile) as target:
        target.write(optical_valid.astype(np.uint8),1)
    foreground=dst[truth]
    counts={str(int(k)):int(v) for k,v in zip(*np.unique(foreground,return_counts=True))}
    result={"EID":eid,"source_scene_id":scene["id"],"source_datetime":scene["properties"]["datetime"],
        "source_asset":url,"source_product":"Copernicus Sentinel-2 L2A SCL via Earth Search",
        "processing_baseline":scene["properties"].get("s2:processing_baseline"),
        "original_SCL_resolution":original_resolution,"resampling":"nearest to published 10m grid",
        "invalid_optical_classes":sorted(INVALID_OPTICAL_CLASSES),
        "cloud_policy":"opaque cloud/cirrus/nodata/defective/snow excluded; shadow retained and reported separately",
        "foreground_class_counts":counts,
        "foreground_kept_by_SCL_rule":float(optical_valid[truth].mean()),
        "whole_frame_kept_by_SCL_rule":float(optical_valid.mean()),
        "foreground_unclassified_fraction":float((foreground==7).mean()),
        "foreground_shadow_fraction":float((foreground==3).mean()),
        "classification_limit":"SCL retention is not independently confirmed visibility; unclassified class 7 and bright-ground/cloud disagreements require visual review",
        "model_predictions_run":0,
        "sha256":{"SCL":hashlib.sha256(scl_path.read_bytes()).hexdigest(),
                  "valid":hashlib.sha256(valid_path.read_bytes()).hexdigest()}}
    (out/f"official_cloud_audit{suffix}.json").write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(result,ensure_ascii=False),flush=True)


if __name__=="__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    parser=argparse.ArgumentParser()
    parser.add_argument("--case",choices=["CL0002","KG0001","BR0001"],required=True)
    parser.add_argument("--sequence",choices=["0","1"],default="0")
    parser.add_argument("--local-scl")
    args=parser.parse_args()
    fetch(args.case,args.sequence,args.local_scl)
