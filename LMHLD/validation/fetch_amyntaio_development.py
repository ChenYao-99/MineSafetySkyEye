# -*- coding: utf-8 -*-
"""Acquire actual four-band development imagery; keep false-colour references separate."""
import hashlib
import json
from pathlib import Path
import sys
import urllib.parse
import requests
import numpy as np
import rasterio
from rasterio.warp import transform_bounds,reproject,Resampling
from rasterio.windows import from_bounds
from rasterio.transform import Affine,array_bounds

ROOT=Path(__file__).resolve().parents[1]/"datasets/mine_benchmark_20261002"
BANDS=("B02","B03","B04","B08")


def fetch():
    context=json.loads((ROOT/"downloads/amyntaio_context.json").read_text(encoding="utf-8"))
    catalogue=json.loads((ROOT/"scene_catalogues/Amyntaio_planetarycomputer.json").read_text(encoding="utf-8-sig"))
    with rasterio.open(context["files"][0]["path"]) as frame:
        crs=frame.crs
        left=np.floor(frame.bounds.left/10)*10; bottom=np.floor(frame.bounds.bottom/10)*10
        right=np.ceil(frame.bounds.right/10)*10; top=np.ceil(frame.bounds.top/10)*10
    width,height=int(round((right-left)/10)),int(round((top-bottom)/10))
    transform=Affine(10,0,left,0,-10,top)
    def coverage(feature):
        asset=feature["assets"]["B02"]
        bounds=array_bounds(*asset.get("proj:shape",feature["properties"].get("proj:shape")),
                           Affine(*asset.get("proj:transform",feature["properties"].get("proj:transform"))[:6]))
        wanted=transform_bounds(crs,f"EPSG:{feature['properties']['proj:epsg']}",left,bottom,right,top)
        intersection=max(0,min(bounds[2],wanted[2])-max(bounds[0],wanted[0]))*max(0,min(bounds[3],wanted[3])-max(bounds[1],wanted[1]))
        return intersection/((wanted[2]-wanted[0])*(wanted[3]-wanted[1]))
    full=[f for f in catalogue["features"] if coverage(f)>=.999]
    pre=[f for f in full if f["properties"]["datetime"][:10]==context["pre_date"]]
    post=[f for f in full if f["properties"]["datetime"][:10]>context["event_date"] and f["properties"].get("eo:cloud_cover",100)<10]
    if not pre or not post:
        raise ValueError("No suitable complete pre/post scene pair")
    selected={"pre":sorted(pre,key=lambda f:(f["properties"].get("eo:cloud_cover",100),f["id"]))[0],
              "post":sorted(post,key=lambda f:(f["properties"]["datetime"],f["id"]))[0]}
    output=ROOT/"development/Amyntaio"
    output.mkdir(parents=True,exist_ok=True)
    record_path=output/"acquisition.json"
    record=json.loads(record_path.read_text(encoding="utf-8")) if record_path.exists() else {
        "case_id":"Amyntaio","role":"development_candidate","event_date":context["event_date"],
        "source":"Microsoft Planetary Computer / Copernicus Sentinel-2 L2A",
        "source_reference":"https://zenodo.org/records/821031","channel_order":"B,G,R,NIR",
        "mask_available":False,"model_training_performed":False,"images":{}}
    for when,scene in selected.items():
        target=output/f"{when}_BGRN_10m.tif"
        if when in record["images"] and target.exists():
            if hashlib.sha256(target.read_bytes()).hexdigest()!=record["images"][when]["sha256"]:
                raise ValueError("Existing development image changed")
            continue
        channels=[]; band_meta=[]
        for band in BANDS:
            asset=scene["assets"][band]
            href=asset["href"]
            endpoint="https://planetarycomputer.microsoft.com/api/sas/v1/sign?href="+urllib.parse.quote(href,safe="")
            # Public, anonymous data service issues read-only short-lived URLs.
            # Never print or persist their SAS query strings.
            response=requests.get(endpoint,timeout=30)
            response.raise_for_status()
            signed=response.json()["href"]
            with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR",GDAL_HTTP_VERSION="1.1",
                              GDAL_HTTP_MAX_RETRY="2",GDAL_HTTP_RETRY_DELAY="1",GDAL_HTTP_TIMEOUT="30",
                              CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tif"):
                with rasterio.open(signed) as src:
                    bounds=transform_bounds(crs,src.crs,left,bottom,right,top)
                    window=from_bounds(*bounds,transform=src.transform).round_offsets().round_lengths()
                    raw=src.read(1,window=window,boundless=True,fill_value=0)
                    dst=np.zeros((height,width),np.uint16)
                    reproject(raw,dst,src_transform=src.window_transform(window),src_crs=src.crs,
                              dst_transform=transform,dst_crs=crs,src_nodata=0,dst_nodata=0,resampling=Resampling.nearest)
            channels.append(dst)
            band_meta.append({"band":band,"unsigned_asset":href,"radiometry":asset.get("raster:bands"),
                              "native_resolution":asset.get("gsd",10)})
            print(f"{when} {band}: read native ROI {dst.shape}",flush=True)
        stack=np.stack(channels)
        with rasterio.open(target,"w",driver="GTiff",height=height,width=width,count=4,dtype="uint16",
                           crs=crs,transform=transform,nodata=0,compress="deflate") as dst:
            dst.write(stack)
            for index,band in enumerate(BANDS,1):
                dst.set_band_description(index,band)
            dst.update_tags(scene_id=scene["id"],acquisition_datetime=scene["properties"]["datetime"],
                            role="development_candidate",channel_order="B,G,R,NIR")
        date=scene["properties"]["datetime"][:10]
        record["images"][when]={"path":str(target),"scene_id":scene["id"],"datetime":scene["properties"]["datetime"],
            "tile_cloud_percentage":scene["properties"].get("eo:cloud_cover"),"bands":band_meta,
            "original_reference_date":context[when+"_date"],"same_date_as_reference":date==context[when+"_date"],
            "crs":str(crs),"transform":list(transform)[:6],"shape":[4,height,width],
            "sha256":hashlib.sha256(target.read_bytes()).hexdigest()}
        record_path.write_text(json.dumps(record,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print("Actual BGRN pair acquired; published failure mask still required before training.",flush=True)


if __name__=="__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    fetch()
