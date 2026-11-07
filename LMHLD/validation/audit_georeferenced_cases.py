# -*- coding: utf-8 -*-
"""Audit published image/boundary pairs without importing or running any detector."""
import argparse
import csv
import hashlib
import json
import sqlite3
import struct
import sys
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import rasterio
from rasterio.features import rasterize
from rasterio.warp import transform_geom

ROOT=Path(__file__).resolve().parents[1]/"datasets/mine_benchmark_20261002"
DATA=ROOT/"raw_sources/dlr/s1s2_landslide_reference_data"
BANDS=("B02","B03","B04","B08")
SCOPE={"CL0002":"open_pit_wall_failure", "KG0001":"mine_adjacent_slope_failure", "BR0001":"tailings_dam_mudflow_auxiliary"}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def wkb_geometry(data,offset=0):
    endian="<" if data[offset]==1 else ">"
    kind=struct.unpack_from(endian+"I",data,offset+1)[0]
    offset+=5
    base=kind%1000
    dims=2+(1 if kind//1000 in (1,2) else 2 if kind//1000==3 else 0)
    if base==3:
        n=struct.unpack_from(endian+"I",data,offset)[0]; offset+=4
        rings=[]
        for _ in range(n):
            points=struct.unpack_from(endian+"I",data,offset)[0]; offset+=4
            values=struct.unpack_from(endian+str(points*dims)+"d",data,offset)
            offset+=points*dims*8
            rings.append([list(values[i:i+2]) for i in range(0,len(values),dims)])
        return {"type":"Polygon","coordinates":rings},offset
    if base==6:
        n=struct.unpack_from(endian+"I",data,offset)[0]; offset+=4
        polygons=[]
        for _ in range(n):
            child,offset=wkb_geometry(data,offset)
            if child["type"]!="Polygon":
                raise ValueError("Invalid MultiPolygon child")
            polygons.append(child["coordinates"])
        return {"type":"MultiPolygon","coordinates":polygons},offset
    raise ValueError(f"Unsupported published geometry type: {kind}")


def gpkg_geometry(data):
    if data[:2]!=b"GP":
        raise ValueError("Not GeoPackage geometry")
    flag=data[3]
    envelope=(flag>>1)&7
    sizes={0:0,1:32,2:48,3:48,4:64}
    if envelope not in sizes or flag&16:
        raise ValueError("Unsupported/empty geometry")
    endian="<" if flag&1 else ">"
    srid=struct.unpack_from(endian+"i",data,4)[0]
    geometry,_=wkb_geometry(data,8+sizes[envelope])
    return geometry,srid


def image(path):
    with rasterio.open(path) as src:
        array=src.read(1)
        valid=src.read_masks(1)>0
        meta={"crs":str(src.crs),"transform":list(src.transform)[:6],"shape":[src.height,src.width],
              "resolution":list(src.res),"dtype":str(array.dtype),"nodata":src.nodata,
              "bounds":list(src.bounds),"tags":src.tags(),"sha256":sha(path)}
        valid &= np.isfinite(array)
        return array,valid,meta,src.transform,src.crs


def rgb_preview(bands):
    rgb=np.stack([bands[2],bands[1],bands[0]],-1).astype(np.float32)
    good=np.isfinite(rgb).all(-1)&(rgb.sum(-1)>0)
    for channel in range(3):
        lo,hi=np.percentile(rgb[...,channel][good],[2,98])
        rgb[...,channel]=np.clip((rgb[...,channel]-lo)/max(hi-lo,1e-6),0,1)*255
    rgb[~good]=0
    return rgb.astype(np.uint8)


def audit(events):
    connection=sqlite3.connect((DATA/"vector_data/landslide_inventory.gpkg").as_uri()+"?mode=ro",uri=True)
    out=ROOT/"quality_audit"
    out.mkdir(exist_ok=True)
    results=[]
    for eid in events:
        folder=DATA/"original_scenes"/eid
        csv_path=folder/f"{eid}__files_meta.csv"
        entries=list(csv.DictReader(csv_path.open(encoding="utf-8-sig",newline=""),delimiter=";"))
        optical=[r for r in entries if r["sensor"]=="S2L2A" and r["band"] in BANDS]
        pre=sorted({r["date"] for r in optical if r["patch_type"].startswith("PRE")})
        post=sorted({r["date"] for r in optical if r["patch_type"].startswith("POST")})
        if len(pre)!=1 or len(post)!=1:
            raise ValueError(f"Unexpected optical dates for {eid}")
        row=connection.execute("select country,region,date,source_link,geom from landslide_inventory where EID=?",(eid,)).fetchall()
        if not row:
            raise ValueError("Event absent from published vector inventory")
        geometries=[]
        for record in row:
            geometry,srid=gpkg_geometry(record[4])
            if srid!=4326:
                raise ValueError("Unexpected vector CRS; do not use erroneous README projection")
            geometries.append(geometry)
        mask_path=folder/f"{eid}__None__None__MASK.tif"
        mask,mvalid,mmeta,transform,crs=image(mask_path)
        unique=np.unique(mask[mvalid])
        if not set(unique.tolist()).issubset({0,1,255}):
            raise ValueError("Unexpected mask values")
        target=mask>0
        projected=[transform_geom("EPSG:4326",crs,g) for g in geometries]
        digitized=rasterize([(g,1) for g in projected],out_shape=mask.shape,transform=transform,dtype="uint8")>0
        intersection=int((target&digitized&mvalid).sum())
        union=int(((target|digitized)&mvalid).sum())
        iou=intersection/max(union,1)
        metadata={"mask":mmeta}
        input_valid=mvalid.copy()
        views={}
        for name,dates in (("pre",pre),("post",post)):
            raster_rows=[r for r in optical if r["date"]==dates[0]]
            channels=[]
            for band in BANDS:
                record=next(r for r in raster_rows if r["band"]==band)
                path=folder/(record["file_name"]+".tif")
                arr,valid,meta,rt,rc=image(path)
                if arr.shape!=mask.shape or rc!=crs or not rt.almost_equals(transform):
                    raise ValueError(f"Raster grid does not match inventory mask: {path.name}")
                channels.append(arr); input_valid &= valid
                metadata[name+"_"+band]=meta
            views[name]=channels
        for key in ("DEM","SLOPE"):
            path=folder/f"{eid}__CDEM__None__{key}.tif"
            arr,valid,meta,rt,rc=image(path)
            if arr.shape!=mask.shape or rc!=crs or not rt.almost_equals(transform):
                raise ValueError("DEM grid mismatch")
            metadata[key]=meta
        # Leave boundary uncertainty out of interior diagnostics. Published masks
        # remain untouched; this is an additional scored-region policy, not truth.
        edge=cv2.morphologyEx(target.astype(np.uint8),cv2.MORPH_GRADIENT,np.ones((5,5),np.uint8))>0
        interior=input_valid&~edge
        case_dir=out/eid
        case_dir.mkdir(exist_ok=True)
        panels=[]
        contours,_=cv2.findContours(target.astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        for key in ("pre","post"):
            preview=rgb_preview(views[key])
            bgr=preview[:,:,::-1].copy()
            cv2.drawContours(bgr,contours,-1,(30,40,240),2)
            cv2.imencode(".png",bgr)[1].tofile(str(case_dir/f"{key}_inventory_overlay.png"))
            panels.append(bgr)
        canvas=np.concatenate(panels,axis=1)
        cv2.imencode(".png",canvas)[1].tofile(str(case_dir/"pair_inventory_review.png"))
        feature={"type":"FeatureCollection","name":eid,"features":[{"type":"Feature","properties":{"EID":eid,"source":"DLR published inventory"},"geometry":g} for g in geometries]}
        (case_dir/"published_boundary_wgs84.geojson").write_text(json.dumps(feature,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
        event_date=datetime.fromisoformat(row[0][2]).date()
        before=datetime.strptime(pre[0],"%d.%m.%Y").date()
        after=datetime.strptime(post[0],"%d.%m.%Y").date()
        if not before<event_date<=after:
            raise ValueError("Pre/post dates do not bracket declared event")
        result={"EID":eid,"country":row[0][0],"region":row[0][1],"event_date":str(event_date),
            "pre_date":str(before),"post_date":str(after),"source_event_link":row[0][3],
            "label_source":"https://zenodo.org/records/17007637","license":"CC BY 4.0",
            "scope":SCOPE[eid],"label_quality":"published georeferenced research inventory; no project field/expert audit",
            "published_mask_vs_vector_IoU":iou,"mask_foreground_pixels":int((target&input_valid).sum()),
            "interior_foreground_pixels":int((target&interior).sum()),"valid_pixels":int(input_valid.sum()),
            "nodata_fraction":float(1-input_valid.mean()),"cloud_mask_available":False,
            "same_day_post":after==event_date,"temporal_review_required":after==event_date,
            "pre_scene_is_not_guaranteed_slide_free":eid in ("CL0002","KG0001"),
            "model_predictions_run":0,"raster_metadata":metadata,
            "geometric_gate_pass":iou>=.85 and int((target&interior).sum())>=50,
            "field_or_expert_validation":False,"human_quality_review":"pending",
            "IoU_interpretation":"raster/vector consistency from the same inventory, not independent ground-truth accuracy"}
        (case_dir/"audit.json").write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
        results.append(result)
        print(json.dumps({k:result[k] for k in ("EID","scope","pre_date","post_date","published_mask_vs_vector_IoU","mask_foreground_pixels","interior_foreground_pixels","geometric_gate_pass")}),flush=True)
    summary={"created_at":datetime.now().isoformat(timespec="seconds"),"model_predictions_run":0,"cases":results}
    (out/"quality_audit.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")


if __name__=="__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    parser=argparse.ArgumentParser()
    parser.add_argument("--events",nargs="+",choices=list(SCOPE),default=list(SCOPE))
    args=parser.parse_args()
    audit(args.events)
