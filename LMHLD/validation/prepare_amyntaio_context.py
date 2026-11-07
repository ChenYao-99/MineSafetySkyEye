# -*- coding: utf-8 -*-
"""Audit false-colour geospatial references; never treat their alpha as NIR."""
import hashlib
import json
from pathlib import Path
import sys
import rasterio
from rasterio.warp import transform_bounds

ROOT=Path(__file__).resolve().parents[1]/"datasets/mine_benchmark_20261002"


def prepare():
    entries=[]
    for when in ("pre","post"):
        path=ROOT/f"downloads/amyntaio_{when}_falsecolor.tif"
        with rasterio.open(path) as src:
            entries.append({"path":str(path),"sha256":hashlib.sha256(path.read_bytes()).hexdigest(),
                "crs":str(src.crs),"bounds":list(src.bounds),"bbox_wgs84":list(transform_bounds(src.crs,"EPSG:4326",*src.bounds)),
                "stored_bands":src.count,"color_interpretation":[c.name for c in src.colorinterp],
                "resolution":list(src.res),"dtype":list(src.dtypes)})
    if entries[0]["bounds"]!=entries[1]["bounds"] or entries[0]["crs"]!=entries[1]["crs"]:
        raise ValueError("Published context frames do not share geometry")
    result={"case_id":"Amyntaio","event_date":"2017-06-10","pre_date":"2017-06-01","post_date":"2017-06-24",
        "bbox_wgs84":entries[0]["bbox_wgs84"],"source":"https://zenodo.org/records/821031",
        "display_channels":["NIR","red","green","alpha"],"allowed_for_model_input":False,
        "reason":"false-colour byte display, not physical B,G,R,NIR reflectance; actual spectral bands required",
        "role":"development_candidate","published_failure_boundary_available":False,"files":entries}
    (ROOT/"downloads/amyntaio_context.json").write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(result,ensure_ascii=False))


if __name__=="__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    prepare()
