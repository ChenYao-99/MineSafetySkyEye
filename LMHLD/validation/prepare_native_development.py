# -*- coding: utf-8 -*-
"""Freeze the first native mine development reference and spatially separate windows."""
import hashlib
import json
from pathlib import Path
import sys
import cv2
import numpy as np
import rasterio

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from finetune_negatives import norm_patch
from mine_data_protocol import assert_development_cases,DATA,REGISTRY

ROOT=Path(__file__).resolve().parents[1]
CASE=DATA/"development/Amyntaio"
OUT=ROOT/"experiments/native_boundary_20261002"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path,value):
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")


def prepare():
    OUT.mkdir(parents=True,exist_ok=True)
    manifest=OUT/"development_data.json"
    if manifest.exists():
        value=json.loads(manifest.read_text(encoding="utf-8"))
        for name,digest in value["frozen_arrays"].items():
            if sha(OUT/name)!=digest:
                raise ValueError("Frozen native development arrays changed")
        return value
    reference=json.loads((CASE/"label_reference.json").read_text(encoding="utf-8"))
    for name,digest in reference["frozen_assets"].items():
        if sha(CASE/name)!=digest:
            raise ValueError("Native development reference asset changed")
    with rasterio.open(CASE/"post_BGRN_10m.tif") as src:
        if tuple(src.descriptions)!=("B02","B03","B04","B08"):
            raise ValueError("Native input must contain actual B/G/R/NIR, never alpha or false colour")
        x=src.read().astype(np.float32)
        transform=src.transform
    with rasterio.open(CASE/"partial_labels.tif") as src:
        y=src.read(1)
    with rasterio.open(CASE/"label_confidence.tif") as src:
        weights=src.read(1)
    if not set(np.unique(y)).issubset({0,1,255}):
        raise ValueError("Partial-label schema must be explicit")
    if np.any(weights[y==255]!=0):
        raise ValueError("Unknown pixels must have zero loss weight")
    windows=[]
    for row in range(0,x.shape[1]-31,16):
        for col in range(0,x.shape[2]-31,16):
            if row%128>96 or col%128>96:
                continue
            target=y[row:row+32,col:col+32]
            valid=target!=255
            if valid.mean()<.5:
                continue
            fraction=float((target[valid]==1).mean())
            kind="positive" if fraction>=.2 else "negative" if fraction<=.01 else "mixed"
            if kind=="mixed":
                continue
            windows.append({"box":[col,row,col+32,row+32],"block":[col//128,row//128],"kind":kind})
    rng=np.random.default_rng(1702)
    pos_blocks=sorted({tuple(w["block"]) for w in windows if w["kind"]=="positive"})
    neg_blocks=sorted({tuple(w["block"]) for w in windows if w["kind"]=="negative"})
    val_blocks={pos_blocks[int(rng.integers(len(pos_blocks)))]}
    neg_available=[b for b in neg_blocks if b not in val_blocks]
    val_blocks.add(neg_available[int(rng.integers(len(neg_available)))])
    training,validation=[],[]
    for w in windows:
        if tuple(w["block"]) in val_blocks:
            validation.append(w)
        else:
            col,row,c1,r1=w["box"]
            # Exclude training image windows within 32 pixels of each val block.
            if any(col<(bx+1)*128+32 and c1>bx*128-32 and row<(by+1)*128+32 and r1>by*128-32 for bx,by in val_blocks):
                continue
            training.append(w)
    for name,rows in (("train",training),("val",validation)):
        if not any(w["kind"]=="positive" for w in rows) or not any(w["kind"]=="negative" for w in rows):
            raise ValueError("Spatial split lacks foreground or weak-negative windows")
        images=np.stack([x[:,w["box"][1]:w["box"][3],w["box"][0]:w["box"][2]] for w in rows])
        labels=np.stack([y[w["box"][1]:w["box"][3],w["box"][0]:w["box"][2]] for w in rows])
        confidence=np.stack([weights[w["box"][1]:w["box"][3],w["box"][0]:w["box"][2]] for w in rows])
        np.save(OUT/f"mine_{name}_x.npy",norm_patch(images))
        np.save(OUT/f"mine_{name}_y.npy",labels)
        np.save(OUT/f"mine_{name}_weight.npy",confidence)
    value={"mine_id":"Amyntaio","input":"actual Sentinel BGRN 10m, post2017-06-28",
        "label_quality":reference["label_quality"],"reference_date":"2017-06-20, 8-day mismatch; 50m uncertainty band",
        "project_visual_review":"source outline and native photo correspondence checked; ordinary project review, no independent expert approval",
        "label_schema":reference["label_schema"],"negative_semantics":"apparently stable event-background, not a safety or no-future-failure assertion",
        "split_unit":"128px spatial blocks with 32px training halo; same mine, not an independent mine test",
        "validation_blocks":[list(b) for b in sorted(val_blocks)],"train_windows":training,"val_windows":validation,
        "overlapping_windows_are_not_independent_events":True,"held_new_mines_used":False,
        "reference_sha256":sha(CASE/"label_reference.json"),
        "frozen_arrays":{p.name:sha(p) for p in OUT.glob("mine_*.npy")}}
    write(manifest,value)
    reg=json.loads(REGISTRY.read_text(encoding="utf-8"))
    reg["cases"]["Amyntaio"]={"mine_id":"Amyntaio","event_date":"2017-06-10","scope":"open_pit_wall_failure",
        "role":"development","quality_review_status":"accepted_for_development_reference_only",
        "raw_source":"Microsoft Planetary Computer / Copernicus S2L2A","label_source":reference["boundary_source"],
        "label_quality":reference["label_quality"],"independent_expert_or_field_audit":False,
        "use_all_times_with_same_mine_role":True,"model_predictions_run":0,
        "development_manifest":"../../experiments/native_boundary_20261002/development_data.json"}
    write(REGISTRY,reg)
    assert_development_cases(["Amyntaio"])
    reference["quality_review_status"]="accepted_for_development_reference_only"
    # Do not change the frozen label-reference document; record acceptance in the registry.
    print(f"Frozen native spatial dev split: train={len(training)}, val={len(validation)}, validation blocks={sorted(val_blocks)}",flush=True)
    print({name:sum(w["kind"]==name for w in training) for name in ("positive","negative")},flush=True)
    return value


if __name__=="__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    prepare()
