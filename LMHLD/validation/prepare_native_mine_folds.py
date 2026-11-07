# -*- coding: utf-8 -*-
"""Freeze two native development mines for whole-mine transfer diagnostics."""
import hashlib
import json
import sys
from pathlib import Path
import numpy as np
import rasterio
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from finetune_negatives import norm_patch
from mine_data_protocol import DATA, REGISTRY, registry, assert_development_cases, verify_assets

OUT=ROOT/'experiments/native_mine_transfer_20261003'
CASES=('Amyntaio','LasCruces')


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare():
    assert_development_cases(list(CASES)); verify_assets()
    OUT.mkdir(parents=True,exist_ok=True); target=OUT/'development_data.json'
    if target.exists():
        value=json.loads(target.read_text(encoding='utf-8'))
        for name,digest in value['frozen_arrays'].items():
            if sha(OUT/name)!=digest: raise ValueError('Frozen mine-transfer array changed')
        for eid,digest in value['references'].items():
            if sha(DATA/f'development/{eid}/label_reference.json')!=digest: raise ValueError('Frozen reference changed')
        return value
    value={'case_ids':list(CASES),'references':{},'cases':{},'frozen_arrays':{},
           'role':'development transfer diagnostics; both mines already observed, not fresh blind tests',
           'split_unit':'whole mine: Amyntaio->LasCruces and LasCruces->Amyntaio',
           'label_schema':{'0':'weak event-background','1':'published-reference interior','255':'unknown'},
           'held_mine_used_for_checkpoint_selection':False,'reserved_mines_read_for_prediction':False}
    for eid in CASES:
        folder=DATA/f'development/{eid}'
        ref=json.loads((folder/'label_reference.json').read_text(encoding='utf-8'))
        for name,digest in ref['frozen_assets'].items():
            if sha(folder/name)!=digest: raise ValueError('Original reference asset changed')
        if ref['model_predictions_during_labeling']!=0: raise ValueError('Labeling requires no model prediction')
        with rasterio.open(folder/'post_BGRN_10m.tif') as src:
            if src.descriptions!=('B02','B03','B04','B08') or src.count!=4: raise ValueError('Physical four-band input required')
            x=src.read().astype(np.float32)
        with rasterio.open(folder/'partial_labels.tif') as src: y=src.read(1)
        with rasterio.open(folder/'label_confidence.tif') as src: w=src.read(1)
        if set(np.unique(y))!={0,1,255} or np.any(w[y==255]!=0): raise ValueError('Unknown-pixel support invalid')
        rows=[]
        for row in range(0,x.shape[1]-31,16):
            for col in range(0,x.shape[2]-31,16):
                lab=y[row:row+32,col:col+32]; valid=lab!=255
                if valid.mean()<.5: continue
                f=float((lab[valid]==1).mean()); kind='positive' if f>=.2 else 'negative' if f<=.01 else 'mixed'
                if kind!='mixed': rows.append({'box':[col,row,col+32,row+32],'kind':kind})
        if not all(any(r['kind']==k for r in rows) for k in ('positive','negative')): raise ValueError('No trainable positive/negative support')
        arrays={'x':norm_patch(np.stack([x[:,r['box'][1]:r['box'][3],r['box'][0]:r['box'][2]] for r in rows])),
                'y':np.stack([y[r['box'][1]:r['box'][3],r['box'][0]:r['box'][2]] for r in rows]),
                'weight':np.stack([w[r['box'][1]:r['box'][3],r['box'][0]:r['box'][2]] for r in rows])}
        for name,array in arrays.items():
            path=OUT/f'{eid}_{name}.npy'; np.save(path,array); value['frozen_arrays'][path.name]=sha(path)
        value['references'][eid]=sha(folder/'label_reference.json')
        value['cases'][eid]={'train_windows':rows,'positive_windows':sum(r['kind']=='positive' for r in rows),
            'negative_windows':sum(r['kind']=='negative' for r in rows),
            'overlapping_windows_are_not_independent_cases':True,'label_quality':ref['label_quality']}
        print(f'{eid}: {len(rows)} windows, {value["cases"][eid]["positive_windows"]} positive, {value["cases"][eid]["negative_windows"]} weak negative',flush=True)
    target.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    reg=registry(); case=reg['cases']['LasCruces']
    case.update(quality_review_status='accepted_for_development_reference_only',
        label_reference='development/LasCruces/label_reference.json',
        project_visual_review='Original pre/post, source overlay and conservative labels reviewed; independent expert false',
        unresolved_quality=['original expert GIS unavailable','boundary digitization and 2-day reference-date uncertainty'])
    verify_assets(reg); REGISTRY.write_text(json.dumps(reg,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    return value


if __name__=='__main__':
    sys.stdout.reconfigure(encoding='utf-8'); prepare()
