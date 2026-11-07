# -*- coding: utf-8 -*-
"""Freeze explicit native development mines, centres and16/32/64px label views."""
import hashlib
import json
import sys
from pathlib import Path
import cv2
import numpy as np
import rasterio
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from finetune_negatives import norm_patch
from mine_data_protocol import DATA,REGISTRY,registry,verify_assets,assert_development_cases

CASES=('Amyntaio','LasCruces','Hpakant')
SIZES=(16,32,64)
OUT=ROOT/'experiments/three_mine_context_20261003'


def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()


def prepare():
    assert_development_cases(list(CASES)); verify_assets()
    OUT.mkdir(parents=True,exist_ok=True); path=OUT/'development_data.json'
    if path.exists():
        data=json.loads(path.read_text(encoding='utf-8'))
        for name,digest in data['frozen_arrays'].items():
            if sha(OUT/name)!=digest: raise ValueError('Frozen multi-context array changed')
        for eid,digest in data['references'].items():
            if sha(DATA/f'development/{eid}/label_reference.json')!=digest: raise ValueError('Frozen reference changed')
        return data
    data={'case_ids':list(CASES),'native_window_sizes':list(SIZES),'model_input_size':32,'references':{},'cases':{},'frozen_arrays':{},
          'role':'previously observed development mines; whole mine train2/dev1, not final blind tests',
          'centre_sampling':'8px native grid; centre must have known1/0; all3 scales >=5% known support',
          'image_resampling':'16->32 INTER_LINEAR,64->32 INTER_AREA; no new spatial detail claimed',
          'label_resampling':'nearest for classes/confidence, unknown remains255 and weight0',
          'new_reserved_mines_read_for_prediction':False}
    for eid in CASES:
        folder=DATA/f'development/{eid}'; ref=json.loads((folder/'label_reference.json').read_text(encoding='utf-8'))
        for name,digest in ref['frozen_assets'].items():
            if sha(folder/name)!=digest: raise ValueError('Original image or conservative label changed')
        with rasterio.open(folder/'post_BGRN_10m.tif') as src:
            if src.descriptions!=('B02','B03','B04','B08') or src.res!=(10,10): raise ValueError('Actual10m physical BGRN required')
            image=src.read().transpose(1,2,0).astype(np.float32)
        with rasterio.open(folder/'partial_labels.tif') as src: y=src.read(1)
        with rasterio.open(folder/'label_confidence.tif') as src: w=src.read(1)
        centres=[]
        for cy in range(32,image.shape[0]-31,8):
            for cx in range(32,image.shape[1]-31,8):
                if y[cy,cx]==255: continue
                if any(np.mean(y[cy-s//2:cy+s//2,cx-s//2:cx+s//2]!=255)<.05 for s in SIZES): continue
                centres.append({'xy':[cx,cy],'kind':'positive' if y[cy,cx]==1 else 'negative'})
        if not all(any(c['kind']==k for c in centres) for k in ('positive','negative')): raise ValueError('No support for class-balanced sampling')
        for size in SIZES:
            xs=[]; ys=[]; ws=[]
            for c in centres:
                cx,cy=c['xy']; sl=np.s_[cy-size//2:cy+size//2,cx-size//2:cx+size//2]
                native=image[sl]
                if size!=32: native=cv2.resize(native,(32,32),interpolation=cv2.INTER_LINEAR if size<32 else cv2.INTER_AREA)
                label=cv2.resize(y[sl],(32,32),interpolation=cv2.INTER_NEAREST)
                confidence=cv2.resize(w[sl],(32,32),interpolation=cv2.INTER_NEAREST)
                xs.append(native.transpose(2,0,1)); ys.append(label); ws.append(confidence)
            arrays={'x':norm_patch(np.stack(xs)),'y':np.stack(ys),'weight':np.stack(ws)}
            if np.any(arrays['weight'][arrays['y']==255]!=0): raise ValueError('Interpolation introduced unknown supervision')
            for name,a in arrays.items():
                dest=OUT/f'{eid}_size{size}_{name}.npy'; np.save(dest,a); data['frozen_arrays'][dest.name]=sha(dest)
        data['cases'][eid]={'centres':centres,'positive_centres':sum(c['kind']=='positive' for c in centres),
            'negative_centres':sum(c['kind']=='negative' for c in centres),'overlaps_not_independent_cases':True,
            'known_positive_pixels':ref['positive_pixels'],'label_quality':ref['label_quality']}
        data['references'][eid]=sha(folder/'label_reference.json')
        print(f'{eid}: {len(centres)} centres, pos={data["cases"][eid]["positive_centres"]}, neg={data["cases"][eid]["negative_centres"]}',flush=True)
    path.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    reg=registry(); reg['cases']['Hpakant'].update(quality_review_status='accepted_for_development_reference_only',
        label_reference='development/Hpakant/label_reference.json',native_model_predictions_run=0,
        project_visual_review='Original native post and source/reference overlay checked; ordinary project review, not independent expert GIS',
        unresolved_quality=['original expert GIS unavailable','multiple panel transform uncertainty','same-calendar-day different sensor/time'])
    verify_assets(reg); REGISTRY.write_text(json.dumps(reg,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    return data


if __name__=='__main__':
    sys.stdout.reconfigure(encoding='utf-8'); prepare()
