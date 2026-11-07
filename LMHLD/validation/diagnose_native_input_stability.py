# -*- coding: utf-8 -*-
"""Bounded diagnostic of deployment normalization/grid sensitivity on development only."""
import os
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; os.environ.setdefault('TORCH_HOME',str(ROOT/'cache/torch')); os.environ['SMP_DISABLE_VERSION_CHECK']='1'
import json
import sys
import numpy as np
import rasterio
sys.path.insert(0,str(ROOT))
from detector import LandslideDetector
from finetune_negatives import norm_patch
from run_rgb_joint_study import CURRENT,BASE,sha,write
from evaluate_mine_study import metrics
from mine_data_protocol import DATA,assert_development_cases,verify_assets
from native_context_inference import context_probability

OUT=ROOT/'experiments/native_input_stability_20261003'
CASES=['Amyntaio','LasCruces','Hpakant']


def predict(detector,image,mode,copy_red=False):
    image=image.copy()
    if copy_red: image[:,:,3]=image[:,:,2]
    if mode=='per_channel_nonoverlap': return detector.detect_image(image,patch_norm='minmax',max_batch=128)
    if mode=='per_channel_overlap': return context_probability(detector,image,[32],batch=128)
    h,w=image.shape[:2]; ps=32
    padded=np.pad(image,((0,(-h)%ps),(0,(-w)%ps),(0,0)),mode='reflect')
    if mode=='scene_shared_percentile':
        lo,hi=np.percentile(image,[2,98]); padded=np.clip((padded-lo)/max(hi-lo,1)*255,0,255)
    coordinates=[(y,x) for y in range(0,padded.shape[0],32) for x in range(0,padded.shape[1],32)]
    out=np.zeros(padded.shape[:2],np.float32)
    for start in range(0,len(coordinates),128):
        sub=coordinates[start:start+128]
        features=np.stack([padded[y:y+32,x:x+32].transpose(2,0,1) for y,x in sub])
        if mode=='patch_shared':
            lo=features.min((1,2,3),keepdims=True); hi=features.max((1,2,3),keepdims=True)
            features=(features-lo)/(hi-lo+1e-6)*255
        prob=detector.predict_batch(features)
        for (y,x),p in zip(sub,prob): out[y:y+32,x:x+32]=p
    return out[:h,:w]


def run():
    assert_development_cases(CASES); verify_assets(); OUT.mkdir(exist_ok=True); final=OUT/'results.json'
    if final.exists(): print('Reusing complete stability diagnostic'); return
    modes=['per_channel_nonoverlap','per_channel_overlap','patch_shared','scene_shared_percentile']
    plan={'cases':CASES,'weights':{str(p):sha(p) for p in (CURRENT,BASE)},'preprocessing':modes,'threshold':.5,
          'role':'development-only diagnostic, alternative preprocessing not validated for deployment',
          'no_training_or_threshold_changes':True,'no_reserved_predictions':True}
    write(OUT/'protocol.json',plan); records=[]
    for model,weights in (('current',CURRENT),('source',BASE)):
        det=LandslideDetector(str(weights),patch_size=32,threshold=.5)
        for eid in CASES:
            folder=DATA/f'development/{eid}'
            with rasterio.open(folder/'post_BGRN_10m.tif') as src: image=src.read().transpose(1,2,0).astype(np.float32)
            with rasterio.open(folder/'partial_labels.tif') as src: y=src.read(1)
            for mode in modes:
                results={}
                for name,copy in (('NIR',False),('RGB',True)):
                    p=predict(det,image,mode,copy); results[name]=metrics(p>=.5,y==1,y!=255)
                    np.save(OUT/f'{model}_{eid}_{mode}_{name}.npy',p)
                records.append({'model':model,'case':eid,'mode':mode,'metrics':results})
                print(f'{model}/{eid}/{mode}: RGB recall={results["RGB"]["recall"]:.3f} F1={results["RGB"]["F1"]:.3f} weakBG={results["RGB"]["background_FPR"]:.3f}',flush=True)
    write(final,{'plan':plan,'records':records})


if __name__=='__main__':
    sys.stdout.reconfigure(encoding='utf-8'); run()
