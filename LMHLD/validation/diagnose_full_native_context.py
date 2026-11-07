# -*- coding: utf-8 -*-
"""Keep native spatial detail at64/128 input; development-only fixed-weight diagnostic."""
import os
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];os.environ.setdefault('TORCH_HOME',str(ROOT/'cache/torch'));os.environ['SMP_DISABLE_VERSION_CHECK']='1'
import json
import sys
import numpy as np
import rasterio
sys.path.insert(0,str(ROOT))
from detector import LandslideDetector
from run_rgb_joint_study import CURRENT,BASE,sha,write
from evaluate_mine_study import metrics
from mine_data_protocol import DATA,assert_development_cases,verify_assets


def run():
    ids=['Amyntaio','LasCruces','Hpakant'];assert_development_cases(ids);verify_assets()
    out=ROOT/'experiments/full_native_context_20261003';out.mkdir(exist_ok=True)
    if (out/'results.json').exists(): print('Reusing complete full-context diagnostic');return
    plan={'cases':ids,'input_sizes':[32,64,128],'native_context_not_resized':True,'threshold':.5,
          'normalization':'per-channel minmax each native window','stride':'nonoverlapping fixednative grid',
          'weights':{str(w):sha(w) for w in (CURRENT,BASE)},'no_reserved_prediction':True,'role':'development diagnostic, no automatic promotion'}
    write(out/'protocol.json',plan);rows=[]
    for model,weights in (('current',CURRENT),('source',BASE)):
        det=LandslideDetector(str(weights),patch_size=32,threshold=.5)
        for eid in ids:
            case=DATA/f'development/{eid}'
            with rasterio.open(case/'post_BGRN_10m.tif') as src: image=src.read().transpose(1,2,0).astype(np.float32)
            with rasterio.open(case/'partial_labels.tif') as src: y=src.read(1)
            for size in plan['input_sizes']:
                det.patch_size=size;values={}
                for mode,copy in (('NIR',False),('RGB',True)):
                    x=image.copy()
                    if copy:x[:,:,3]=x[:,:,2]
                    p=det.detect_image(x,patch_norm='minmax',max_batch=16)
                    values[mode]=metrics(p>=.5,y==1,y!=255)
                    np.save(out/f'{model}_{eid}_size{size}_{mode}.npy',p)
                rows.append({'model':model,'case':eid,'input_size':size,'metrics':values})
                print(f'{model}/{eid}/{size}: RGBrec={values["RGB"]["recall"]:.3f}, F1={values["RGB"]["F1"]:.3f}, weakBG={values["RGB"]["background_FPR"]:.3f}',flush=True)
    write(out/'results.json',{'plan':plan,'records':rows})


if __name__=='__main__':sys.stdout.reconfigure(encoding='utf-8');run()
