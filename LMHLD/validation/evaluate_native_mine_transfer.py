# -*- coding: utf-8 -*-
"""Evaluate fixed whole-mine transfer weights after all training has completed."""
import os
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
os.environ.setdefault('TORCH_HOME',str(ROOT/'cache/torch'))
os.environ['SMP_DISABLE_VERSION_CHECK']='1'
import json
import sys
from datetime import datetime
import numpy as np
import rasterio
import torch
sys.path.insert(0,str(ROOT))
from detector import LandslideDetector
from finetune_negatives import DATA_ROOT,NEG_NPY,norm_patch
from evaluate_mine_study import metrics,predict_patches,prepare_background_holdout
from run_rgb_joint_study import CURRENT,BASE,sha,write
from prepare_native_mine_folds import OUT,prepare
from mine_data_protocol import DATA,assert_development_cases,verify_assets


def native_probabilities(detector,post,copy_red=False):
    image=post.copy()
    if copy_red: image[:,:,3]=image[:,:,2]
    # These weights were trained on per-patch [0,255], never raw uint16 DN.
    return detector.detect_image(image,patch_norm='minmax',max_batch=128)


def evaluate():
    prepare(); verify_assets()
    plan=json.loads((OUT/'training_protocol.json').read_text(encoding='utf-8'))
    work=[{'id':'baseline_current','weights':str(CURRENT),'arm':'current_mine','seed':42,'held_cases':['Amyntaio','LasCruces']},
          {'id':'baseline_source','weights':str(BASE),'arm':'source_base','seed':42,'held_cases':['Amyntaio','LasCruces']}]
    for fold in plan['folds']:
        key=fold['train_mine']+'_to_'+fold['held_development_mine']
        for arm in plan['arms']:
            for seed in plan['seeds']:
                path=OUT/key/f'train_{arm}_seed{seed}.json'; r=json.loads(path.read_text(encoding='utf-8'))
                if r['completed_epochs']!=15 or r['protocol_sha256']!=sha(OUT/'training_protocol.json') or sha(Path(r['weights']))!=r['weights_sha256']:
                    raise ValueError('All fixed-epoch runs must finish with frozen weights before evaluation')
                if r['held_mine_predictions_during_training']!=0: raise ValueError('Held mine was predicted during training')
                work.append({**r,'id':key+'_'+arm+'_seed'+str(seed),'held_cases':[fold['held_development_mine']]})
    dest=OUT/'evaluation'; dest.mkdir(exist_ok=True); final=dest/'results.json'; partial=dest/'results_partial.json'
    frozen={'registered_at':datetime.now().isoformat(timespec='seconds'),'training_protocol_sha256':sha(OUT/'training_protocol.json'),
            'development_data_sha256':sha(OUT/'development_data.json'),'weights':{w['id']:sha(Path(w['weights'])) for w in work},
            'thresholds':[.5,.4],'primary':.5,'automatic_promotion':False,
            'native_input_normalization':'minmax per channel per32px tile to [0,255], same as training',
            'score_role':'whole held development mine, all known published-reference interiors and weak backgrounds; unknown excluded',
            'both_mines_previously_observed':True,'not_new_blind_test':True,'reserved_test_predictions':0}
    fplan=dest/'evaluation_plan.json'
    if fplan.exists():
        previous=json.loads(fplan.read_text(encoding='utf-8'))
        if any(previous.get(k)!=frozen[k] for k in ('training_protocol_sha256','development_data_sha256','weights','native_input_normalization')): raise ValueError('Frozen evaluation input changed')
        frozen=previous
    else: write(fplan,frozen)
    if final.exists(): print('Reusing complete mine-transfer evaluation'); return
    region=Path(DATA_ROOT)/'Emilia-Romagna_Italy_32'
    sx=norm_patch(np.load(region/'test_images.npy').astype(np.float32)); sy=np.load(region/'test_labels.npy')[:,0]>0
    split=json.loads((ROOT/'experiments/nir_train/neg_split_seed42.json').read_text(encoding='utf-8'))
    backgrounds=prepare_background_holdout(np.load(NEG_NPY),split['holdout_idx'])
    old=json.loads(partial.read_text(encoding='utf-8'))['models'] if partial.exists() else []
    done={r['id'] for r in old}
    for row in work:
        if row['id'] in done: continue
        detector=LandslideDetector(row['weights'],patch_size=32,threshold=.5)
        nir=predict_patches(detector,sx,batch=128); proxy=sx.copy(); proxy[:,3]=proxy[:,2]
        rp=predict_patches(detector,proxy,batch=128); bg=predict_patches(detector,backgrounds,batch=128)>=.5
        result={k:v for k,v in row.items() if k!='history'}
        result.update(weights_sha256=frozen['weights'][row['id']],source_test=metrics(nir>=.5,sy),
            source_RGB_proxy_test=metrics(rp>=.5,sy),generic_background_holdout={'n':len(backgrounds),
            'block_activation':float((bg.sum((1,2))>=10).mean()),'pixel_activation':float(bg.mean()),
            'limit':'historical same-pool pseudo-background, not independent mine FPR'},cases={})
        for eid in row['held_cases']:
            assert_development_cases([eid]); folder=DATA/f'development/{eid}'
            with rasterio.open(folder/'post_BGRN_10m.tif') as src: post=src.read().transpose(1,2,0).astype(np.float32)
            with rasterio.open(folder/'partial_labels.tif') as src: y=src.read(1)
            values={}
            for mode,copy_red in (('measured_NIR',False),('RGB_proxy',True)):
                probability=native_probabilities(detector,post,copy_red=copy_red)
                np.save(dest/f'{row["id"]}_{eid}_{mode}_probability.npy',probability)
                values[mode]={f'@{threshold:g}':metrics(probability>=threshold,y==1,y!=255) for threshold in (.5,.4)}
            result['cases'][eid]=values
        old.append(result); write(partial,{'plan':frozen,'models':old})
        diagnostic={eid:{k:round(values['RGB_proxy']['@0.5'][k],4) for k in ('F1','recall','background_FPR')} for eid,values in result['cases'].items()}
        print(f'{row["id"]}: {json.dumps(diagnostic)}, genericBG={result["generic_background_holdout"]["block_activation"]:.3f}',flush=True)
        del detector; torch.cuda.empty_cache()
    write(final,{'completed_at':datetime.now().isoformat(timespec='seconds'),'plan':frozen,'models':old})
    print('Whole-mine development transfer complete; reserved test remained closed',flush=True)


if __name__=='__main__':
    sys.stdout.reconfigure(encoding='utf-8'); evaluate()
