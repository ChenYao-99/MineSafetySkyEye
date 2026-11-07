# -*- coding: utf-8 -*-
"""All frozen3-mine folds, no final-test access, no epoch/threshold selection from results."""
import os
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
os.environ.setdefault('TORCH_HOME',str(ROOT/'cache/torch')); os.environ['SMP_DISABLE_VERSION_CHECK']='1'
import json
import argparse
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
from prepare_three_mine_context import OUT,CASES,prepare
from mine_data_protocol import DATA,assert_development_cases,verify_assets
from native_context_inference import context_probability
DATA_OUT=OUT


def evaluate():
    prepare(); verify_assets(); p=json.loads((OUT/'training_protocol.json').read_text(encoding='utf-8'))
    work=[{'id':'baseline_current','arm':'current_mine','weights':str(CURRENT),'seed':42,'held_cases':list(CASES),'sizes':[32]},
          {'id':'baseline_source','arm':'source_base','weights':str(BASE),'seed':42,'held_cases':list(CASES),'sizes':[32]}]
    for held in CASES:
        for arm in p['arms']:
            for seed in p['seeds']:
                row=json.loads((OUT/f'held_{held}'/f'train_{arm}_seed{seed}.json').read_text(encoding='utf-8'))
                if row['completed_epochs']!=20 or row['protocol_sha256']!=sha(OUT/'training_protocol.json') or sha(Path(row['weights']))!=row.get('weights_sha256'):
                    raise ValueError('All registered fixed-epoch runs must finish before held evaluation')
                if row['held_mine_predictions_during_training']!=0: raise ValueError('Held mine predicted during training')
                work.append({k:v for k,v in {**row,'id':f'held_{held}_{arm}_seed{seed}','held_cases':[held],'sizes':p['inference_contexts'][arm]}.items() if k!='history'})
    dest=OUT/'evaluation'; dest.mkdir(exist_ok=True); fplan=dest/'evaluation_plan.json'; final=dest/'results.json'; partial=dest/'results_partial.json'
    plan={'registered_at':datetime.now().isoformat(timespec='seconds'),'training_protocol_sha256':sha(OUT/'training_protocol.json'),
          'weights':{r['id']:sha(Path(r['weights'])) for r in work},'data_sha256':sha(DATA_OUT/'development_data.json'),
          'primary_threshold':.5,'diagnostic_thresholds':[.3,.4,.5],'native_sliding_stride':'half native context size, identical for paired32px arms',
          'native_resampling':'resize context to32, per-channel minmax0..255, resize output to native context then average overlapping windows',
          'native_scale_fusion':'equal probability mean across16/32/64 for multiscale',
          'source_regression_limit':'observed32px historical source/negative arrays only; intrinsic weights32px regression, not full multiscale source-scene validation',
          'labels':'all known published-reference interior and project-reviewed weak background, unknown excluded',
          'not_new_blind_testing':True,'reserved_final_predictions':0,'engineering_gates':p['engineering_gates']}
    if fplan.exists():
        prior=json.loads(fplan.read_text(encoding='utf-8'))
        if any(prior[k]!=plan[k] for k in ('weights','data_sha256','training_protocol_sha256')): raise ValueError('Frozen context evaluation changed')
        plan=prior
    else: write(fplan,plan)
    if final.exists(): print('Reusing complete context evaluation'); return
    region=Path(DATA_ROOT)/'Emilia-Romagna_Italy_32'; sx=norm_patch(np.load(region/'test_images.npy').astype(np.float32)); sy=np.load(region/'test_labels.npy')[:,0]>0
    split=json.loads((ROOT/'experiments/nir_train/neg_split_seed42.json').read_text(encoding='utf-8')); neg=prepare_background_holdout(np.load(NEG_NPY),split['holdout_idx'])
    rows=json.loads(partial.read_text(encoding='utf-8'))['models'] if partial.exists() else []; done={r['id'] for r in rows}
    for row in work:
        if row['id'] in done: continue
        det=LandslideDetector(row['weights'],patch_size=32,threshold=.5)
        nir=predict_patches(det,sx,batch=128); proxy=sx.copy(); proxy[:,3]=proxy[:,2]; rp=predict_patches(det,proxy,batch=128); bg=predict_patches(det,neg,batch=128)>=.5
        result={**row,'weights_sha256':plan['weights'][row['id']],'source32_NIR':metrics(nir>=.5,sy),'source32_RGB':metrics(rp>=.5,sy),
                'generic32_background':{'block_activation':float((bg.sum((1,2))>=10).mean()),'pixel_activation':float(bg.mean()),'n':len(neg)},'cases':{}}
        for eid in row['held_cases']:
            assert_development_cases([eid]); folder=DATA/f'development/{eid}'
            with rasterio.open(folder/'post_BGRN_10m.tif') as src: post=src.read().transpose(1,2,0).astype(np.float32)
            with rasterio.open(folder/'partial_labels.tif') as src: y=src.read(1)
            values={}
            for mode,copy in (('measured_NIR',False),('RGB_proxy',True)):
                probability=context_probability(det,post,row['sizes'],copy_red=copy,batch=128)
                np.save(dest/f'{row["id"]}_{eid}_{mode}_probability.npy',probability)
                values[mode]={f'@{t:g}':metrics(probability>=t,y==1,y!=255) for t in plan['diagnostic_thresholds']}
            result['cases'][eid]=values
        rows.append(result); write(partial,{'plan':plan,'models':rows})
        key=row['held_cases'][0]; display=result['cases'][key]['RGB_proxy']['@0.5']
        print(f'{row["id"]} nativeRGB {key} F1={display["F1"]:.4f}, recall={display["recall"]:.4f}, weakBG={display["background_FPR"]:.4f}, generic32={result["generic32_background"]["block_activation"]:.3f}',flush=True)
        del det; torch.cuda.empty_cache()
    write(final,{'completed_at':datetime.now().isoformat(timespec='seconds'),'plan':plan,'models':rows})
    print(f'{len(work)-2} registered ablations and2 baselines complete; reserved final test remained closed',flush=True)


if __name__=='__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    parser=argparse.ArgumentParser(); parser.add_argument('--source-init',action='store_true'); args=parser.parse_args()
    if args.source_init: OUT=ROOT/'experiments/three_mine_source_init_20261003'
    evaluate()
