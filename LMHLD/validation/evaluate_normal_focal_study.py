# -*- coding: utf-8 -*-
"""Evaluate complete finite32/128 fixed-epoch folds and observed full-scene backgrounds."""
import os
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];os.environ.setdefault('TORCH_HOME',str(ROOT/'cache/torch'));os.environ['SMP_DISABLE_VERSION_CHECK']='1'
import json,sys
import numpy as np
import rasterio
import torch
import cv2
sys.path.insert(0,str(ROOT))
from detector import LandslideDetector
from finetune_negatives import DATA_ROOT,NEG_NPY,norm_patch
from evaluate_mine_study import metrics,predict_patches,prepare_background_holdout
from run_rgb_joint_study import CURRENT,BASE,sha,write
from mine_data_protocol import DATA,assert_development_cases,verify_assets

STUDY=ROOT/'experiments/normal_focal_study_20261003';CASES=('Amyntaio','LasCruces','Hpakant')


def evaluate():
    verify_assets();p=json.loads((STUDY/'protocol.json').read_text(encoding='utf-8'))
    work=[]
    for held in CASES:
        for size in p['sizes']:
            for seed in p['seeds']:
                r=json.loads((STUDY/f'held_{held}'/f'size{size}_seed{seed}.json').read_text(encoding='utf-8'))
                if r['epochs']!=15 or r['protocol_sha']!=sha(STUDY/'protocol.json') or sha(Path(r['weights']))!=r.get('weights_sha'):raise ValueError('Complete frozen finite runs required')
                if not all(np.isfinite(h['loss']) for h in r['history']):raise ValueError('Numerically invalid training is not accuracy evidence')
                work.append({k:v for k,v in {**r,'id':f'held{held}_size{size}_seed{seed}'}.items() if k!='history'})
    out=STUDY/'evaluation';out.mkdir(exist_ok=True);final=out/'results.json';partial=out/'results_partial.json'
    if final.exists():print('Reusing complete full-resolution study');return
    plan={'protocol_sha':sha(STUDY/'protocol.json'),'weights':{r['id']:r['weights_sha'] for r in work},'thresholds':[.5,.4],
       'source_regression':'original32px observed arrays, weight-level capability only; full native-size background scenes separately reported',
       'native_input':'actual32 or128 without spatial resize; per-channel minmax; fixed nonoverlap grid',
       'role':'previously observed3-mine development validation','new_reserved_predictions':0,
       'engineering_gate_registered_before_held_predictions':{'primary_threshold':.5,'RGB_macro_recall_min':.7,'RGB_worst_mine_recall_min':.5,
          'weak_background_FPR_max':.15,'source_NIR_retention_min':.85,'source_RGB_retention_min':.95,'generic32_block_activation_max':.1,
          'full_observed_background_proxy_max':.1,'reviewed_normal_activation_max':.15,'competition_award_threshold':False}}
    write(out/'plan.json',plan)
    region=Path(DATA_ROOT)/'Emilia-Romagna_Italy_32';sx=norm_patch(np.load(region/'test_images.npy').astype(np.float32));sy=np.load(region/'test_labels.npy')[:,0]>0
    split=json.loads((ROOT/'experiments/nir_train/neg_split_seed42.json').read_text(encoding='utf-8'));neg=prepare_background_holdout(np.load(NEG_NPY),split['holdout_idx'])
    backgrounds=[ROOT/'experiments/domain_gap/negatives_img'/f'{n}.png' for n in ('beijing_city','huabei_farmland','tianjin_port')]
    saved=json.loads(partial.read_text(encoding='utf-8')) if partial.exists() else None
    if saved and saved['plan']!=plan:raise ValueError('Evaluation resume plan changed')
    rows=saved['models'] if saved else [];done={r['id'] for r in rows}
    for r in work:
        if r['id'] in done:continue
        det=LandslideDetector(r['weights'],patch_size=32,threshold=.5)
        src=predict_patches(det,sx,batch=128);rgb=sx.copy();rgb[:,3]=rgb[:,2];srgb=predict_patches(det,rgb,batch=128);bg=predict_patches(det,neg,batch=128)>=.5
        item={**r,'source32_NIR':metrics(src>=.5,sy),'source32_RGB':metrics(srgb>=.5,sy),
              'generic32_background':{'block_activation':float((bg.sum((1,2))>=10).mean()),'n':len(bg)},'case':{},'full_background_proxy':[]}
        held=r['held'];assert_development_cases([held]);folder=DATA/f'development/{held}'
        with rasterio.open(folder/'post_BGRN_10m.tif') as f:image=f.read().transpose(1,2,0).astype(np.float32)
        with rasterio.open(folder/'partial_labels.tif') as f:y=f.read(1)
        det.patch_size=r['size']
        for mode,copy in (('NIR',False),('RGB',True)):
            x=image.copy()
            if copy:x[:,:,3]=x[:,:,2]
            probability=det.detect_image(x,patch_norm='minmax',max_batch=16)
            item['case'][mode]={f'@{t:g}':metrics(probability>=t,y==1,y!=255) for t in plan['thresholds']}
            with rasterio.open(DATA/f'review_v2_20261003/{held}/reviewed_normal_mask.tif') as f:normal=f.read(1)>0
            item.setdefault('reviewed_normal',{})[mode]={f'@{t:g}':{'activation':float((probability[normal]>=t).mean()),'pixels':int(normal.sum()),'weak_reference_not_field_FPR':True} for t in plan['thresholds']}
            np.save(out/f'{r["id"]}_{mode}.npy',probability)
        for path in backgrounds:
            if not path.exists():continue
            x=cv2.imdecode(np.fromfile(path,np.uint8),1);x=np.concatenate([x,x[:,:,2:3]],2).astype(np.float32)
            q=det.detect_image(x,patch_norm='minmax',max_batch=16)
            item['full_background_proxy'].append({'source':path.name,'predicted_fraction':float((q>=.5).mean()),
                'no_ground_truth_or_field_FPR_claim':True,'source_GSD_not_matching10m_mine':True})
        rows.append(item);write(partial,{'plan':plan,'models':rows})
        m=item['case']['RGB']['@0.5'];print(f'{r["id"]}: RGBrec={m["recall"]:.3f}, F1={m["F1"]:.3f}, weakBG={m["background_FPR"]:.3f}, generic32={item["generic32_background"]["block_activation"]:.3f}, newNormal={item["reviewed_normal"]["RGB"]["@0.5"]["activation"]:.3f}',flush=True)
        del det;torch.cuda.empty_cache()
    write(final,{'plan':plan,'models':rows,'production_model_changed':False,'new_independent_expert_feedback':False})


if __name__=='__main__':sys.stdout.reconfigure(encoding='utf-8');evaluate()
