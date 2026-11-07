# -*- coding: utf-8 -*-
"""Matched whole-mine development transfer; fixed final epoch, never held-mine selection."""
import os
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
os.environ.setdefault('TORCH_HOME',str(ROOT/'cache/torch'))
os.environ.setdefault('HF_HOME',str(ROOT/'cache/huggingface'))
os.environ['SMP_DISABLE_VERSION_CHECK']='1'
import json
import random
import sys
import time
from datetime import datetime
import numpy as np
import torch
sys.path.insert(0,str(ROOT))
from finetune_negatives import DS
from run_native_boundary_study import PartialDataset,partial_loss
from run_rgb_joint_study import model_at,validation,sha,write,CURRENT,DEVICE
from prepare_native_mine_folds import prepare,OUT,CASES
from mine_data_protocol import assert_development_cases,verify_assets,REGISTRY

OLD=ROOT/'experiments/core_ablation_20261001'
MODELS=ROOT/'models_prior/native_mine_transfer_20261003'
ARMS=('weak_patch','partial_boundary')


def protocol():
    data=prepare(); assert_development_cases(list(CASES)); verify_assets()
    original=json.loads((OLD/'protocol.json').read_text(encoding='utf-8'))
    for name,digest in original['frozen_files'].items():
        if sha(OLD/name)!=digest: raise ValueError('Frozen source data changed')
    path=OUT/'training_protocol.json'
    if path.exists():
        plan=json.loads(path.read_text(encoding='utf-8'))
        if plan['development_data_sha256']!=sha(OUT/'development_data.json') or plan['initial_sha256']!=sha(CURRENT): raise ValueError('Frozen transfer protocol changed')
        return plan
    plan={'registered_at':datetime.now().isoformat(timespec='seconds'),'arms':list(ARMS),'seeds':[42,43,44],
        'folds':[{'train_mine':a,'held_development_mine':b} for a,b in (CASES,CASES[::-1])],
        'epochs':15,'updates_per_epoch':120,'common_batch':24,'mine_batch':8,
        'sampling':'12 source positive +12 generic background, 4 mine positive +4 mine weak negative',
        'supervision':'paired true BGRN and NIR:=R, unknown255 weight0; .5BCE+.5Dice',
        'source_guard_weight':.5,'source_guard':'frozen current teacher Bernoulli KL on common source/background only, not mine positives',
        'optimizer':'SGD .0002 momentum.9 decay.0001 ExponentialLR.96',
        'checkpoint_rule':'FIXED FINAL EPOCH15 for every run, no mining/held-mine metric selection',
        'primary_threshold':.5,'secondary_threshold':.4,
        'engineering_diagnostic_gates':{'held_development_reference_recall_min':.5,'held_weak_background_FPR_max':.15,
            'source_NIR_current_retention_min':.85,'source_RGB_current_retention_min':.95,'generic_background_block_max':.10},
        'gates_are_not_competition_standards':True,'gates_frozen_before_transfer_predictions':True,
        'promotion':'No automatic promotion; all3 seeds and both fold regressions must pass; two dev mines alone do not establish field performance',
        'initial_weights':str(CURRENT),'initial_sha256':sha(CURRENT),'development_data_sha256':sha(OUT/'development_data.json'),
        'source_protocol_sha256':sha(OLD/'protocol.json'),'registry_sha256':sha(REGISTRY),
        'label_policy_comparison':'Same images, sampler, seed, augmentation and update budget within fold; effective labeled support differs',
        'mine_test_role':'Both historical development mines, not new independent final test',
        'new_reserved_mines_read_for_prediction':False}
    write(path,plan)
    return plan


def train(train_mine,held_mine,arm,seed,plan):
    assert_development_cases([train_mine])
    fold=f'{train_mine}_to_{held_mine}'; dest=OUT/fold; dest.mkdir(exist_ok=True)
    path=dest/f'train_{arm}_seed{seed}.json'; folder=MODELS/fold/f'{arm}_seed{seed}'; folder.mkdir(parents=True,exist_ok=True)
    digest=sha(OUT/'training_protocol.json'); weightpath=folder/'final_model.pth'; resume=folder/'resume.pth'
    if path.exists():
        old=json.loads(path.read_text(encoding='utf-8'))
        if old['protocol_sha256']!=digest: raise ValueError('Transfer run changed')
        if old['completed_epochs']==plan['epochs'] and sha(weightpath)==old['weights_sha256']:
            print(f'Reusing {fold} {arm}/{seed}',flush=True); return
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark=False; torch.backends.cudnn.deterministic=True
    images=np.concatenate([np.load(OLD/'common_x.npy'),np.load(OLD/'ours_extra.npy')])
    labels=np.concatenate([np.load(OLD/'common_y.npy'),np.zeros((2000,32,32),np.float32)])
    pos=np.flatnonzero(labels.mean((1,2))>=.1); neg=np.flatnonzero(labels.mean((1,2))==0)
    common=DS(images,labels,train=True,seed=seed)
    mx,my,mw=[np.load(OUT/f'{train_mine}_{k}.npy') for k in ('x','y','weight')]
    data=json.loads((OUT/'development_data.json').read_text(encoding='utf-8'))
    windows=data['cases'][train_mine]['train_windows']
    mp=np.array([i for i,w in enumerate(windows) if w['kind']=='positive']); mn=np.array([i for i,w in enumerate(windows) if w['kind']=='negative'])
    replay=PartialDataset(mx,my,mw,seed+200)
    vx,vy=np.load(OLD/'val_x.npy'),np.load(OLD/'val_y.npy')
    model=model_at(CURRENT); teacher=model_at(CURRENT).eval(); teacher.requires_grad_(False)
    optimizer=torch.optim.SGD(model.parameters(),lr=.0002,momentum=.9,weight_decay=.0001)
    scheduler=torch.optim.lr_scheduler.ExponentialLR(optimizer,gamma=.96)
    history=[]; begin=1; started=time.time()
    if resume.exists():
        saved=torch.load(resume,map_location=DEVICE,weights_only=False)
        if saved['protocol_sha256']!=digest: raise ValueError('Resume protocol changed')
        model.load_state_dict(saved['model']); optimizer.load_state_dict(saved['optimizer']); scheduler.load_state_dict(saved['scheduler'])
        common.rng.bit_generator.state=saved['common_rng']; replay.rng.bit_generator.state=saved['replay_rng']
        torch.set_rng_state(saved['torch_rng'].cpu()); torch.cuda.set_rng_state_all([r.cpu() for r in saved['cuda_rng']])
        history=saved['history']; begin=saved['epoch']+1
    for epoch in range(begin,plan['epochs']+1):
        rng=np.random.default_rng(seed*10000+epoch); rrng=np.random.default_rng(seed*10000+epoch+1000000)
        model.train(); losses=[]; supports=[]
        for _ in range(plan['updates_per_epoch']):
            ids=np.concatenate([rng.choice(pos,12,replace=True),rng.choice(neg,12,replace=True)])
            pairs=[(*common[int(i)],torch.ones((32,32))) for i in ids]
            for group,kind in ((mp,'positive'),(mn,'negative')):
                pairs += [replay.get(int(i),weak=arm=='weak_patch',kind=kind) for i in rrng.choice(group,4,replace=True)]
            x=torch.stack([p[0] for p in pairs]).to(DEVICE); y=torch.stack([p[1] for p in pairs]).to(DEVICE).unsqueeze(1)
            w=torch.stack([p[2] for p in pairs]).to(DEVICE).unsqueeze(1); proxy=torch.cat([x[:,:3],x[:,2:3]],1)
            logits=model(torch.cat([x,proxy])); loss=partial_loss(logits,torch.cat([y,y]),torch.cat([w,w]))
            source_logits=torch.cat([logits[:24],logits[32:56]])
            with torch.no_grad():
                tl=teacher(torch.cat([x[:24],proxy[:24]])); tp=torch.sigmoid(tl)
                entropy=torch.nn.functional.binary_cross_entropy_with_logits(tl,tp,reduction='none')
            kl=(torch.nn.functional.binary_cross_entropy_with_logits(source_logits,tp,reduction='none')-entropy).mean()
            loss=loss+plan['source_guard_weight']*kl
            optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
            losses.append(float(loss)); supports.append(float(w.mean()))
        scheduler.step(); source=validation(model,vx,vy)
        history.append({'epoch':epoch,'train_loss':float(np.mean(losses)),'mean_supervision_weight':float(np.mean(supports)),
                        'source_internal_validation':source,'elapsed_seconds':round(time.time()-started,2)})
        record={'train_mine':train_mine,'held_development_mine':held_mine,'arm':arm,'seed':seed,
            'completed_epochs':epoch,'protocol_sha256':digest,'weights':str(weightpath),
            'checkpoint_rule':plan['checkpoint_rule'],'held_mine_predictions_during_training':0,'history':history}
        write(path,record)
        torch.save({'model':model.state_dict(),'optimizer':optimizer.state_dict(),'scheduler':scheduler.state_dict(),
            'epoch':epoch,'protocol_sha256':digest,'history':history,'common_rng':common.rng.bit_generator.state,
            'replay_rng':replay.rng.bit_generator.state,'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all()},resume)
        if epoch==1 or epoch%5==0:
            print(f'{fold} {arm}/{seed} {epoch}/15: sourceRGB F1={source["source_RGB_proxy"]["F1"]:.4f}, sourceBG={source["generic_background"]["block_activation"]:.3f}',flush=True)
    torch.save(model.state_dict(),weightpath); record['weights_sha256']=sha(weightpath); write(path,record)
    del model,teacher,optimizer; torch.cuda.empty_cache()


if __name__=='__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    plan=protocol()
    if DEVICE!='cuda': raise ValueError('CUDA required for this fixed budget')
    for fold in plan['folds']:
        for seed in plan['seeds']:
            for arm in plan['arms']:
                train(fold['train_mine'],fold['held_development_mine'],arm,seed,plan)
