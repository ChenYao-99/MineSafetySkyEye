"""Source-only checkpoint selection for a bounded training-budget diagnosis.

The initial 1200 updates exactly replay the previous short pilot. Continuation
keeps optimizer moments, restarts LR at 3e-4, and runs to update 4800. This tests
additional training WITH an explicit schedule change, not a pure step-count effect.
"""
from __future__ import annotations
import argparse
import copy
import hashlib
from pathlib import Path
import sys
import time
import cv2
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from runtime_paths import configure_runtime_paths
configure_runtime_paths()
import cross_region_scale as base
import validate_scale_selector as split_helper

PRIOR=ROOT/'experiments/source_scale_selector_20261019'
HELD=(2,4,3)  # Hokkaido, Wenchuan, Jiuzhaigou: frozen diagnostic cohort, not all six.
CHECKPOINTS=(1200,2400,3600,4800)


def register(out):
    out=Path(out).resolve()
    if out.drive.upper()!='D:' or not out.is_relative_to(ROOT/'experiments'):
        raise ValueError('Output must stay in D:/LMHLD/experiments')
    out.mkdir(parents=True,exist_ok=True)
    parent=base.read(PRIOR/'protocol.json')
    assets={**parent['code_sha256'],str(Path(__file__)):base.sha(__file__),
            str(PRIOR/'protocol.json'):base.sha(PRIOR/'protocol.json'),
            str(PRIOR/'data_manifest.json'):base.sha(PRIOR/'data_manifest.json'),
            str(ROOT/'experiments/nir_train/finetune_T1b_ours.json'):base.sha(ROOT/'experiments/nir_train/finetune_T1b_ours.json')}
    for item in parent['protected_assets']:assets[item['path']]=item['sha256']
    for held in HELD:
        for seed in (42,43):
            path=PRIOR/f'fold_{held}_{base.REGIONS[held][0]}'/f'seed{seed}'/'result.json'
            assets[str(path)]=base.sha(path)
    for path,digest in assets.items():
        if base.sha(path)!=digest:raise ValueError('Frozen dependency changed')
    path=out/'protocol.json'
    if path.exists():
        protocol=base.read(path)
        if protocol['frozen_assets']!=assets:raise ValueError('Frozen code/data changed; use new directory')
        return protocol,parent
    protocol={'registered_at':base.now(),'frozen_assets':assets,
        'question':'Does more source-only training improve source validation and transfer, or merely improve source fit?',
        'held_indices':list(HELD),'held_regions':[base.REGIONS[g][0] for g in HELD],'seeds':[42,43],
        'cohort_reason':'Hokkaido/Wenchuan have many historical weak-response misses; Jiuzhaigou is a stronger positive control. Historically exposed development cohort, not new blind or all-six-region validation.',
        'base_fits':6,'short_updates':1200,'total_updates':4800,'checkpoints':list(CHECKPOINTS),
        'first_stage':'Exact prior fixed-arm initialization/parent sampling/augmentations/AdamW/cosine1200; verify exact state hash at1200',
        'continuation':'Keep AdamW moments and parent RNG sequence; explicitly restart LR at .0003 then cosine to .00001 over3600 additional updates',
        'continuation_lr':.0003,'continuation_eta_min':.00001,
        'causal_scope':'Additional optimization budget plus a declared LR restart; no claim that step count alone explains differences',
        'unchanged':'Same five source regions x512 parents, 64x64 whole-parent input, four true BGRN bands, loss, architecture, normalization, batch20, 4 examples/source/update, paired flip/rotation schedule. No scale gate fitting or native-window change.',
        'source_checkpoint_selection':'At updates1200/2400/3600/4800, maximize equal-source-region native-grid TTA5 F1 at fixed0.5 on source selection half; strict tie chooses earliest. Target and calibration-half metrics cannot select a checkpoint.',
        'source_selection_and_calibration_split':'Reuse prior deterministic meta_split: source validation first shuffled half is selection; other half is threshold calibration. No overlapping original parents.',
        'train_monitor':'First64 kept training rows of each source, fixed whole view, no augmentation, same BCE+.5softDice; report by-region monitor loss at each checkpoint',
        'source_monitor':'Selection-half TTA native F1/precision/recall/FPR at .5 and full-view soft BCE+.5Dice, each source separately and macro',
        'target_evaluation':'After source-only decision is written, evaluate fixed-final4800 and source-selected checkpoint; reuse cached prior TTA5 short result. No retrospective best-target selection.',
        'target_thresholds':'Primary recalibrated on source calibration half to equal-region background FPR<=.02, same0.001 grid; secondary fixed .5. All three recipes use same calibration-half definition.',
        'methods':['short1200','selected_source','final4800'],
        'retention':'Keep at most six source-selected ~100MB states for future exact work, tiny learning curves, compact probability grids. Current optimizer checkpoint saved each1200 updates for resume, deleted after successful fit. No original data copies or retired checkpoint accumulation.',
        'default_model_audit':'Production mine weight is a separate historical adaptation, with30epoch record; recent1200step scratch pilots cannot be called its training history. Historical late loss fluctuates and is not a field/cross-mine convergence proof.',
        'limitations':['Three selected development regions and two seeds, not all-six stable generalization.',
                       'Fixed source512-parent cap tests training budget, not full source-data sufficiency.',
                       'Source-selected checkpoint may overfit source style; target must remain evaluation only.',
                       'Nominal GSD, border/connected-component and original-parent adjacency limits inherited.',
                       'No learned selector reused on new backbone; its probability distribution and state hash differ.',
                       'Official test/mining reserved data and external APIs untouched; production promotion forbidden.'],
        'protected_assets':parent['protected_assets'],'production_changed':False,'official_test_predictions':0,'new_mine_predictions':0,'external_calls':0}
    base.write(path,protocol)
    return protocol,parent


def state_hash(model):
    h=hashlib.sha256()
    for key,value in model.state_dict().items():
        h.update(key.encode());h.update(value.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def start_continuation(optimizer,remaining=3600,lr=.0003,eta_min=.00001):
    import torch
    for group in optimizer.param_groups:group['lr']=lr;group['initial_lr']=lr
    return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,remaining,eta_min=eta_min)


def choose_source_checkpoint(curve):
    if not curve:raise ValueError('No source checkpoints')
    # Only this explicitly source-labelled scalar can select the checkpoint.
    ordered=sorted(curve,key=lambda r:r['update'])
    return max(ordered,key=lambda r:(r['source_selection']['macro_f1'],-r['update']))


def soft_loss(prob,label):
    p=np.clip(np.asarray(prob,np.float64),1e-6,1-1e-6);y=np.asarray(label,np.float64)
    bce=-(y*np.log(p)+(1-y)*np.log1p(-p)).mean()
    dice=(1-(2*(p*y).sum((1,2))+1e-6)/(p.sum((1,2))+y.sum((1,2))+1e-6)).mean()
    return float(bce+.5*dice)


def pixel_metrics(probs,labels,threshold=.5):
    counts=np.zeros(4,np.int64)
    for p,y in zip(probs,labels):
        p=np.asarray(p,np.float64)>=threshold;y=np.asarray(y)>0
        counts+=np.array([(p&y).sum(),(p&~y).sum(),(~p&y).sum(),(~p&~y).sum()],np.int64)
    tp,fp,fn,tn=map(int,counts)
    return {'tp':tp,'fp':fp,'fn':fn,'tn':tn,'f1':2*tp/max(2*tp+fp+fn,1),
            'precision':tp/max(tp+fp,1),'recall':tp/max(tp+fn,1),'background_fpr':fp/max(fp+tn,1)}


def forward(model,images,batch=96,kind=0):
    import torch
    values=[]
    with torch.inference_mode():
        for start in range(0,len(images),batch):
            a=torch.as_tensor(np.ascontiguousarray(images[start:start+batch]),device='cuda').float()
            if kind in (1,2,3):a=a.rot90(kind,(-2,-1))
            elif kind==4:a=a.flip(-1)
            prob=base.inverse_tta(model(a).sigmoid()[:,0],kind)
            if not torch.isfinite(prob).all():raise ValueError('Non-finite model output')
            values.append(prob.cpu().numpy())
    return np.concatenate(values)


def predict_tta(model,images):
    full=forward(model,images);tta=full.copy()
    for kind in (1,2,3,4):tta+=forward(model,images,kind=kind)
    return full,tta/5


def native_probabilities(grids,labels):
    return [cv2.resize(p,(len(y),len(y)),interpolation=cv2.INTER_LINEAR) for p,y in zip(grids,labels)]


def label_grid(labels):
    return np.array([cv2.resize(y.astype(np.float32),(64,64),interpolation=cv2.INTER_AREA if len(y)>64 else cv2.INTER_LINEAR)
                     for y in labels],np.float16)


def source_monitor(model,pack_x,pack_y,groups,val_data,held):
    result={};train_losses=[]
    for g in range(6):
        if g==held:continue
        ids=np.flatnonzero(groups==g)[:64]
        images=pack_x[ids,0].cpu().numpy();labels=pack_y[ids,0,0].cpu().numpy()
        train_losses.append(soft_loss(forward(model,images),labels))
        positions=split_helper.meta_split(len(val_data[g]['labels']),g)[0]
        y=[val_data[g]['labels'][i] for i in positions]
        full,tta=predict_tta(model,val_data[g]['views'][positions,0])
        result[base.REGIONS[g][0]]={**pixel_metrics(native_probabilities(tta,y),y),
            'soft_loss':soft_loss(full,label_grid(y)),'selection_positions':positions.tolist(),
            'train_monitor_loss':train_losses[-1]}
    if set(result)!={base.REGIONS[g][0] for g in range(6) if g!=held}:raise ValueError('Target entered source monitor')
    return {'regions':result,'macro_f1':float(np.mean([v['f1'] for v in result.values()])),
            'macro_soft_loss':float(np.mean([v['soft_loss'] for v in result.values()])),
            'macro_train_monitor_loss':float(np.mean(train_losses))}


def save_torch(path,value):
    import torch
    temporary=Path(str(path)+'.tmp');torch.save(value,temporary);temporary.replace(path)


def train(model_dir,x,y,groups,val_data,held,seed,protocol,parent):
    import torch
    import torch.nn.functional as F
    import segmentation_models_pytorch as smp
    fold_seed=seed+held*100;torch.manual_seed(fold_seed);torch.cuda.manual_seed_all(fold_seed)
    model=smp.UnetPlusPlus(encoder_name='resnet34',encoder_weights=None,in_channels=4,classes=1).cuda()
    optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,1200,eta_min=.00001)
    rng=np.random.default_rng(fold_seed);visits=np.zeros(6,np.int64);schedule=bytearray();losses=[];curve=[];step0=0
    phase='short';training_seconds=0.
    reference=base.read(PRIOR/f'fold_{held}_{base.REGIONS[held][0]}'/f'seed{seed}'/'result.json')
    resume=model_dir/'resume_training.pth';best_path=model_dir/'source_selected.pth'
    if resume.exists():
        cp=torch.load(resume,map_location='cpu',weights_only=True)
        if cp['protocol_sha256']!=base.sha(model_dir.parents[1]/'protocol.json'):raise ValueError('Checkpoint protocol mismatch')
        model.load_state_dict(cp['model']);phase=cp['phase']
        if phase=='continuation':scheduler=start_continuation(optimizer)
        optimizer.load_state_dict(cp['optimizer']);scheduler.load_state_dict(cp['scheduler'])
        rng.bit_generator.state=cp['rng'];visits=np.array(cp['visits'],np.int64);schedule=bytearray(cp['schedule'])
        losses=cp['losses'];curve=cp['curve'];step0=cp['step'];training_seconds=cp['training_seconds']
        torch.set_rng_state(cp['torch_rng']);torch.cuda.set_rng_state_all(cp['cuda_rng'])
        if curve and not best_path.exists():raise ValueError('Selected source state missing')
        print('RESUME',base.REGIONS[held][0],seed,'update',step0,flush=True)
    model.train();torch.cuda.synchronize();started=time.perf_counter()
    for step in range(step0,protocol['total_updates']):
        ids=base.sample_batch(groups,held,rng,4)
        # Consume identical proposed-view RNG even though only whole view is used.
        proposed=np.where(rng.random(len(ids))<.5,0,rng.integers(1,5,len(ids)))
        k,flip=int(rng.integers(0,4)),bool(rng.integers(0,2))
        schedule.extend(np.asarray([*ids,k,int(flip)],np.int64).tobytes());visits+=np.bincount(groups[ids],minlength=6)
        a=x[ids,0].float().rot90(k,(-2,-1));b=y[ids,0].float().rot90(k,(-2,-1))
        if flip:a,b=a.flip(-1),b.flip(-1)
        optimizer.zero_grad(set_to_none=True);logits=model(a);p=logits.sigmoid()
        dice=(1-(2*(p*b).sum((2,3))+1e-6)/(p.sum((2,3))+b.sum((2,3))+1e-6)).mean()
        loss=F.binary_cross_entropy_with_logits(logits,b)+.5*dice
        if not torch.isfinite(loss):raise ValueError('Non-finite loss')
        loss.backward();optimizer.step();scheduler.step();losses.append(float(loss.detach()))
        update=step+1
        if update%300==0:print('TRAIN',base.REGIONS[held][0],seed,update,'loss100',round(float(np.mean(losses[-100:])),5),flush=True)
        if update in protocol['checkpoints']:
            model.eval();digest=state_hash(model)
            if update==1200:
                if digest!=reference['coefficients']['feature_gate']['base_model_state_sha256']:raise ValueError('Exact1200 state replay failed')
                if hashlib.sha256(schedule).hexdigest()!=reference['training']['parent_schedule_sha256']:raise ValueError('Sampling prefix changed')
            monitor=source_monitor(model,x,y,groups,val_data,held)
            row={'update':update,'model_state_sha256':digest,'mean_train_loss_last100':float(np.mean(losses[-100:])),
                 'source_selection':monitor,'completed_at':base.now()}
            curve.append(row)
            best=choose_source_checkpoint(curve)
            if best['update']==update:save_torch(best_path,{k:v.detach().cpu() for k,v in model.state_dict().items()})
            base.write(model_dir/'source_curve.json',curve)
            print('SOURCE',base.REGIONS[held][0],seed,update,'macroF1',round(monitor['macro_f1'],5),
                  'train_monitor_loss',round(monitor['macro_train_monitor_loss'],5),'source_loss',round(monitor['macro_soft_loss'],5),flush=True)
            if update==1200:
                scheduler=start_continuation(optimizer,3600,protocol['continuation_lr'],protocol['continuation_eta_min']);phase='continuation'
            torch.cuda.synchronize();training_seconds+=time.perf_counter()-started
            save_torch(resume,{'protocol_sha256':base.sha(model_dir.parents[1]/'protocol.json'),
                'model':model.state_dict(),'optimizer':optimizer.state_dict(),'scheduler':scheduler.state_dict(),'phase':phase,
                'rng':rng.bit_generator.state,'visits':visits.tolist(),'schedule':bytes(schedule),'losses':losses,'curve':curve,'step':update,
                'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all(),'training_seconds':training_seconds})
            model.train();started=time.perf_counter()
    if visits[held]!=0 or visits.tolist()!=[0 if g==held else 19200 for g in range(6)]:raise ValueError('Training source exclusion failed')
    return model.eval(),curve,{'updates':4800,'visits_per_region':visits.tolist(),'sampling_sha256':hashlib.sha256(schedule).hexdigest(),
            'seconds_including_source_monitor':training_seconds,'first1200_exact_replay':True,'optimizer_moments_preserved_at_restart':True}


def evaluate(model,val_data,held):
    probabilities={};selected={};grids={}
    for g in range(6):
        positions=np.arange(len(val_data[g]['labels'])) if g==held else split_helper.meta_split(len(val_data[g]['labels']),g)[1]
        labels=[val_data[g]['labels'][i] for i in positions]
        _,tta=predict_tta(model,val_data[g]['views'][positions,0]);grids[g]=tta
        probabilities[g]=native_probabilities(tta,labels)
        selected[g]={'labels':labels,'components':[val_data[g]['components'][i] for i in positions]}
    threshold,calibration=base.source_threshold(probabilities,selected,[g for g in range(6) if g!=held],held)
    d=selected[held]
    rows=[base.parent_counts(p,y,c,threshold) for p,y,c in zip(probabilities[held],d['labels'],d['components'])]
    fixed=[base.parent_counts(p,y,c,.5) for p,y,c in zip(probabilities[held],d['labels'],d['components'])]
    return {'threshold':threshold,'calibration':calibration,'held_out':base.aggregate_counts(rows),
            'held_out_at_fixed_05':base.aggregate_counts(fixed),'per_parent_counts':rows,
            'calibration_parent_count_by_region':{base.REGIONS[g][0]:len(selected[g]['labels']) for g in range(6) if g!=held}},grids


def summarize(out,protocol):
    rows=[base.read(p) for p in sorted(Path(out).glob('fold_*/seed*/result.json'))]
    if len(rows)!=6:raise ValueError('Six diagnostic fits required')
    methods=('short1200','selected_source','final4800');metrics=('f1','small_cover50','small_iou50','background_fpr')
    macro={name:{k:float(np.mean([r['methods'][name]['held_out'][k] for r in rows])) for k in metrics} for name in methods}
    source=[]
    for r in rows:
        first,last=r['source_curve'][0],r['source_curve'][-1]
        source.append({'region':r['held_out'],'seed':r['seed'],'selected_update':r['selection']['selected_update'],
            'source_selection_f1_short':first['source_selection']['macro_f1'],'source_selection_f1_final':last['source_selection']['macro_f1'],
            'train_monitor_loss_short':first['source_selection']['macro_train_monitor_loss'],
            'train_monitor_loss_final':last['source_selection']['macro_train_monitor_loss'],
            'source_selection_loss_short':first['source_selection']['macro_soft_loss'],
            'source_selection_loss_final':last['source_selection']['macro_soft_loss'],
            'target_selected_delta':{k:r['methods']['selected_source']['held_out'][k]-r['methods']['short1200']['held_out'][k] for k in metrics},
            'target_final_delta':{k:r['methods']['final4800']['held_out'][k]-r['methods']['short1200']['held_out'][k] for k in metrics}})
    for path,digest in protocol['frozen_assets'].items():
        if base.sha(path)!=digest:raise ValueError('Frozen dependency changed')
    result={'completed_at':base.now(),'status':'completed','fits':6,'total_updates':28800,
        'checkpoint_evaluations':24,'source_and_target_diagnostics':source,'target_macro':macro,
        'scope':'Hokkaido/Wenchuan/Jiuzhaigou x2seeds, chosen development diagnostic cohort. Not full six-region or blind mine acceptance.',
        'continuation_includes_lr_restart':True,'production_changed':False,'official_test_predictions':0,'new_mine_predictions':0,'external_calls':0}
    base.write(Path(out)/'results.json',result);base.write(Path(out)/'state.json',{k:v for k,v in result.items() if k not in ('source_and_target_diagnostics','target_macro')})
    return result


def run(out):
    import torch
    protocol,parent=register(out);base_protocol=base.read(parent['parent_protocol'])
    torch.set_num_threads(4);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    if not torch.cuda.is_available():raise RuntimeError('CUDA unavailable')
    x,y,groups,data,manifest=base.prepare_data(out,base_protocol)
    old_manifest=base.read(PRIOR/'data_manifest.json')
    if {k:v for k,v in manifest.items() if k!='registered_at'}!={k:v for k,v in old_manifest.items() if k!='registered_at'}:raise ValueError('Frozen raw rows changed')
    x,y=torch.from_numpy(x).cuda(),torch.from_numpy(y).cuda();completed=0
    base.write(Path(out)/'runtime.json',{'torch':torch.__version__,'cuda':torch.version.cuda,'device':torch.cuda.get_device_name(0),
        'threads':4,'deterministic_cudnn':True,'tf32':False,'cache_drive':'D:'})
    for held in protocol['held_indices']:
        region=base.REGIONS[held][0]
        for seed in protocol['seeds']:
            directory=Path(out)/f'fold_{held}_{region}'/f'seed{seed}';directory.mkdir(parents=True,exist_ok=True)
            destination=directory/'result.json'
            if destination.exists():
                prior=base.read(destination)
                if base.sha(directory/'source_selected.pth')!=prior['selected_weights_sha256']:raise ValueError('Completed selected state changed')
                completed+=1;continue
            base.write(Path(out)/'state.json',{'status':'running','updated_at':base.now(),'held_out':region,'seed':seed,'completed':completed,'total':6,'production_changed':False})
            model,curve,training=train(directory,x,y,groups,data,held,seed,protocol,parent)
            selected=choose_source_checkpoint(curve)
            selection={'selected_at':base.now(),'selected_update':selected['update'],'source_macro_f1':selected['source_selection']['macro_f1'],
                'selection_regions':[base.REGIONS[g][0] for g in range(6) if g!=held],
                'target_used':False,'calibration_half_used_for_selection':False,
                'selected_state_sha256':selected['model_state_sha256'],'weights_sha256':base.sha(directory/'source_selected.pth')}
            base.write(directory/'checkpoint_selection.json',selection)
            # Target is first inferred only after all source-only choices are persisted.
            final,final_grids=evaluate(model,data,held)
            if selected['update']==4800:
                chosen=copy.deepcopy(final);chosen_grids=final_grids
            elif selected['update']==1200:
                old=base.read(PRIOR/f'fold_{held}_{region}'/f'seed{seed}'/'result.json')
                chosen=copy.deepcopy(old['methods']['tta5']);chosen_grids=None
            else:
                model.load_state_dict(torch.load(directory/'source_selected.pth',map_location='cpu',weights_only=True));model.eval()
                if state_hash(model)!=selected['model_state_sha256']:raise ValueError('Selected state identity mismatch')
                chosen,chosen_grids=evaluate(model,data,held)
            old=base.read(PRIOR/f'fold_{held}_{region}'/f'seed{seed}'/'result.json')
            cache={'final_target':final_grids[held],'original_target_indices':np.array(data[held]['indices'],np.int64)}
            if chosen_grids is not None:
                if selected['update']!=4800:cache['selected_target']=chosen_grids[held]
                for g in range(6):
                    if g!=held:cache[f'selected_source_calibration_{g}']=chosen_grids[g]
            np.savez_compressed(directory/'probability_grids.npz',**cache)
            base.write(destination,{'held_index':held,'held_out':region,'seed':seed,'completed_at':base.now(),'training':training,
                'selection':selection,'source_curve':curve,'methods':{'short1200':old['methods']['tta5'],'selected_source':chosen,'final4800':final},
                'selected_weights_sha256':base.sha(directory/'source_selected.pth'),
                'probability_grids_sha256':base.sha(directory/'probability_grids.npz'),'target_evaluation_after_source_selection':True})
            (directory/'resume_training.pth').unlink(missing_ok=True)
            del model;torch.cuda.empty_cache();completed+=1
            print('SUFFICIENCY_COMPLETED',completed,'/6',region,seed,'source_selected',selected['update'],flush=True)
    result=summarize(out,protocol)
    print('COMPLETE',str({'fits':6,'target_macro':result['target_macro']}),flush=True)


if __name__=='__main__':
    sys.stdout.reconfigure(encoding='utf-8');p=argparse.ArgumentParser();p.add_argument('--output',required=True)
    p.add_argument('--stage',choices=('register','run','summarize'),default='run');a=p.parse_args()
    if a.stage=='register':register(a.output)
    elif a.stage=='summarize':summarize(a.output,register(a.output)[0])
    else:run(a.output)
