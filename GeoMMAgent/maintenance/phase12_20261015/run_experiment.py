"""Frozen paired pilot: reconstruction vs task supervision vs privileged distillation."""
import argparse
import json
import hashlib
import sys
import time
from pathlib import Path
from datetime import datetime, timezone, timedelta
from task_rgb import RGBTaskAdapter, task_loss, torch, F
import numpy as np

ROOT=Path('D:/LMHLD')
HERE=Path(__file__).resolve().parent
OUT=ROOT/'experiments/task_rgb_adapter_20261015'
BASE=ROOT/'models_prior/real_ft_T1b_ours_seed42/best_model.pth'
sys.path.insert(0,str(ROOT))
from detector import LandslideDetector
from data_paths import resolve_data_root


def now():return datetime.now(timezone(timedelta(hours=8))).isoformat(timespec='seconds')
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def write(p,value):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def array_sha(x):return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()
def normalize(x):
    x=np.asarray(x,np.float32)
    lo=x.min((2,3),keepdims=True);hi=x.max((2,3),keepdims=True)
    return (x-lo)/(hi-lo+1e-6)*255


def prepare():
    OUT.mkdir(parents=True,exist_ok=True)
    protocol_path=OUT/'protocol.json'
    pack_path=OUT/'source_pack.npz'
    if protocol_path.exists():
        protocol=read(protocol_path)
        if not pack_path.exists() or sha(pack_path)!=protocol['pack_sha256']:
            raise ValueError('Frozen experiment pack missing or changed')
        return protocol
    regions=[('native','Emilia-Romagna_Italy_32'),('native','Gorkha_Nepal_32'),
             ('tiled','Hokkaido_Japan_32'),('tiled','Jiuzhaigou_China_32'),
             ('tiled','Wenchuan_China_32'),('tiled','Taiwan_China_32')]
    configuration={'registered_at':now(),'client_work_date':'2026-10-15',
        'hypothesis':'Optimize RGB-only latent for segmentation, rather than infer that lower NIR MSE gives better detection',
        'arms':['reconstruction','task_supervised','task_distilled'],'seeds':[42,43],
        'updates':400,'batch_size':48,'optimizer':{'name':'AdamW','lr':.001,'weight_decay':.0001},
        'schedule':'cosine to 0; fixed final update; no epoch/checkpoint selection',
        'detector_frozen':True,'detector_batchnorm_frozen':True,
        'normalization':'exact deployed per-patch per-channel minmax 0..255; adapter BGR/255',
        'architecture':'same 3x3,3x3,dilated3x3 width16 residual fourth-channel adapter all arms; max residual .5; exact red-copy initialization',
        'regularization':'.01 * latent-vs-red MSE in every arm',
        'task_objective':'BCE + .5 per-image soft Dice',
        'distillation_objective':'.25*T^2 Bernoulli KL, T2, only true-NIR teacher correct and confident >=.8/<=.2 on source train labels',
        'reconstruction_objective':'MSE against normalized real NIR, same input rows/augmentations/updates',
        'batch_rule':'8 patches per region, identical across arms for each seed; paired quarter rotations and flips',
        'train_selection':'512 positive-containing and 512 all-background train patches per region; capped to available without replacement',
        'validation_selection':'fixed random256 original val rows per region; no test files; previously seen source validation',
        'mine_regression_cases':['Amyntaio','LasCruces','Hpakant','CN_XJ2023'],
        'mine_inputs_in_adapter_training':False,'mine_results_may_select_checkpoint_or_threshold':False,
        'thresholds':[.4,.5],'primary_threshold':.4,
        'selection_rule':'task-distilled family only; pass both seeds macro source-val F1 gain >=.01 vs red-copy and every region drop <=.02; additionally every seen-mine recall drop <=.02 and weak-normal activation rise <=.02; if fail no deployment',
        'retained_component_rule':'at most one family seed42 adapter if eligible, otherwise delete all pilot weights',
        'new_unopened_mine_predictions':0,'new_external_generation_calls':0,
        'new_Noto_source_training_or_predictions':0,'default_weight_change_allowed':False,
        'model_claim':'latent fourth-channel proxy, not recovered/measured NIR; no physical vegetation/safety inference',
        'references':['https://openaccess.thecvf.com/content_cvpr_2016/html/Hoffman_Learning_With_Side_CVPR_2016_paper.html',
                      'https://arxiv.org/abs/1511.03643'],
        'novelty_claim':'application hypothesis and controlled ablation only; modality distillation is established',
        'base_model':{'path':str(BASE),'sha256':sha(BASE)},
        'protected_registries':[{'path':str(p),'sha256':sha(p)} for p in [ROOT/'model_registry.json',ROOT/'datasets/input_reserve_registry.json',ROOT/'datasets/mine_benchmark_20261002/mine_case_registry.json']],
        'model_fits_planned':6,'fit_compute_limits':'same updates/examples; task arms require frozen-detector backprop and cost more FLOPs; timings reported'}
    write(HERE/'plan.json',configuration) # Hypothesis and budget recorded before data/model outputs.
    xs,ys,vs,vy,vg,provenance=[],[],[],[],[],[]
    for region_index,(kind,region) in enumerate(regions):
        folder=Path(resolve_data_root(kind))/region
        record={'region':region,'kind':kind,'splits':{}}
        for split in ['train','val']:
            xp,yp=folder/(split+'_images.npy'),folder/(split+'_labels.npy')
            x,y=np.load(xp,mmap_mode='r'),np.load(yp,mmap_mode='r')
            if x.shape[1:]!=(4,32,32):raise ValueError('Unexpected source input layout')
            rng=np.random.default_rng(20261015+region_index*11+(split=='val'))
            yy=np.asarray(y).reshape(len(y),-1)
            if not set(np.unique(yy)).issubset({0.,1.}):raise ValueError('Unexpected source label schema')
            if split=='train':
                positive=np.flatnonzero(np.any(yy>0,axis=1));negative=np.flatnonzero(~np.any(yy>0,axis=1))
                ids=np.concatenate([rng.choice(pool,min(len(pool),512),replace=False) for pool in [positive,negative]])
            else:ids=rng.choice(len(y),min(len(y),256),replace=False)
            raw=np.array(x[ids],dtype=np.float32);labels=np.array(y[ids],dtype=np.float32).reshape(-1,1,32,32)
            if not np.isfinite(raw).all():raise ValueError('Nonfinite source pixels')
            norm=normalize(raw)
            record['splits'][split]={'indices':ids.tolist(),'input_path':str(xp),'label_path':str(yp),
                'input_shape':list(x.shape),'selected_raw_sha256':array_sha(raw),'selected_labels_sha256':array_sha(labels),
                'unique_parent_samples':len(set(ids.tolist())),'test_files_read':0}
            if split=='train':xs.append(norm);ys.append(labels)
            else:vs.append(norm);vy.append(labels);vg.extend([region_index]*len(ids))
        provenance.append(record)
        print(region,'train',len(xs[-1]),'val',len(vs[-1]),flush=True)
    np.savez_compressed(pack_path,train_x=np.concatenate(xs),train_y=np.concatenate(ys),
         train_group=np.concatenate([np.full(len(x),i,np.int64) for i,x in enumerate(xs)]),
         val_x=np.concatenate(vs),val_y=np.concatenate(vy),val_group=np.array(vg))
    protocol={**configuration,'provenance':provenance,'pack_path':str(pack_path),'pack_sha256':sha(pack_path),
              'train_patches':sum(map(len,xs)),'validation_patches':sum(map(len,vs)),
              'code_sha256':{p.name:sha(p) for p in [HERE/'task_rgb.py',Path(__file__)]}}
    write(protocol_path,protocol)
    write(HERE/'state.json',{'status':'algorithm_ablation_registered','protocol':str(protocol_path),
          'new_completed_fits':0,'default_weight_changed':False,'new_blind_predictions':0})
    return protocol


def model():
    detector=LandslideDetector(str(BASE),threshold=.4)
    trunk=detector.model.eval()
    trunk.requires_grad_(False)
    return detector,trunk


def metrics(prob,labels,threshold):
    p,t=prob>=threshold,labels==1
    tp,fp,fn,tn=[int(a.sum()) for a in [p&t,p&~t,~p&t,~p&~t]]
    return {'tp':tp,'fp':fp,'fn':fn,'tn':tn,'f1':2*tp/max(2*tp+fp+fn,1),
            'recall':tp/max(tp+fn,1),'precision':tp/max(tp+fp,1),'background_activation':fp/max(fp+tn,1)}


def evaluate(trunk,pack,adapter=None,real_nir=False):
    probabilities=[];mse=0
    with torch.no_grad():
        for i in range(0,len(pack['val_x']),96):
            x=torch.as_tensor(pack['val_x'][i:i+96],device='cuda')
            current=x.clone()
            if not real_nir:
                latent=x[:,2:3]/255 if adapter is None else adapter(x[:,:3]/255)
                mse+=float(F.mse_loss(latent,x[:,3:4]/255,reduction='sum'))
                current[:,3:4]=latent*255
            probabilities.append(torch.sigmoid(trunk(current))[:,0].cpu().numpy())
    probs=np.concatenate(probabilities);labels=pack['val_y'][:,0];groups=pack['val_group']
    return {'thresholds':{str(t):{'micro':metrics(probs,labels,t),
              'regions':[metrics(probs[groups==g],labels[groups==g],t) for g in range(6)],
              'macro_f1':float(np.mean([metrics(probs[groups==g],labels[groups==g],t)['f1'] for g in range(6)]))}
              for t in [.4,.5]},'latent_vs_true_nir_mse':mse/labels.size if not real_nir else 0}


def run():
    protocol=prepare()
    for row in [protocol['base_model'],*protocol['protected_registries']]:
        if sha(row['path'])!=row['sha256']:raise ValueError('Frozen default or registry changed')
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    pack=np.load(protocol['pack_path'])
    detector,trunk=model()
    baseline_path=OUT/'source_baselines.json'
    if baseline_path.exists():baseline=read(baseline_path)
    else:
        baseline={'red_copy':evaluate(trunk,pack),'real_nir_context':evaluate(trunk,pack,real_nir=True)}
        write(baseline_path,baseline)
    print('BASELINE source RGB macroF1',baseline['red_copy']['thresholds']['0.4']['macro_f1'],flush=True)
    x=torch.as_tensor(pack['train_x'],device='cuda')
    y=torch.as_tensor(pack['train_y'],device='cuda')
    teacher_path=OUT/'train_teacher_logits.npy'
    if teacher_path.exists():teacher=np.load(teacher_path)
    else:
        with torch.no_grad():teacher=np.concatenate([trunk(x[i:i+96]).cpu().numpy() for i in range(0,len(x),96)])
        np.save(teacher_path,teacher)
    teacher=torch.as_tensor(teacher,device='cuda')
    pools=[np.flatnonzero(pack['train_group']==g) for g in range(6)]
    completed=[]
    for seed in protocol['seeds']:
        for arm in protocol['arms']:
            directory=OUT/f'{arm}_seed{seed}';directory.mkdir(exist_ok=True)
            result_path=directory/'result.json';weights=directory/'adapter.pth'
            if result_path.exists():
                result=read(result_path)
                if sha(weights)!=result['weights_sha256']:raise ValueError('Finished adapter changed')
                completed.append(result);continue
            torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
            adapter=RGBTaskAdapter().cuda().train()
            optimizer=torch.optim.AdamW(adapter.parameters(),lr=.001,weight_decay=.0001)
            scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=protocol['updates'])
            rng=np.random.default_rng(seed);history=[];begin=time.monotonic()
            for step in range(protocol['updates']):
                ids=np.concatenate([rng.choice(pool,8,replace=False) for pool in pools])
                xx,yy,tt=x[ids].clone(),y[ids].clone(),teacher[ids].clone()
                k=int(rng.integers(0,4))
                if k:xx,yy,tt=[torch.rot90(a,k,(-2,-1)) for a in [xx,yy,tt]]
                if rng.random()<.5:xx,yy,tt=[a.flip(-1) for a in [xx,yy,tt]]
                if rng.random()<.5:xx,yy,tt=[a.flip(-2) for a in [xx,yy,tt]]
                # Teacher is not spatially equivariant. Recompute on augmented
                # true-NIR input in both task arms; never rotate teacher predictions.
                if arm!='reconstruction':
                    with torch.no_grad():tt=trunk(xx)
                optimizer.zero_grad(set_to_none=True)
                latent=adapter(xx[:,:3]/255)
                regularizer=.01*F.mse_loss(latent,xx[:,2:3]/255)
                if arm=='reconstruction':
                    loss=F.mse_loss(latent,xx[:,3:4]/255)+regularizer
                    distill=loss.new_zeros(())
                else:
                    mixed=torch.cat([xx[:,:3],latent*255],dim=1)
                    logits=trunk(mixed)
                    loss,distill=task_loss(logits,yy,tt if arm=='task_distilled' else None)
                    loss=loss+regularizer
                if not torch.isfinite(loss):raise ValueError('Nonfinite training loss')
                loss.backward()
                if not all(p.grad is None or torch.isfinite(p.grad).all() for p in adapter.parameters()):
                    raise ValueError('Nonfinite adapter gradient')
                torch.nn.utils.clip_grad_norm_(adapter.parameters(),1)
                optimizer.step();scheduler.step()
                if step%100==0 or step+1==protocol['updates']:
                    row={'update':step+1,'loss':float(loss),'distill_kl':float(distill),'seconds':time.monotonic()-begin}
                    history.append(row);print(arm,seed,row,flush=True)
            adapter.eval();torch.save(adapter.cpu().state_dict(),weights);adapter.cuda()
            evaluation=evaluate(trunk,pack,adapter)
            result={'arm':arm,'seed':seed,'actual_optimizer_updates':protocol['updates'],
                    'parameters':sum(p.numel() for p in adapter.parameters()),'seconds':time.monotonic()-begin,
                    'weights':str(weights),'weights_sha256':sha(weights),'protocol_sha256':sha(OUT/'protocol.json'),
                    'history':history,'source_validation':evaluation,'frozen_detector_has_gradients':any(p.grad is not None for p in trunk.parameters()),
                    'mine_samples_trained':0,'independent_test_claim':False}
            write(result_path,result);completed.append(result)
            print('FINISHED',arm,seed,'source macroF1',evaluation['thresholds']['0.4']['macro_f1'],flush=True)
            write(HERE/'state.json',{'status':'algorithm_ablation_training','protocol':str(OUT/'protocol.json'),
                  'new_completed_fits':len(completed),'default_weight_changed':False,'new_blind_predictions':0})
            del adapter;torch.cuda.empty_cache()
    write(OUT/'training_results.json',{'completed_at':now(),'fits':completed,'new_completed_fits':len(completed),
          'total_optimizer_updates':sum(r['actual_optimizer_updates'] for r in completed),
          'source_baselines':baseline,'mine_evaluation_pending':True})


if __name__=='__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=['prepare','run'])
    args=parser.parse_args()
    if args.command=='prepare':prepare()
    else:run()
