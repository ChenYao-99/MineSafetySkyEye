"""Structural follow-up after latent-band regression failed; fixed budget, no new data."""
import sys
import time
from pathlib import Path
from run_experiment import ROOT,HERE,OUT,BASE,read,write,sha,now,metrics,torch,F
from task_rgb import task_loss
from rgb_feature_residual import RGBFeatureResidual,FrozenRGBFeatureModel
from detector import LandslideDetector
import numpy as np
import rasterio
sys.path.insert(0,str(ROOT/'validation'))
from mine_data_protocol import assert_development_cases

FOLLOW=OUT/'feature_residual'


def source_validation(model,pack):
    probabilities=[]
    with torch.no_grad():
        for start in range(0,len(pack['val_x']),96):
            x=torch.as_tensor(pack['val_x'][start:start+96],device='cuda')
            probabilities.append(torch.sigmoid(model(x))[:,0].cpu().numpy())
    probs=np.concatenate(probabilities);labels=pack['val_y'][:,0];group=pack['val_group']
    result={}
    for t in [.4,.5]:
        regions=[metrics(probs[group==g],labels[group==g],t) for g in range(6)]
        result[str(t)]={'micro':metrics(probs,labels,t),'regions':regions,'macro_f1':float(np.mean([r['f1'] for r in regions]))}
    return {'thresholds':result}


def mine_validation(detector):
    plan=read(OUT/'mine_evaluation_plan.json')
    result=[]
    assert_development_cases([row['mine'] for row in plan['cases']])
    for row in plan['cases']:
        for key in ['image','label','normal']:
            if sha(row[key+'_path'])!=row[key+'_sha256']:raise ValueError('Frozen mine regression source changed')
        with rasterio.open(row['image_path']) as src:image=src.read().transpose(1,2,0).astype(np.float32)
        with rasterio.open(row['label_path']) as src:labels=src.read(1)
        with rasterio.open(row['normal_path']) as src:normal=src.read(1)>0
        image[:,:,3]=image[:,:,2]
        probability=detector.detect_image(image,patch_norm='minmax',max_batch=128)
        values={}
        for threshold in [.4,.5]:
            hit=probability>=threshold;pos,bg=labels==1,labels==0
            values[str(threshold)]={'known_positive_recall':float(hit[pos].mean()),
                'old_weak_background_activation':float(hit[bg].mean()),
                'reviewed_normal_activation':float(hit[normal].mean()),'unknown_pixels_scored':0}
        result.append({'mine':row['mine'],'thresholds':values})
    return result


def main():
    FOLLOW.mkdir(exist_ok=True)
    parent=read(OUT/'protocol.json')
    path=FOLLOW/'protocol.json'
    if not path.exists():
        write(path,{'registered_at':now(),'parent_protocol_sha256':sha(OUT/'protocol.json'),
            'previous_mine_results_observed':True,'development_hypothesis_selection_disclosed':True,
            'hypothesis':'A single proxy channel is a bottleneck; train RGB multiscale feature residual directly',
            'architecture':'8ch projections from 1/2,1/4,1/8,1/16 encoder plus full-size decoder; 16ch residual head; frozen detector and BN',
            'arms':['feature_supervised','feature_distilled'],'seeds':[42,43],'updates':400,
            'same_parent_training_pack_sha256':parent['pack_sha256'],'same_source_batch_augmentations_optimizer_and_task_losses':True,
            'residual_regularizer':'.001 * squared residual logit mean',
            'checkpoint_selection':'fixed last update only; no search',
            'selection_gate':'both distilled seeds source macroF1 >= red-copy+.01; each source region drop <=.02; each seen-mine positive recall drop <=.02; both normal activation rises <=.02 vs red-copy; no field/independent claim',
            'thresholds':[.4,.5],'primary_threshold':.4,'new_fits_planned':4,
            'mine_training_inputs':0,'new_unopened_test_predictions':0,'Noto_training_or_predictions':0,
            'production_promotion_allowed':False,'new_external_generation_calls':0,
            'code_sha256':{p.name:sha(p) for p in [Path(__file__),HERE/'rgb_feature_residual.py']}})
    protocol=read(path)
    if sha(OUT/'source_pack.npz')!=protocol['same_parent_training_pack_sha256']:raise ValueError('Source pack changed')
    for row in [parent['base_model'],*parent['protected_registries']]:
        if sha(row['path'])!=row['sha256']:raise ValueError('Frozen runtime changed')
    torch.set_num_threads(4);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    pack=np.load(OUT/'source_pack.npz')
    x=torch.as_tensor(pack['train_x'],device='cuda');y=torch.as_tensor(pack['train_y'],device='cuda')
    pools=[np.flatnonzero(pack['train_group']==g) for g in range(6)]
    detector=LandslideDetector(str(BASE),threshold=.4)
    trunk=detector.model.eval().requires_grad_(False)
    finished=[]
    for seed in protocol['seeds']:
        for arm in protocol['arms']:
            root=FOLLOW/f'{arm}_seed{seed}';root.mkdir(exist_ok=True)
            result_path=root/'result.json';weights=root/'head.pth'
            if result_path.exists():
                record=read(result_path)
                if sha(weights)!=record['weights_sha256']:raise ValueError('Completed residual head changed')
                finished.append(record);continue
            torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
            head=RGBFeatureResidual().cuda().train()
            wrapped=FrozenRGBFeatureModel(trunk,head)
            optimizer=torch.optim.AdamW(head.parameters(),lr=.001,weight_decay=.0001)
            scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=400)
            rng=np.random.default_rng(seed);history=[];begin=time.monotonic()
            for step in range(400):
                ids=np.concatenate([rng.choice(pool,8,replace=False) for pool in pools])
                xx,yy=x[ids].clone(),y[ids].clone()
                k=int(rng.integers(0,4))
                if k:xx,yy=[torch.rot90(a,k,(-2,-1)) for a in [xx,yy]]
                if rng.random()<.5:xx,yy=[a.flip(-1) for a in [xx,yy]]
                if rng.random()<.5:xx,yy=[a.flip(-2) for a in [xx,yy]]
                optimizer.zero_grad(set_to_none=True)
                with torch.no_grad():teacher=trunk(xx) if arm=='feature_distilled' else None
                logits,residual=wrapped.components(xx)
                loss,kl=task_loss(logits,yy,teacher)
                loss=loss+.001*residual.square().mean()
                if not torch.isfinite(loss):raise ValueError('Nonfinite feature residual loss')
                loss.backward()
                if not all(p.grad is None or torch.isfinite(p.grad).all() for p in head.parameters()):raise ValueError('Nonfinite head gradient')
                torch.nn.utils.clip_grad_norm_(head.parameters(),1)
                optimizer.step();scheduler.step()
                if step%100==0 or step==399:
                    record={'step':step+1,'loss':float(loss),'seconds':time.monotonic()-begin}
                    history.append(record);print(arm,seed,record,flush=True)
            head.eval();torch.save(head.cpu().state_dict(),weights);head.cuda()
            source=source_validation(wrapped,pack)
            detector.model=wrapped
            mines=mine_validation(detector)
            detector.model=trunk
            record={'arm':arm,'seed':seed,'updates':400,'parameters':sum(p.numel() for p in head.parameters()),
                  'weights':str(weights),'weights_sha256':sha(weights),'protocol_sha256':sha(path),'history':history,
                  'source_validation':source,'mines':mines,'seconds':time.monotonic()-begin,
                  'trunk_has_gradients':any(p.grad is not None for p in trunk.parameters()),'mine_inputs_trained':0}
            write(result_path,record);finished.append(record)
            write(FOLLOW/'results_partial.json',{'fits':finished})
            print('FINISHED',arm,seed,'source macroF1',source['thresholds']['0.4']['macro_f1'],
                  'mine recalls',[r['thresholds']['0.4']['known_positive_recall'] for r in mines],
                  'mine normal',[r['thresholds']['0.4']['reviewed_normal_activation'] for r in mines],flush=True)
            del wrapped,head;torch.cuda.empty_cache()
    write(FOLLOW/'results.json',{'completed_at':now(),'fits':finished,'new_fits':4,'optimizer_updates':1600,
          'no_unopened_tests_or_Noto_inputs':True,'default_weight_changed':False})


if __name__=='__main__':
    sys.stdout.reconfigure(encoding='utf-8');main()
