"""Start from the existing stronger RGB completion component; bounded residual only."""
import sys
import time
from pathlib import Path
from run_experiment import ROOT,HERE,OUT,BASE,read,write,sha,now,metrics,torch,F
from task_rgb import task_loss
from detector import LandslideDetector
from nir_synth_utils import load_synth
from run_feature_residual import mine_validation
import numpy as np

FOLLOW=OUT/'legacy_task_refinement'


class LegacyTaskAdapter(torch.nn.Module):
    def __init__(self,legacy):
        super().__init__()
        self.legacy=legacy.eval().requires_grad_(False)
        self.residual=torch.nn.Sequential(torch.nn.Conv2d(4,16,3,padding=1),torch.nn.GELU(),
             torch.nn.Conv2d(16,16,3,padding=1),torch.nn.GELU(),
             torch.nn.Conv2d(16,16,3,padding=2,dilation=2),torch.nn.GELU(),torch.nn.Conv2d(16,1,1))
        torch.nn.init.zeros_(self.residual[-1].weight);torch.nn.init.zeros_(self.residual[-1].bias)

    def train(self,mode=True):
        super().train(mode);self.legacy.eval();return self

    def components(self,bgr):
        if bgr.ndim!=4 or bgr.shape[1]!=3:raise ValueError('RGB adapter cannot consume measured NIR')
        with torch.no_grad():base=self.legacy(bgr).clamp(0,1)
        delta=.25*torch.tanh(self.residual(torch.cat([bgr,base],1)))
        return (base+delta).clamp(0,1),delta

    def forward(self,bgr):return self.components(bgr)[0]


def source_probabilities(trunk,pack,adapter=None):
    outputs=[]
    with torch.no_grad():
        for start in range(0,len(pack['val_x']),96):
            x=torch.as_tensor(pack['val_x'][start:start+96],device='cuda')
            x[:,3:4]=x[:,2:3] if adapter is None else adapter(x[:,:3]/255)*255
            outputs.append(torch.sigmoid(trunk(x))[:,0].cpu().numpy())
    return np.concatenate(outputs)


def source_metrics(prob,pack):
    labels=pack['val_y'][:,0];groups=pack['val_group'];result={}
    for t in [.4,.5]:
        regions=[metrics(prob[groups==g],labels[groups==g],t) for g in range(6)]
        result[str(t)]={'micro':metrics(prob,labels,t),'regions':regions,
                        'macro_f1':float(np.mean([r['f1'] for r in regions]))}
    return {'thresholds':result}


class SynthDetector:
    def __init__(self,detector,adapter,blend=1):self.detector,self.adapter,self.blend=detector,adapter,blend
    def detect_image(self,image,**kwargs):
        refined=self.detector.detect_image(image,nir_synth_model=self.adapter,**kwargs)
        if self.blend==1:return refined
        current=self.detector.detect_image(image,**kwargs)
        return (1-self.blend)*current+self.blend*refined


def main():
    FOLLOW.mkdir(exist_ok=True)
    parent=read(OUT/'protocol.json');path=FOLLOW/'protocol.json'
    legacy_path=ROOT/'models_prior/nir_synth_seed42/best_model.pth'
    if not path.exists():
        write(path,{'registered_at':now(),'parent_protocol_sha256':sha(OUT/'protocol.json'),
          'hypothesis':'Task correction must beat the stronger existing MSE synthesizer, not just copy-red',
          'previous_development_outputs_observed':True,'new_independent_evaluation_claim':False,
          'initial_component':{'path':str(legacy_path),'sha256':sha(legacy_path)},
          'architecture':'frozen legacy synthesizer + zero-initialized width16 residual, max .25 in normalized domain',
          'arms':['legacy_mse_refine','legacy_task_refine','legacy_task_distilled'],'seeds':[42,43],
          'updates':400,'optimizer':'AdamW lr.001 weight_decay.0001; cosine; final update only',
          'data_batches_and_augmentation':'exact parent source indices, 8/region, same seeds, no mine training',
          'regularizer':'.05 * bounded residual squared mean',
          'task_loss':'same BCE+.5 Dice, distilled arm adds same reliable-teacher KL',
          'pack_sha256':parent['pack_sha256'],
          'source_success':'distilled both seeds >= legacy macroF1+.01; each source region drop <=.02 at.4',
          'mine_success':'each mine recall drop <=.02 and both weak normal activation rise <=.02 vs original copy-red at.4',
          'fixed_blend_diagnostic':'.85 existing RGB + .15 refined RGB, inherited blend weight; also report same blend with legacy; no blend or threshold search',
          'retention':'keep seed42 distilled residual if both source gates pass, even if mine gates fail; offline research only, no production promotion; discard other checkpoint payloads',
          'default_weight_change_allowed':False,'new_unopened_mine_predictions':0,
          'new_Noto_training_or_predictions':0,'new_external_generation_calls':0,
          'fixed_primary_threshold':.4,'secondary_threshold':.5,'maximum_new_fits':6,
          'code_sha256':{Path(__file__).name:sha(__file__)}})
    protocol=read(path)
    if sha(OUT/'source_pack.npz')!=protocol['pack_sha256']:raise ValueError('Source pack changed')
    for row in [parent['base_model'],protocol['initial_component'],*parent['protected_registries']]:
        if sha(row['path'])!=row['sha256']:raise ValueError('Frozen component changed')
    torch.set_num_threads(4);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    pack=np.load(OUT/'source_pack.npz');x=torch.as_tensor(pack['train_x'],device='cuda');y=torch.as_tensor(pack['train_y'],device='cuda')
    pools=[np.flatnonzero(pack['train_group']==g) for g in range(6)]
    detector=LandslideDetector(str(BASE),threshold=.4);trunk=detector.model.eval().requires_grad_(False)
    legacy=load_synth(dev='cuda').eval().requires_grad_(False)
    current_prob=source_probabilities(trunk,pack)
    legacy_prob=source_probabilities(trunk,pack,legacy)
    write(FOLLOW/'baselines.json',{'red_copy':source_metrics(current_prob,pack),
         'legacy_full':source_metrics(legacy_prob,pack),'legacy_blend15':source_metrics(.85*current_prob+.15*legacy_prob,pack),
         'legacy_blend15_mines':mine_validation(SynthDetector(detector,legacy,.15))})
    finished=[]
    for seed in protocol['seeds']:
        for arm in protocol['arms']:
            root=FOLLOW/f'{arm}_seed{seed}';root.mkdir(exist_ok=True)
            result_path=root/'result.json';weights=root/'residual.pth'
            if result_path.exists():
                record=read(result_path)
                if sha(weights)!=record['weights_sha256']:raise ValueError('Completed residual changed')
                finished.append(record);continue
            torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
            adapter=LegacyTaskAdapter(legacy).cuda().train()
            optimizer=torch.optim.AdamW(adapter.residual.parameters(),lr=.001,weight_decay=.0001)
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
                latent,delta=adapter.components(xx[:,:3]/255)
                regularizer=.05*delta.square().mean()
                if arm=='legacy_mse_refine':loss=F.mse_loss(latent,xx[:,3:4]/255)+regularizer
                else:
                    with torch.no_grad():teacher=trunk(xx) if arm=='legacy_task_distilled' else None
                    mixed=torch.cat([xx[:,:3],latent*255],dim=1)
                    loss,_=task_loss(trunk(mixed),yy,teacher)
                    loss=loss+regularizer
                if not torch.isfinite(loss):raise ValueError('Nonfinite legacy task loss')
                loss.backward()
                if not all(p.grad is None or torch.isfinite(p.grad).all() for p in adapter.residual.parameters()):raise ValueError('Nonfinite residual gradient')
                torch.nn.utils.clip_grad_norm_(adapter.residual.parameters(),1)
                optimizer.step();scheduler.step()
                if step%100==0 or step==399:
                    record={'step':step+1,'loss':float(loss),'seconds':time.monotonic()-begin}
                    history.append(record);print(arm,seed,record,flush=True)
            adapter.eval();torch.save(adapter.residual.cpu().state_dict(),weights);adapter.residual.cuda()
            prob=source_probabilities(trunk,pack,adapter)
            source=source_metrics(prob,pack);blend_source=source_metrics(.85*current_prob+.15*prob,pack)
            mines=mine_validation(SynthDetector(detector,adapter))
            blend_mines=mine_validation(SynthDetector(detector,adapter,.15))
            record={'arm':arm,'seed':seed,'updates':400,'parameters':sum(p.numel() for p in adapter.residual.parameters()),
                   'weights':str(weights),'weights_sha256':sha(weights),'protocol_sha256':sha(path),'history':history,
                   'source_validation':source,'blend15_source_validation':blend_source,'mines':mines,'blend15_mines':blend_mines,
                   'seconds':time.monotonic()-begin,'frozen_trunk_has_gradients':any(p.grad is not None for p in trunk.parameters()),
                   'frozen_legacy_has_gradients':any(p.grad is not None for p in legacy.parameters()),'mine_inputs_trained':0}
            write(result_path,record);finished.append(record)
            write(FOLLOW/'results_partial.json',{'fits':finished})
            print('FINISHED',arm,seed,'source macroF1',source['thresholds']['0.4']['macro_f1'],
                  'mine recall',[r['thresholds']['0.4']['known_positive_recall'] for r in mines],
                  'blend normal',[r['thresholds']['0.4']['reviewed_normal_activation'] for r in blend_mines],flush=True)
            del adapter;torch.cuda.empty_cache()
    write(FOLLOW/'results.json',{'completed_at':now(),'fits':finished,'new_fits':6,'optimizer_updates':2400,
          'no_unopened_tests_or_Noto_inputs':True,'default_weight_changed':False})


if __name__=='__main__':
    sys.stdout.reconfigure(encoding='utf-8');main()
