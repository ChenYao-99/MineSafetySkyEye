"""Frozen nested-source scale-gate pilot on six historically exposed regions."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np
import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from runtime_paths import configure_runtime_paths
configure_runtime_paths()
import cross_region_scale as base
from scale_selector import FEATURES, METADATA_FEATURES, feature_map, fuse

PRIOR = ROOT/'experiments/cross_region_scale_20261019_v2'
METHODS = ('tta5','mean_multiscale5','constant_gate','feature_gate','feature_gate_no_gsd')
HEADS = METHODS[2:]


def register(out):
    out = Path(out).resolve()
    if out.drive.upper() != 'D:' or not out.is_relative_to(ROOT/'experiments'):
        raise ValueError('Use D:/LMHLD/experiments')
    out.mkdir(parents=True,exist_ok=True)
    parent = base.read(PRIOR/'protocol.json')
    code = [Path(__file__),ROOT/'scale_selector.py',Path(base.__file__)]
    hashes = {str(p):base.sha(p) for p in code}
    if hashes[str(Path(base.__file__))] != parent['runner_sha256']:
        raise ValueError('Prior frozen runner changed')
    for row in parent['protected_assets']:
        if base.sha(row['path']) != row['sha256']: raise ValueError('Protected asset changed')
    path = out/'protocol.json'
    if path.exists():
        protocol = base.read(path)
        if hashes != protocol['code_sha256']: raise ValueError('Frozen code changed; new directory required')
        if (base.sha(PRIOR/'protocol.json')!=protocol['parent_protocol_sha256'] or
                base.sha(PRIOR/'data_manifest.json')!=protocol['parent_manifest_sha256']):
            raise ValueError('Frozen parent protocol or dataset manifest changed')
        return protocol,parent
    protocol = {
        'registered_at':base.now(),'code_sha256':hashes,
        'question':'Can a source-trained input-conditioned scale gate beat both a source-trained constant gate and the stronger same-budget single-scale TTA baseline?',
        'parent_protocol':str(PRIOR/'protocol.json'),'parent_protocol_sha256':base.sha(PRIOR/'protocol.json'),
        'parent_manifest_sha256':base.sha(PRIOR/'data_manifest.json'),
        'regions':parent['regions'],'seeds':[42,43],
        'base_fits':12,'base_recipe':'Replay ONLY fixed arm of parent, same parents/seed/random initialization/1200 steps; no production or prior fitted weights loaded',
        'base_updates_each':1200,'base_updates_total':14400,
        'source_meta_split':'Within each source region, deterministic shuffled original validation positions: floor(N/2) meta-fit; rest threshold calibration. Target whole validation is used only after fitted coefficients are frozen.',
        'meta_split_seed':20261019,'meta_fit_calibration_overlap_allowed':False,
        'head_arms':list(HEADS),'head_steps':600,'head_per_region_batch':1024,
        'head_optimizer':{'name':'Adam','lr':.02},'head_regularization':.001,
        'head_initialization':'All weights/bias zero; exact 50:50 probability mean before fitting',
        'features':list(FEATURES),'normalization':'Mean/std on sampled source meta-fit features only; std floor .001; clip standardized features +/-8',
        'feature_exclusions':'No labels, region identifier, coordinates, dates or target-domain normalization statistics in inference features. No-GSD ablation zeros the last two metadata features.',
        'head_objective':'Importance-weighted native-pixel BCE; source interior-small foreground x4; other foreground/background x1; .001*(sum(weights^2)+bias^2)',
        'pixel_sampling':'Per source meta-fit parent, sample 64 pixels with replacement from each nonempty stratum: interior-small foreground, other foreground, background. Importance weights correct pixel/parent/region sampling; equal source regions per update.',
        'constant_control':'Same sampled pixels, labels, optimization steps and regularization; all input weights frozen at zero, learns only shared bias',
        'feature_gate':'18 inputs + bias, outputs sigmoid local weight per native pixel; convex blend of global and four-quadrant local probabilities',
        'inference_budget':'tta5 uses five full-parent views; other methods use one global+four local views. Same base network and 64x64 model inputs; head operations/timing reported separately, not called zero cost.',
        'calibration':'For each arm independently, lowest 0.001-grid threshold with equal-region mean background FPR<=.02 on source calibration HALF only; meta-fit and target parents excluded.',
        'methods':list(METHODS),'main_method':'feature_gate',
        'comparators':['tta5','constant_gate'],'secondary_ablations':['mean_multiscale5','feature_gate_no_gsd'],
        'stability_gate':parent['stability_gate'],
        'extra_gate':'Feature gate must also gain >=.01 macro small_cover50 vs learned constant in EACH seed, preserve F1 >=-.005 and macro FPR rise <=.002 vs constant.',
        'metrics':parent['other_metrics'],'small_definition':parent['primary_small_definition'],
        'no_grid_search_or_checkpoint_selection':True,
        'history':'All six original validation domains were development-exposed in the parent experiment; not a new blind test. Method choice is informed by parent results.',
        'limitations':['Nominal GSD, original tile border/adjacency and connected-component limitations inherited from parent.',
                      'Source meta-fit reuses part of official validation as learning data; disjoint calibration and target fold are explicit.',
                      'Small fixed-budget pilot, not an architecture ceiling or field/mining safety acceptance.',
                      'Gating/mixture is established; no original algorithm or paper-level novelty claim.'],
        'protected_assets':parent['protected_assets'],
        'retention':'Save tiny gate coefficients and complete scores/manifests; discard base checkpoints and full probability/feature caches.',
        'production_change_allowed':False,'official_test_files_used':0,'new_mine_predictions':0,
        'external_calls':0,'downloads':0,
    }
    base.write(path,protocol)
    return protocol,parent


def meta_split(n,group):
    order = np.random.default_rng(20261019+131*group).permutation(n)
    return np.sort(order[:n//2]),np.sort(order[n//2:])


def source_keys(keys,held):
    if set(keys) != set(range(6))-{held}: raise ValueError('Exactly five source regions required; target forbidden')


def predict_components(model,data,native_side,batch=96):
    import torch
    views = data['views']
    def predict(array,kind=0):
        chunks=[]
        with torch.inference_mode():
            for start in range(0,len(array),batch):
                a=torch.from_numpy(np.ascontiguousarray(array[start:start+batch])).cuda().float()
                if kind in (1,2,3): a=a.rot90(kind,(-2,-1))
                elif kind==4: a=a.flip(-1)
                p=base.inverse_tta(model(a).sigmoid()[:,0],kind)
                if not torch.isfinite(p).all(): raise ValueError('Non-finite prediction')
                chunks.append(p.cpu().numpy())
        return np.concatenate(chunks)
    started=time.perf_counter()
    full=predict(views[:,0]); tta=full.copy()
    for k in (1,2,3,4): tta+=predict(views[:,0],k)
    tta/=5
    quads=predict(views[:,1:].reshape(-1,4,64,64)).reshape(-1,4,64,64)
    result={'global':[],'local':[],'tta5':[]}
    p=native_side; h=p//2
    for i in range(len(views)):
        local=np.empty((p,p),np.float32)
        for q in range(4):
            y,x=q//2*h,q%2*h
            local[y:y+h,x:x+h]=cv2.resize(quads[i,q],(h,h),interpolation=cv2.INTER_LINEAR)
        result['global'].append(cv2.resize(full[i],(p,p),interpolation=cv2.INTER_LINEAR))
        result['tta5'].append(cv2.resize(tta[i],(p,p),interpolation=cv2.INTER_LINEAR))
        result['local'].append(local)
    return result,time.perf_counter()-started


def sampled_parent(features,p,q,gt,components,rng):
    small = components['small'][components['labels']]
    strata = (small, (gt>0)&~small, gt==0)
    xs,ps,qs,ys,ws=[],[],[],[],[]
    flat=features.reshape(-1,len(FEATURES)); total=gt.size
    for j,mask in enumerate(strata):
        pool=np.flatnonzero(mask)
        if not len(pool): continue
        ids=rng.choice(pool,64,replace=True)
        xs.append(flat[ids]);ps.append(p.ravel()[ids]);qs.append(q.ravel()[ids]);ys.append(gt.ravel()[ids])
        ws.append(np.full(64,len(pool)/total/64*(4 if j==0 else 1),np.float32))
    return [np.concatenate(v) for v in (xs,ps,qs,ys,ws)]


def make_source_pools(val_data,components,manifest,held,seed):
    pools={}; rows={}
    for g in range(6):
        if g==held: continue
        data=val_data[g]; fit,cal=meta_split(len(data['labels']),g)
        assert not set(fit)&set(cal)
        rows[g]={'meta_fit_positions':fit.tolist(),'calibration_positions':cal.tolist(),
                 'meta_fit_original_indices':[data['indices'][i] for i in fit],
                 'calibration_original_indices':[data['indices'][i] for i in cal]}
        raw=np.load(manifest['regions'][g]['splits']['val']['image_path'],mmap_mode='r')
        samples=[]; rng=np.random.default_rng(seed+held*100+g*17+9000)
        for i in fit:
            p,q=components[g]['global'][i],components[g]['local'][i]
            f=feature_map(raw[data['indices'][i]],p,q,data['gsd'])
            samples.append(sampled_parent(f,p,q,data['labels'][i],data['components'][i],rng))
        pack=[np.concatenate([s[k] for s in samples]) for k in range(5)]
        pack[4] *= len(pack[0])/len(fit)
        pools[g]=pack
    source_keys(pools,held)
    return pools,rows


def fit_gate(pools,held,seed,arm,protocol):
    import torch
    source_keys(pools,held)
    ordered=sorted(pools)
    x,p,q,y,w=[np.concatenate([pools[g][k] for g in ordered]).astype(np.float32) for k in range(5)]
    groups=np.concatenate([np.full(len(pools[g][0]),g,np.int64) for g in ordered])
    mean=x.mean(0); std=np.maximum(x.std(0),.001)
    z=np.clip((x-mean)/std,-8,8)
    mask=np.ones(len(FEATURES),np.float32)
    if arm=='constant_gate':mask[:]=0
    elif arm=='feature_gate_no_gsd':mask[list(METADATA_FEATURES)]=0
    z=torch.as_tensor(z,device='cuda');p,q,y,w=[torch.as_tensor(v,device='cuda') for v in (p,q,y,w)]
    mask=torch.as_tensor(mask,device='cuda')
    weights=torch.nn.Parameter(torch.zeros(len(FEATURES),device='cuda'))
    bias=torch.nn.Parameter(torch.zeros((),device='cuda'))
    opt=torch.optim.Adam([weights,bias],lr=protocol['head_optimizer']['lr'])
    rng=np.random.default_rng(seed+held*100+5000); trace=hashlib.sha256(); losses=[]
    started=time.perf_counter()
    for step in range(protocol['head_steps']):
        ids=np.concatenate([rng.choice(np.flatnonzero(groups==g),protocol['head_per_region_batch'],replace=True) for g in ordered])
        if np.any(groups[ids]==held):raise AssertionError('Target entered gate fit')
        trace.update(np.asarray(ids,np.int64).tobytes())
        opt.zero_grad(set_to_none=True)
        alpha=torch.sigmoid(z[ids]@(weights*mask)+bias)
        mixture=((1-alpha)*p[ids]+alpha*q[ids]).clamp(1e-6,1-1e-6)
        bce=-(y[ids]*mixture.log()+(1-y[ids])*torch.log1p(-mixture))
        loss=(w[ids]*bce).mean()+protocol['head_regularization']*((weights*mask).square().sum()+bias.square())
        if not torch.isfinite(loss):raise ValueError('Non-finite gate objective')
        loss.backward();opt.step();losses.append(float(loss.detach()))
    torch.cuda.synchronize()
    coefficients={'features':list(FEATURES),'mean':mean.tolist(),'std':std.tolist(),
                  'weights':(weights*mask).detach().cpu().tolist(),'bias':float(bias.detach()),
                  'arm':arm,'source_regions':[base.REGIONS[g][0] for g in ordered],
                  'held_out_excluded':True,'active_parameters':int(mask.sum())+1}
    return coefficients,{'seconds':time.perf_counter()-started,'updates':len(losses),
            'sample_schedule_sha256':trace.hexdigest(),'source_pixels':len(x),
            'source_regions':coefficients['source_regions'],'target_rows':0,
            'loss_first50':float(np.mean(losses[:50])),'loss_last50':float(np.mean(losses[-50:]))}


def evaluate(components,heads,val_data,manifest,held):
    maps={name:{} for name in METHODS}; selected={}; alphas={arm:[] for arm in HEADS}
    for g,data in val_data.items():
        positions=np.arange(len(data['labels'])) if g==held else meta_split(len(data['labels']),g)[1]
        selected[g]={'labels':[data['labels'][i] for i in positions],
                     'components':[data['components'][i] for i in positions]}
        for name in METHODS: maps[name][g]=[]
        raw=np.load(manifest['regions'][g]['splits']['val']['image_path'],mmap_mode='r')
        for i in positions:
            p,q=components[g]['global'][i],components[g]['local'][i]
            maps['tta5'][g].append(components[g]['tta5'][i])
            maps['mean_multiscale5'][g].append(.5*(p+q))
            features=feature_map(raw[data['indices'][i]],p,q,data['gsd'])
            for arm in HEADS:
                prob,alpha=fuse(features,p,q,heads[arm]);maps[arm][g].append(prob)
                if g==held: alphas[arm].append(float(alpha.mean()))
    result={}
    for name in METHODS:
        threshold,calibration=base.source_threshold(maps[name],selected,[g for g in range(6) if g!=held],held)
        target=selected[held]
        rows=[base.parent_counts(p,y,c,threshold) for p,y,c in zip(maps[name][held],target['labels'],target['components'])]
        fixed=[base.parent_counts(p,y,c,.5) for p,y,c in zip(maps[name][held],target['labels'],target['components'])]
        result[name]={'threshold':threshold,'calibration':calibration,'held_out':base.aggregate_counts(rows),
                      'held_out_at_fixed_05':base.aggregate_counts(fixed),'per_parent_counts':rows,
                      'calibration_parent_count_by_region':{base.REGIONS[g][0]:len(selected[g]['labels']) for g in range(6) if g!=held}}
        if name in alphas:
            result[name]['target_mean_local_weight']=float(np.mean(alphas[name]))
    return result


def summarize(out,protocol):
    rows=[base.read(p) for p in sorted(Path(out).glob('fold_*/seed*/result.json'))]
    if len(rows)!=12:raise ValueError('All twelve folds/seeds required')
    metrics=('small_cover50','small_iou50','f1','background_fpr')
    comparisons={}
    for comparator in ('tta5','constant_gate','mean_multiscale5','feature_gate_no_gsd'):
        regional=[]; by_seed=[]
        for seed in protocol['seeds']:
            rr=[]
            for r in rows:
                if r['seed']!=seed:continue
                a,b=r['methods']['feature_gate']['held_out'],r['methods'][comparator]['held_out']
                item={'region':r['held_out'],'seed':seed,'candidate':{k:a[k] for k in metrics},
                      'baseline':{k:b[k] for k in metrics},'delta':{k:a[k]-b[k] for k in metrics}}
                rr.append(item);regional.append(item)
            by_seed.append({'seed':seed,'candidate_macro':{k:float(np.mean([r['candidate'][k] for r in rr])) for k in metrics},
                'baseline_macro':{k:float(np.mean([r['baseline'][k] for r in rr])) for k in metrics},
                'macro_delta':{k:float(np.mean([r['delta'][k] for r in rr])) for k in metrics},
                'candidate_worst_region_fpr':max(r['candidate']['background_fpr'] for r in rr),
                'worst_fpr_increase':max(r['delta']['background_fpr'] for r in rr)})
        comparisons[comparator]={'regions':regional,'by_seed':by_seed}
    primary=comparisons['tta5'];gate=protocol['stability_gate']
    positive=sum(all(next(r for r in primary['regions'] if r['region']==name and r['seed']==seed)['delta']['small_cover50']>0
                     for seed in protocol['seeds']) for name,_ in base.REGIONS)
    checks={
        'small_gain_each_seed':all(r['macro_delta']['small_cover50']>=.02 for r in primary['by_seed']),
        'four_regions_positive_both_seeds':positive>=4,
        'f1_preserved_vs_tta':all(r['macro_delta']['f1']>=-.005 for r in primary['by_seed']),
        'macro_fpr_increase_bounded':all(r['macro_delta']['background_fpr']<=.002 for r in primary['by_seed']),
        'regional_fpr_increase_bounded':all(r['worst_fpr_increase']<=.01 for r in primary['by_seed']),
        'candidate_macro_fpr_bounded':all(r['candidate_macro']['background_fpr']<=.03 for r in primary['by_seed']),
        'candidate_worst_fpr_bounded':all(r['candidate_worst_region_fpr']<=.05 for r in primary['by_seed']),
        'beats_fitted_constant_small':all(r['macro_delta']['small_cover50']>=.01 for r in comparisons['constant_gate']['by_seed']),
        'f1_preserved_vs_constant':all(r['macro_delta']['f1']>=-.005 for r in comparisons['constant_gate']['by_seed']),
        'fpr_preserved_vs_constant':all(r['macro_delta']['background_fpr']<=.002 for r in comparisons['constant_gate']['by_seed']),
    }
    if any(base.sha(row['path'])!=row['sha256'] for row in protocol['protected_assets']):raise ValueError('Protected asset changed')
    result={'completed_at':base.now(),'status':'completed','base_fits':12,'base_updates':14400,
        'head_fits':36,'head_updates':21600,'comparisons':comparisons,'stability_checks':checks,
        'stability_gate_passed':all(checks.values()),'regions_positive_both_seeds':positive,
        'production_changed':False,'new_official_test_predictions':0,'new_mine_predictions':0,'external_calls':0,
        'scope':'Source-only nested fit/calibration on development-exposed validation, not new blind mining acceptance'}
    base.write(Path(out)/'results.json',result)
    base.write(Path(out)/'state.json',{k:v for k,v in result.items() if k!='comparisons'})
    return result


def run(out):
    import torch
    protocol,parent=register(out)
    torch.set_num_threads(4);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    if not torch.cuda.is_available():raise RuntimeError('CUDA unavailable')
    x,y,groups,val_data,manifest=base.prepare_data(out,parent)
    prior_manifest=base.read(PRIOR/'data_manifest.json')
    if {k:v for k,v in manifest.items() if k!='registered_at'}!={k:v for k,v in prior_manifest.items() if k!='registered_at'}:
        raise ValueError('Parent rows/pixels changed')
    x,y=torch.from_numpy(x).cuda(),torch.from_numpy(y).cuda()
    base.write(Path(out)/'runtime.json',{'torch':torch.__version__,'cuda':torch.version.cuda,
        'device':torch.cuda.get_device_name(0),'threads':torch.get_num_threads(),
        'deterministic_cudnn':True,'tf32':False,'cache_drive':'D:',
        'features':len(FEATURES),'parameters_full_gate':len(FEATURES)+1})
    completed=0
    for held,(region,_) in enumerate(base.REGIONS):
        for seed in protocol['seeds']:
            directory=Path(out)/f'fold_{held}_{region}'/f'seed{seed}'
            directory.mkdir(parents=True,exist_ok=True)
            path=directory/'result.json'
            if path.exists():completed+=1;continue
            base.write(Path(out)/'state.json',{'status':'running','updated_at':base.now(),
                'held_out':region,'seed':seed,'completed':completed,'total':12,'production_changed':False})
            model,training=base.train_model(x,y,groups,held,seed,'fixed',parent)
            prior=base.read(PRIOR/f'fold_{held}_{region}'/f'seed{seed}'/'result_fixed.json')
            for k in ('initial_state_sha256','parent_schedule_sha256','visits_per_region'):
                if training[k]!=prior['training'][k]:raise ValueError('Base replay differs from parent')
            components={};prediction_seconds={}
            for g,data in val_data.items():
                side=manifest['regions'][g]['splits']['val']['image_shape'][-1]
                components[g],prediction_seconds[base.REGIONS[g][0]]=predict_components(model,data,side)
            pools,split_rows=make_source_pools(val_data,components,manifest,held,seed)
            heads={};head_training={}
            for arm in HEADS:
                heads[arm],head_training[arm]=fit_gate(pools,held,seed,arm,protocol)
            if len({v['sample_schedule_sha256'] for v in head_training.values()})!=1:
                raise ValueError('Unpaired head sampling')
            state_hash=hashlib.sha256()
            for key,value in model.state_dict().items():
                state_hash.update(key.encode());state_hash.update(value.detach().cpu().numpy().tobytes())
            for arm in HEADS: heads[arm]['base_model_state_sha256']=state_hash.hexdigest()
            # Only after freezing the heads, compare the fixed-0.5 baseline
            # against the parent run. This never selects a fitted component.
            d=val_data[held]
            fixed=base.aggregate_counts([base.parent_counts(p,gt,c,.5) for p,gt,c in
                  zip(components[held]['tta5'],d['labels'],d['components'])])
            if fixed['counts']!=prior['methods']['tta5']['held_out_at_fixed_05']['counts']:
                raise ValueError('Baseline reconstruction mismatch; do not interpret gate scores')
            methods=evaluate(components,heads,val_data,manifest,held)
            base.write(path,{'held_index':held,'held_out':region,'seed':seed,'completed_at':base.now(),
                'training':training,'head_training':head_training,'coefficients':heads,
                'source_meta_split':{base.REGIONS[g][0]:v for g,v in split_rows.items()},
                'baseline_fixed05_reproduced':True,'prediction_seconds':prediction_seconds,'methods':methods})
            del model,components,pools
            torch.cuda.empty_cache();completed+=1
            print('SELECTOR_COMPLETED',completed,'/12',region,seed,flush=True)
    result=summarize(out,protocol)
    print('COMPLETE',json.dumps({'fits':12,'stability_gate_passed':result['stability_gate_passed'],
                               'checks':result['stability_checks']}),flush=True)


if __name__=='__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    p=argparse.ArgumentParser();p.add_argument('--output',required=True)
    p.add_argument('--stage',choices=('register','run','summarize'),default='run');a=p.parse_args()
    if a.stage=='register':register(a.output)
    elif a.stage=='summarize':summarize(a.output,register(a.output)[0])
    else:run(a.output)
