"""Frozen-model missed-component diagnosis; no new model/threshold selection.

Replays discarded source-only trunks exactly, then reuses saved source-fitted
gates and thresholds. Ground truth is used only for retrospective diagnosis.
"""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
from pathlib import Path
import sys
import time
import numpy as np
import cv2

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from runtime_paths import configure_runtime_paths
configure_runtime_paths()
import cross_region_scale as base
import validate_scale_selector as selector
from scale_selector import feature_map,fuse

PRIOR=ROOT/'experiments/source_scale_selector_20261019'
CATEGORIES=('routing_loss','complementarity_loss','threshold_gap','weak_branch_response')


def register(out):
    out=Path(out).resolve()
    if out.drive.upper()!='D:' or not out.is_relative_to(ROOT/'experiments'):
        raise ValueError('Use D:/LMHLD/experiments')
    out.mkdir(parents=True,exist_ok=True)
    parent=base.read(PRIOR/'protocol.json')
    assets={**parent['code_sha256'],str(PRIOR/'protocol.json'):base.sha(PRIOR/'protocol.json'),
            str(PRIOR/'data_manifest.json'):base.sha(PRIOR/'data_manifest.json'),
            str(PRIOR/'verification.json'):base.sha(PRIOR/'verification.json'),
            str(Path(__file__)):base.sha(__file__)}
    for row in parent['protected_assets']:assets[row['path']]=row['sha256']
    for path,digest in assets.items():
        if base.sha(path)!=digest:raise ValueError('Frozen asset changed')
    for path in sorted(PRIOR.glob('fold_*/seed*/result.json')):assets[str(path)]=base.sha(path)
    if len(list(PRIOR.glob('fold_*/seed*/result.json')))!=12:raise ValueError('Twelve completed parent fits required')
    path=out/'protocol.json'
    if path.exists():
        protocol=base.read(path)
        if assets!=protocol['frozen_assets']:raise ValueError('Frozen inputs/code changed; use a new directory')
        return protocol,parent
    protocol={'registered_at':base.now(),'frozen_assets':assets,'regions':parent['regions'],
        'seeds':[42,43],'question':'Which fixed-model misses are limited by branch support, fusion, threshold, or input detail?',
        'model_policy':'Replay twelve exact fixed trunks (1200 steps each) solely to reconstruct discarded models; no new fitting recipe, head update, threshold search or production selection.',
        'exact_replay_guard':'Model state hash must equal saved gate backbone hash; every parent TTA/gate count at saved thresholds must match prior result exactly.',
        'primary_population':'All 787 original interior <=2500 nominal-m2 GT connected components; two seeds =1574 correlated object-seed observations, not 1574 independent landslides.',
        'response_partition':{
            'routing_loss':'Gate misses half coverage at its frozen threshold, but at least one raw branch reaches half at that SAME threshold.',
            'complementarity_loss':'Neither branch alone reaches half but pixelwise max does at same threshold. Label-informed diagnostic ceiling, not a deployable algorithm.',
            'threshold_gap':'Max of branches misses at frozen threshold but reaches half at fixed reference 0.5. Not authorization to lower threshold.',
            'weak_branch_response':'Even max of branches fails half coverage at fixed reference 0.5 and frozen threshold; finite-budget model observation, not proof of impossibility.',
        },
        'geometry':'Resize EACH GT object with same AREA/LINEAR kernels as model image preprocessing. Global/native-quadrant 64px peak occupancy, effective area and seam splits. Peak<0.5 means no majority-target model cell, NOT proof of no visual signal.',
        'contrast':'Compare normalized four-band object-minus-10m-ring mean vector norm before/after actual FP16 input-view resampling; ring excludes all GT. Nominal radius clipped to1..16px; ring<5px or original contrast<0.02 flagged, not invented.',
        'paired_detail_probe':'Deterministic 64-native-pixel windows (entire32px parent enlarged to64 where needed), same parent normalization/FOV/weights. Sharp vs same window reduced to half width then linearly restored. Edge windows shifted in bounds; overlap probabilities averaged. No GT-centered crops.',
        'probe_thresholds':'Fixed .5 and the saved source-calibrated gate threshold; never optimized on target labels. Probe has additional compute and no separate source calibration, so diagnostic only.',
        'causal_limits':'Paired sharp-vs-blur differs only in smoothing at same window context. Change from original branches to native windows ALSO changes context. Occupancy/contrast associations alone do not establish cause.',
        'cache':'Lossless FP32 64px global, four quadrant and TTA-average probability grids, original row IDs, hashes; no original image copies or trunk weights. Later attribution can replay from cache without training.',
        'cache_use':'Development diagnosis only; these exposed target predictions cannot be relabeled as new independent test evidence or enter training of the corresponding held-out fold.',
        'retention_budget':'Network-grid caches about408MB uncompressed for both seeds; compressed on D. Probe maps not persisted, retain object/pixel statistics.',
        'example_rule':'First seed42 miss by region/index/component in each response category, plus first sharp-only rescue and first both-view subpixel miss, maximum six examples; no best-looking image selection.',
        'history':'Same exposed LMHLD validation domains, not new blind mine acceptance.',
        'default_changed':False,'official_test_predictions':0,'new_mine_predictions':0,'cloud_calls':0,'downloads':0}
    base.write(path,protocol)
    return protocol,parent


def boxes(side):
    size=min(side,64)
    positions=list(range(0,side-size+1,size))
    if positions[-1]!=side-size:positions.append(side-size)
    return [(x,y,x+size,y+size) for y in positions for x in positions]


def normalize(raw):
    raw=np.asarray(raw,np.float32);lo=raw.min((1,2),keepdims=True);span=raw.max((1,2),keepdims=True)-lo
    return np.divide(raw-lo,span,out=np.zeros_like(raw),where=span>0)


def resize_image(chw,size,interpolation):
    return cv2.resize(chw.transpose(1,2,0),(size,size),interpolation=interpolation).transpose(2,0,1)


def paired_inputs(normalized):
    side=normalized.shape[-1]; sharp=[];blur=[]
    for x0,y0,x1,y1 in boxes(side):
        window=normalized[:,y0:y1,x0:x1]
        sharp.append(resize_image(window,64,cv2.INTER_LINEAR))
        low=resize_image(window,(x1-x0)//2,cv2.INTER_AREA)
        blur.append(resize_image(low,64,cv2.INTER_LINEAR))
    return np.asarray(sharp,np.float16),np.asarray(blur,np.float16)


def native_probe(model,normalized):
    import torch
    sharp,blur=paired_inputs(normalized)
    with torch.inference_mode():
        inputs=torch.from_numpy(np.concatenate([sharp,blur])).cuda().float()
        probability=model(inputs).sigmoid()[:,0].cpu().numpy()
    if not np.isfinite(probability).all():raise ValueError('Invalid paired probe')
    side=normalized.shape[-1];n=len(sharp)
    outputs=[]
    for part in [probability[:n],probability[n:]]:
        value=np.zeros((side,side),np.float32);weight=np.zeros_like(value)
        for grid,(x0,y0,x1,y1) in zip(part,boxes(side)):
            value[y0:y1,x0:x1]+=cv2.resize(grid,(x1-x0,y1-y0),interpolation=cv2.INTER_LINEAR)
            weight[y0:y1,x0:x1]+=1
        if np.any(weight==0):raise AssertionError('Unobserved probe pixel')
        outputs.append(value/weight)
    return outputs[0],outputs[1],2*n


def predict_grids(model,views):
    import torch
    def predict(array,kind=0):
        values=[]
        with torch.inference_mode():
            for start in range(0,len(array),96):
                a=torch.from_numpy(np.ascontiguousarray(array[start:start+96])).cuda().float()
                if kind in (1,2,3):a=a.rot90(kind,(-2,-1))
                elif kind==4:a=a.flip(-1)
                p=base.inverse_tta(model(a).sigmoid()[:,0],kind)
                if not torch.isfinite(p).all():raise ValueError('Invalid reconstruction')
                values.append(p.cpu().numpy())
        return np.concatenate(values)
    full=predict(views[:,0]);tta=full.copy()
    for k in (1,2,3,4):tta+=predict(views[:,0],k)
    tta/=5
    local=predict(views[:,1:].reshape(-1,4,64,64)).reshape(-1,4,64,64)
    return full,local,tta


def restore(full,local,tta,side):
    p=cv2.resize(full,(side,side),interpolation=cv2.INTER_LINEAR)
    q=np.empty_like(p);half=side//2
    for j in range(4):
        y,x=j//2*half,j%2*half
        q[y:y+half,x:x+half]=cv2.resize(local[j],(half,half),interpolation=cv2.INTER_LINEAR)
    t=cv2.resize(tta,(side,side),interpolation=cv2.INTER_LINEAR)
    return p,q,t


def reconstructed_input(views,side):
    global_input=resize_image(views[0].astype(np.float32),side,cv2.INTER_LINEAR)
    local_input=np.empty_like(global_input);half=side//2
    for j in range(4):
        y,x=j//2*half,j%2*half
        local_input[:,y:y+half,x:x+half]=resize_image(views[j+1].astype(np.float32),half,cv2.INTER_LINEAR)
    return global_input,local_input


def occupancy(mask):
    side=len(mask);half=side//2
    interpolation=cv2.INTER_AREA if side>64 else cv2.INTER_LINEAR
    global_mask=cv2.resize(mask.astype(np.float32),(64,64),interpolation=interpolation)
    local=[];pieces=[]
    for y in (0,half):
        for x in (0,half):
            part=mask[y:y+half,x:x+half];pieces.append(int(part.sum()))
            local.append(cv2.resize(part.astype(np.float32),(64,64),interpolation=cv2.INTER_AREA if half>64 else cv2.INTER_LINEAR))
    local=np.stack(local)
    a,b=float(global_mask.max()),float(local.max())
    return {'global_peak_occupancy':a,'local_peak_occupancy':b,
        'global_effective_model_pixels':float(global_mask.sum()),'local_effective_model_pixels':float(local.sum()),
        'global_majority_cells':int((global_mask>=.5).sum()),'local_majority_cells':int((local>=.5).sum()),
        'both_views_subpixel':a<.5 and b<.5,'local_restores_majority_cell':a<.5 and b>=.5,
        'quadrant_pieces':sum(v>0 for v in pieces),'largest_quadrant_fraction':max(pieces)/max(sum(pieces),1)}


def coverage(prob,mask,threshold):
    return float((np.asarray(prob,np.float64)[mask]>=threshold).mean())


def response_category(p,q,g,mask,threshold):
    a,b,c=[coverage(v,mask,threshold) for v in (p,q,g)]
    union=coverage(np.maximum(p,q),mask,threshold)
    reference=coverage(np.maximum(p,q),mask,.5)
    if c>=.5:return 'hit'
    if max(a,b)>=.5:return 'routing_loss'
    if union>=.5:return 'complementarity_loss'
    if reference>=.5:return 'threshold_gap'
    return 'weak_branch_response'


def contrast_measure(normalized,global_input,local_input,mask,all_gt,gsd):
    radius=int(np.clip(round(10/gsd),1,16))
    ring=cv2.dilate(mask.astype(np.uint8),cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2*radius+1,2*radius+1))).astype(bool)&~all_gt.astype(bool)
    row={'ring_radius_px':radius,'ring_pixels':int(ring.sum())}
    if ring.sum()<5:
        return {**row,'native_contrast':None,'global_contrast_retention':None,'local_contrast_retention':None}
    norms=[float(np.linalg.norm(x[:,mask].mean(1)-x[:,ring].mean(1))) for x in (normalized,global_input,local_input)]
    row.update(native_contrast=norms[0],global_input_contrast=norms[1],local_input_contrast=norms[2])
    row['global_contrast_retention']=norms[1]/norms[0] if norms[0]>=.02 else None
    row['local_contrast_retention']=norms[2]/norms[0] if norms[0]>=.02 else None
    return row


def analyze_parent(raw,gt,comp,views,p,q,tta,gate,sharp,blur,threshold,region,index,seed):
    normalized=normalize(raw);global_input,local_input=reconstructed_input(views,len(gt))
    rows=[]
    for component in np.flatnonzero(comp['small']):
        mask=comp['labels']==component
        geom=occupancy(mask)
        row={'object_id':f'{region}/{index}/{int(component)}','region':region,'original_parent_index':int(index),
             'component_id':int(component),'seed':seed,'native_area_pixels':int(mask.sum()),
             'nominal_area_m2':float(mask.sum()*comp['gsd']**2),'threshold':threshold,
             'category':response_category(p,q,gate,mask,threshold),'geometry':geom,
             'contrast':contrast_measure(normalized,global_input,local_input,mask,gt,comp['gsd'])}
        for name,v in [('global',p),('local',q),('max_diagnostic',np.maximum(p,q)),('tta',tta),('gate',gate),('sharp_probe',sharp),('blur_probe',blur)]:
            values=v[mask]
            row[name]={'coverage_at_gate_threshold':coverage(v,mask,threshold),'coverage_at_05':coverage(v,mask,.5),
                       'probability_median':float(np.median(values)),'probability_p90':float(np.quantile(values,.9))}
        row['sharp_only_probe_recovery']=(row['category']!='hit' and row['sharp_probe']['coverage_at_gate_threshold']>=.5 and row['blur_probe']['coverage_at_gate_threshold']<.5)
        row['sharp_and_blur_probe_recovery']=(row['category']!='hit' and row['sharp_probe']['coverage_at_gate_threshold']>=.5 and row['blur_probe']['coverage_at_gate_threshold']>=.5)
        row['sharp_only_probe_recovery_at05']=(row['category']!='hit' and row['sharp_probe']['coverage_at_05']>=.5 and row['blur_probe']['coverage_at_05']<.5)
        rows.append(row)
    return rows


def record_examples(out,objects,sharp,blur):
    """Preserve only first predeclared examples; never affect any prediction."""
    if not objects or objects[0]['seed']!=42:return
    path=Path(out)/'examples.json'
    examples=base.read(path) if path.exists() else []
    reasons={r['reason'] for r in examples}
    for row in objects:
        if row['category']=='hit':continue
        options=[row['category']]
        if row['sharp_only_probe_recovery']:options.append('sharp_only_probe_recovery')
        if row['geometry']['both_views_subpixel']:options.append('both_views_subpixel')
        for reason in options:
            if reason in reasons:continue
            folder=Path(out)/'example_probes';folder.mkdir(exist_ok=True)
            target=folder/f"{row['region']}_{row['original_parent_index']}.npz"
            if not target.exists():np.savez_compressed(target,sharp=sharp,blur=blur)
            examples.append({'reason':reason,'object_id':row['object_id'],'region':row['region'],
                'original_parent_index':row['original_parent_index'],'component_id':row['component_id'],
                'seed':42,'probe_path':str(target.relative_to(out)),'threshold':row['threshold']})
            reasons.add(reason)
    if len(examples)>6:raise AssertionError('Too many diagnostic examples')
    base.write(path,examples)


def summarize(out,protocol):
    results=[base.read(p) for p in sorted(Path(out).glob('fold_*/seed*/result.json'))]
    if len(results)!=12:raise ValueError('All twelve exact reconstructions required')
    rows=[v for r in results for v in r['objects']]
    if len(rows)!=1574:raise AssertionError('Object count changed')
    counts=Counter(r['category'] for r in rows);misses=[r for r in rows if r['category']!='hit']
    unique={}
    for row in rows:unique.setdefault(row['object_id'],[]).append(row)
    if len(unique)!=787 or any(len(v)!=2 for v in unique.values()):raise AssertionError('Seed pairing failed')
    regional=[]
    for region,_ in base.REGIONS:
        rr=[r for r in rows if r['region']==region];mm=[r for r in rr if r['category']!='hit']
        regional.append({'region':region,'object_seed_observations':len(rr),'misses':len(mm),
            'categories':dict(Counter(r['category'] for r in mm)),
            'misses_subpixel_both_views':sum(r['geometry']['both_views_subpixel'] for r in mm),
            'misses_local_restores_majority':sum(r['geometry']['local_restores_majority_cell'] for r in mm),
            'sharp_only_probe_recoveries':sum(r['sharp_only_probe_recovery'] for r in mm)})
    probe={}
    for name in ('sharp_at_gate','blur_at_gate','sharp_at05','blur_at05'):
        probe[name]={k:float(np.mean([r['probe_metrics'][name][k] for r in results])) for k in ('f1','background_fpr')}
    contrasts=[r for r in misses if r['contrast']['global_contrast_retention'] is not None]
    summary={'completed_at':base.now(),'status':'completed','replayed_base_models':12,'replay_updates':14400,
        'new_optimization_recipes':0,'head_updates':0,'threshold_changes':0,
        'unique_label_components':len(unique),'object_seed_observations':len(rows),'hits':counts.get('hit',0),'misses':len(misses),
        'miss_categories':{k:counts.get(k,0) for k in CATEGORIES},
        'stable_missed_objects_both_seeds':sum(all(r['category']!='hit' for r in pair) for pair in unique.values()),
        'stable_hit_objects_both_seeds':sum(all(r['category']=='hit' for r in pair) for pair in unique.values()),
        'stable_weak_branch_objects_both_seeds':sum(all(r['category']=='weak_branch_response' for r in pair) for pair in unique.values()),
        'miss_geometry':{'both_views_subpixel':sum(r['geometry']['both_views_subpixel'] for r in misses),
            'local_restores_majority_cell':sum(r['geometry']['local_restores_majority_cell'] for r in misses),
            'local_already_has_majority_cell':sum(r['geometry']['local_peak_occupancy']>=.5 for r in misses),
            'crosses_quadrant_seam':sum(r['geometry']['quadrant_pieces']>1 for r in misses)},
        'miss_contrast':{'available':len(contrasts),'global_retention_below_half':sum(r['contrast']['global_contrast_retention']<.5 for r in contrasts),
            'global_loss_local_recovered':sum(r['contrast']['global_contrast_retention']<.5 and r['contrast']['local_contrast_retention']>=.75 for r in contrasts)},
        'paired_probe':{'sharp_only_recoveries_at_gate_threshold':sum(r['sharp_only_probe_recovery'] for r in misses),
            'sharp_and_blur_recoveries_at_gate_threshold':sum(r['sharp_and_blur_probe_recovery'] for r in misses),
            'sharp_only_recoveries_at05':sum(r['sharp_only_probe_recovery_at05'] for r in misses),
            'old_gate_hits_lost_by_sharp_probe':sum(r['category']=='hit' and r['sharp_probe']['coverage_at_gate_threshold']<.5 for r in rows),
            'all_sharp_probe_hits_at_gate_threshold':sum(r['sharp_probe']['coverage_at_gate_threshold']>=.5 for r in rows),
            'all_blur_probe_hits_at_gate_threshold':sum(r['blur_probe']['coverage_at_gate_threshold']>=.5 for r in rows),
            'pixel_macro_metrics':probe,'diagnostic_only_not_calibrated_deployable_method':True},
        'regional':regional,'all_exact_replay_checks_passed':all(r['exact_prior_counts_reproduced'] for r in results),
        'cache_bytes':sum(p.stat().st_size for p in Path(out).glob('fold_*/seed*/probability_grids.npz')),
        'production_changed':False,'official_test_predictions':0,'new_mine_predictions':0,'external_calls':0,
        'limits':'Response partition is numerical diagnosis at frozen thresholds, not exclusive physical causes. Geometry is label occupancy; image contrast and paired probe provide separate evidence. Additional native-window compute and context prevent claims of deployable recall gains. Same development data.'}
    if sum(summary['miss_categories'].values())!=summary['misses']:raise AssertionError('Miss partition incomplete')
    for path,digest in protocol['frozen_assets'].items():
        if base.sha(path)!=digest:raise ValueError('Frozen asset changed')
    base.write(Path(out)/'results.json',summary)
    base.write(Path(out)/'state.json',{k:v for k,v in summary.items() if k not in ('regional','limits')})
    return summary


def run(out):
    import torch
    protocol,parent=register(out)
    base_protocol=base.read(parent['parent_protocol'])
    torch.set_num_threads(4);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    if not torch.cuda.is_available():raise RuntimeError('CUDA unavailable')
    x,y,groups,data,manifest=base.prepare_data(out,base_protocol)
    prior_manifest=base.read(PRIOR/'data_manifest.json')
    if {k:v for k,v in manifest.items() if k!='registered_at'}!={k:v for k,v in prior_manifest.items() if k!='registered_at'}:raise ValueError('Raw rows changed')
    x,y=torch.from_numpy(x).cuda(),torch.from_numpy(y).cuda()
    completed=0
    for held,(region,gsd) in enumerate(base.REGIONS):
        d=data[held];d['components']=[{**c,'gsd':gsd} for c in d['components']]
        record=manifest['regions'][held]['splits']['val'];raw=np.load(record['image_path'],mmap_mode='r')
        for seed in protocol['seeds']:
            directory=Path(out)/f'fold_{held}_{region}'/f'seed{seed}';directory.mkdir(parents=True,exist_ok=True)
            result_path=directory/'result.json';cache_path=directory/'probability_grids.npz'
            prior=base.read(PRIOR/f'fold_{held}_{region}'/f'seed{seed}'/'result.json')
            if result_path.exists():
                old=base.read(result_path)
                if not cache_path.exists() or base.sha(cache_path)!=old['cache_sha256']:raise ValueError('Diagnostic cache changed')
                completed+=1;continue
            base.write(Path(out)/'state.json',{'status':'running','updated_at':base.now(),'held_out':region,'seed':seed,'completed':completed,'total':12,'production_changed':False})
            model,training=base.train_model(x,y,groups,held,seed,'fixed',base_protocol)
            state_hash=hashlib.sha256()
            for key,value in model.state_dict().items():state_hash.update(key.encode());state_hash.update(value.detach().cpu().numpy().tobytes())
            if state_hash.hexdigest()!=prior['coefficients']['feature_gate']['base_model_state_sha256']:raise ValueError('Backbone exact-state replay failed')
            for k in ('initial_state_sha256','parent_schedule_sha256','visits_per_region'):
                if training[k]!=prior['training'][k]:raise ValueError('Training replay mismatch')
            full,local,tta=predict_grids(model,d['views'])
            np.savez_compressed(cache_path,global_grid=full,local_grids=local,tta_grid=tta,original_indices=np.array(d['indices'],np.int64))
            objects=[];gate_counts=[];tta_counts=[];probe_counts={k:[] for k in ('sharp_at_gate','blur_at_gate','sharp_at05','blur_at05')}
            probe_views=0;threshold=prior['methods']['feature_gate']['threshold']
            started=time.perf_counter()
            for i,(index,gt,c,views) in enumerate(zip(d['indices'],d['labels'],d['components'],d['views'])):
                p,q,t=restore(full[i],local[i],tta[i],len(gt))
                image=np.asarray(raw[index]);gate,alpha=fuse(feature_map(image,p,q,gsd),p,q,prior['coefficients']['feature_gate'])
                gc=base.parent_counts(gate,gt,c,threshold);tc=base.parent_counts(t,gt,c,prior['methods']['tta5']['threshold'])
                if gc!=prior['methods']['feature_gate']['per_parent_counts'][i] or tc!=prior['methods']['tta5']['per_parent_counts'][i]:raise ValueError('Per-parent reproduction mismatch')
                gate_counts.append(gc);tta_counts.append(tc)
                sharp,blur,num=native_probe(model,normalize(image));probe_views+=num
                for key,value,thr in [('sharp_at_gate',sharp,threshold),('blur_at_gate',blur,threshold),('sharp_at05',sharp,.5),('blur_at05',blur,.5)]:
                    probe_counts[key].append(base.parent_counts(value,gt,c,thr))
                parent_objects=analyze_parent(image,gt,c,views,p,q,t,gate,sharp,blur,threshold,region,index,seed)
                record_examples(out,parent_objects,sharp,blur)
                objects.extend(parent_objects)
            base.write(result_path,{'held_index':held,'held_out':region,'seed':seed,'completed_at':base.now(),
                'training_replay':training,'backbone_state_sha256':state_hash.hexdigest(),'cache_sha256':base.sha(cache_path),
                'original_indices':d['indices'],'saved_thresholds':{k:prior['methods'][k]['threshold'] for k in ('tta5','feature_gate')},
                'gate_metrics_reproduced':base.aggregate_counts(gate_counts),'tta_metrics_reproduced':base.aggregate_counts(tta_counts),
                'exact_prior_counts_reproduced':True,'probe_model_views':probe_views,'diagnostic_seconds':time.perf_counter()-started,
                'probe_metrics':{k:base.aggregate_counts(v) for k,v in probe_counts.items()},'objects':objects})
            del model;torch.cuda.empty_cache();completed+=1
            print('ATTRIBUTION_COMPLETED',completed,'/12',region,seed,'small_objects',len(objects),flush=True)
    result=summarize(out,protocol)
    print('COMPLETE',str({'misses':result['misses'],'categories':result['miss_categories'],'cache_bytes':result['cache_bytes']}),flush=True)


if __name__=='__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--stage',choices=('register','run','summarize'),default='run');a=p.parse_args()
    if a.stage=='register':register(a.output)
    elif a.stage=='summarize':summarize(a.output,register(a.output)[0])
    else:run(a.output)
