"""Source-only candidate screening; reuses frozen CNN weights and target grids."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

import numpy as np
import cv2

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from runtime_paths import configure_runtime_paths
configure_runtime_paths()
import component_filter as filt
import full_source_training as parent

base=parent.base
PRIOR=ROOT/'experiments/full_source_20261020'
ARMS=('area_guard','confidence_guard','learned_guard','learned_no_guard')
METHODS=('tta5','multiscale_raw','anchor_only',*ARMS)
CAL={'macro_cover_loss':.005,'region_cover_loss':.02,'macro_f1_loss':.005,
     'macro_iou_loss':.005,'region_iou_loss':.02,'quantile_intervals':100}


def register(out):
    out=Path(out).resolve()
    if out.drive.upper()!='D:' or not out.is_relative_to(ROOT/'experiments'):
        raise ValueError('Use D:/LMHLD/experiments')
    out.mkdir(parents=True,exist_ok=True)
    predecessor=base.read(PRIOR/'protocol.json');frozen=dict(predecessor['frozen_assets'])
    for p in (Path(__file__),ROOT/'component_filter.py',ROOT/'tests/test_component_screen.py',
              PRIOR/'protocol.json',PRIOR/'data_manifest.json',PRIOR/'results.json',PRIOR/'scale_results.json'):
        frozen[str(p)]=base.sha(p)
    for g in predecessor['held_indices']:
        for seed in predecessor['seeds']:
            d=PRIOR/f'fold_{g}_{base.REGIONS[g][0]}'/f'seed{seed}'
            for name in ('result.json','scale_result.json','scale_heads.json','source_selected.pth',
                         'scale_target_grids.npz','scale_calibration_histograms.npz'):
                frozen[str(d/name)]=base.sha(d/name)
    for p,h in frozen.items():
        if base.sha(p)!=h:raise ValueError(f'Frozen predecessor changed: {p}')
    path=out/'protocol.json'
    if path.exists():
        result=base.read(path)
        if result['frozen_assets']!=frozen:raise ValueError('Frozen experiment changed; use a new directory')
        return result
    result={'registered_at':base.now(),'frozen_assets':frozen,'held_indices':predecessor['held_indices'],
        'seeds':predecessor['seeds'],'fits':6,'question':'Can source-calibrated whole-component screening remove multiscale background fragments while retaining its small-object gains?',
        'base_maps':'Frozen selected backbone; one global+four quadrant views; 50:50 global/local probability mean and original source2% pixel threshold.',
        'anchor':'At least one component pixel reaches the source-calibrated2% global-single-view threshold. Guarded methods retain such components unconditionally.',
        'tiny_guard':'Components of one pixel are retained; matches predecessor false-component metric which counts area>=2. No component splitting, filling or merging.',
        'classifier':'Source-only region/class-balanced L2 logistic regression C1, 16 probability/shape features, source region-balanced mean/std, clip+/-8. Any GT overlap positive, zero GT overlap negative.',
        'fit_and_calibration':'Reuse original source selection half for classifier fitting; other half calibrates pixel anchors and component cutoff. Prior pixel mean threshold is frozen. Joint reuse of calibration is explicit; no formal risk guarantee.',
        'features':list(filt.FEATURES),'feature_exclusions':'No truth, region IDs, coordinates, dates, raw appearance or target normalization statistics in inference features.',
        'source_calibration':dict(CAL),'cutoff_choice':'101 source score quantiles plus keep-all and remove-all. Minimize equal-region false components/Mpx under source coverage/F1 constraints, ties lower cutoff; check exact strict IoU and back off to next feasible cutoff.',
        'main_method':'learned_guard','methods':list(METHODS),
        'ablation':'learned_no_guard uses the identical classifier and its own source-calibrated cutoff, with only global anchor protection disabled.',
        'diagnostic_gate':{'min_false_component_reduction_each_seed':.20,'max_cover_loss_each_seed':.01,
            'max_region_cover_loss':.03,'max_iou_loss_each_seed':.01,'max_f1_loss_each_seed':.005,
            'max_false_component_ratio_vs_tta_each_seed':1.10,
            'min_extra_false_component_reduction_vs_confidence_each_seed':.05},
        'target_access':'Persist source fit/calibration policy before loading target grids/labels in this round. Existing target development results are exposed history, never fit/calibrate on held region.',
        'inference_budget':'Primary uses the existing five CNN views plus CPU CCL/features/logistic scoring; no extra CNN views. Cached TTA5 is a separate five-view reference. Component preparation and scoring timings reported separately.',
        'scope':'Three historically exposed regions x2 seeds, not new blind, all-six-region, mine accident boundary or field acceptance. Standard component filtering, no original-method claim.',
        'retention':'Small JSON coefficients/calibration, compressed source component maps/features for replay, target keep decisions and results only. No new CNN weights or raw-image copies.',
        'production_changed':False,'official_test_predictions':0,'new_mine_predictions':0,
        'external_calls':0,'downloads':0,'new_cnn_training_updates':0}
    base.write(path,result);return result


def full_calibration_metrics(pool,keep):
    rows={g:[] for g in pool.groups};offset=0
    for c in pool.cases:
        n=len(c.prediction.areas)
        mask=filt.component_mask(c.prediction,keep[offset:offset+n]);offset+=n
        rows[c.group].append(base.parent_counts(mask,c.truth,c.gt_components,.5))
    result={str(g):base.aggregate_counts(v) for g,v in rows.items()}
    valid=[r for r in result.values() if r['small_cover50'] is not None]
    return {'regions':result,'macro_f1':float(np.mean([r['f1'] for r in result.values()])),
            'macro_cover50':float(np.mean([r['small_cover50'] for r in valid])) if valid else 0.,
            'macro_iou50':float(np.mean([r['small_iou50'] for r in valid])) if valid else 0.,
            'macro_false_components_per_mpix':float(np.mean([r['false_components_per_mpix'] for r in result.values()]))}


def assert_fast_matches_exact(fast,exact):
    for g,r in fast['regions'].items():
        e=exact['regions'][g]
        for k in ('tp','fp','fn','tn','small_n'):
            if r[k]!=e['counts'][k]:raise ValueError('Fast calibration counts differ')
        if r['small_cover50_count']!=e['counts']['small_cover50']:raise ValueError('Fast coverage differs')
        if abs(r['false_components_per_mpix']-e['false_components_per_mpix'])>1e-9:
            raise ValueError('Fast false component rate differs')


def fast_feasible(candidate,baseline,limits):
    if candidate['macro_f1']<baseline['macro_f1']-limits['macro_f1_loss']-1e-12:return False
    if candidate['macro_cover50']<baseline['macro_cover50']-limits['macro_cover_loss']-1e-12:return False
    for g,b in baseline['regions'].items():
        a=candidate['regions'][g]
        if b['small_cover50'] is not None and a['small_cover50']<b['small_cover50']-limits['region_cover_loss']-1e-12:return False
    return True


def strict_feasible(candidate,baseline,limits):
    if candidate['macro_iou50']<baseline['macro_iou50']-limits['macro_iou_loss']-1e-12:return False
    for g,b in baseline['regions'].items():
        if b['small_iou50'] is not None and candidate['regions'][g]['small_iou50']<b['small_iou50']-limits['region_iou_loss']-1e-12:return False
    return True


def calibrate(pool,scores,protect_anchor,baseline_exact,limits,force_keep_all=False):
    scores=np.asarray(scores,np.float64)
    if scores.shape!=pool.area.shape or not np.isfinite(scores).all() or np.any(scores<0):
        raise ValueError('Invalid source component scores')
    guard=pool.guard if protect_anchor else pool.area<2
    eligible=scores[~guard]
    if not len(eligible) or force_keep_all:
        cuts=np.array([-1.],np.float64)
    else:
        cuts=np.unique(np.r_[-1.,np.quantile(eligible,np.linspace(0,1,limits['quantile_intervals']+1)),np.nextafter(eligible.max(),np.inf)])
    baseline_fast=pool.summary(np.ones(len(scores),bool));assert_fast_matches_exact(baseline_fast,baseline_exact)
    candidates=[];trace=[]
    for cutoff in cuts:
        keep=guard|(scores>=cutoff);summary=pool.summary(keep)
        ok=fast_feasible(summary,baseline_fast,limits)
        trace.append({'cutoff':float(cutoff),'macro_f1':summary['macro_f1'],
            'macro_cover50':summary['macro_cover50'],'macro_false_components_per_mpix':summary['macro_false_components_per_mpix'],'fast_feasible':ok})
        if ok:candidates.append((summary['macro_false_components_per_mpix'],float(cutoff),summary))
    exact_attempts=[]
    for _,cutoff,fast in sorted(candidates,key=lambda v:(v[0],v[1])):
        keep=guard|(scores>=cutoff);exact=full_calibration_metrics(pool,keep);assert_fast_matches_exact(fast,exact)
        ok=strict_feasible(exact,baseline_exact,limits)
        exact_attempts.append({'cutoff':cutoff,'macro_iou50':exact['macro_iou50'],'strict_feasible':ok})
        if ok:
            return {'cutoff':cutoff,'protect_anchor':protect_anchor,'calibration':exact,
                    'source_threshold_candidates':trace,'exact_strict_attempts':exact_attempts,
                    'target_used':False}
    raise ValueError('Keep-all source calibration fallback failed')


def source_group(manifest,g):
    v=manifest['regions'][g]['splits']['val']
    x=np.load(v['image_path'],mmap_mode='r');y=np.load(v['label_path'],mmap_mode='r')
    views=[];labels=[]
    for i,hx,hy in zip(v['indices'],v['raw_parent_sha256'],v['label_parent_sha256']):
        if base.array_sha(x[i])!=hx or base.array_sha(y[i])!=hy:raise ValueError('Frozen source row changed')
        views.append(base.make_views(x[i])[0]);labels.append(y[i,0].astype(np.uint8))
    return {'views':np.stack(views),'labels':labels,'indices':v['indices'],
            'components':[base.gt_components(a,manifest['regions'][g]['gsd_m']) for a in labels]}


def native_maps(global_grids,quad_grids,side):
    p=[];q=[];half=side//2
    for globalp,quad in zip(global_grids,quad_grids):
        local=np.empty((side,side),np.float32)
        for i in range(4):
            y,x=i//2*half,i%2*half
            local[y:y+half,x:x+half]=cv2.resize(quad[i],(half,half),interpolation=cv2.INTER_LINEAR)
        p.append(cv2.resize(globalp,(side,side),interpolation=cv2.INTER_LINEAR));q.append(local)
    return p,q


def source_predictions(model,data):
    p=parent.short.forward(model,data['views'][:,0])
    q=parent.short.forward(model,data['views'][:,1:].reshape(-1,4,64,64)).reshape(-1,4,64,64)
    return native_maps(p,q,len(data['labels'][0]))


def build_source_cases(data,predictions,manifest,held,pixel_threshold,anchor_threshold):
    fit={};cal={};packed={};records=[]
    for g,d in sorted(data.items()):
        a,b=parent.gate.meta_split(len(d['labels']),g);fit[g]=[];cal[g]=[];fitset=set(a.tolist())
        for i,(p,q,y,c) in enumerate(zip(*predictions[g],d['labels'],d['components'])):
            pred=filt.extract_components(p,q,pixel_threshold,anchor_threshold,manifest['regions'][g]['gsd_m'])
            obs=filt.observed(pred,y,c,g,i);split='fit' if i in fitset else 'calibration'
            (fit[g] if split=='fit' else cal[g]).append(obs)
            key=f'g{g}_p{i}'
            packed[key+'_labels']=pred.labels;packed[key+'_features']=pred.features;packed[key+'_anchor']=pred.anchored
            records.append({'key':key,'group':g,'position':i,'original_index':d['indices'][i],'split':split,
                            'label_sha256':manifest['regions'][g]['splits']['val']['label_parent_sha256'][i]})
    filt.source_keys(fit,held);filt.source_keys(cal,held)
    return fit,cal,packed,records


def score_cases(cases_by_group,classifier,arm):
    return np.concatenate([filt.component_scores(c.prediction,classifier,arm) for _,cases in sorted(cases_by_group.items()) for c in cases])


def selected_keep(pred,classifier,policies,method):
    if method=='multiscale_raw':return np.ones(len(pred.areas),bool)
    if method=='anchor_only':return pred.anchored|(pred.areas<2)
    if method not in ARMS:raise ValueError('Unknown screening method')
    policy=policies[method]
    return filt.keep_components(pred,filt.component_scores(pred,classifier,method),policy['cutoff'],policy['protect_anchor'])


def run_fit(out,manifest,held,seed,protocol):
    import torch
    import segmentation_models_pytorch as smp
    name=base.REGIONS[held][0];directory=Path(out)/f'fold_{held}_{name}'/f'seed{seed}'
    directory.mkdir(parents=True,exist_ok=True);previous=PRIOR/directory.parent.name/directory.name
    old=base.read(previous/'scale_result.json');original=base.read(previous/'result.json')
    choice=old['backbone_choice'];weight=Path(choice['chosen_weight'])
    expected=base.read(previous/'scale_heads.json')
    if base.sha(weight)!=expected['weights_sha256']:raise ValueError('Frozen backbone identity changed')
    model=smp.UnetPlusPlus(encoder_name='resnet34',encoder_weights=None,in_channels=4,classes=1).cuda()
    model.load_state_dict(torch.load(weight,map_location='cpu',weights_only=True));model.eval()
    if parent.short.state_hash(model)!=expected['base_state_sha256']:raise ValueError('Frozen model state changed')
    started=time.perf_counter();data={};predictions={};global_hist={};mean_hist={}
    for g in range(6):
        if g==held:continue
        data[g]=source_group(manifest,g);predictions[g]=source_predictions(model,data[g])
        pos=parent.gate.meta_split(len(data[g]['labels']),g)[1]
        labels=[data[g]['labels'][i] for i in pos]
        p,q=predictions[g]
        global_hist[g]=parent.source_histogram([p[i] for i in pos],labels)
        mean_hist[g]=parent.source_histogram([.5*(p[i]+q[i]) for i in pos],labels)
    anchor_threshold,anchor_cal=parent.threshold_from_histograms(global_hist,held,.02)
    pixel_threshold,pixel_cal=parent.threshold_from_histograms(mean_hist,held,.02)
    if pixel_threshold!=old['methods']['mean_multiscale5']['threshold'] or pixel_cal!=old['methods']['mean_multiscale5']['calibration']:
        raise ValueError('Multiscale source threshold did not reproduce')
    fit,cal,packed,records=build_source_cases(data,predictions,manifest,held,pixel_threshold,anchor_threshold)
    classifier=filt.fit_source_classifier(fit,held)
    pool=filt.CalibrationPool(cal,held)
    baseline=full_calibration_metrics(pool,np.ones(len(pool.area),bool))
    policies={}
    for arm in ARMS:
        scores=score_cases(cal,classifier,arm)
        policies[arm]=calibrate(pool,scores,arm!='learned_no_guard',baseline,protocol['source_calibration'],
            force_keep_all=classifier['kind']=='keep_all' and arm.startswith('learned'))
    np.savez_compressed(directory/'source_components.npz',**packed)
    snapshot={'frozen_at':base.now(),'held_index':held,'seed':seed,'source_groups':sorted(data),
        'pixel_threshold':pixel_threshold,'pixel_calibration':pixel_cal,
        'anchor_threshold':anchor_threshold,'anchor_calibration':anchor_cal,
        'classifier':classifier,'policies':policies,'baseline_source_calibration':baseline,
        'source_components_sha256':base.sha(directory/'source_components.npz'),'source_records':records,
        'base_weight':str(weight),'base_weight_sha256':expected['weights_sha256'],
        'base_state_sha256':expected['base_state_sha256'],'source_seconds':time.perf_counter()-started,
        'target_used':False,'protocol_sha256':base.sha(Path(out)/'protocol.json')}
    base.write(directory/'filter_policy.json',snapshot)
    del model,data,predictions,fit,cal,packed,pool;torch.cuda.empty_cache()
    # This round's target maps/labels are opened only after the source policy is saved.
    target_opened_at=base.now();cache=np.load(previous/'scale_target_grids.npz')
    v=manifest['regions'][held]['splits']['val'];indices=np.array(v['indices'],np.int64)
    if not np.array_equal(cache['original_target_indices'],indices):raise ValueError('Target grid order changed')
    labels_file=np.load(v['label_path'],mmap_mode='r');side=v['image_shape'][-1]
    global_maps,local_maps=native_maps(cache['global_grid'],cache['local_grids'],side)
    counts={method:[] for method in METHODS};decisions={};component_timing=0.;decision_timing=0.
    for i,(p,q) in enumerate(zip(global_maps,local_maps)):
        started=time.perf_counter()
        pred=filt.extract_components(p,q,pixel_threshold,anchor_threshold,manifest['regions'][held]['gsd_m'])
        component_timing+=time.perf_counter()-started
        gt=labels_file[indices[i]]
        if base.array_sha(gt)!=v['label_parent_sha256'][i]:raise ValueError('Target label changed')
        truth=gt[0].astype(np.uint8);components=base.gt_components(truth,manifest['regions'][held]['gsd_m'])
        tta=cv2.resize(cache['tta_grid'][i],(side,side),interpolation=cv2.INTER_LINEAR)
        counts['tta5'].append(base.parent_counts(tta,truth,components,old['methods']['tta5']['threshold']))
        for method in METHODS[1:]:
            started=time.perf_counter();keep=selected_keep(pred,classifier,policies,method)
            mask=filt.component_mask(pred,keep);decision_timing+=time.perf_counter()-started
            decisions[f'p{i}_{method}']=keep
            counts[method].append(base.parent_counts(mask,truth,components,.5))
            if np.any(mask&~(pred.labels>0)):raise ValueError('Screening introduced foreground')
    methods={method:{'held_out':base.aggregate_counts(rows),'per_parent_counts':rows} for method,rows in counts.items()}
    for method,old_name in (('tta5','tta5'),('multiscale_raw','mean_multiscale5')):
        if methods[method]['held_out']!=old['methods'][old_name]['held_out'] or methods[method]['per_parent_counts']!=old['methods'][old_name]['per_parent_counts']:
            raise ValueError('Frozen baseline did not reproduce')
    for method in METHODS[2:]:
        for before,after in zip(counts['multiscale_raw'],counts[method]):
            if any(after[k]>before[k] for k in ('fp','tp','small_cover50','false_components_ge2px')):
                raise ValueError('Whole-component removal monotonicity violated')
    np.savez_compressed(directory/'target_decisions.npz',**decisions)
    base.write(directory/'result.json',{'held_index':held,'held_out':name,'seed':seed,'completed_at':base.now(),
        'target_opened_at':target_opened_at,'policy_frozen_at':snapshot['frozen_at'],
        'policy_sha256':base.sha(directory/'filter_policy.json'),'target_decisions_sha256':base.sha(directory/'target_decisions.npz'),
        'source_component_cache_sha256':snapshot['source_components_sha256'],'methods':methods,
        'baseline_reproduced':True,'component_preparation_seconds':component_timing,
        'all_six_decision_methods_seconds':decision_timing,'target_rows':len(indices),
        'target_not_used_for_fit_or_calibration':True})
    cache.close()
    print('COMPONENT_COMPLETED',name,seed,'cutoff',policies['learned_guard']['cutoff'],flush=True)


def summarize(out,protocol):
    rows=[base.read(p) for p in sorted(Path(out).glob('fold_*/seed*/result.json'))]
    if len(rows)!=6:raise ValueError('Require all six component fits')
    metrics=('f1','small_cover50','small_iou50','background_fpr','false_components_per_mpix')
    macro={m:{k:float(np.mean([r['methods'][m]['held_out'][k] for r in rows])) for k in metrics} for m in METHODS}
    by_seed=[];limits=protocol['diagnostic_gate']
    for seed in protocol['seeds']:
        sub=[r for r in rows if r['seed']==seed]
        means={m:{k:float(np.mean([r['methods'][m]['held_out'][k] for r in sub])) for k in metrics} for m in METHODS}
        a,b=means['learned_guard'],means['multiscale_raw'];ref=means['tta5'];conf=means['confidence_guard']
        reduction=1-a['false_components_per_mpix']/max(b['false_components_per_mpix'],1e-12)
        extra=(conf['false_components_per_mpix']-a['false_components_per_mpix'])/max(b['false_components_per_mpix'],1e-12)
        worst_loss=max(r['methods']['multiscale_raw']['held_out']['small_cover50']-r['methods']['learned_guard']['held_out']['small_cover50'] for r in sub)
        checks={'false_component_reduction':reduction>=limits['min_false_component_reduction_each_seed'],
            'cover_preserved':a['small_cover50']>=b['small_cover50']-limits['max_cover_loss_each_seed'],
            'region_cover_preserved':worst_loss<=limits['max_region_cover_loss'],
            'iou_preserved':a['small_iou50']>=b['small_iou50']-limits['max_iou_loss_each_seed'],
            'f1_preserved':a['f1']>=b['f1']-limits['max_f1_loss_each_seed'],
            'near_tta_candidate_count':a['false_components_per_mpix']<=limits['max_false_component_ratio_vs_tta_each_seed']*ref['false_components_per_mpix'],
            'beats_simple_confidence':extra>=limits['min_extra_false_component_reduction_vs_confidence_each_seed']}
        by_seed.append({'seed':seed,'means':means,'false_component_reduction':reduction,
            'extra_reduction_vs_confidence':extra,'worst_region_cover_loss':worst_loss,'checks':checks})
    for p,h in protocol['frozen_assets'].items():
        if base.sha(p)!=h:raise ValueError('Frozen/protected asset changed')
    passed=all(ok for r in by_seed for ok in r['checks'].values())
    result={'completed_at':base.now(),'status':'completed','fits':6,'target_macro':macro,
        'by_seed':by_seed,'diagnostic_gate_passed':passed,'scope':protocol['scope'],
        'production_changed':False,'official_test_predictions':0,'new_mine_predictions':0,
        'external_calls':0,'new_cnn_training_updates':0}
    base.write(Path(out)/'results.json',result)
    base.write(Path(out)/'state.json',{k:v for k,v in result.items() if k not in ('target_macro','by_seed')})
    return result


def run(out):
    protocol=register(out);torch=parent.setup_runtime();manifest=base.read(PRIOR/'data_manifest.json');completed=0
    for g in protocol['held_indices']:
        for seed in protocol['seeds']:
            directory=Path(out)/f'fold_{g}_{base.REGIONS[g][0]}'/f'seed{seed}'
            if (directory/'result.json').exists():
                old=base.read(directory/'result.json')
                if base.sha(directory/'filter_policy.json')!=old['policy_sha256']:raise ValueError('Completed policy changed')
                completed+=1;continue
            base.write(Path(out)/'state.json',{'status':'running','updated_at':base.now(),'held_out':base.REGIONS[g][0],
                'seed':seed,'completed':completed,'total':6,'production_changed':False})
            run_fit(out,manifest,g,seed,protocol);completed+=1
    result=summarize(out,protocol)
    print('COMPONENT_COMPLETE',{'gate':result['diagnostic_gate_passed'],'macro':result['target_macro']},flush=True)


if __name__=='__main__':
    sys.stdout.reconfigure(encoding='utf-8');p=argparse.ArgumentParser();p.add_argument('--output',required=True)
    p.add_argument('--stage',choices=('register','run'),default='run');a=p.parse_args()
    if a.stage=='register':register(a.output)
    else:run(a.output)
