"""Frozen candidate acceptance on additional, previously unselected val parents.

Same three geographic events; additional original parents, not new regions or
field incidents. No acceptance labels can fit weights, rules or thresholds.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
import sys
import numpy as np
import cv2

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from runtime_paths import configure_runtime_paths
configure_runtime_paths()
import component_filter as filt
import component_screen as screen

base=screen.base
PRIOR=ROOT/'experiments/component_screen_20261020'
METHODS=('tta5','multiscale_raw','area_guard','confidence_guard')


def select_additional(manifest,held_indices):
    seen={h for r in manifest['regions'] for h in r['splits']['val']['raw_parent_sha256']}
    trained={r['sha256'] for r in manifest['kept_train']}
    grouped=defaultdict(list);excluded=[]
    for g in held_indices:
        region=manifest['regions'][g];selected=set(region['splits']['val']['indices'])
        for i,h in enumerate(region['all_official_val_image_sha256']):
            if i in selected:continue
            row={'group':g,'index':i,'image_sha256':h}
            if h in seen or h in trained:
                excluded.append({**row,'reason':'matches_prior_selected_or_train'});continue
            grouped[h].append(row)
    keep=[]
    for rows in grouped.values():
        if len({r['group'] for r in rows})>1:
            excluded.extend({**r,'reason':'cross_region_duplicate'} for r in rows);continue
        rows.sort(key=lambda r:r['index']);keep.append(rows[0])
        excluded.extend({**r,'reason':'same_region_duplicate'} for r in rows[1:])
    return sorted(keep,key=lambda r:(r['group'],r['index'])),excluded


def register(out):
    out=Path(out).resolve()
    if out.drive.upper()!='D:' or not out.is_relative_to(ROOT/'experiments'):
        raise ValueError('Use D:/LMHLD/experiments')
    out.mkdir(parents=True,exist_ok=True)
    verified=base.read(PRIOR/'verification.json')
    if verified['status']!='passed' or verified['verified_fits']!=6:
        raise ValueError('Verify development component experiment first')
    prior=base.read(PRIOR/'protocol.json');manifest=base.read(screen.PRIOR/'data_manifest.json')
    frozen=dict(prior['frozen_assets'])
    for p in (Path(__file__),ROOT/'tests/test_component_acceptance.py',PRIOR/'protocol.json',
              PRIOR/'results.json',PRIOR/'verification.json'):
        frozen[str(p)]=base.sha(p)
    for g in prior['held_indices']:
        label_path=Path(manifest['regions'][g]['splits']['val']['label_path'])
        if label_path.name!='val_labels.npy':raise ValueError('Acceptance uses validation files only')
        frozen[str(label_path)]=base.sha(label_path)  # Byte integrity only; no label-derived choice.
        for seed in prior['seeds']:
            folder=PRIOR/f'fold_{g}_{base.REGIONS[g][0]}'/f'seed{seed}'
            frozen[str(folder/'filter_policy.json')]=base.sha(folder/'filter_policy.json')
    for p,h in frozen.items():
        if base.sha(p)!=h:raise ValueError('Frozen dependency changed')
    samples,excluded=select_additional(manifest,prior['held_indices'])
    result={'registered_at':base.now(),'frozen_assets':frozen,'held_indices':prior['held_indices'],
        'seeds':prior['seeds'],'primary_method':'area_guard','methods':list(METHODS),
        'candidate_selection':'Area guard chosen from the completed development comparison. No further model/method/cutoff changes based on acceptance outcomes.',
        'sample_selection':'All unselected validation indices of the same held regions; exclude image hashes appearing in any prior selected val or train, deduplicate before reading label content.',
        'samples':samples,'excluded_samples':excluded,'different_parents':len(samples),
        'limits':{k:v for k,v in prior['diagnostic_gate'].items() if k!='min_extra_false_component_reduction_vs_confidence_each_seed'},
        'extra_limits':{'min_cover_gain_vs_tta_each_seed':.02,'min_iou_gain_vs_tta_each_seed':.02},
        'frozen_policies':'Use each old fold/seed policy unchanged; no acceptance-image calibration or source statistics recomputed.',
        'scope':'Additional parents unused by the current training/selection/calibration runners, but same three regions/events. Earlier project-wide exposure and geographic adjacency not independently established. Not new-region or field blind validation.',
        'official_test_predictions':0,'new_mine_predictions':0,'external_calls':0,'new_cnn_training_updates':0,
        'production_changed':False,'inference_budget':'Five views per multiscale method; separate TTA5 comparator shares its global view, nine total extracted views for the comparison.'}
    path=out/'protocol.json'
    if path.exists():
        previous=base.read(path)
        if {k:v for k,v in previous.items() if k!='registered_at'}!={k:v for k,v in result.items() if k!='registered_at'}:
            raise ValueError('Acceptance protocol changed; no adaptive reruns')
        return previous
    base.write(path,result);return result


def assess(means,worst_cover_loss,limits,extra):
    a,b,t=means['area_guard'],means['multiscale_raw'],means['tta5']
    reduction=1-a['false_components_per_mpix']/max(b['false_components_per_mpix'],1e-12)
    checks={'false_components_reduced':reduction>=limits['min_false_component_reduction_each_seed'],
        'coverage_preserved':a['small_cover50']>=b['small_cover50']-limits['max_cover_loss_each_seed'],
        'region_coverage_preserved':worst_cover_loss<=limits['max_region_cover_loss'],
        'iou_preserved':a['small_iou50']>=b['small_iou50']-limits['max_iou_loss_each_seed'],
        'f1_preserved':a['f1']>=b['f1']-limits['max_f1_loss_each_seed'],
        'near_tta_candidate_count':a['false_components_per_mpix']<=limits['max_false_component_ratio_vs_tta_each_seed']*t['false_components_per_mpix'],
        'coverage_gain_vs_tta':a['small_cover50']>=t['small_cover50']+extra['min_cover_gain_vs_tta_each_seed'],
        'iou_gain_vs_tta':a['small_iou50']>=t['small_iou50']+extra['min_iou_gain_vs_tta_each_seed']}
    return reduction,checks


def summarize(out,protocol):
    rows=[base.read(p) for p in sorted(Path(out).glob('fold_*/seed*/result.json'))]
    if len(rows)!=6:raise ValueError('Require all six frozen acceptance runs')
    metrics=('f1','small_cover50','small_iou50','background_fpr','false_components_per_mpix')
    def macro(sub,method):
        return {m:float(np.mean([r['methods'][method]['held_out'][m] for r in sub])) for m in metrics}
    means={method:macro(rows,method) for method in METHODS};by_seed=[]
    for seed in protocol['seeds']:
        sub=[r for r in rows if r['seed']==seed];mm={method:macro(sub,method) for method in METHODS}
        worst=max(r['methods']['multiscale_raw']['held_out']['small_cover50']-r['methods']['area_guard']['held_out']['small_cover50'] for r in sub)
        reduction,checks=assess(mm,worst,protocol['limits'],protocol['extra_limits'])
        by_seed.append({'seed':seed,'means':mm,'false_component_reduction':reduction,
                        'worst_region_cover_loss':worst,'checks':checks})
    for p,h in protocol['frozen_assets'].items():
        if base.sha(p)!=h:raise ValueError('Frozen acceptance dependency changed')
    passed=all(ok for r in by_seed for ok in r['checks'].values())
    result={'completed_at':base.now(),'status':'completed','different_validation_parents':protocol['different_parents'],
        'validation_parent_seed_predictions':sum(r['target_rows'] for r in rows),
        'target_macro':means,'by_seed':by_seed,'acceptance_passed':passed,'primary_method':'area_guard',
        'scope':protocol['scope'],'official_test_predictions':0,'new_mine_predictions':0,
        'new_cnn_training_updates':0,'external_calls':0,'production_changed':False}
    base.write(Path(out)/'results.json',result)
    base.write(Path(out)/'state.json',{k:v for k,v in result.items() if k not in ('target_macro','by_seed')})
    return result


def run(out):
    protocol=register(out);torch=screen.parent.setup_runtime();manifest=base.read(screen.PRIOR/'data_manifest.json')
    import segmentation_models_pytorch as smp
    completed=0
    for held in protocol['held_indices']:
        region=manifest['regions'][held];meta=region['splits']['val']
        if Path(meta['image_path']).name!='val_images.npy':raise ValueError('Only additional val images permitted')
        samples=[r for r in protocol['samples'] if r['group']==held]
        indices=np.array([r['index'] for r in samples],np.int64)
        raw=np.load(meta['image_path'],mmap_mode='r');labels=np.load(meta['label_path'],mmap_mode='r')
        views=[];truth=[];label_hashes=[]
        for s in samples:
            x,y=raw[s['index']],labels[s['index']]
            if base.array_sha(x)!=s['image_sha256']:raise ValueError('Frozen additional image changed')
            if not np.isfinite(y).all() or not ((y==0)|(y==1)).all():raise ValueError('Invalid additional label; abort acceptance')
            views.append(base.make_views(x)[0]);truth.append(y[0].astype(np.uint8));label_hashes.append(base.array_sha(y))
        data={'views':np.stack(views),'labels':truth}
        for seed in protocol['seeds']:
            directory=Path(out)/f'fold_{held}_{base.REGIONS[held][0]}'/f'seed{seed}'
            directory.mkdir(parents=True,exist_ok=True)
            if (directory/'result.json').exists():
                prior=base.read(directory/'result.json')
                if base.sha(directory/'prediction_grids.npz')!=prior['prediction_grids_sha256']:
                    raise ValueError('Completed acceptance cache changed')
                completed+=1;continue
            source=PRIOR/directory.parent.name/directory.name
            policy=base.read(source/'filter_policy.json')
            baseline=base.read(screen.PRIOR/directory.parent.name/directory.name/'scale_result.json')
            base.write(Path(out)/'state.json',{'status':'running','updated_at':base.now(),'held_out':base.REGIONS[held][0],
                'seed':seed,'completed':completed,'total':6,'different_parents':protocol['different_parents']})
            model=smp.UnetPlusPlus(encoder_name='resnet34',encoder_weights=None,in_channels=4,classes=1).cuda()
            if base.sha(policy['base_weight'])!=policy['base_weight_sha256']:raise ValueError('Frozen weight changed')
            model.load_state_dict(torch.load(policy['base_weight'],map_location='cpu',weights_only=True));model.eval()
            if screen.parent.short.state_hash(model)!=policy['base_state_sha256']:raise ValueError('Model state mismatch')
            native,grids,seconds=screen.parent.component_grids(model,data)
            counts={m:[] for m in METHODS};decisions={}
            for i,y in enumerate(truth):
                gc=base.gt_components(y,region['gsd_m'])
                counts['tta5'].append(base.parent_counts(native['tta5'][i],y,gc,baseline['methods']['tta5']['threshold']))
                pred=filt.extract_components(native['global'][i],native['local'][i],policy['pixel_threshold'],policy['anchor_threshold'],region['gsd_m'])
                for method in METHODS[1:]:
                    keep=screen.selected_keep(pred,policy['classifier'],policy['policies'],method)
                    decisions[f'p{i}_{method}']=keep
                    counts[method].append(base.parent_counts(filt.component_mask(pred,keep),y,gc,.5))
                for method in METHODS[2:]:
                    if any(counts[method][-1][k]>counts['multiscale_raw'][-1][k] for k in ('tp','fp','small_cover50','false_components_ge2px')):
                        raise ValueError('Whole-component monotonicity failed')
            grids['original_target_indices']=indices
            np.savez_compressed(directory/'prediction_grids.npz',**grids)
            np.savez_compressed(directory/'component_decisions.npz',**decisions)
            methods={m:{'held_out':base.aggregate_counts(c),'per_parent_counts':c} for m,c in counts.items()}
            base.write(directory/'result.json',{'held_index':held,'held_out':base.REGIONS[held][0],'seed':seed,'completed_at':base.now(),
                'target_rows':len(indices),'original_indices':indices.tolist(),'label_sha256':label_hashes,
                'methods':methods,'policy_sha256':base.sha(source/'filter_policy.json'),
                'prediction_grids_sha256':base.sha(directory/'prediction_grids.npz'),
                'component_decisions_sha256':base.sha(directory/'component_decisions.npz'),
                'cnn_comparison_seconds':seconds,'no_acceptance_calibration':True,'primary_method':'area_guard'})
            del model,native,grids;torch.cuda.empty_cache();completed+=1
            print('ADDITIONAL_ACCEPTANCE',completed,'/6',base.REGIONS[held][0],seed,'parents',len(indices),flush=True)
    result=summarize(out,protocol)
    print('ACCEPTANCE_COMPLETE',{'passed':result['acceptance_passed'],'macro':result['target_macro']},flush=True)


if __name__=='__main__':
    sys.stdout.reconfigure(encoding='utf-8');p=argparse.ArgumentParser();p.add_argument('--output',required=True)
    p.add_argument('--stage',choices=('register','run'),default='run');a=p.parse_args()
    if a.stage=='register':register(a.output)
    else:run(a.output)
