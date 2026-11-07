"""A new fixed subset of historical source validation plus exact runtime parity."""
import sys
import time
from pathlib import Path
from run_experiment import ROOT,HERE,OUT,BASE,read,write,sha,now,normalize,array_sha,metrics,torch
import numpy as np
from data_paths import resolve_data_root
from detector import LandslideDetector,ModelComponentError
from nir_synth_utils import load_synth
from rgb_completion_fusion import load_rgb_completion_fusion
from rgb_adapter import load_guarded_adapter


def main():
    root=OUT/'fixed_fusion_verification';root.mkdir(exist_ok=True)
    parent=read(OUT/'protocol.json');plan_path=root/'protocol.json'
    if not plan_path.exists():
        selection=[]
        for i,region in enumerate(parent['provenance']):
            row=region['splits']['val']
            used=set(row['indices']);pool=np.array([j for j in range(row['input_shape'][0]) if j not in used])
            rng=np.random.default_rng(2026101500+i)
            ids=rng.choice(pool,min(512,len(pool)),replace=False)
            selection.append({'region':region['region'],'input_path':row['input_path'],'label_path':row['label_path'],
                              'indices':ids.tolist(),'excluded_previous_selected_indices':row['indices']})
        write(plan_path,{'registered_at':now(),'selection':selection,'source_origin':'remaining original source validation rows; historical validation exposure not excluded',
          'not_independent_geography_or_blind_acceptance':True,'previous_development_results_observed':True,
          'candidates':['current_red_copy','legacy_full_completion','existing_rgb_guarded','fixed_completion_blend15'],
          'primary_threshold':.4,'secondary_threshold':.5,'fixed_blend':.15,'new_training':0,
          'decision_rule':'verification only; no retuning, no production default change; retain runnable offline fixed-fusion candidate if every source-region F1 delta >=-.02 and macro gain >=.01 vs current, while four old mine recall drops <=.02 and normal rises <=.02',
          'Noto_event_inputs':0,'unopened_mine_inputs':0})
    plan=read(plan_path)
    config_path=OUT/'rgb_completion_fusion.json'
    if not config_path.exists():
        write(config_path,{'schema_version':1,'input_kind':'rgb_proxy','patch_size':32,'threshold':.4,
              'numerical_protocol':{'dtype':'float32','cuda_tf32':False,'cudnn_deterministic':True},
              'patch_norm':'minmax','completion_weight':.15,'status':'offline_development_candidate',
              'production_registered':False,'independent_acceptance_completed':False,
              'components':{'current':{'path':str(BASE.relative_to(ROOT/'models_prior')),'sha256':sha(BASE)},
                 'completion':{'path':'nir_synth_seed42/best_model.pth','sha256':sha(ROOT/'models_prior/nir_synth_seed42/best_model.pth')}},
              'warning':'latent proxy is not measured NIR; no spectral index or physical/safety interpretation from proxy'})
    torch.set_num_threads(4);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    current=LandslideDetector(str(BASE),threshold=.4)
    legacy=load_synth(dev='cuda')
    guarded=load_guarded_adapter(ROOT/'models_prior/rgb_guarded_20261001/adapter.json',ROOT/'models_prior',device='cuda')
    groups=[];outputs={key:[] for key in plan['candidates']};labels=[];provenance=[]
    for region_index,row in enumerate(plan['selection']):
        if set(row['indices']) & set(row['excluded_previous_selected_indices']):raise ValueError('Additional validation rows overlap initial subset')
        raw=np.array(np.load(row['input_path'],mmap_mode='r')[row['indices']],dtype=np.float32)
        y=np.array(np.load(row['label_path'],mmap_mode='r')[row['indices']],dtype=np.float32).reshape(-1,32,32)
        x=normalize(raw);labels.append(y);groups.extend([region_index]*len(x))
        provenance.append({'region':row['region'],'selected_raw_sha256':array_sha(raw),'selected_labels_sha256':array_sha(y),'n':len(x)})
        for start in range(0,len(x),96):
            batch=x[start:start+96].copy();batch[:,3]=batch[:,2]
            with torch.no_grad():
                raw_prob=current.predict_batch(batch)
                tx=torch.as_tensor(batch,device='cuda')
                proxy=legacy(tx[:,:3]/255).clamp(0,1)
                mixed=batch.copy();mixed[:,3]=proxy[:,0].cpu().numpy()*255
                completed=current.predict_batch(mixed)
                outputs['current_red_copy'].append(raw_prob)
                outputs['legacy_full_completion'].append(completed)
                outputs['existing_rgb_guarded'].append(guarded.predict_batch(batch))
                outputs['fixed_completion_blend15'].append(.85*raw_prob+.15*completed)
    yy=np.concatenate(labels);gg=np.asarray(groups);source_results={}
    for key,chunks in outputs.items():
        prob=np.concatenate(chunks);values={}
        for t in [.4,.5]:
            regions=[metrics(prob[gg==g],yy[gg==g],t) for g in range(6)]
            values[str(t)]={'micro':metrics(prob,yy,t),'regions':regions,'macro_f1':float(np.mean([r['f1'] for r in regions]))}
        source_results[key]={'thresholds':values}
    del guarded;torch.cuda.empty_cache()
    # Match the callable candidate against the separately computed exact blend.
    candidate=load_rgb_completion_fusion(config_path,ROOT/'models_prior',device='cuda')
    batch=x[:37].copy();batch[:,3]=batch[:,2]
    with torch.no_grad():
        p0=current.predict_batch(batch);tx=torch.as_tensor(batch,device='cuda');proxy=legacy(tx[:,:3]/255).clamp(0,1)
        mixed=batch.copy();mixed[:,3]=proxy[:,0].cpu().numpy()*255
        expected=.85*p0+.15*current.predict_batch(mixed)
        observed=candidate.predict_batch(batch)
    max_error=float(np.max(np.abs(expected-observed)))
    if max_error>1e-6:raise ValueError('Candidate probability fusion differs from frozen evaluation')
    rejected_actual_nir=rejected_implicit_input=rejected_nested_component=False
    actual=batch.copy();actual[:,3]=actual[:,2]+1
    try:candidate.predict_batch(actual)
    except ValueError:rejected_actual_nir=True
    frame=batch[0].transpose(1,2,0)
    try:candidate.detect_image(frame,patch_norm='minmax')
    except ValueError:rejected_implicit_input=True
    try:candidate.detect_image(frame,input_kind='rgb_proxy',patch_norm='minmax',nir_synth_model=legacy)
    except ValueError:rejected_nested_component=True
    # A failed component must fail visibly rather than substitute copy-red output.
    original=candidate.completion
    class Broken(torch.nn.Module):
        def forward(self,x):raise RuntimeError('intentional verification failure')
    candidate.completion=Broken()
    rejected_failure=False
    try:candidate.predict_batch(batch)
    except ModelComponentError:rejected_failure=True
    candidate.completion=original
    if not all([rejected_actual_nir,rejected_implicit_input,rejected_nested_component,rejected_failure]):raise ValueError('Explicit-input/failure contract failed')
    # Same non-multiple dimensions, normalization, and padding as streaming service.
    frame=np.pad(frame,((0,5),(0,3),(0,0)),mode='reflect')
    t0=time.monotonic();p0=current.detect_image(frame,patch_norm='minmax',max_batch=7);base_s=time.monotonic()-t0
    p1=current.detect_image(frame,patch_norm='minmax',max_batch=7,nir_synth_model=legacy)
    t0=time.monotonic();observed=candidate.detect_image(frame,input_kind='rgb_proxy',patch_norm='minmax',max_batch=7);candidate_s=time.monotonic()-t0
    frame_error=float(np.max(np.abs(.85*p0+.15*p1-observed)))
    if frame_error>1e-6:raise ValueError('Streaming grid blend differs from offline comparison')
    result={'completed_at':now(),'protocol_sha256':sha(plan_path),'validation_patches':len(yy),
          'source_results':source_results,'selected_source_provenance':provenance,
          'runtime_checks':{'batch_probability_max_error':max_error,'streaming_probability_max_error':frame_error,
              'actual_nir_rejected':rejected_actual_nir,'implicit_input_rejected':rejected_implicit_input,
              'nested_synth_rejected':rejected_nested_component,'failed_component_not_silently_substituted':rejected_failure,
              'tiny_frame_seconds':{'current':base_s,'fusion':candidate_s},'performance_benchmark_claim':False},
          'new_fits':0,'new_unopened_mine_predictions':0,'Noto_inputs':0,'default_weight_changed':False}
    write(root/'results.json',result)
    print(__import__('json').dumps({'new_validation_patches':len(yy),'macro_f1':{key:r['thresholds']['0.4']['macro_f1'] for key,r in source_results.items()},
          'runtime_max_error':max(max_error,frame_error)}),flush=True)


if __name__=='__main__':
    sys.stdout.reconfigure(encoding='utf-8');main()
