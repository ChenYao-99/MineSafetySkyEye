"""One negative-reference quantile rule, never a threshold sweep or blind selection."""
from pathlib import Path
from datetime import datetime,timezone,timedelta
import json,sys
ROOT=Path('D:/LMHLD');sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'validation'))
from runtime_paths import configure_runtime_paths
configure_runtime_paths()
import numpy as np
from model_registry import file_sha256
from mine_data_protocol import verify_assets,registry,reserved_case_ids
SOURCE=ROOT/'experiments/fixed_temporal_mean_20261005';BASE=ROOT/'experiments/cause_terrain_round_20261005'
OUT=ROOT/'experiments/global_work_reference_calibration_20261005'

def load(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def write(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')

def run():
    OUT.mkdir(exist_ok=True);verify_assets();v=registry();assert all(v['cases'][c]['model_predictions_run']==0 for c in reserved_case_ids(v))
    source=load(SOURCE/'results.json');cases=[row['held'] for row in source['cases']]
    plan={'registered_at':datetime.now(timezone(timedelta(hours=8))).isoformat(),
          'role':'post-result global operatingpoint calibration on alreadyobserved development;not a new method ablation or blindtest',
          'candidate_equalweight_model_scores_sha256':file_sha256(SOURCE/'results.json'),
          'implementation_sha256':file_sha256(__file__),'cases':cases,
          'criterion':'max of per-mine 90thpercentile reviewednormal scores using higher quantile;onefloat32 successor for tie exclusion;singleglobalthreshold',
          'target_development_weak_work_activation':.10,'source_reference_confidence':.25,
          'reference_labels_not_verified_slope_safety_or_accidentabsence':True,
          'accident_positive_labels_used_for_threshold_choice':False,'threshold_or_weight_sweep':False,
          'mine_specific_threshold_or_predictor_routing':False,
          'complete_positive_recall_and_reference_background_activation_reported_afterthresholdfrozen':True,
          'case_scores_are_not_calibrated_posterior_probabilities':True,
          'new_supervised_fits':0,'reserved_case_predictions':0,'production_default_changed':False,
          'single_new_operatingpoint_only_no_secondcalibration_if_acceptancefails':True}
    path=OUT/'protocol.json'
    if path.exists():
        old=load(path);assert all(old.get(k)==val for k,val in plan.items() if k!='registered_at')
    else:write(path,plan)
    scores={};quantiles=[]
    for case in cases:
        score_path=SOURCE/(case+'_mean_score.npy');score=np.load(score_path)
        observed=np.load(BASE/'inputs'/case/'support.npy')[0]>0
        normal=(np.load(BASE/'inputs'/case/'normal_mask.npy')>0)&observed
        q=float(np.quantile(score[normal],.9,method='higher'))
        quantiles.append({'case':case,'reviewed_normal_pixels':int(normal.sum()),'q90_higher':q,
                          'score_sha256':file_sha256(score_path),
                          'normal_mask_sha256':file_sha256(BASE/'inputs'/case/'normal_mask.npy')})
        scores[case]=(score,observed,normal)
    threshold=float(np.nextafter(np.float32(max(row['q90_higher'] for row in quantiles)),np.float32(np.inf)))
    frozen={'threshold':threshold,'quantiles':quantiles,'protocol_sha256':file_sha256(path),
            'positive_labels_not_consumed_for_selection':True,'legal_finite_threshold':bool(np.isfinite(threshold) and 0<threshold<1)}
    path=OUT/'threshold_before_positive_scoring.json'
    if path.exists():assert load(path)==frozen
    else:write(path,frozen)
    if not frozen['legal_finite_threshold']:
        write(OUT/'results.json',{'threshold_selection':frozen,'acceptance_pass':False,'reason':'no valid global threshold under fixedrule'});return
    rows=[]
    for case in cases:
        score,observed,normal=scores[case];target=np.load(BASE/'inputs'/case/'old_reference.npy');known=(target!=255)&observed
        positive=(target==1)&known;background=(target==0)&known;candidate=(score>=threshold)&observed
        tp=int((candidate&positive).sum());fp=int((candidate&background).sum())
        row={'case':case,'reference_recall':tp/int(positive.sum()),'reviewed_normal_activation':float(candidate[normal].mean()),
             'weak_background_FPR':fp/int(background.sum()),'TP':tp,'FN':int(positive.sum())-tp,'FP':fp,
             'unchanged_positive_reference_pixels':int(positive.sum()),'threshold':threshold}
        rows.append(row);np.save(OUT/(case+'_candidate_mask.npy'),candidate)
    summary={'macro_reference_recall':float(np.mean([r['reference_recall'] for r in rows])),
             'worst_reference_recall':min(r['reference_recall'] for r in rows),
             'worst_reviewed_normal_activation':max(r['reviewed_normal_activation'] for r in rows),
             'worst_weak_background_FPR':max(r['weak_background_FPR'] for r in rows)}
    summary['development_reference_gates_pass']=summary['macro_reference_recall']>=.7 and summary['worst_reference_recall']>=.5 and summary['worst_reviewed_normal_activation']<=.15 and summary['worst_weak_background_FPR']<=.15
    write(OUT/'results.json',{'completed_at':datetime.now(timezone(timedelta(hours=8))).isoformat(),'threshold_selection':frozen,
          'threshold_record_sha256':file_sha256(OUT/'threshold_before_positive_scoring.json'),'cases':rows,'summary':summary,
          'same_development_references_used_for_negative_calibration_not_independent_test':True,
          'source_or_multi_seed_or_actualservice_admission_not_completed':True,'production_default_changed':False,
          'new_independent_accident_boundary_or_unseen_accident_accuracy_claim':False,'new_CNN_supervised_fits':0})
    print('Fixedglobalthreshold',threshold,json.dumps(summary),flush=True)
    for row in rows:print(row['case'],row['reference_recall'],row['reviewed_normal_activation'],flush=True)

if __name__=='__main__':run()
