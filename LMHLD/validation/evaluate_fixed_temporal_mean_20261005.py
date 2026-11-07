"""One fixed equal-weight score ensemble; no mining identity, fitting or sweep."""
from pathlib import Path
from datetime import datetime,timezone,timedelta
import json,sys
ROOT=Path('D:/LMHLD');sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'validation'))
from runtime_paths import configure_runtime_paths
configure_runtime_paths()
import numpy as np
from model_registry import file_sha256
from mine_data_protocol import registry,reserved_case_ids,verify_assets
BASE=ROOT/'experiments/cause_terrain_round_20261005';WORK=ROOT/'experiments/tropical_work_matched_20261005'
OUT=ROOT/'experiments/fixed_temporal_mean_20261005'

def load(p):return json.loads(p.read_text(encoding='utf-8'))
def write(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')

def run():
    OUT.mkdir(exist_ok=True);verify_assets();v=registry();assert all(v['cases'][c]['model_predictions_run']==0 for c in reserved_case_ids(v))
    base=load(BASE/'results.json');work=load(WORK/'results.json')
    plan={'registered_at':datetime.now(timezone(timedelta(hours=8))).isoformat(),'role':'post-result observed development diagnostic;not new blindtest',
          'base_results_sha256':file_sha256(BASE/'results.json'),'work_results_sha256':file_sha256(WORK/'results.json'),
          'implementation_sha256':file_sha256(__file__),'policy':'0.5*basebinaryscore+0.5*sourceworkbinaryscore;candidate>=.5;identical allmines',
          'scores_not_independently_calibrated_probability':True,'threshold':.5,'component_weight':.5,
          'case_specific_weights_or_component_selection':False,'new_supervised_fits':0,'threshold_or_weight_sweep':False,
          'source_or_service_regression':False,'reserved_predictions':0,'production_default_changed':False,
          'bounded_fixed_ensemble_rules_this_round':['hardagreement with review','equalarithmeticmean'],
          'no_more_ensemble_rules_after_this_diagnostic':True}
    path=OUT/'protocol.json'
    if path.exists():
        old=load(path);assert all(old.get(k)==v for k,v in plan.items() if k!='registered_at')
    else:write(path,plan)
    rows=[]
    for row in work['models']:
        if row['arm']!='source_supported_work':continue
        held=row['held'];control=next(r for r in base['models'] if r['held']==held and r['arm']=='binary_temporal')
        p=BASE/(control['id']+'_accident_probability.npy');q=WORK/(row['id']+'_accident_probability.npy')
        a,b=np.load(p),np.load(q);score=(a+b)*.5;folder=BASE/'inputs'/held
        observed=np.load(folder/'support.npy')[0]>0;target=np.load(folder/'old_reference.npy');known=(target!=255)&observed
        positive=(target==1)&known;background=(target==0)&known;normal=(np.load(folder/'normal_mask.npy')>0)&observed
        candidate=(score>=.5)&observed;tp=int((candidate&positive).sum());fp=int((candidate&background).sum())
        measure={'held':held,'reference_recall':tp/int(positive.sum()),'reviewed_normal_activation':float(candidate[normal].mean()),
                 'weak_background_FPR':fp/int(background.sum()),'TP':tp,'FN':int(positive.sum())-tp,'FP':fp,
                 'base_weights_sha256':control['weights_sha256'],'work_weights_sha256':row['weights_sha256'],
                 'base_probability_sha256':file_sha256(p),'work_probability_sha256':file_sha256(q),
                 'normal_reference_is_weak_not_safety':True,'new_blind_accuracy_claim':False}
        score[~observed]=0;np.save(OUT/(held+'_mean_score.npy'),score);rows.append(measure)
    summary={'macro_reference_recall':float(np.mean([r['reference_recall'] for r in rows])),
             'worst_reference_recall':min(r['reference_recall'] for r in rows),
             'worst_reviewed_normal_activation':max(r['reviewed_normal_activation'] for r in rows),
             'worst_weak_background_FPR':max(r['weak_background_FPR'] for r in rows)}
    summary['joint_reference_gates_pass']=summary['macro_reference_recall']>=.7 and summary['worst_reference_recall']>=.5 and summary['worst_reviewed_normal_activation']<=.15 and summary['worst_weak_background_FPR']<=.15
    write(OUT/'results.json',{'completed_at':datetime.now(timezone(timedelta(hours=8))).isoformat(),'cases':rows,'summary':summary,
                            'protocol_sha256':file_sha256(path),'new_CNN_supervised_fits':0,'production_default_changed':False,
                            'observed_work_region_already_used_for_prior_diagnostic':True,'independent_blind_accident_runs':0})
    print(json.dumps(summary),flush=True)
    for row in rows:print(row['held'],row['reference_recall'],row['reviewed_normal_activation'],flush=True)

if __name__=='__main__':run()
