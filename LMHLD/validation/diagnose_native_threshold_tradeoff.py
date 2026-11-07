# -*- coding: utf-8 -*-
"""Post-hoc development diagnostic on frozen saved probabilities; no new inference."""
from pathlib import Path
import json
import sys
import numpy as np
import rasterio

ROOT=Path(__file__).resolve().parents[1];STUDY=ROOT/'experiments/full_resolution_study_safe_20261003'
DATA=ROOT/'datasets/mine_benchmark_20261002';CASES=('Amyntaio','LasCruces','Hpakant')


def run():
    result=json.loads((STUDY/'evaluation/results.json').read_text(encoding='utf-8'));rows=[]
    grid=[.5,.525,.55,.575,.6,.625,.65,.675,.7,.725,.75,.775,.8,.825,.85,.875,.9,.925,.95,.975]
    for r in result['models']:
        if r['size']!=128:continue
        eid=r['held']
        with rasterio.open(DATA/f'development/{eid}/partial_labels.tif') as src:y=src.read(1)
        probability=np.load(STUDY/f'evaluation/{r["id"]}_RGB.npy')
        if not np.isfinite(probability).all():raise ValueError('Invalid saved probability')
        pos=probability[y==1];neg=probability[y==0];values=[]
        for t in grid:
            tp=int((pos>=t).sum());fp=int((neg>=t).sum());fn=len(pos)-tp
            values.append({'threshold':t,'recall':tp/len(pos),'weak_background_FPR':fp/len(neg),'F1':2*tp/max(2*tp+fp+fn,1)})
        rows.append({'id':r['id'],'case':eid,'seed':r['seed'],'grid':values,
            'positive_score_quantiles':np.quantile(pos,[.1,.25,.5,.75,.9]).tolist(),'weak_background_score_quantiles':np.quantile(neg,[.1,.25,.5,.75,.9]).tolist()})
    thresholds=[]
    for t in grid:
        seeds=[]
        for seed in (42,43,44):
            mm=[next(v for v in r['grid'] if v['threshold']==t) for r in rows if r['seed']==seed]
            mean=float(np.mean([m['recall'] for m in mm]));worst=min(m['recall'] for m in mm);bg=max(m['weak_background_FPR'] for m in mm)
            seeds.append({'seed':seed,'macro_recall':mean,'worst_recall':worst,'worst_weak_background_FPR':bg,'foreground_background_gates_pass':bool(mean>=.7 and worst>=.5 and bg<=.15)})
        thresholds.append({'threshold':t,'seeds':seeds,'all3_passed':all(s['foreground_background_gates_pass'] for s in seeds)})
    out=STUDY/'threshold_diagnostic.json';value={'role':'post_hoc_observed_development_diagnostic_not_independent_validation',
        'weights_changed':False,'production_threshold_changed':False,'reserved_predictions':0,
        'source_regression_at_new_threshold_not_checked':True,'no_deployment_or_new_blind_claim':True,'models':rows,'thresholds':thresholds}
    out.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    candidates=[t['threshold'] for t in thresholds if t['all3_passed']]
    print('Posthoc unified RGB thresholds passing recall/weak-background gates:',candidates)
    for eid in CASES:
        rs=[r for r in rows if r['case']==eid];t=.75
        mm=[next(v for v in r['grid'] if v['threshold']==t) for r in rs]
        print(eid,'@.75 mean recall',round(float(np.mean([v['recall'] for v in mm])),4),'weakBG',round(float(np.mean([v['weak_background_FPR'] for v in mm])),4))


if __name__=='__main__':sys.stdout.reconfigure(encoding='utf-8');run()
