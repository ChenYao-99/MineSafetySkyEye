# -*- coding: utf-8 -*-
"""Document known supervised origins and verify no reserved case enters training plans."""
import hashlib
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'datasets/mine_benchmark_20261002'


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def audit():
    core=json.loads((ROOT/'experiments/core_ablation_20261001/protocol.json').read_text(encoding='utf-8'))
    origins=core['common_train_provenance']
    regions=sorted({r['region'] for r in origins if r['kind']=='lmhld'})
    expected={'Emilia-Romagna_Italy_32','Gorkha_Nepal_32','Jiuzhaigou_China_32','Taiwan_China_32','Wenchuan_China_32'}
    if set(regions)!=expected: raise ValueError('Unexpected supervised source geography')
    generic=json.loads((ROOT/'experiments/domain_gap/negatives_meta.json').read_text(encoding='utf-8'))
    extra=json.loads((ROOT/'experiments/mining_neg/negatives_t1_meta.json').read_text(encoding='utf-8'))
    plan=json.loads((ROOT/'experiments/three_mine_context_20261003/training_protocol.json').read_text(encoding='utf-8'))
    for f in plan['folds']:
        if set(f['train_mines'])&{'CL0002','KG0001'} or f['held_development_mine'] in f['train_mines']:
            raise ValueError('Mine role leakage')
    record={'known_source_regions':regions,'source_band_evidence':'Liu2025 LMHLD sectionIIIC, four bands blue/green/red/near-infrared',
        'initial_base':'Emilia-Romagna Italy upload seed42; supervised base training region separate from reserved mines',
        'historical_generic_scene_counts':generic['per_scene'],'additional_mine_background':extra['ours']['half2_sources'],
        'additional_arid_scene':extra['half1']['sources'],'Baige_is_training_not_unseen':True,
        'current_development_mines':['Amyntaio','LasCruces','Hpakant'],
        'reserved_candidates':['CL0002','KG0001'],'reserved_case_geographies':'Chile MinistroHales / Kyrgyz KaraKeche, different from known supervised origins',
        'no_reserved_case_in_explicit_training_plans':True,'DLR_checkpoints_not_used':True,
        'limits':['Audit of known supervised provenance, not proof about opaque generic pretraining data.',
                  'Generic background annotations are weak and previously reused; not independent mine negative truth.',
                  'No claim of external expert assessment or independent absolute boundary accuracy.'],
        'frozen_documents':{str(p.relative_to(ROOT)):sha(p) for p in (ROOT/'experiments/core_ablation_20261001/protocol.json',
          ROOT/'experiments/nir_train/finetune_T1b_ours.json',ROOT/'experiments/three_mine_context_20261003/training_protocol.json')}}
    (DATA/'training_provenance_audit.json').write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print('Known training provenance audited; reserved cases not in explicit supervised plans')


if __name__=='__main__': audit()
