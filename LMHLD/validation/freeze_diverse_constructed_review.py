from pathlib import Path
from datetime import datetime
import json,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from model_registry import file_sha256
DATA=ROOT/'datasets/diverse_operation_pairs_20261004';OUT=DATA/'review_v1_20261004';path=OUT/'review_manifest.json'
p=json.loads(path.read_text(encoding='utf-8'))
for name,digest in p['frozen_assets'].items():
    if file_sha256(OUT/name)!=digest:raise ValueError('Reference modified before visual acceptance')
if not all(r['pair_alignment']['raw_pair_gate_pass'] for r in p['sites'].values()):raise ValueError('Some pairs failed independent ground alignment')
if set(p['training_sites'])&set(p['normal_holdout_sites']):raise ValueError('Mine-level role leak')
p.update(status='accepted_project_weak_appearance_reference',visual_reviewed_at=datetime.now().isoformat(timespec='seconds'),
         accepted_sites=list(p['sites']),expert_feedback_received=0,geotechnical_stability_verified=False,
         source_split_preserved=True,normal_holdout_never_in_training=True,
         training_known_pixels=sum(p['sites'][s]['normal_pixels'] for s in p['training_sites']),
         normal_holdout_known_pixels=sum(p['sites'][s]['normal_pixels'] for s in p['normal_holdout_sites']),
         review_limit='constructed surfaces visible at both dates; absence of subtle movement or hidden failure is not proven',
         uncertainty_policy='cloud/shadow/SCL-water/uncertain faces and all unselected pixels remain unknown')
path.write_bytes((json.dumps(p,ensure_ascii=False,indent=2)+'\n').encode('utf-8'))
print(json.dumps({'train_pixels':p['training_known_pixels'],'holdout_pixels':p['normal_holdout_known_pixels'],'expert_review':False}))
