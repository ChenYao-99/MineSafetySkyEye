"""Accept visually inspected low-confidence constructed surfaces only when raw pair gate passes."""
from pathlib import Path
import json,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from model_registry import file_sha256
DATA=ROOT/'datasets/arid_operation_pairs_20261004'
OUT=DATA/'review_v1_20261004'


def freeze():
    path=OUT/'review_manifest.json';p=json.loads(path.read_text(encoding='utf-8'))
    for name,digest in p['frozen_assets'].items():
        if file_sha256(OUT/name)!=digest:raise ValueError('Draft reference changed after visual inspection')
    accepted=[];pending=[]
    for site,entry in p['sites'].items():
        if entry['pair_alignment']['raw_pair_gate_pass'] and entry['normal_pixels']>0:
            entry.update(status='accepted_auxiliary_weak_appearance_only',project_original_and_overlay_visual_review=True,independent_expert_review=False)
            accepted.append(site)
        else:
            entry.update(status='quarantined_pair_alignment_pending',project_original_and_overlay_visual_review=True,independent_expert_review=False)
            pending.append(site)
    p.update(status='partially_accepted_project_weak_reference',accepted_sites=accepted,pending_sites=pending,
             actual_normal_surface_count=sum(len(p['sites'][s]['polygons']) for s in accepted),
             actual_known_normal_pixels=sum(p['sites'][s]['normal_pixels'] for s in accepted),
             expert_feedback_received=0,stability_certified=False)
    path.write_bytes((json.dumps(p,ensure_ascii=False,indent=2)+'\n').encode('utf-8'))
    print(json.dumps({'accepted_sites':accepted,'pending_sites':pending,'weak_normal_pixels':p['actual_known_normal_pixels'],'expert_feedback':0}))


if __name__=='__main__':freeze()
