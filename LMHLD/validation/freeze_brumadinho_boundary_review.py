from pathlib import Path
from datetime import datetime
import json,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from model_registry import file_sha256
OUT=ROOT/'datasets/mine_benchmark_20261002/development/Brumadinho/official_boundary_review_v1_20261004'
path=OUT/'boundary_audit.json';p=json.loads(path.read_text(encoding='utf-8'))
for name,digest in p['derived_asset_sha256'].items():
    if file_sha256(OUT/name)!=digest:raise ValueError('Boundary review asset changed')
for name,digest in p['original_assets'].items():
    if file_sha256(ROOT/'datasets/mine_benchmark_20261002'/name)!=digest:raise ValueError('Original scene changed')
p.update(visual_review_completed_at=datetime.now().isoformat(timespec='seconds'),
         visual_review_status='accepted official georeferenced auxiliary boundary corroboration; training/scoring still pending',
         visible_post_plume_matches_both_outlines=True,
         local_occlusion_observed='bright cloud over lower/southwest plume corridor in Feb1 crop; full image is not completely weather-clear',
         unresolved_acceptance=['exact Pleiades acquisition date','local cloud/shadow mask before positive-pixel scoring','underlying source independence from research inventory'],
         primary_pit_wall_gap_closed=False,independent_field_or_project_expert_GIS=False)
path.write_bytes((json.dumps(p,ensure_ascii=False,indent=2)+'\n').encode('utf-8'))
print('Official coordinate boundary reviewed and frozen; no segmentation labels, inference or final-test acceptance')
