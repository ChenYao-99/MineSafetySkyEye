"""Record explicit visual acceptance of the scoped external research core only."""
from pathlib import Path
from datetime import datetime
import json,sys
ROOT=Path('D:/LMHLD');sys.path.insert(0,str(ROOT))
from model_registry import file_sha256
from mine_data_protocol import assert_development_cases,verify_assets
OUT=ROOT/'datasets/arid_accident_operation_20261004/ZA_GAMS2020S/pixel_reference_v1_wang2026'
SOURCE=OUT/'reference_review_pending.json'
assert_development_cases(['ZA_GAMS2020S']);verify_assets()
v=json.loads(SOURCE.read_text(encoding='utf-8'))
for path,digest in v['frozen_dependencies'].items():
 if file_sha256(path)!=digest:raise ValueError('Frozen dependency changed')
for name,digest in v['frozen_assets'].items():
 if file_sha256(OUT/name)!=digest:raise ValueError('Frozen partial reference changed')
if (OUT/'reference.json').exists():raise SystemExit('Already frozen; do not rewrite')
v.update(reviewed_at=datetime.now().isoformat(timespec='seconds'),review_status='accepted_development_partial_published_research_reference',
  accepted_for_development_positive_core_supervision=True,accepted_for_known_core_recall_evaluation=True,
  visual_review={'performed_by':'project agent; no independent specialist credential claimed','source_outline_and_explicit2020event_heading_reviewed':True,
    'original_printed_latitude_ticks_read_from_intact_label_pixels':True,'all30_ground_holdouts_per_original_date_viewed':True,
    'original_pre_post_partial_reference_overlay_viewed':True,'original_dated2020_12_03_SkySat_image_viewed':True,
    'obvious_cloud_or_snow_on_known_core_observed':False,'secondary_failure_dust_runout_and_unknown_GE_base_date_not_promoted':True,
    'visibility_review_is_not_field_ground_truth':True},
  hard_gap_status='Scoped traceable development core obtained; complete incident boundary and original independent surveyGIS still absent')
(OUT/'reference.json').write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
registry=ROOT/'datasets/mine_benchmark_20261002/mine_case_registry.json';r=json.loads(registry.read_text(encoding='utf-8'))
c=r['cases']['ZA_GAMS2020S'];c.update(partial_pixel_reference=str(OUT/'reference.json'),partial_pixel_reference_sha256=file_sha256(OUT/'reference.json'),
  partial_research_positive_pixels=v['positive_pixels'],partial_research_reference_accepted=True,complete_independent_accident_GIS=False)
registry.write_text(json.dumps(r,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
sys.stdout.reconfigure(encoding='utf-8');print('Accepted38 conservative native10m development pixels;492685 unknown;zero background;no fullIoU truth')
