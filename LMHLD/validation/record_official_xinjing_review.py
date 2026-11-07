"""Record visual examination without promoting a schematic to accident GIS truth."""
from pathlib import Path
from datetime import datetime
import json,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from model_registry import file_sha256
DIR=ROOT/'datasets/mine_benchmark_20261002/development/Xinjing/independent_official_20261004'
manifest=DIR/'source_manifest.json';p=json.loads(manifest.read_text(encoding='utf-8'))
if file_sha256(DIR/'official_accident_report.pdf')!=p['pdf_sha256']:raise ValueError('Official attachment changed')
ref=ROOT/'datasets/mine_benchmark_20261002/development/Xinjing/reference_v1_20261004/label_reference.json'
reference=json.loads(ref.read_text(encoding='utf-8'))
paper_area=reference['full_footprint_m2'];official_area=p['numeric_extent_reported']['approximate_damage_area_m2']
p.update(reviewed_at=datetime.now().isoformat(timespec='seconds'),use_status='accepted independent official event/extent corroboration only; no new segmentation ground truth',
         visual_review={'figure1_pdf_page7':'pre-event mining/design/property outlines schematic; no georeferenced accident polygon or labelled collapse boundary',
                        'figures2_and3_pdf_page15':'oblique operational images around13:00/13:12; no orthorectification or ground-control metadata',
                        'damage_extent_pdf_page16':'approximate final damage area and dimensions; final timing not equivalent to Feb23 satellite footprint'},
         source_reference_sha256=file_sha256(ref),author_derived_footprint_m2=paper_area,
         area_relative_difference_from_official_approximation=abs(paper_area-official_area)/official_area,
         scalar_area_agreement_is_not_boundary_accuracy=True,existing_labels_or_original_images_modified=False,
         no_model_based_registration=True,expert_georeferenced_boundary_obtained=False,
         source_group_independence_limit='official source independent of project; papers may cite the same official dimensions, so dimensions are not a second independent surveyed polygon')
manifest.write_bytes((json.dumps(p,ensure_ascii=False,indent=2)+'\n').encode('utf-8'))
print(json.dumps({'official_corroboration':True,'new_GIS_ground_truth':False,'area_difference':p['area_relative_difference_from_official_approximation']}))
