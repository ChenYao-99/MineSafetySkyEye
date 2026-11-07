"""Fixed processing-version correction, preserving strict-intersection sensitivity results.

Legacy Sen2Cor weather classifiers can falsely identify bright bare terrain as cloud.
The official processing documentation supports using a consistent current classifier.
No detector output/metric is used; all data/registration thresholds remain unchanged.
"""
from pathlib import Path
ROOT=Path('D:/LMHLD');BASE=ROOT/'validation/fetch_cl_earthsearch_preimage_v2.py';OUT=ROOT/'validation/fetch_cl_earthsearch_preimage_v3.py'
text=BASE.read_text(encoding='utf-8')
replacements={
 'chronology_safe_preimage_v2_earthsearch_20261005':'chronology_safe_preimage_v3_consistent_quality_20261005',
 "'originalpost_multi_version_mask_preserved':True":"'originalpost_legacy_intersection_preserved_as_separate_sensitivity':True,'primary_quality_policy':'latest processing version for both capture dates, selected by processingbaseline not predictions','primary_post_quality_processing_baseline':'05.00'",
 "for source in [Q/'exact_date_quality_v1_20261005/post_SCL_original_grid.tif',*sorted(Q.glob('official_post_SCL*.tif'))]:":"for source in [Q/'official_post_SCL_sequence1.tif']:"
}
for old,new in replacements.items():
 if text.count(old)!=1:raise ValueError('Unexpected v2 acquisition code')
 text=text.replace(old,new)
if OUT.exists() and OUT.read_text(encoding='utf-8')!=text:raise ValueError('Existing v3 protocol code changed')
OUT.write_text(text,encoding='utf-8');print('Uniform latest-processing quality rule saved;old strict masks/results stay intact')
