"""Isolate reciprocal matching at original confidence; keep all data-admission gates."""
from pathlib import Path
import sys,json
ROOT=Path('D:/LMHLD');sys.path.insert(0,str(ROOT))
from model_registry import file_sha256
def run():
    parent=ROOT/'validation/register_kef_author_outline_v2.py';text=parent.read_text(encoding='utf-8')
    replace={'reference_registration_v2_reciprocal_confidence':'reference_registration_v3_reciprocal_original_confidence',
        "'matcher_confidence_min':.8,'reciprocal_confidence_min':.8":"'matcher_confidence_min':.5,'reciprocal_confidence_min':.5,'single_factor_vs_v1_reciprocity_only':True",
        'if cc[i]<.8':'if cc[i]<.5','mutual=(rc>=.8)':'mutual=(rc>=.5)'}
    for old,new in replace.items():
        if old not in text:raise ValueError('Frozen parent changed')
        text=text.replace(old,new)
    dst=ROOT/'validation/register_kef_author_outline_v3.py'
    if dst.exists() and dst.read_text(encoding='utf-8')!=text:raise ValueError('Frozen singlefactor protocol changed')
    if not dst.exists():dst.write_text(text,encoding='utf-8')
    (ROOT/'datasets/kef_essnoun_development_20261005/reciprocal_singlefactor_v3_provenance.json').write_text(json.dumps({
        'parent':str(parent),'parent_sha256':file_sha256(parent),'derived':str(dst),'derived_sha256':file_sha256(dst),
        'reason':'v2dualchange confidenceplusreciprocity produced14points;isolate reciprocalcycle at v1confidence.5',
        'unique90_holdout30_median1p5_p903_within3p9_acceptance_unchanged':True,'no_label_or_detector_output_used':True},indent=2)+'\n',encoding='utf-8')
if __name__=='__main__':run()
