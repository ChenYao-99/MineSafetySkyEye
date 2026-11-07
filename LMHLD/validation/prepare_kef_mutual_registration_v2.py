"""Data-QA revision using reciprocal correspondence, never post-fit holdout filtering."""
from pathlib import Path
import json,sys
ROOT=Path('D:/LMHLD');sys.path.insert(0,str(ROOT))
from model_registry import file_sha256
def run():
    parent=ROOT/'validation/register_kef_author_outline_20261005.py';text=parent.read_text(encoding='utf-8')
    replacements={
      'reference_registration_v1_wang2026':'reference_registration_v2_reciprocal_confidence',
      "'matcher_confidence_min':.5":"'matcher_confidence_min':.8,'reciprocal_confidence_min':.8,'cycle_source_px_max':1,'cycle_native_px_max':1,'v1_failure_retained':True,'pre_fit_match_quality_only':True",
      "with torch.inference_mode():output=model({'image0':x,'image1':y,'mask0':m,'mask1':n})":
      "with torch.inference_mode():\n            output=model({'image0':x,'image1':y,'mask0':m,'mask1':n})\n            reverse=model({'image0':y,'image1':x,'mask0':n,'mask1':m})",
      "unique=[]\n        for i in np.argsort(-cc):":
      "ra=reverse['keypoints1'].cpu().numpy();rb=reverse['keypoints0'].cpu().numpy();rc=reverse['confidence'].cpu().numpy()\n        unique=[]\n        for i in np.argsort(-cc):",
      "if cc[i]<.5 or not(0<=sx<mask.shape[1]":
      "if cc[i]<.8 or not(0<=sx<mask.shape[1]",
      "if np.linalg.norm(b-expected[i])>15:continue":
      "if np.linalg.norm(b-expected[i])>15:continue\n            mutual=(rc>=.8)&(np.linalg.norm(ra-a,axis=1)<=1)&(np.linalg.norm(rb-b,axis=1)<=1)\n            if not mutual.any():continue",
      'del output,y,n;torch.cuda.empty_cache()':'del output,reverse,y,n;torch.cuda.empty_cache()'}
    for old,new in replacements.items():
        if old not in text:raise ValueError('Parent frozen registration changed')
        text=text.replace(old,new)
    dst=ROOT/'validation/register_kef_author_outline_v2.py'
    if dst.exists() and dst.read_text(encoding='utf-8')!=text:raise ValueError('Frozen reciprocal method changed')
    if not dst.exists():dst.write_text(text,encoding='utf-8')
    out=ROOT/'datasets/kef_essnoun_development_20261005/reciprocal_registration_provenance.json'
    out.write_text(json.dumps({'parent':str(parent),'parent_sha256':file_sha256(parent),'derived':str(dst),'derived_sha256':file_sha256(dst),
        'reason':'v1has manyambiguousmatches;pre-fit reciprocalconfidence/cyclequality;all30heldpoints allocatedafterfilter butbeforefit',
        'same_numeric_acceptance_gates_and_spatial_holdout_allocation':True,'detector_outputs_or_post_fit_heldout_errors_used_to_discard_points':False},indent=2)+'\n',encoding='utf-8')
if __name__=='__main__':run()
