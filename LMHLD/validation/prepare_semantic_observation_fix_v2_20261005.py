"""Version the observation-contract correction without editing frozen experiments."""
from pathlib import Path
import json,sys
ROOT=Path('D:/LMHLD');sys.path.insert(0,str(ROOT))
from model_registry import file_sha256
def derive(parent,dest,replacements):
    source=parent.read_text(encoding='utf-8')
    for old,new in replacements.items():
        if old not in source:raise ValueError('Frozen parent template changed: '+old[:70])
        source=source.replace(old,new)
    if dest.exists() and dest.read_text(encoding='utf-8')!=source:raise ValueError('Frozen observation fix changed')
    if not dest.exists():dest.write_text(source,encoding='utf-8')
    return {'parent':str(parent),'parent_sha256':file_sha256(parent),'derived':str(dest),'derived_sha256':file_sha256(dest),'literal_replacements':replacements}
def run():
    quality_helper='''def main_epoch_support(eid,period,image):
 folder=DATA/'development'/('Xinjing' if eid=='CN_XJ2023' else eid)
 with rasterio.open(folder/(period+'_BGRN_10m.tif')) as f:valid=f.dataset_mask()>0
 valid &= image[:,:,:3].sum(2)>0
 scl_path=folder/(period+'_SCL_10m.tif')
 if scl_path.exists():
  with rasterio.open(scl_path) as f:valid &= ~np.isin(f.read(1),[0,1,3,8,9,10,11])
 return valid
'''
    encoder_replacements={
        'experiments/semantic_structure_inputs_20261005':'experiments/semantic_structure_inputs_v2_20261005',
        'def run():':quality_helper+'\ndef run():',
        "'source_inputs_resized':False":"'source_inputs_resized':False,'preparation_code_sha256':file_sha256(__file__),'main_semantic_support_per_epoch_not_pair_intersection':True,'Jags_unclassified_SCL7_not_observed':True,'old_development_SCL7_policy_unchanged_not_clear_weather_certification':True",
        "for period,a in [('pre',pre),('post',post)]:jobs.append((eid,period,a,available))":
        "for period,a in [('pre',pre),('post',post)]:\n   jobs.append((eid,period,a,main_epoch_support(eid,period,a)))\n   folder=DATA/'development'/('Xinjing' if eid=='CN_XJ2023' else eid)\n   for src in [DATA/'radiometry_audit_20261004'/('Xinjing' if eid=='CN_XJ2023' else eid)/(period+'_BGRN_calibrated_tags.tif'),folder/(period+'_SCL_10m.tif')]:\n    if src.exists():m['source_dependencies'][str(src)]=file_sha256(src)",
        "valid &= ~np.isin(f.read(1),[0,1,3,8,9,10,11])\n   m['source_dependencies']":
        "valid &= ~np.isin(f.read(1),[0,1,3,8,9,10,11]+([7] if eid=='ZA_JAGS2022' else []))\n   m['source_dependencies']"}
    head_replacements={
        'semantic_structure_matched_20261005':'semantic_structure_matched_v2_20261005',
        'semantic_structure_inputs_20261005':'semantic_structure_inputs_v2_20261005',
        "available &= ~np.isin(array(folder/f'{period}_SCL_10m.tif'),[0,1,3,8,9,10,11])":
        "available &= ~np.isin(array(folder/f'{period}_SCL_10m.tif'),[0,1,3,8,9,10,11]+([7] if c=='ZA_JAGS2022' else []))",
        "if encoded['reserved_inputs']!=0:raise ValueError('Reserved representation was observed')":
        "if encoded['reserved_inputs']!=0:raise ValueError('Reserved representation was observed')\n for name,digest in encoded['source_dependencies'].items():\n  if file_sha256(name)!=digest:raise ValueError('Frozen actual imagery or observation support changed')",
        "'PCA_fit_held_labels_or_vectors':False":
        "'observation_contract_fix':'semantics per date own support;Jags7unknownpre;old outcomes retained;all labels samples weights and budgets unchanged','PCA_fit_held_labels_or_vectors':False"}
    rows=[derive(ROOT/'validation/prepare_semantic_structure_inputs_20261005.py',ROOT/'validation/prepare_semantic_structure_inputs_v2_20261005.py',encoder_replacements),
          derive(ROOT/'validation/run_semantic_structure_matched_study_20261005.py',ROOT/'validation/run_semantic_structure_matched_study_v2_20261005.py',head_replacements)]
    output=ROOT/'validation/semantic_observation_fix_v2_20261005.json'
    output.write_text(json.dumps({'reason':'48 post-valid Las corepixels lost by joint semantic mask;371 Jags preSCL7 pixels wrongly pair-available',
                                  'old_results_retained':True,'changed_labels_or_sampling':False,'reserved_predictions':0,'derived_scripts':rows},indent=2)+'\n',encoding='utf-8')
if __name__=='__main__':run()
