"""Freeze the current default runtime and pre-register a single independent audit."""
from datetime import datetime
from pathlib import Path
import hashlib,json,shutil,sys

ROOT=Path(__file__).resolve().parents[1];ML=Path('D:/LMHLD')
sys.path.insert(0,str(ROOT))
from project_runtime import configure_runtime_paths
configure_runtime_paths()
sys.path.insert(0,str(ROOT/'validation'))
sys.path.insert(0,str(ML/'validation'))
from build_portable_release_20261007 import build
from mine_data_protocol import register_final_run,registry,verify_assets

OUT=ROOT/'cache/independent_acceptance_20261007'
EVAL=ML/'experiments/frozen_default_independent_audit_20261007'


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024**2),b''):h.update(block)
    return h.hexdigest()


def write(path,value):
    with Path(path).open('x',encoding='utf-8') as stream:json.dump(value,stream,ensure_ascii=False,indent=2)


def main():
    assert not OUT.exists() and not EVAL.exists(),'Never replace a frozen version or its independent audit'
    inventory=registry();verify_assets(inventory)
    case=inventory['cases']['CL0002']
    assert case['model_predictions_run']==0 and case['quality_review_status']=='accepted'
    reference=Path(case['benchmark_reference']);contract=Path(case['benchmark_usage_contract'])
    assert sha(reference)==case['benchmark_reference_sha256'] and sha(contract)==case['benchmark_usage_contract_sha256']
    ref=json.loads(reference.read_text(encoding='utf-8'))
    for path,digest in ref['frozen_dependencies'].items():assert sha(path)==digest,path
    for relative,digest in ref['frozen_assets'].items():assert sha(reference.parent/relative)==digest,relative
    EVAL.mkdir(parents=True)
    build(OUT)
    release=json.loads((OUT/'portable_release_manifest.json').read_text(encoding='utf-8'))
    for relative,digest in release['files'].items():assert sha(Path(release['release'])/relative)==digest
    frozen_ml=Path(release['release'])/'LMHLD'
    weight=frozen_ml/'models_prior/real_ft_T1b_ours_seed42/best_model.pth'
    post=Path(next(path for path in ref['frozen_dependencies'] if path.endswith('post_producer_BGRN_original_grid.tif')))
    labels=reference.parent/'labels_post_reference_interior.tif'
    input_protocol={'version':1,'frozen_at':datetime.now().astimezone().isoformat(),
        'model_tag':'mine','patch_size':32,'threshold':.4,'patch_norm':'minmax','input_scale':1,
        'analysis_grid':'native','nir_synth':False,'instance_tau':0,'max_batch':128,
        'reference_scope':'post_observed_reference','input_mode':'post_only',
        'primary_prediction':'production-reported connected components >=10 native pixels',
        'raw_threshold_mask':'secondary diagnostic only;cannot select another output',
        'scored_labels':{'positive':1,'annotation_background':0,'unknown':255},
        'input':{'path':str(post),'sha256':sha(post)},'labels':{'path':str(labels),'sha256':sha(labels)},
        'reference':{'path':str(reference),'sha256':sha(reference)},'contract':{'path':str(contract),'sha256':sha(contract)},
        'safety_or_unmasked_boundary_accuracy_claim':False,'threshold_or_model_selection_after_results':False}
    write(EVAL/'input_protocol.json',input_protocol)
    acceptance=ROOT/'cache/spatial_reference_acceptance_20261007/final_acceptance.json'
    evidence=json.loads(acceptance.read_text(encoding='utf-8'))
    assert evidence['passed'] and not evidence['default_weights_changed']
    selection={'status':'accepted_for_frozen_version_audit','evaluation_purpose':'frozen_version_independent_audit',
        'development_gates_passed':False,'source_regression_passed':True,'reserved_predictions_used_for_selection':0,
        'reserved_results_may_select_or_tune_model':False,'threshold':.4,'selected_weights_sha256':[sha(weight)],
        'selection_basis':'current deployed default, retained unchanged despite failed development joint gate',
        'user_authorization':{'source':'user_message','instruction':'冻结版本吧，先完成一次独立验收，再针对验收暴露的问题补数据，然后量化完整复核任务的收益。'},
        'input_protocol':{'path':str(EVAL/'input_protocol.json'),'sha256':sha(EVAL/'input_protocol.json')},
        'source_regression_evidence':{'path':str(acceptance),'sha256':sha(acceptance)},
        'runtime_snapshot_manifest':{'path':str(OUT/'portable_release_manifest.json'),'sha256':sha(OUT/'portable_release_manifest.json')}}
    write(EVAL/'selection_record.json',selection)
    plan={'evaluation_purpose':'frozen_version_independent_audit','case_ids':['CL0002'],
        'weights':[{'path':str(weight),'sha256':sha(weight)}],'threshold':.4,
        'metrics':['precision_on_scored_published_reference','recall_on_scored_published_reference','IoU_on_scored_published_reference'],
        'reference_scopes':{'CL0002':'post_observed_reference'},'input_modes':{'CL0002':'post_only'},
        'source_regression_passed':True,
        'development_selection_record':{'path':str(EVAL/'selection_record.json'),'sha256':sha(EVAL/'selection_record.json')},
        'result_independent_of_passing_performance':True,'number_of_logical_model_predictions':1,
        'other_reserved_cases_opened':False,'no_ROI_or_threshold_or_best_seed_selection':True,
        'diagnostics_not_safety_metrics':['confusion_counts','activation_on_annotation_background','probability_distribution','small_component_filter_effect'],
        'performance_reference_targets':{'recall':.70,'precision':.50,'IoU':.35},
        'target_basis':'predeclared internal screening research targets,not official AIC or safety criteria',
        'followup_development_rule':'CL0002 and same-mine neighbouring images remain unavailable for training or tuning;acquire different mines'}
    write(EVAL/'plan.json',plan)
    runner=ML/'validation/run_frozen_default_independent_audit_20261007.py'
    shutil.copy2(runner,EVAL/'frozen_evaluation_runner.py')
    lock=register_final_run(EVAL/'plan.json',EVAL)
    freeze={'frozen_at':input_protocol['frozen_at'],'case':'CL0002','planned_predictions':1,
        'release_manifest_sha256':sha(OUT/'portable_release_manifest.json'),
        'protocol_sha256':sha(EVAL/'input_protocol.json'),'plan_sha256':sha(EVAL/'plan.json'),
        'runner_sha256':sha(EVAL/'frozen_evaluation_runner.py'),
        'one_shot_lock':str(ML/'datasets/mine_benchmark_20261002/final_evaluation_once.json'),
        'one_shot_lock_sha256':sha(ML/'datasets/mine_benchmark_20261002/final_evaluation_once.json'),
        'development_performance_claimed_passed':False,'snapshot_default_weight_sha256':sha(weight)}
    write(EVAL/'freeze.json',freeze)
    print('FROZEN',release['release'],len(release['files']),'files;CL0002 single prediction registered;failed development status preserved',flush=True)


if __name__=='__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    main()
