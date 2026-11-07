"""Record the user-directed development tradeoff without rewriting old gates."""
from pathlib import Path
from datetime import datetime,timezone,timedelta
import sys,json
ROOT=Path('D:/LMHLD');sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'validation'))
from runtime_paths import configure_runtime_paths
configure_runtime_paths()
from model_registry import file_sha256
from mine_data_protocol import DATA,registry,reserved_case_ids,verify_assets
from finalize_cause_terrain_round_20261005 import protected_runtime
OUT=ROOT/'experiments/recall_priority_policy_20261006'

def load(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def write(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')

def run():
    verify_assets();v=registry();protected=protected_runtime()
    reserved={c:v['cases'][c]['model_predictions_run'] for c in sorted(reserved_case_ids(v))}
    assert len(reserved)==4 and not any(reserved.values()) and not (DATA/'final_evaluation_once.json').exists()
    high=ROOT/'experiments/fixed_temporal_mean_20261005/results.json'
    low=ROOT/'experiments/global_work_reference_calibration_20261005/results.json'
    a=load(high);b=load(low)
    assert not a['summary']['joint_reference_gates_pass'] and not b['summary']['development_reference_gates_pass']
    p=ROOT/'experiments/fixed_temporal_mean_20261005/protocol.json'
    assert file_sha256(p)==a['protocol_sha256']
    rule=load(p);assert rule['threshold']==.5 and rule['component_weight']==.5 and not rule['case_specific_weights_or_component_selection']
    OUT.mkdir(exist_ok=True)
    policy={'adopted_at':datetime.now(timezone(timedelta(hours=8))).isoformat(timespec='seconds'),
        'authorization':'user directs an explicit recall-versus-false-positive tradeoff on2026-10-06',
        'scope':'screeningdevelopment with humanreview;not confirmed mine-safety alarm or awardcertification',
        'primary_objective':'maximize accident positive-core recall subject to operation-reference activation ceiling',
        'development_constraints':{'macro_known_positive_core_recall_min':.80,'each_mine_known_positive_core_recall_min':.70,
            'each_mine_reviewed_normal_reference_activation_max':.25,'each_mine_observed_weak_background_activation_max':.25},
        'secondary_improvement_target':{'reviewed_normal_reference_activation_max':.15},
        'no_zero_recall_mine_hidden_by_mean':True,
        'normal_reference_activation_is_pixel_fraction_not_real_event_alarm_false_positive_rate':True,
        'positive_core_recall_is_not_complete_accident_boundary_or_event_sensitivity':True,
        'original_15percent_joint_gates_and_historical_failures_preserved':True,
        'priority_rule':'meet every development constraint;then rank recall;among comparable recall prefer lower review burden',
        'selected_existing_development_diagnostic':str(high),'selected_result_sha256':file_sha256(high),
        'fixed_rule_protocol':str(p),'fixed_rule_protocol_sha256':file_sha256(p),
        'rule':'equalmean ofbasebinary/sourcework components;>=.5;sameallmines;no thresholdsorweightsweep',
        'selected_reference_summary':a['summary'],'alternative_low_activation_summary':b['summary'],
        'alternative_result_sha256':file_sha256(low),
        'policy_and_route_selected_after_observing_development_results':True,
        'independent_fresh_test_or_multi_seed_validation_passed':False,
        'additional_activation_and_instance_review_burden_regressions_required':True,
        'CV_fold_components_not_one_deployable_model_and_never_routed_by_known_mine_identity':True,
        'runtime_default_model_replacement_authorized_by_this_diagnostic_alone':False,
        'next':'freeze this one existing recallpriority route;verify fullcomponents andalloperations underrepeatedtraining;buildone mineagnostic runtimecandidate;then independent frozen acceptance andservice checks;keep sourcingdatedindependentboundaries;no new threshold/fusion sweeps',
        'new_training_or_prediction_calls':0,'reserved_predictions':reserved,'production_default_changed':False,
        'protected_runtime_artifacts':protected,'all_new_storage_D':True}
    s=a['summary'];c=policy['development_constraints']
    policy['existing_observed_reference_point_satisfies_new_development_constraints']=bool(
        s['macro_reference_recall']>=c['macro_known_positive_core_recall_min'] and s['worst_reference_recall']>=c['each_mine_known_positive_core_recall_min']
        and s['worst_reviewed_normal_activation']<=c['each_mine_reviewed_normal_reference_activation_max']
        and s['worst_weak_background_FPR']<=c['each_mine_observed_weak_background_activation_max'])
    assert policy['existing_observed_reference_point_satisfies_new_development_constraints']
    write(OUT/'selection_policy.json',policy)
    # Update only the future priority, never counters, old result files or runtime.
    state=load(DATA/'STATE.json');assert state['completed_native_training_runs']==state['completed_native_evaluated_runs']==218
    assert state['completed_lightweight_feature_diagnostics']==119
    state.update(updated_at=policy['adopted_at'],active_model_selection_policy=str(OUT/'selection_policy.json'),
        model_optimization_priority='accident_recall_first_with_bounded_normal_reference_activation',
        recall_priority_development_constraints=c,recall_priority_existing_development_candidate=str(high),
        original_joint_gate_failures_reclassified=False,recall_priority_runtime_candidate_admitted=False,
        next=policy['next'],next_local_work=policy['next'])
    write(DATA/'STATE.json',state)
    print('Recallpriority adopted:macro>=.80,eachmine>=.70,normal/bgactivation<=.25;aim.15;existingpoint84.6/70.5/20.9;oldgatesunchanged;defaultunchanged;0newpredictions',flush=True)

if __name__=='__main__':run()
