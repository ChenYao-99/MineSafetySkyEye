"""Preserve performed fits and fix future replay without rewriting frozen code."""
from pathlib import Path
from datetime import datetime,timezone,timedelta
import json,sys
ROOT=Path('D:/LMHLD');sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'validation'))
from model_registry import file_sha256
OUT=ROOT/'experiments/madenkoy_presence_matched_20261005'

def load(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def run():
    audit=load(OUT/'actual_optimizer_step_audit.json');assert len(audit['rows'])==3
    original=audit['rows'][0]['trace'];zero=audit['rows'][1]['trace'];disconnected=audit['rows'][2]['trace']
    assert all(a['loss']==b['loss'] for a,b in zip(original,disconnected))
    assert any(a['loss']!=b['loss'] for a,b in zip(original,zero))
    assert all(row['actual_optimizer_steps']==24 for row in audit['rows'])
    base=ROOT/'validation/run_madenkoy_presence_matched_20261005.py';text=base.read_text(encoding='utf-8')
    text=text.replace('from accident_presence_supervision import positive_presence_loss',
                      'from accident_presence_supervision import positive_presence_loss\nfrom auxiliary_objective import combine_optional_objective')
    text=text.replace('loss=base+coefficient*presence','loss=combine_optional_objective(base,presence,coefficient)')
    text=text.replace("experiments/madenkoy_presence_matched_20261005'","experiments/madenkoy_presence_corrected_control_20261005'")
    text=text.replace("models_prior/madenkoy_presence_matched_20261005'","models_prior/madenkoy_presence_corrected_control_20261005'")
    text=text.replace("cache/madenkoy_presence_resume_20261005'","cache/madenkoy_presence_corrected_resume_20261005'")
    text=text.replace("GENERIC=ROOT/", "INPUT=ROOT/'experiments/madenkoy_presence_matched_20261005/inputs'\nGENERIC=ROOT/")
    text=text.replace("OUT/'inputs/manifest.json'", "INPUT/'manifest.json'").replace("OUT/'inputs'/", "INPUT/")
    text=text.replace("'planned_native_CNN_fits':8", "'planned_native_CNN_fits':4")
    text=text.replace("assert file_sha256(ROOT/'accident_presence_supervision.py')==plan['weak_presence_loss_sha256']",
                      "assert file_sha256(ROOT/'accident_presence_supervision.py')==plan['weak_presence_loss_sha256']\n    assert file_sha256(ROOT/'auxiliary_objective.py')==plan['optional_objective_sha256']")
    text=text.replace("scaler=torch.amp.GradScaler('cuda',enabled=DEVICE=='cuda');checkpoint=", "actual={'steps':0};optimizer.register_step_post_hook(lambda *_:actual.__setitem__('steps',actual['steps']+1))\n            scaler=torch.amp.GradScaler('cuda',enabled=DEVICE=='cuda');checkpoint=")
    text=text.replace("history=saved['history'];start_epoch=saved['epoch']+1", "history=saved['history'];start_epoch=saved['epoch']+1;actual['steps']=saved['actual_optimizer_steps']")
    text=text.replace("'manifest_sha256':file_sha256(OUT/'run_manifest.json')},checkpoint)", "'manifest_sha256':file_sha256(OUT/'run_manifest.json'),'actual_optimizer_steps':actual['steps']},checkpoint)")
    text=text.replace("'metrics':measure,'already_observed_work_region_reference_activation'", "'optimizer_updates_attempted':480,'actual_optimizer_steps':actual['steps'],'metrics':measure,'already_observed_work_region_reference_activation'")
    path=ROOT/'validation/run_madenkoy_presence_corrected_control_20261005.py'
    corrected=ROOT/'experiments/madenkoy_presence_corrected_control_20261005';corrected.mkdir(exist_ok=True)
    if path.exists() and path.read_text(encoding='utf-8')!=text:
        assert not (corrected/'run_manifest.json').exists() and not (corrected/'results_partial.json').exists(), 'Never change code after any corrected fit'
        (OUT/'unexecuted_replay_bootstrap_fix.json').write_text(json.dumps({
            'previous_unexecuted_script_sha256':file_sha256(path),
            'reason':'connect frozen original input manifest,create runnable4control protocol,andadd real optimizerstep telemetry before any replayfit',
            'corrected_fit_started':False,'original_performed_protocols_or_weights_changed':False},indent=2)+'\n',encoding='utf-8')
        previous=OUT/'control_correction.json'
        if previous.exists() and not (OUT/'control_correction_before_replay_bootstrap_fix.json').exists():
            (OUT/'control_correction_before_replay_bootstrap_fix.json').write_bytes(previous.read_bytes())
    path.write_text(text,encoding='utf-8')
    proposal=load(OUT/'protocol.json');proposal.update(registered_at=datetime.now(timezone(timedelta(hours=8))).isoformat(),
          arms=['case_input_control'],maximum_initial_CNN_fits=4,
          original_protocol_sha256=file_sha256(OUT/'protocol.json'),
          control_zero_auxiliary_loss_completely_disconnected=True,
          optional_objective_sha256=file_sha256(ROOT/'auxiliary_objective.py'),
          required_actual_optimizer_steps_per_fit=480,
          role='full correctedcontrol replay only;preserve 4existingpresence treatments;not another modelsearch')
    protocol=corrected/'protocol.json'
    if protocol.exists():
        old=load(protocol);assert all(old.get(k)==v for k,v in proposal.items() if k!='registered_at')
    else:protocol.write_text(json.dumps(proposal,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    performed=load(OUT/'results.json');parent=load(ROOT/'experiments/tropical_work_matched_20261005/results.json');rows=[]
    for row in performed['models']:
        if row['arm']!='weak_positive_presence':continue
        reference=next(x for x in parent['models'] if x['arm']=='source_supported_work' and x['held']==row['held'])
        rows.append({'held':row['held'],'weak_presence_metrics':row['metrics'],
                     'frozen_prior_reference_id':reference['id'],'frozen_prior_reference_metrics':reference['metrics'],
                     'weak_presence_work_reference_activation':row['already_observed_work_region_reference_activation'],
                     'full_corrected_matched_control_retraining_performed':False})
    receipt={'performed_original_results_sha256':file_sha256(OUT/'results.json'),
      'actual_step_audit_sha256':file_sha256(OUT/'actual_optimizer_step_audit.json'),
      'original_control_status':'4completednativefits with zeroauxiliarybackward;not numericallyequivalent puredata controls;not valid toclaim matched superiority',
      'weak_presence_treatment_fits_completed':4,'original_control_fits_completed':4,'historical_fitcounts_not_removed':True,
      'corrected_replay_script':str(path),'corrected_replay_script_sha256':file_sha256(path),
      'optional_objective_sha256':file_sha256(ROOT/'auxiliary_objective.py'),
      'validated_scope':'24updateprefixsameGPUenvironment;disabledauxiliarydisconnected gives exactparentlosssequence;no full480updateorcrossdriverreproducibilityclaim',
      'short_audit_no_AMP_skips_not_proof_no_skip_in_everyhistoricalfit':True,
      'supplementary_prior_reference_comparison':rows,'new_CNN_fits_after_correction':0,
      'all_presence_treatments_failed_joint_acceptance_no_new_training_expansion':True,
      'new_accuracy_gain_or_production_promotion_claim':False}
    (OUT/'control_correction.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print('Frozen original8fits preserved;zeroobjective backward corrected forfuture;24stepexactprefixverified;no matchinggainclaim')

if __name__=='__main__':run()
