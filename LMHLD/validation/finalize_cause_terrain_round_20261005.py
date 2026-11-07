"""Verify the bounded temporal/cause experiment and retire only rejected checkpoints."""
from pathlib import Path
from datetime import datetime, timezone, timedelta
import argparse
import json
import shutil
import sys

ROOT = Path('D:/LMHLD')
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'validation'))
from runtime_paths import configure_runtime_paths
configure_runtime_paths()
import numpy as np
from model_registry import ModelRegistry, file_sha256
from mine_data_protocol import registry, reserved_case_ids, verify_assets

OUT = ROOT / 'experiments/cause_terrain_round_20261005'
DATA = ROOT / 'datasets/mine_benchmark_20261002'
FOCUS = ROOT / 'experiments/reviewed_normal_focus_20261005'
BN = ROOT / 'experiments/label_free_scene_BN_20261005'


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def now():
    return datetime.now(timezone(timedelta(hours=8))).isoformat(timespec='seconds')


def frozen(path, digest):
    assert file_sha256(path) == digest, f'Frozen file changed: {path}'


def protected_runtime():
    inventory = load(ROOT / 'model_registry.json')
    runtime = ModelRegistry(ROOT / 'model_registry.json', ROOT / 'models_prior')
    protected = {}
    for key, entry in inventory['artifacts'].items():
        path = ROOT / 'models_prior' / entry['path']
        runtime.verify(key, path, entry['protocol'])
        protected[str(path.resolve())] = entry['sha256']
    adapter = load(ROOT / 'models_prior/rgb_guarded_20261001/adapter.json')
    for entry in adapter['components'].values():
        path = ROOT / 'models_prior' / entry['path']
        frozen(path, entry['sha256'])
        protected[str(path.resolve())] = entry['sha256']
    assert inventory['artifacts']['upload:mine']['sha256'] == '0d8d8a0540419d4409951ed91e69499fac206b68303f3feec08a32f0841cebef'
    return protected


def verify_round():
    inventory = registry()
    verify_assets(inventory)
    reserved = {case: inventory['cases'][case]['model_predictions_run']
                for case in sorted(reserved_case_ids(inventory))}
    assert len(reserved) == 4 and all(count == 0 for count in reserved.values())
    assert not (DATA / 'final_evaluation_once.json').exists()
    protected = protected_runtime()
    result, focus, bn = load(OUT / 'results.json'), load(FOCUS / 'results.json'), load(BN / 'results.json')
    protocol = load(OUT / 'training_protocol.json')
    manifest = load(OUT / 'inputs/manifest.json')
    frozen(OUT / 'round_plan.json', protocol['round_plan_sha256'])
    frozen(OUT / 'inputs/manifest.json', protocol['input_manifest_sha256'])
    frozen(ROOT / 'validation/run_cause_terrain_CNN_20261005.py', protocol['implementation_sha256'])
    frozen(ROOT / 'cause_temporal_network.py', protocol['network_loss_sha256'])
    frozen(ROOT / 'cause_supervision.py', manifest['target_code_sha256'])
    frozen(ROOT / 'historical_terrain_features.py', manifest['terrain_feature_code_sha256'])
    frozen(ROOT / 'cache/torch/cause_encoder/resnet18-f37072fd.pth', protocol['generic_encoder_sha256'])
    terrain_path = ROOT / 'datasets/historical_terrain_development_20261005'
    frozen(terrain_path / 'acquisition.json', manifest['terrain_acquisition_sha256'])
    terrain = load(terrain_path / 'acquisition.json')
    assert len(terrain['cases']) == 11 and not (set(terrain['cases']) & set(reserved))
    terrain_assets = 0
    for case, entry in terrain['cases'].items():
        for name, digest in entry['assets_sha256'].items():
            frozen(terrain_path / case / name, digest)
            terrain_assets += 1
    for name, digest in manifest['assets'].items():
        path = (OUT / 'inputs' / name).resolve()
        assert path.is_relative_to((OUT / 'inputs').resolve())
        frozen(path, digest)
    for case, entry in manifest['cases'].items():
        assert case not in reserved
        for name, digest in entry['source_dependencies'].items():
            assert Path(name).drive.upper() == 'D:'
            frozen(name, digest)
        directory = OUT / 'inputs' / case
        support = np.load(directory / 'support.npy', mmap_mode='r')
        for target, confidence in [('binary_accident', 'binary_confidence'),
                                   ('independent_accident', 'independent_confidence'),
                                   ('work', 'work_confidence')]:
            y, weight = np.load(directory / (target + '.npy')), np.load(directory / (confidence + '.npy'))
            assert set(np.unique(y)) <= {0, 1, 255}
            assert np.isfinite(weight).all() and np.all((weight >= 0) & (weight <= 1))
            assert np.all(weight[y == 255] == 0)
            assert np.all(weight[support[0] == 0] == 0)
    annual = OUT / 'inputs/Amrun_annual_SLATS2324'
    work, zone = np.load(annual / 'work.npy'), np.load(annual / 'work_evaluation_zone.npy')
    allowed = np.load(annual / 'training_allowed.npy')
    independent = np.load(annual / 'independent_accident.npy')
    assert np.all(independent[work == 1] == 255)
    assert np.all(allowed[zone > 0] == 0)
    annual_cut = manifest['cases']['Amrun_annual_SLATS2324']['work_evaluation_column_from']
    assert not np.any(allowed[:, annual_cut-96:])
    assert result['completed_new_CNN_fits'] == 12 and len(result['models']) == 12
    assert focus['completed_new_CNN_fits'] == 4 and len(focus['models']) == 4
    assert bn['new_supervised_fits'] == 0 and len(bn['models']) == 4
    assert not focus['joint_primary_gates_pass'] and not bn['joint_primary_gates_pass']
    assert all(not family['joint_primary_gates_pass'] for family in result['families'].values())
    schedule_rows = {}
    for fold in protocol['folds']:
        held = fold['held']
        rows = [row for row in result['models'] if row['held'] == held]
        assert len(rows) == 3
        train = rows[0]['training_groups']
        assert held not in train and not (set(train) & set(reserved))
        assert train[:3] == fold['train']
        path = OUT / (held + '_patch_schedule.npz')
        for row in rows:
            assert row['training_groups'] == train and row['epochs'] == 10 and row['updates'] == 480
            frozen(path, row['patch_schedule_sha256'])
        with np.load(path) as schedule:
            assert len(schedule['x']) == 3840
            source_index, xs, ys = schedule['source_index'], schedule['x'], schedule['y']
            assert np.all((source_index >= 0) & (source_index < len(train)))
            annual_mask = source_index == train.index('Amrun_annual_SLATS2324')
            assert np.all(xs[annual_mask] + 48 < annual_cut)
            for index, case in enumerate(train):
                selected = source_index == index
                permissible = np.load(OUT / 'inputs' / case / 'training_allowed.npy', mmap_mode='r')
                assert np.all(permissible[ys[selected], xs[selected]] > 0)
            schedule_rows[held] = {'native_training_crops': 3840,
                                   'annual_training_crops': int(annual_mask.sum()),
                                   'no_training_crop_intersects_annual_evaluation_zone': True,
                                   'held_mine_training_crops': 0}
    focus_protocol = load(FOCUS / 'protocol.json')
    frozen(ROOT / 'validation/run_reviewed_normal_focus_20261005.py', focus_protocol['implementation_sha256'])
    frozen(OUT / 'training_protocol.json', focus_protocol['parent_protocol_sha256'])
    frozen(OUT / 'results.json', focus_protocol['parent_results_sha256'])
    bn_protocol = load(BN / 'protocol.json')
    frozen(ROOT / 'label_free_scene_bn.py', bn_protocol['statistic_code_sha256'])
    frozen(ROOT / 'validation/evaluate_label_free_BN_20261005.py', bn_protocol['implementation_sha256'])
    for row in bn['models']:
        assert row['calibration']['parameter_updates'] == 0 and not row['calibration']['target_labels_consumed']
        assert row['reserved_predictions'] == 0
    comparison = load(OUT / 'same_pixel_runtime_comparison.json')
    assert comparison['reference_pixels'] == 32641 and comparison['reserved_predictions'] == 0
    for row in comparison['candidate_replays']:
        frozen(row['probability'], row['probability_sha256'])
    return {'protected_runtime_artifacts': protected, 'reserved_predictions': reserved,
            'frozen_native_input_arrays_verified': len(manifest['assets']),
            'historical_terrain_cases': 11, 'historical_terrain_assets_verified': terrain_assets,
            'training_schedules': schedule_rows, 'primary': result, 'focus': focus, 'bn': bn,
            'same_pixel_comparison': comparison}


def prepare_cleanup():
    verified = verify_round()
    retained, rejected = [], []
    primary = verified['primary']
    for row in primary['models'] + verified['focus']['models']:
        path = Path(row['weights']).resolve()
        roots = [ROOT / 'models_prior/cause_terrain_round_20261005',
                 ROOT / 'models_prior/reviewed_normal_focus_20261005']
        assert any(path.is_relative_to(root.resolve()) for root in roots)
        assert path.name == 'final_model.pth' and not path.is_symlink()
        assert str(path) not in verified['protected_runtime_artifacts']
        item = {'id': row['id'], 'path': str(path), 'sha256': row['weights_sha256']}
        if row.get('arm') == 'binary_temporal':
            frozen(path, row['weights_sha256'])
            retained.append(item)
        else:
            assert path.exists(), 'Prepare cleanup must precede removing rejected weights'
            frozen(path, row['weights_sha256'])
            item.update(bytes=path.stat().st_size, reason='failed joint development gates;retrain from frozen protocol if needed')
            rejected.append(item)
    assert len(retained) == 4 and len(rejected) == 12
    write(OUT / 'rejected_checkpoint_cleanup_plan.json', {
        'prepared_at': now(), 'deletion_authorized_by_prior_user_cleanup_request': True,
        'exact_files_only': True, 'retained_candidates': retained, 'rejected_files': rejected,
        'raw_inputs_labels_probabilities_protocols_and_runtime_artifacts_to_be_preserved': True,
        'planned_bytes': sum(row['bytes'] for row in rejected),
        'verification_before_cleanup': {key: value for key, value in verified.items()
                                        if key not in {'primary', 'focus', 'bn', 'same_pixel_comparison'}},
    })
    print('Verified inputs and protected assets; cleanup plan:', len(rejected), 'rejected;4 candidates retained')


def complete():
    verified = verify_round()
    plan = load(OUT / 'rejected_checkpoint_cleanup_plan.json')
    cleanup = load(OUT / 'rejected_checkpoint_cleanup.json')
    assert cleanup['deleted_files'] == len(plan['rejected_files']) == 12
    assert cleanup['deleted_bytes'] == plan['planned_bytes']
    for row in plan['retained_candidates']:
        frozen(row['path'], row['sha256'])
    for row in plan['rejected_files']:
        assert not Path(row['path']).exists()
    resume = ROOT / 'cache/cause_terrain_resume_20261005'
    assert not resume.exists() or not list(resume.iterdir())
    state_path = DATA / 'STATE.json'
    state = load(state_path)
    if state.get('latest_completed_cause_terrain_round') is None:
        assert state['completed_native_training_runs'] == state['completed_native_evaluated_runs'] == 164
        state['completed_native_training_runs'] = state['completed_native_evaluated_runs'] = 180
    else:
        assert state['latest_completed_cause_terrain_round'] == str(OUT / 'verification.json')
        assert state['completed_native_training_runs'] == state['completed_native_evaluated_runs'] == 180
    assert state['completed_lightweight_feature_diagnostics'] == 119
    result = verified['primary']
    receipt = {
        'verified_at': now(), 'status': 'bounded temporal pivot completed;binary candidate promising;all joint gates still fail',
        'new_completed_native_CNN_fits': 16, 'primary_matched_CNN_fits': 12,
        'post_result_reviewed_normal_followup_CNN_fits': 4,
        'new_label_free_statistic_diagnostics': 4, 'new_supervised_fits_in_statistic_diagnostics': 0,
        'native_CNN_total': 180, 'lightweight_feature_diagnostics_total': 119,
        'unit_and_observation_contract_tests': {'command': 'python -m unittest tests.test_independent_cause_supervision tests.test_historical_terrain_and_scene_statistics -v',
                                               'completed_tests': 9, 'exit_code': 0},
        'data_and_protocol_checks': {key: value for key, value in verified.items()
                                    if key not in {'primary', 'focus', 'bn', 'same_pixel_comparison'}},
        'primary_families': result['families'],
        'normal_focus': {key: value for key, value in verified['focus'].items() if key != 'models'},
        'label_free_BN': {key: value for key, value in verified['bn'].items() if key != 'models'},
        'same_pixel_runtime_comparison': str(OUT / 'same_pixel_runtime_comparison.json'),
        'same_pixel_default_activation': verified['same_pixel_comparison']['default_expected_work_activation_same_pixels'],
        'same_pixel_candidate_mean_activation': verified['same_pixel_comparison']['candidate_macro_expected_work_activation_same_pixels'],
        'retained_candidates': plan['retained_candidates'], 'checkpoint_cleanup': cleanup,
        'deleted_checkpoints_not_available_for_replay_without_retraining': True,
        'normal_focus_protocol_typographic_key_960_is96_native_pixels': True,
        'scope': 'repeatedly observed development leave-one-mine-out;not fresh blind or field testing',
        'normal_and_annual_references_are_weak_expected_operation_not_verified_safety': True,
        'multi_seed_threshold_sweep_and_per_mine_result_cherrypicking_performed': False,
        'new_complete_independent_accident_boundaries': 0, 'new_original_professional_survey_GIS': 0,
        'new_project_field_trials_or_expert_feedback': 0,
        'new_cloud_calls': 0, 'materials_generated': False,
        'production_default_changed': False, 'production_candidate_source_and_service_regression_not_completed': True,
        'all_downloads_model_caches_and_test_temp_files_on_D': True,
        'free_D_GiB': round(shutil.disk_usage(ROOT).free / 1024**3, 2),
        'next': 'continue fixed native temporal binary baseline;prioritize independently supported tropical excavation/haul-road competing-cause examples and light-arid pitwall positives;stop this cause-head/oldDSM branch;do not reopen held development per-mine routing or final tests',
    }
    write(OUT / 'verification.json', receipt)
    state.update(updated_at=receipt['verified_at'],
                 phase='mature_native_temporal_candidate_retained_all_joint_gates_still_fail',
                 current_study=str(OUT / 'training_protocol.json'),
                 new_native_CNN_fits_current_phase=16, new_native_training_runs_planned=16,
                 latest_completed_cause_terrain_round=str(OUT / 'verification.json'),
                 latest_experimental_temporal_candidate_metrics=result['families']['binary_temporal'],
                 latest_historical_terrain_development=str(ROOT / 'datasets/historical_terrain_development_20261005/acquisition.json'),
                 historical_terrain_context_not_current_survey_or_deformation=True,
                 latest_rejected_diagnostic_weight_cleanup=str(OUT / 'rejected_checkpoint_cleanup.json'),
                 free_D_GiB=receipt['free_D_GiB'],
                 default_model_changed=False, new_final_test_predictions=0,
                 current_model_selection_allowed_on_new_test=False, running_training_session=None,
                 running_evaluation_session=None, running_download_session=None,
                 running_training_script=None, running_download_script=None,
                 next=receipt['next'], next_local_work=receipt['next'])
    state['artifact_retention']['cause_terrain_rejected_checkpoints'] = {
        'receipt': str(OUT / 'rejected_checkpoint_cleanup.json'), 'deleted_files': 12,
        'retained_candidate_checkpoints': 4, 'freed_MiB': cleanup['deleted_bytes'] / 1024**2,
        'sources_labels_predictions_and_protected_runtime_artifacts_removed': False,
    }
    write(state_path, state)
    print('Complete:16CNN fits;180total;9tests;4retained;12rejected removed;4reserves0;default unchanged')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--prepare-cleanup', action='store_true')
    parser.add_argument('--complete', action='store_true')
    args = parser.parse_args()
    if args.prepare_cleanup:
        prepare_cleanup()
    elif args.complete:
        complete()
    else:
        parser.error('Choose --prepare-cleanup or --complete')
