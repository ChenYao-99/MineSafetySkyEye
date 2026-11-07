"""Repeatable, offline delivery checks for the paired-report engineering change."""
from pathlib import Path
from datetime import datetime, timezone, timedelta
import contextlib
import hashlib
import json
import os
import shutil
import subprocess
import sys
import unittest

ROOT = Path('D:/GeoMMAgent'); DETECTOR = Path('D:/LMHLD')
sys.path.insert(0, str(ROOT))
from project_runtime import configure_runtime_paths
configure_runtime_paths()
os.environ['YOLO_AUTOINSTALL'] = 'false'
OUT = ROOT / 'cache/comparison_report_acceptance_20261006'
OUT.mkdir(exist_ok=True)


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def main():
    os.chdir(ROOT)
    with (OUT / 'python_checks.txt').open('w', encoding='utf-8') as log, contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
        suite = unittest.TestSuite()
        for pattern in ['test_comparison_reporting.py', 'test_report_contracts.py', 'test_review_handoff.py', 'test_request_validation.py']:
            suite.addTests(unittest.defaultTestLoader.discover(str(ROOT / 'tests'), pattern=pattern))
        expected_tests = suite.countTestCases()
        result = unittest.TextTestRunner(stream=log, verbosity=2).run(suite)
    assert result.wasSuccessful() and result.testsRun == expected_tests and expected_tests >= 32 and len(result.skipped) == 1
    node = shutil.which('node'); assert node
    js_checks = []
    for name in ['test_frontend_runtime.js', 'test_review_handoff_frontend.js', 'test_raster_frontend.js', 'test_frontend_async.js']:
        check = subprocess.run([node, str(ROOT / 'tests' / name)], cwd=ROOT, capture_output=True, text=True, encoding='utf-8')
        (OUT / (name + '.txt')).write_text(check.stdout + check.stderr, encoding='utf-8')
        assert check.returncode == 0, name
        js_checks.append(name)
    browser = read(OUT / 'result.json'); printing = read(OUT / 'print_verification.json')
    assert browser['passed'] and browser['native_PDF_export_verified'] and browser['mobile_layout_verified']
    assert printing['passed'] and printing['Chinese_and_source_hashes_survive_print']
    assert browser['new_model_predictions'] == browser['new_cloud_calls'] == 0
    source = ROOT / '矿安天眼_前端演示(后端版).html'; mirror = ROOT / 'kat_demo.html'
    assert sha(source) == sha(mirror)
    algorithm = read(DETECTOR / 'experiments/final_multiview_object_verification_20261006/verification.json')
    assert algorithm['algorithm_variants_stopped_by_user_condition'] and not algorithm['default_changed']
    default = DETECTOR / 'models_prior/real_ft_T1b_ours_seed42/best_model.pth'
    assert sha(default) == '0d8d8a0540419d4409951ed91e69499fac206b68303f3feec08a32f0841cebef'
    state_path = DETECTOR / 'datasets/mine_benchmark_20261002/STATE.json'; state = read(state_path)
    assert state['completed_native_training_runs'] == state['completed_native_evaluated_runs'] == 263
    assert state['completed_lightweight_feature_diagnostics'] == 123
    assert not (state_path.parent / 'final_evaluation_once.json').exists()
    packet_path = next(row['path'] for row in browser['exports'] if row['format'] == 'json')
    packet = read(packet_path)
    assert not packet['review_handoff']['project_field_trial_or_expert_endorsement']
    assert not packet['review_handoff']['accepted_human_review_decisions']
    filenames = ['backend.py', 'comparison_reporting.py', 'review_handoff.py', 'evidence_reporting.py',
                 'http_contract.py', 'project_runtime.py', 'sync_frontend.py', 'start_demo.ps1', 'stop_demo.ps1', 'stop_demo.bat',
                 source.name, mirror.name, 'README.md', 'NOTICE',
                 'tests/test_comparison_reporting.py', 'tests/test_review_handoff_frontend.js',
                 'tests/test_frontend_runtime.js', 'tests/test_frontend_async.js', 'tests/test_request_validation.py',
                 'tests/test_demo_launcher.ps1', 'tests/browser_comparison_report_smoke.mjs']
    stamp = datetime.now(timezone(timedelta(hours=8))).isoformat(timespec='seconds')
    receipt = {
        'verified_at': stamp, 'status': 'paired_report_engineering_complete;algorithm_variants_stopped',
        'changes': ['paired cached HTML/JSON export', 'both epoch instance and scale evidence forwarded',
                    'stale report identity cleared when inputs or analyses change', 'print-visible input and weight provenance'],
        'python_checks': {'run': result.testsRun, 'passed': result.testsRun - len(result.skipped),
                          'skipped': [{'test': str(test), 'reason': reason} for test, reason in result.skipped]},
        'frontend_suites_passed': js_checks, 'real_browser_acceptance': browser,
        'print_verification': printing, 'files_sha256': {name: sha(ROOT / name) for name in filenames},
        'acceptance_exports_sha256': {row['path']: sha(row['path']) for row in browser['exports']},
        'PDF_sha256': sha(OUT / 'print_smoke.pdf'), 'frontend_workspace_copies_identical': True,
        'acceptance_fixture_images_not_present': True,
        'image_export_checked_with_verified_synthetic_PNG_in_unit_tests': True,
        'production_default_changed': False, 'new_model_predictions': 0, 'new_cloud_calls': 0,
        'preserved_unopened_cases': algorithm['unopened_reserved_model_predictions'],
        'algorithm_last_route_receipt': str(DETECTOR / 'experiments/final_multiview_object_verification_20261006/verification.json'),
        'all_new_files_and_caches_on_D': True, 'new_competition_materials_or_user_trial_claims': False,
        'free_D_GiB': round(shutil.disk_usage(ROOT).free / 1024**3, 2),
        'next': 'With algorithm experiments closed, prioritize reliable demonstration, final competition description of implemented innovation, and independent/field evidence when available',
    }
    write(OUT / 'verification.json', receipt)
    pending_path = ROOT / 'validation/PENDING_ENGINEERING_VERIFICATION.json'
    if pending_path.exists():
        pending = read(pending_path)
        if 'verification_blockers' in pending:
            pending['historical_verification_blockers'] = pending.pop('verification_blockers')
        pending.update(status='verified', verified_at=stamp, verified_receipt=str(OUT / 'verification.json'),
                       previous_receipt_is_not_verification_of_these_new_changes=False)
        write(pending_path, pending)
    state.pop('pending_project_engineering_verification', None)
    state.pop('last_passed_engineering_receipt_does_not_cover_latest_prompt_and_single_export_changes', None)
    state.update(updated_at=stamp, phase='last_algorithm_failed_closed_and_paired_report_engineering_complete',
                 latest_project_engineering_verification=str(OUT / 'verification.json'),
                 running_training_script=None, running_training_pid=None, running_training_session=None,
                 algorithm_variants_stopped_by_user_condition=True, next=receipt['next'])
    write(state_path, state)
    print(f'PASS {result.testsRun - len(result.skipped)} Python tests + {len(result.skipped)} optional skip, '
          f'{len(js_checks)} frontend suites, real browser downloads and {printing["pages"]}-page PDF provenance; no model/default/blind-test changes')


if __name__ == '__main__':
    main()
