"""Recheck the frozen reporting study and current archive handoff, without inference.

This is engineering evidence on already observed inputs, not a user trial, geological
accuracy score or new independent mine test. Historical receipts remain untouched.
"""
from pathlib import Path
from datetime import datetime
from unittest.mock import patch
import base64
import copy
import hashlib
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from project_runtime import configure_runtime_paths
configure_runtime_paths()
import backend as service
from case_archive import CaseArchive
from runtime_identity import runtime_fingerprint

BANK = Path('D:/LMHLD/experiments/same_input_reporting_workflow_20261005')
OUT = ROOT / 'cache' / 'final_workflow_evidence_20261007'
STATE = Path('D:/LMHLD/datasets/mine_benchmark_20261002/STATE.json')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def forbidden(*args, **kwargs):
    raise AssertionError('This evidence check permits no detector or cloud request')


def run():
    OUT.mkdir(parents=True, exist_ok=True)
    state_before = digest(STATE)
    protocol = read_json(BANK / 'reporting_protocol.json')
    bank_protocol = read_json(BANK / 'bank_protocol.json')
    weight = Path('D:/LMHLD/models_prior/real_ft_T1b_ours_seed42/best_model.pth')
    actual_weight_sha256 = digest(weight)
    assert actual_weight_sha256 == bank_protocol['selected_weights_sha256']
    bank = read_json(BANK / 'detector_bank.json')
    assert digest(BANK / 'detector_bank.json') == protocol['bank_sha256']
    raw = read_json(BANK / 'reporting_results.json')
    assert digest(BANK / 'reporting_protocol.json') == raw['protocol_sha256']
    interpretation = read_json(BANK / 'reporting_score_interpretation.json')
    assert raw['actual_paid_calls_initialwriter_shared_not_triplecounted'] == 19
    comparison = {}
    for arm, summary in raw['summary'].items():
        rows = [row for row in interpretation['data'] if row['arm'] == arm]
        substantive = sum(len(row['categories']['substantive_field_errors']) for row in rows)
        format_errors = sum(sum(len(value) for key, value in row['categories'].items()
                                if key != 'substantive_field_errors') for row in rows)
        assert sum(row['strict_errors'] for row in rows) == summary['wrong_field_count']
        assert substantive == interpretation['summary'][arm]['substantive_field_errors']
        comparison[arm] = {
            'tasks': summary['tasks'], 'exact_contract_tasks': summary['report_contract_passed_tasks'],
            'format_alias_missing_container_or_default_fields': format_errors,
            'substantive_field_errors': substantive,
            'mechanical_summary_error_cases': summary['summary_contract_error_cases'],
            'unprovided_citation_cases': summary['unprovided_citation_cases'],
            'logical_model_calls': summary['logical_model_calls'],
            'logical_tokens': summary['logical_token_total'],
        }

    # An isolated, fresh DB for every invocation; never write fixture cases to production.
    run_dir = OUT / datetime.now().strftime('run_%Y%m%d_%H%M%S_%f')
    run_dir.mkdir()
    store = CaseArchive(run_dir / 'acceptance.sqlite3')
    rows = []
    with patch.object(service, 'case_archive', store), \
         patch.object(service, 'detector_request', side_effect=forbidden), \
         patch.object(service, 'analyze_with_agents', side_effect=forbidden):
        client = service.app.test_client()
        case = client.post('/api/cases', json={'mine_name': '内部工程验收',
            'title': '冻结开发结果的归档复核验收', 'notes': '不是专业用户意见、现场试用或独立矿区测试'}).get_json()
        for task in bank['tasks']:
            det = copy.deepcopy(task['detection'])
            paired = 't1_ratio' in det['stats']
            saved = service.cache_report({'region': 'compare' if paired else 'upload', 'detection': det,
                'final_answer': '工程验收：保留同次量测，变化原因等待复核。', 'knowledge': [],
                'generation': {'status': 'fallback', 'source': 'detection_summary'},
                'quality': {'degraded': True}, 'agents': []}, case['case_id'])
            rid = saved['report_id']
            decision = {'expected_revision': 0, 'target_key': 'report', 'status': 'insufficient_evidence',
                'reviewer': '内部工程验收', 'evidence': '冻结开发输入；本条为功能验收模拟意见',
                'note': '需要原图和外部资料继续核查；不能用本记录证明现场效果'}
            reviewed = client.post(f'/api/archive/reports/{rid}/reviews', json=decision)
            assert reviewed.status_code == 200, reviewed.get_json()
            assert client.post(f'/api/archive/reports/{rid}/reviews', json=decision).status_code == 409
            response = client.post('/api/review_record', json={'report_id': rid, 'meters_per_pixel': 999})
            assert response.status_code == 200, response.get_json()
            body = response.get_json()
            record = json.loads(base64.b64decode(body['content_b64']))
            assert record == body['record']
            assert record['trace']['request_id'] == det['request_id'] and record['trace']['trace_complete']
            assert record['measurement_scale'] == (det.get('scale_meta') or {})
            assert record['review_revision'] == 1 and len(record['review_history']) == 1
            assert not record['project_field_trial_or_expert_endorsement']
            assert not record['accepted_human_review_decisions'][0]['reviewer_identity_verified']
            assert not record['accepted_human_review_decisions'][0]['interpretation_is_verified_ground_truth']
            restored = CaseArchive(store.path).get_report(rid)
            assert restored['detection'] == saved['detection']
            assert restored['objective_facts'] == saved['objective_facts']
            assert restored['archive']['snapshot_sha256'] == saved['archive']['snapshot_sha256']
            if paired:
                assert record['input_evidence']['t1']['acquisition_datetime'] == det['band_meta']['t1']['acquisition_datetime']
                assert abs(record['summary']['net_change_percentage_points'] - det['stats']['net_ratio'] * 100) < 1e-8
                html = client.post('/api/report_compare', json={'report_id': rid, 'format': 'html'})
                assert html.status_code == 200
                text = base64.b64decode(html.get_json()['content_b64']).decode('utf-8')
                assert rid in text and '内部工程验收' in text
            if task['fixture_kind'].startswith('synthetic'):
                assert record['summary']['area_ratio'] is None
            if task['id'] == 'LasCruces_RGB_metadata_missing':
                assert any(row['code'] == 'PHYSICAL_INDEX_UNAVAILABLE' for row in record['pending_checks'])
            rows.append({'task_id': task['id'], 'fixture_kind': task['fixture_kind'], 'report_id': rid,
                'snapshot_sha256': saved['archive']['snapshot_sha256'], 'passed': True,
                'same_request_weight_scale_and_machine_facts': True, 'review_conflict_409': True,
                'SQLite_reopen_preserves_snapshot': True, 'JSON_contains_review_without_groundtruth_promotion': True})
        backup = CaseArchive(store.backup())
        for row in rows:
            assert backup.get_report(row['report_id'])['archive']['snapshot_sha256'] == row['snapshot_sha256']
        service._report_cache.clear()
    assert digest(STATE) == state_before
    primary = ROOT / '矿安天眼_前端演示(后端版).html'
    assert primary.read_bytes() == (ROOT / 'kat_demo.html').read_bytes()
    receipt = {
        'verified_at': datetime.now().astimezone().isoformat(timespec='seconds'), 'passed': True,
        'scope': 'current local archive/export contract on7 frozen observed inputs plus1 named synthetic stress fixture',
        'new_cloud_calls': 0, 'new_detector_predictions': 0, 'new_training_or_threshold_tuning': 0,
        'new_independent_mine_test': False, 'human_trial_or_expert_endorsement': False,
        'production_archive_untouched': True, 'benchmark_STATE_unchanged_sha256': state_before,
        'current_default_weight_sha256': actual_weight_sha256,
        'same_input_reporting_study': {'historical_cloud_calls': 19, 'arms': comparison,
            'single_writer_shared_across_arms': True,
            'fixed_fields_correct_by_construction_not_AI_accuracy': True,
            'original_full_GeoMMAgent_not_benchmarked': True,
            'semantic_geological_report_correctness_not_scored': True,
            'reporting_protocol_sha256': digest(BANK / 'reporting_protocol.json'),
            'detector_bank_sha256': digest(BANK / 'detector_bank.json')},
        'current_archive_handoff': {'passed_tasks': len(rows), 'rows': rows,
            'database': str(store.path), 'consistent_backup_verified': True},
        'current_backend_runtime_sha256': runtime_fingerprint(ROOT, 'backend'),
        'current_frontend_sha256': digest(primary),
        'related_evidence': {
            'responsive_real_browser': str(ROOT / 'cache/archive_layout_20261007/reopen_result.json'),
            'earlier_frozen_release_onehour_workload': str(ROOT / 'cache/deployment_soak_20261006/final_acceptance.json')},
    }
    (OUT / 'verification.json').write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'passed_tasks': len(rows), 'comparison': comparison, 'receipt': str(OUT / 'verification.json'),
                      'new_cloud_or_detector_calls': 0}, ensure_ascii=False))


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    run()
