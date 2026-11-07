"""Same-request paired export, sensitive-field exclusion and standalone HTML safety."""
import base64
import copy
import io
import json
import os
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ['YOLO_AUTOINSTALL'] = 'false'
import backend as service
from archive_test_support import isolate_archive
from comparison_reporting import export_comparison
from PIL import Image


class ComparisonReporting(unittest.TestCase):
    def setUp(self):
        isolate_archive(self,service)
        service._report_cache.clear()
        self.client = service.app.test_client()
        picture = io.BytesIO(); Image.new('RGB', (8, 6), '#147b95').save(picture, format='PNG')
        image = base64.b64encode(picture.getvalue()).decode('ascii')
        self.result = service.cache_report({
            'region': 'compare', 'final_answer': '<script>malicious()</script> 中文报告',
            'generation': {'status': 'fallback', 'source': 'detection_summary', 'reason': 'PRIVATE_PROVIDER_ERROR'},
            'quality': {'degraded': True, 'api_key': 'PRIVATE_KEY'},
            'detection': {'request_id': 'paired-request', 'model_tag': 'mine', 'threshold': .4,
                'model_info': {'weights_sha256': 'a' * 64, 'api_key': 'PRIVATE_KEY'},
                'stats': {'t1_ratio': .1, 't2_ratio': .2, 'net_ratio': .1,
                          'n_added_regions': 2, 'n_expanded_regions': 3, 'n_removed_regions': 1, 'valid_pixels': 48},
                'registration': {'status': 'aligned', 'diagnostic': '<img src=x onerror=malicious()>'},
                'input_quality': {'status': 'usable_observation', 'api_key': 'PRIVATE_KEY'},
                'band_meta': {epoch: {'acquisition_datetime': date, 'input_sha256': letter * 64}
                              for epoch, date, letter in [('t1', '2019-01-12T11:00:00Z', 'b'), ('t2', '2019-01-27T11:00:00Z', 'c')]},
                'overlay_b64': image, 't1_overlay_b64': image, 't2_overlay_b64': image,
                'change_instances': {'instances': [{'id': 2, 'area_px': 7, 'change': 'added'}],
                                     'removed_instances': [{'id': 1, 'area_px': 3, 'change': 'removed'}],
                                     'summary': {'n_added': 1, 'n_removed': 1}}},
            'knowledge': [{'id': 'REF-1', 'title': '<script>bad()</script>', 'source_url': 'javascript:malicious()'},
                          {'id': 'REF-2', 'title': '参考资料', 'source_url': 'https://example.org/source'}],
        })

    def export(self, format):
        # Accidentally recomputing analysis or invoking the detector fails the test.
        with patch.object(service, 'run_pipeline_compare', side_effect=AssertionError('must not rerun')), \
             patch.object(service, 'detector_request', side_effect=AssertionError('must not infer')), \
             patch.object(service, 'analyze_with_agents', side_effect=AssertionError('must not call cloud')):
            response = self.client.post('/api/report_compare', json={'report_id': self.result['report_id'], 'format': format})
        self.assertEqual(response.status_code, 200)
        body = response.get_json()
        return body, base64.b64decode(body['content_b64']).decode('utf-8')

    def test_json_preserves_request_times_weights_measurements_and_pending_review(self):
        body, text = self.export('json'); record = json.loads(text)
        handoff = record['review_handoff']
        self.assertEqual(body['report_id'], self.result['report_id'])
        self.assertEqual(handoff['trace']['request_id'], 'paired-request')
        self.assertEqual(handoff['trace']['weights_sha256'], ['a' * 64])
        self.assertEqual(record['objective_facts'], self.result['objective_facts'])
        self.assertEqual(record['generated_at'], self.result['generated_at'])
        self.assertEqual(handoff['input_evidence']['t2']['acquisition_datetime'], '2019-01-27T11:00:00Z')
        self.assertEqual(handoff['registration'], self.result['detection']['registration'])
        self.assertEqual(handoff['review_items'][0]['review_status'], 'not_reviewed')
        self.assertEqual(handoff['accepted_human_review_decisions'], [])
        self.assertFalse(handoff['project_field_trial_or_expert_endorsement'])
        self.assertFalse(record['illustrations_included_in_json'])
        self.assertNotIn('PRIVATE_KEY', text); self.assertNotIn('PRIVATE_PROVIDER_ERROR', text)
        self.assertNotIn('overlay_b64', text)

    def test_html_escapes_text_blocks_active_links_and_embeds_three_images(self):
        body, text = self.export('html')
        self.assertEqual(body['mime_type'], 'text/html')
        self.assertIn('双期影像复核报告', text); self.assertIn('同次变化量测', text)
        self.assertIn('10.0 个百分点', text); self.assertIn('检测事实汇总（自动分析未完成）', text)
        self.assertIn('&lt;script&gt;', text); self.assertNotIn('<script>', text)
        self.assertNotIn('href="javascript:', text)
        self.assertIn('href="https://example.org/source"', text)
        self.assertEqual(text.count('<img src="data:image/png;base64,'), 3)
        self.assertNotIn('PRIVATE_KEY', text); self.assertNotIn('PRIVATE_PROVIDER_ERROR', text)
        self.assertIn('2019-01-27T11:00:00Z', text)
        self.assertIn('<h2>结果来源标识</h2>', text)
        self.assertIn('模型权重 SHA-256', text)

    def test_invalid_image_is_omitted_without_losing_report(self):
        self.result['detection']['overlay_b64'] = base64.b64encode(b'<svg onload="malicious()"/>').decode()
        text = base64.b64decode(export_comparison(self.result)['content_b64']).decode()
        self.assertNotIn('data:image/svg', text)
        self.assertIn('本次结果未提供可用图像', text)

    def test_missing_invalid_expired_and_wrong_slot_reports_do_not_generate(self):
        with patch.object(service, 'run_pipeline_compare', side_effect=AssertionError('must not rerun')):
            for payload in ([], {}, {'report_id': []}, {'report_id': self.result['report_id'], 'format': 'png'}):
                self.assertEqual(self.client.post('/api/report_compare', json=payload).status_code, 400)
            self.assertEqual(self.client.post('/api/report_compare', json={'report_id': 'missing'}).status_code, 410)
            original = service._report_cache[self.result['report_id']]
            service._report_cache[self.result['report_id']] = (original[0] - 1801, original[1])
            self.assertEqual(self.client.post('/api/report_compare', json={'report_id': self.result['report_id']}).status_code, 200)
            wrong=copy.deepcopy(self.result);wrong['region']='upload'
            wrong=service.cache_report(wrong)
            self.assertEqual(self.client.post('/api/report_compare', json={'report_id': wrong['report_id']}).status_code, 400)

    def test_incomplete_trace_is_rejected(self):
        invalid=copy.deepcopy(self.result);invalid['detection']['model_info']={}
        invalid=service.cache_report(invalid)
        self.assertEqual(self.client.post('/api/report_compare', json={'report_id': invalid['report_id']}).status_code, 422)

    def test_real_pipeline_forwards_both_epoch_regions_to_cached_export(self):
        detector = copy.deepcopy(self.result['detection'])
        detector['change_overlay_b64'] = detector.pop('overlay_b64')
        detector['scale_meta'] = {'source': 'projected_geotiff', 'meters_per_pixel': 10}
        detector['size'] = [8, 6]
        detector['t2_rgb_b64']=detector['t2_overlay_b64']
        quality = {'generation': {'status': 'fallback', 'source': 'detection_summary'}, 'degraded': True}
        with patch.object(service, 'detector_request', return_value=detector), \
             patch.object(service, 'analyze_with_agents', return_value=([], '实际接口离线事实汇总', None, quality)):
            response = self.client.post('/api/pipeline_compare', json={'image_t1': 'original1', 'image_t2': 'original2'})
        self.assertEqual(response.status_code, 200)
        body = response.get_json()
        self.assertEqual(body['detection']['change_instances'], detector['change_instances'])
        self.assertEqual(body['detection']['size'], [8, 6])
        self.assertEqual(body['detection']['t2_rgb_b64'],detector['t2_rgb_b64'])
        self.assertEqual(body['review_handoff']['measurement_scale'], detector['scale_meta'])
        items = body['review_handoff']['review_items']
        self.assertEqual({item['review_item_key'] for item in items}, {'t2:2', 't1:1'})
        self.result = body
        _, text = self.export('json'); exported = json.loads(text)['review_handoff']
        self.assertEqual(exported['review_items'], items)
        self.assertEqual(exported['summary']['matched_instance_summary'], {'n_added': 1, 'n_removed': 1})

    def test_pair_prompt_uses_percentage_points_and_does_not_certify_stability(self):
        detector = copy.deepcopy(self.result['detection'])
        detector['change_overlay_b64'] = detector.pop('overlay_b64')
        for field in ('n_added_regions', 'n_expanded_regions', 'n_removed_regions'):
            detector['stats'][field] = 0
        quality = {'generation': {'status': 'fallback', 'source': 'detection_summary'}}
        with patch.object(service, 'detector_request', return_value=detector), \
             patch.object(service, 'analyze_with_agents', return_value=([], '事实汇总', None, quality)) as analyze:
            result = service.run_pipeline_compare('original1', 'original2')
        question = analyze.call_args.args[0]
        self.assertIn('净变化 +10.00 个百分点', question)
        self.assertIn('模型掩膜像素占比', question)
        self.assertIn('差分连通区域不等于事故数量或匹配实例数量', question)
        self.assertNotIn('风险区域基本稳定', question)
        self.assertNotIn('净变化 +10.0%', question)
        self.assertEqual(result['detection']['stats'], detector['stats'])


if __name__ == '__main__':
    unittest.main(verbosity=2)
