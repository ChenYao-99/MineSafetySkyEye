"""Malformed API requests must fail before any detector or cloud work."""
import os
import sys
import unittest
from unittest.mock import patch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ['YOLO_AUTOINSTALL'] = 'false'
import backend as service
from archive_test_support import isolate_archive


class RequestValidation(unittest.TestCase):
    def setUp(self):
        isolate_archive(self,service)
        self.client = service.app.test_client()

    def test_primitive_types_rejected_before_expensive_work(self):
        with patch.object(service, 'detector_request', side_effect=AssertionError('must not proxy')), \
             patch.object(service, 'analyze_with_agents', side_effect=AssertionError('must not call cloud')):
            for endpoint, data in [('/api/pipeline', {'num': 'bad'}), ('/api/pipeline', {'sample_index': True}),
                                   ('/api/pipeline', {'fuse': 'false'}), ('/api/knowledge', {'query': {}}),
                                   ('/api/analyze', {'question': []}), ('/api/pipeline_upload', {'image': {}}),
                                   ('/api/pipeline_compare', {'image_t1': 3, 'image_t2': 'image'})]:
                with self.subTest(endpoint=endpoint, data=data):
                    self.assertEqual(self.client.post(endpoint, json=data).status_code, 400)

    def test_oversized_requests_are_json_413(self):
        with patch.dict(service.app.config, {'MAX_CONTENT_LENGTH': 32}), \
             patch.object(service, 'detector_request', side_effect=AssertionError('must not proxy')):
            response = self.client.post('/api/detect_upload', json={'image': 'A' * 64})
        self.assertEqual(response.status_code, 413)
        self.assertEqual(response.get_json()['max_request_bytes'], 32)

    def test_missing_selected_sample_does_not_fall_back_to_other_sample(self):
        response_data = {'samples': [{'index': 2, 'risk_level': 'red'}]}
        with patch.object(service, 'detector_request', return_value=response_data) as detector, \
             patch.object(service, 'analyze_with_agents', side_effect=AssertionError('must not analyze a different sample')):
            response = self.client.post('/api/pipeline', json={'sample_index': 9})
        self.assertEqual(response.status_code, 422)
        self.assertEqual(detector.call_args.args[1]['start'], 9)
        self.assertEqual(detector.call_args.args[1]['num'], 1)
        self.assertNotIn('detection', response.get_json())

    def test_unexpected_server_failure_is_json_and_does_not_expose_diagnostics(self):
        with patch.object(service.case_archive,'list_cases',side_effect=RuntimeError('PRIVATE_PROVIDER_DIAGNOSTIC')):
            response=self.client.get('/api/cases')
        self.assertEqual(response.status_code,500);self.assertEqual(response.get_json()['status_code'],500)
        self.assertNotIn('PRIVATE_PROVIDER_DIAGNOSTIC',response.get_data(as_text=True))


if __name__ == '__main__':
    unittest.main(verbosity=2)
