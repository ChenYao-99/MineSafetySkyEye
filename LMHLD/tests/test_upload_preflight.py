"""Compressed raster limits must be checked before pixel allocation/model loading."""
import base64
import io
import os
import sys
import unittest
from unittest.mock import patch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from runtime_paths import configure_runtime_paths
configure_runtime_paths()
from PIL import Image
import detector_service as service
from upload_utils import decode_upload, positive_number


class UploadPreflight(unittest.TestCase):
    def test_oversized_compressed_header_is_rejected_before_decode(self):
        data = io.BytesIO(); Image.new('RGB', (64, 64)).save(data, 'PNG')
        encoded = base64.b64encode(data.getvalue()).decode()
        with patch.dict(os.environ, {'KAT_MAX_IMAGE_PIXELS': '32'}), \
             patch('upload_utils.cv2.imdecode', side_effect=AssertionError('must not allocate raster')):
            with self.assertRaisesRegex(ValueError, '尺寸'):
                decode_upload(encoded)

    def test_boolean_scale_is_not_accepted_as_one_meter(self):
        with self.assertRaises(ValueError):
            positive_number(True, 'meters_per_pixel')

    def test_413_response_releases_capacity_without_loading_model(self):
        with patch.dict(service.app.config, {'MAX_CONTENT_LENGTH': 32}), \
             patch.object(service, '_load_upload_detector', side_effect=AssertionError('must not load model')):
            response = service.app.test_client().post('/api/detect_upload', json={'image': 'A' * 64})
        self.assertEqual(response.status_code, 413)
        self.assertTrue(response.is_json)
        self.assertTrue(response.headers.get('X-KAT-Request-ID'))
        self.assertTrue(service._inference_lock.acquire(blocking=False))
        service._inference_lock.release()


if __name__ == '__main__':
    unittest.main(verbosity=2)
