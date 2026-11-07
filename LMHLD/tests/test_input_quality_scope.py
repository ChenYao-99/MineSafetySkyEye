"""Reject uninformative analysis areas before loading a detection model."""
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from rasterio.transform import from_origin

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import detector_service as service
from input_quality import InputQualityError, inspect_quality, require_usable_input
from test_geotiff_upload import Stub, tiff_data


class InputQualityScopeTests(unittest.TestCase):
    def setUp(self):
        self.client = service.app.test_client()
        self.image = np.full((64, 64, 4), 100, np.float32)
        self.image[:16, :16, 0] = np.arange(16)[None, :] + 120
        self.roi = np.zeros((64, 64), bool)
        self.roi[32:48, 32:48] = True

    def test_texture_outside_requested_roi_does_not_validate_uniform_roi(self):
        self.assertEqual(inspect_quality(self.image, {})['status'], 'usable_observation')
        with self.assertRaises(InputQualityError) as caught:
            require_usable_input(self.image, {}, self.roi)
        self.assertEqual(caught.exception.quality['observed_pixels'], 256)
        self.assertFalse(caught.exception.quality['visible_has_spatial_information'])

    def test_invalid_roi_pixels_do_not_supply_spatial_information(self):
        image = self.image.copy()
        image[32:40, 32:48, 0] = 240
        valid = np.ones((64, 64), bool)
        valid[32:40, 32:48] = False
        with self.assertRaises(InputQualityError):
            require_usable_input(image, {'_valid_mask': valid}, self.roi)

    def test_informative_roi_is_accepted_with_its_own_observed_counts(self):
        image = self.image.copy()
        image[32:48, 32:48, 0] = np.arange(16)[None, :] + 100
        quality = require_usable_input(image, {}, self.roi)
        self.assertEqual(quality['scope_pixels'], 256)
        self.assertEqual(quality['observed_pixels'], 256)
        self.assertTrue(quality['visible_has_spatial_information'])

    def test_roi_api_rejects_before_model_loading_and_releases_capacity(self):
        encoded = tiff_data(self.image.transpose(2, 0, 1))
        with patch.object(service, '_load_upload_detector', return_value=(Stub(), 'minmax', 1)) as loader:
            response = self.client.post('/api/detect_upload', json={
                'image': encoded, 'roi': [32, 32, 48, 48]})
        self.assertEqual(response.status_code, 422, response.get_json().get('error'))
        self.assertNotIn('risk_level', response.get_json())
        loader.assert_not_called()
        self.assertFalse(service._inference_lock.locked())

    def pair(self, common_texture=False):
        first = np.full((5, 64, 64), 100, np.uint16)
        second = first.copy()
        first[4] = second[4] = 5
        first[4, :, 56:] = 9
        second[4, :, :8] = 9
        first[0, :, :8] = np.arange(8)[None, :] + 130
        second[0, :, 56:] = np.arange(8)[None, :] + 140
        if common_texture:
            first[0, :, 16:32] = np.arange(16)[None, :] + 120
            second[0, :, 16:32] = np.arange(16)[None, :] + 130
        descriptions = ('B02', 'B03', 'B04', 'B08', 'SCL')
        return {'image_t1': tiff_data(first, descriptions=descriptions),
                'image_t2': tiff_data(second, descriptions=descriptions)}

    def test_pair_rejects_texture_that_only_exists_outside_common_observation(self):
        with patch.object(service, '_load_upload_detector', return_value=(Stub(), 'minmax', 1)) as loader:
            response = self.client.post('/api/compare', json=self.pair())
        self.assertEqual(response.status_code, 422, response.get_json().get('error'))
        self.assertNotIn('stats', response.get_json())
        self.assertEqual(response.get_json()['input_quality']['analysis_scope'], 'common_valid_pixels')
        self.assertEqual(response.get_json()['input_quality']['epoch'], 't1')
        loader.assert_not_called()
        self.assertFalse(service._inference_lock.locked())

    def test_pair_preserves_source_quality_and_reports_common_scope_quality(self):
        with patch.object(service, '_load_upload_detector', return_value=(Stub(), 'minmax', 1)):
            response = self.client.post('/api/compare', json=self.pair(common_texture=True))
        self.assertEqual(response.status_code, 200, response.get_json().get('error'))
        body = response.get_json()
        self.assertEqual(body['stats']['valid_pixels'], 64 * 48)
        for epoch in ('t1', 't2'):
            self.assertEqual(body['input_quality'][epoch]['observed_pixels'], 64 * 56)
            quality = body['input_quality']['common_valid'][epoch]
            self.assertEqual(quality['observed_pixels'], 64 * 48)
            self.assertEqual(quality['analysis_scope'], 'common_valid_pixels')
            self.assertTrue(quality['visible_has_spatial_information'])

    def test_pair_checks_second_epoch_common_scope_independently(self):
        payload = self.pair()
        first = np.full((5, 64, 64), 100, np.uint16)
        first[4] = 5
        first[4, :, 56:] = 9
        first[0, :, 16:32] = np.arange(16)[None, :] + 120
        payload['image_t1'] = tiff_data(first, descriptions=('B02', 'B03', 'B04', 'B08', 'SCL'))
        with patch.object(service, '_load_upload_detector', return_value=(Stub(), 'minmax', 1)) as loader:
            response = self.client.post('/api/compare', json=payload)
        self.assertEqual(response.status_code, 422, response.get_json().get('error'))
        self.assertEqual(response.get_json()['input_quality']['epoch'], 't2')
        loader.assert_not_called()

    def test_reference_grid_rejects_texture_lost_by_actual_averaging(self):
        image = np.full((4, 128, 128), 100, np.uint16)
        image[0] += (np.indices((128, 128)).sum(axis=0) % 2).astype(np.uint16) * 100
        payload = {'image': tiff_data(image, transform=from_origin(754950, 4156920, 2.5, 2.5)),
                   'analysis_grid': 'reference_10m'}
        with patch.object(service, '_load_upload_detector', return_value=(Stub(), 'minmax', 1)) as loader:
            response = self.client.post('/api/detect_upload', json=payload)
        self.assertEqual(response.status_code, 422, response.get_json().get('error'))
        self.assertEqual(response.get_json()['input_quality']['analysis_scope'], 'reference_10m')
        loader.assert_not_called()

    def test_reference_grid_quality_is_evaluated_in_requested_roi(self):
        image = np.repeat(np.repeat(self.image.transpose(2, 0, 1), 4, axis=1), 4, axis=2)
        # Detail inside ROI disappears at 10m; retained texture outside must not validate it.
        checker = np.indices((64, 64)).sum(axis=0) % 2
        image[0, 128:192, 128:192] = 100 + checker * 100
        payload = {'image': tiff_data(image, transform=from_origin(754950, 4156920, 2.5, 2.5)),
                   'roi': [128, 128, 192, 192], 'analysis_grid': 'reference_10m'}
        with patch.object(service, '_load_upload_detector', return_value=(Stub(), 'minmax', 1)) as loader:
            response = self.client.post('/api/detect_upload', json=payload)
        self.assertEqual(response.status_code, 422, response.get_json().get('error'))
        self.assertEqual(response.get_json()['input_quality']['analysis_scope'], 'reference_10m')
        loader.assert_not_called()


if __name__ == '__main__':
    unittest.main(verbosity=2)
