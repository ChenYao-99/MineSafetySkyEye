"""Meaningful endpoint checks with real image decoding/geometry and a stub predictor."""
import base64
import os
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import cv2
import numpy as np
import detector_service as service
from detector import LandslideDetector
from geo_tools import compute_spectral_index
from upload_utils import decode_upload, align_periods,period_chronology


def encoded(image):
    ok, raw = cv2.imencode(".png", image)
    assert ok
    return base64.b64encode(raw).decode()


class Predictor(LandslideDetector):
    def __init__(self):
        self.threshold, self.device, self.patch_size = .4, "cpu", 32
        self.calls = []

    def detect_image(self, image, **kwargs):
        self.calls.append(kwargs)
        return (image[..., 0] > 128).astype(np.float32)


class UploadContracts(unittest.TestCase):
    def setUp(self):
        self.client = service.app.test_client()
        self.predictor = Predictor()
        self.loader = patch.object(service, "_load_upload_detector", return_value=(self.predictor, "minmax", 1))
        self.loader.start()
        self.addCleanup(self.loader.stop)
        self.image = np.zeros((64, 64, 3), np.uint8)
        self.image[4:20, 4:20, 0] = 200
        self.image[..., 2] = 30

    def detect(self, **options):
        return self.client.post("/api/detect_upload", json={"image": encoded(self.image), **options})

    def test_roi_stats_geometry_and_quant_share_one_scope(self):
        whole = self.detect().get_json()
        scoped = self.detect(roi={"x0": 0, "y0": 0, "x1": .5, "y1": .5, "unit": "fraction"}).get_json()
        self.assertEqual(whole["stats"]["area_ratio"], .0625)
        self.assertEqual(scoped["stats"]["area_ratio"], .25)
        self.assertEqual(scoped["risk_level"], "red")
        self.assertEqual(scoped["quant"]["region"]["area_ratio"], scoped["stats"]["area_ratio"])
        self.assertEqual(scoped["stats"]["num_regions"], 1)
        self.assertEqual(scoped["stats"]["regions"][0]["area_px"], 256)
        overlay = cv2.imdecode(np.frombuffer(base64.b64decode(scoped["overlay_b64"]), np.uint8), cv2.IMREAD_COLOR)
        np.testing.assert_array_equal(overlay[0, 0], [217, 111, 30])
        # A usable negative ROI has spatial information even without candidates.
        # An entirely constant ROI is rejected by the input-quality guard.
        self.image[32:, 32:, 1] = np.arange(32, dtype=np.uint8)[None, :]
        empty_response = self.detect(roi=[.5, .5, 1, 1])
        self.assertEqual(empty_response.status_code, 200)
        empty = empty_response.get_json()
        self.assertEqual(empty["stats"]["num_regions"], 0)
        self.assertEqual(empty["stats"]["regions"], [])
        self.assertEqual(empty["risk_level"], "blue")

    def test_rgb_does_not_offer_physical_ndvi(self):
        result = self.detect().get_json()
        self.assertFalse(result["quant"]["ndvi"]["available"])
        self.assertNotIn("mean", result["quant"]["ndvi"])
        self.assertEqual(result["band_meta"]["nir_source"], "red_proxy")

    def test_rgba_alpha_is_never_nir(self):
        rgba = np.concatenate([self.image, np.full((64, 64, 1), 255, np.uint8)], axis=-1)
        image, meta = decode_upload(encoded(rgba))
        np.testing.assert_array_equal(image[..., 3], image[..., 2])
        self.assertFalse(meta["spectral_indices_available"])

    def test_explicit_four_band_input_requires_radiometry_for_physical_ndvi(self):
        data = np.zeros((8, 8, 4), np.uint16)
        data[..., 2], data[..., 3] = 20, 60
        data[:4, :4, 0] = 10  # NDVI is constant, but a detection image needs spatial information.
        result = self.client.post("/api/detect_upload", json={"image": encoded(data), "channel_order": "BGRN"})
        self.assertEqual(result.status_code, 200)
        self.assertEqual(result.get_json()['band_meta']['nir_source'],'measured')
        self.assertFalse(result.get_json()["quant"]["ndvi"]["available"])
        self.assertNotIn('mean',result.get_json()['quant']['ndvi'])

    def test_unknown_model_and_missing_weights_do_not_fall_back(self):
        self.assertEqual(self.detect(model_tag="unknown").status_code, 400)
        with patch.object(service, "_load_upload_detector", return_value=None):
            self.assertEqual(self.detect(model_tag="mine").status_code, 503)

    def test_invalid_gsd_and_roi_are_rejected(self):
        for gsd in (-1, "nan", "inf", "wrong"):
            with self.subTest(gsd=gsd):
                self.assertEqual(self.detect(meters_per_pixel=gsd).status_code, 400)
        self.assertEqual(self.detect(roi=[0, 0, .001, .001]).status_code, 400)

    def test_large_upload_limit_and_batch_limit(self):
        with patch.dict(os.environ, {"KAT_MAX_IMAGE_PIXELS": "32"}):
            self.assertEqual(self.detect().status_code, 400)
        with patch.dict(os.environ, {"KAT_MAX_BATCH": "64"}):
            self.assertEqual(self.detect().status_code, 200)
            self.assertEqual(self.predictor.calls[-1]["max_batch"], 64)

    def test_requested_synth_must_be_available(self):
        with patch.object(service, "_load_nir_synth", return_value=None):
            self.assertEqual(self.detect(nir_synth=True).status_code, 503)

    def test_compare_uses_selected_model_and_valid_area(self):
        result = self.client.post("/api/compare", json={"image_t1": encoded(self.image),
            "image_t2": encoded(self.image), "model_tag": "realfit"})
        self.assertEqual(result.status_code, 200)
        body = result.get_json()
        self.assertEqual(body["model_tag"], "realfit")
        self.assertEqual(body["registration"]["status"], "identical")
        self.assertEqual(body["stats"]["net_ratio"], 0)
        self.assertFalse(body["quant"]["ndvi_t1"]["available"])

    def test_unverifiable_registration_does_not_generate_changes(self):
        result = self.client.post("/api/compare", json={"image_t1": encoded(np.zeros((64, 64, 3), np.uint8)),
            "image_t2": encoded(np.ones((64, 64, 3), np.uint8))})
        self.assertEqual(result.status_code, 422)
        self.assertNotIn("stats", result.get_json())

    def test_prealigned_does_not_resize_different_extents(self):
        result = self.client.post("/api/compare", json={"image_t1": encoded(self.image),
            "image_t2": encoded(self.image[:32]), "registration": "prealigned"})
        self.assertEqual(result.status_code, 400)

    def test_auto_registration_corrects_translation(self):
        rng = np.random.default_rng(42)
        target = rng.integers(0, 256, (512, 512, 3), np.uint8)
        shifted = cv2.warpAffine(target, np.float32([[1, 0, 7], [0, 1, 5]]), (512, 512))
        t1, _ = decode_upload(encoded(shifted))
        t2, _ = decode_upload(encoded(target))
        _, valid, meta = align_periods(t1, t2)
        self.assertEqual(meta["status"], "registered")
        self.assertAlmostEqual(meta["transform"][0][2], -7, delta=1)
        self.assertAlmostEqual(meta["transform"][1][2], -5, delta=1)
        self.assertGreater(float(valid.mean()), .9)

    def test_spectral_index_rejects_a_plain_three_band_array(self):
        result = compute_spectral_index(self.image)
        self.assertFalse(result["available"])
        self.assertFalse(compute_spectral_index(np.ones((8,8,4),np.float32))['available'])
        path='D:/LMHLD/datasets/mine_benchmark_20261002/development/LasCruces/post_BGRN_10m.tif'
        self.assertTrue(compute_spectral_index(path)['available'])

    def test_invalid_reflectance_excluded_without_inventing_SWIR(self):
        data=np.zeros((1,3,4),np.float32)
        data[0,:,2]=[.1,-.1,0];data[0,:,3]=[.3,.2,0]
        result=compute_spectral_index(data,'ndvi',nir_source='measured')
        self.assertEqual(result['mean'],.5)
        self.assertEqual(result['index_valid_pixels'],1)
        self.assertEqual(result['index_excluded_pixels'],2)
        self.assertFalse(compute_spectral_index(data,'bsi',nir_source='measured')['available'])

    def test_reversed_same_time_and_unknown_chronology_are_distinguished(self):
        first={'acquisition_datetime':'2023-02-23T03:00:00Z'}
        second={'acquisition_datetime':'2023-02-15T03:00:00Z'}
        with self.assertRaises(ValueError):period_chronology(first,second)
        with self.assertRaises(ValueError):period_chronology(first,first,False)
        self.assertEqual(period_chronology(first,first,True)['status'],'same_observation')
        self.assertEqual(period_chronology({},second)['status'],'not_verified')


if __name__ == "__main__":
    unittest.main(verbosity=2)
