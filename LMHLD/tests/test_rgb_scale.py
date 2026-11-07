"""Scientific guards for missing-band ablation and partial mine references."""
import sys
from pathlib import Path
import unittest
import numpy as np
sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parents[1]/'validation')]
from runtime_paths import configure_runtime_paths
configure_runtime_paths()
import rgb_scale as rgb


class RGBScaleTests(unittest.TestCase):
    def test_rgb_cannot_observe_actual_nir(self):
        x = np.random.default_rng(7).normal(size=(4, 32, 32)).astype(np.float32)
        expected = rgb.make_input_views(x, 'rgb_proxy')
        x[3] = np.nan
        np.testing.assert_array_equal(expected, rgb.make_input_views(x, 'rgb_proxy'))
        np.testing.assert_array_equal(expected[:, 2], expected[:, 3])
        np.testing.assert_array_equal(expected, rgb.make_input_views(x[:3], 'rgb_proxy'))
        with self.assertRaises(ValueError): rgb.make_input_views(x, 'actual_bgrn')

    def test_mode_and_normalization_are_explicit(self):
        x = np.ones((4, 32, 32), np.float32)
        with self.assertRaises(ValueError): rgb.make_input_views(x, None)
        views = rgb.make_input_views(x, 'rgb_proxy')
        self.assertEqual(float(views.max()), 0.)
        import torch
        model = torch.nn.Conv2d(4, 1, 1)
        with self.assertRaises(ValueError): rgb.infer_grids(model, views[None]+255)

    def test_tta_inverse_and_batching_do_not_change_pointwise_model(self):
        import torch
        model = torch.nn.Conv2d(4, 1, 1).eval()
        raw = np.random.default_rng(8).random((4, 32, 32)).astype(np.float32)
        x = np.stack([rgb.make_input_views(raw, 'rgb_proxy')]*3)
        a = rgb.infer_grids(model, x, batch=2); b = rgb.infer_grids(model, x, batch=11)
        for k in a: np.testing.assert_allclose(a[k], b[k], atol=1e-7, rtol=0)
        np.testing.assert_allclose(a['global_grid'], a['tta_grid'], atol=1e-7, rtol=0)

    def test_scene_native_tiles_cover_nonmultiple_edges_once(self):
        import torch
        model = torch.nn.Conv2d(4, 1, 1).eval()
        x = np.random.default_rng(9).random((4, 35, 67)).astype(np.float32)
        grids, positions = rgb.scene_grids(model, x, 'rgb_proxy', parent_batch=2)
        self.assertEqual(len(positions), 6)
        policy = {'pixel_threshold': 0., 'anchor_threshold': 0., 'tta_threshold': 0.,
                  'area_policy': {'cutoff': -1., 'protect_anchor': True}}
        masks = rgb.scene_masks(grids, (35, 67), 10., policy)
        self.assertTrue(all(m.shape == (35, 67) and m.all() for m in masks.values()))
        with self.assertRaises(ValueError): rgb.scene_masks(grids, (64, 128), 10., policy)

    def test_unknown_reference_is_never_scored_as_background(self):
        truth = np.array([[1, 0, 255], [1, 0, 255]], np.uint8)
        normal = np.array([[0, 1, 0], [0, 0, 0]], np.uint8)
        valid = np.ones_like(truth, bool); valid[1, 0] = False
        a = rgb.partial_reference_counts(np.ones_like(valid), truth, normal, valid)
        self.assertEqual(a['positive_core']['pixels'], 1)
        self.assertEqual(a['weak_background']['pixels'], 2)
        self.assertEqual(a['unknown_pixels_excluded'], 2)
        b = normal.copy(); b[0, 2] = 1
        independent = rgb.partial_reference_counts(valid, truth, b, valid)
        self.assertEqual(independent['reviewed_normal']['pixels'], 2)
        self.assertEqual(independent['reviewed_normal_from_partial_unknown_pixels'], 1)
        self.assertEqual(independent['weak_background']['pixels'], 2)
        b[0, 0] = 1
        with self.assertRaises(ValueError): rgb.partial_reference_counts(valid, truth, b, valid)

    def test_invalid_fill_uses_only_image_and_keeps_valid_pixels(self):
        x = np.arange(4*4*4, dtype=np.float32).reshape(4, 4, 4)
        valid = np.ones((4, 4), bool); valid[2, 2] = False; x[:, 2, 2] = np.nan
        filled = rgb.fill_invalid(x, valid)
        self.assertTrue(np.isfinite(filled).all())
        np.testing.assert_array_equal(filled[:, valid], x[:, valid])


if __name__ == '__main__': unittest.main()
