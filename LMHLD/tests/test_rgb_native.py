"""Native RGB and continuous-experiment scientific guards."""
from pathlib import Path
import copy
import sys
import unittest
import numpy as np
sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parents[1]/'validation')]
from runtime_paths import configure_runtime_paths
configure_runtime_paths()
import rgb_native as native
import rgb_native_training as exp


class NativeRGBTests(unittest.TestCase):
    def test_preprocessing_never_reads_nir_and_retains_three_channels(self):
        x = np.random.default_rng(7).random((4, 128, 128)).astype(np.float32)
        a, _ = native.make_views(x)
        x[3] = np.nan; b, _ = native.make_views(x)
        self.assertEqual(a.shape, (5, 3, 64, 64)); np.testing.assert_array_equal(a, b)
        np.testing.assert_array_equal(a, native.make_views(x[:3])[0])

    def test_native_bgr_matches_visible_channels_of_frozen_preprocessing(self):
        x = np.random.default_rng(8).random((4, 32, 32)).astype(np.float32)
        y = (x[2] > .7).astype(np.uint8)
        a, b = native.make_views(x, y); old_a, old_b = exp.base.make_views(x, y)
        np.testing.assert_array_equal(a, old_a[:, :3]); np.testing.assert_array_equal(b, old_b)

    def test_paired_sampler_excludes_held_and_matches_old_full_schedule(self):
        groups = np.repeat(np.arange(6), [513, 700, 600, 900, 800, 1000])
        a = np.random.default_rng(242); p = np.random.default_rng(242)
        old = np.random.default_rng(242); paired_old = np.random.default_rng(242)
        for _ in range(27):
            ids, views, k, flip = exp.draw_step(groups, 2, a, p)
            old_ids, _, _ = exp.parent.legacy_augmentation(groups, 2, old)
            _, ok, of = exp.parent.legacy_augmentation(np.repeat(np.arange(6), 512), 2, paired_old)
            np.testing.assert_array_equal(ids, old_ids); self.assertEqual((k, flip), (ok, of))
            self.assertFalse(np.any(groups[ids] == 2)); self.assertTrue(np.isin(views, range(5)).all())

    def test_sampling_resume_restores_both_streams(self):
        groups = np.repeat(np.arange(6), 600); a = np.random.default_rng(42); p = np.random.default_rng(42)
        for _ in range(7): exp.draw_step(groups, 3, a, p)
        states = (copy.deepcopy(a.bit_generator.state), copy.deepcopy(p.bit_generator.state))
        expected = exp.draw_step(groups, 3, a, p)
        b = np.random.default_rng(); q = np.random.default_rng(); b.bit_generator.state, q.bit_generator.state = states
        actual = exp.draw_step(groups, 3, b, q)
        for i in (0, 1): np.testing.assert_array_equal(expected[i], actual[i])
        self.assertEqual(expected[2:], actual[2:])

    def test_three_band_inference_rejects_proxy_and_preserves_tta(self):
        import torch
        torch.manual_seed(7); model = torch.nn.Conv2d(3, 1, 1).eval()
        views = np.stack([native.make_views(np.random.default_rng(3).random((3, 32, 32)))[0]]*2)
        p = native.infer_grids(model, views, batch=1)
        np.testing.assert_allclose(p['global_grid'], p['tta_grid'], atol=1e-7, rtol=0)
        with self.assertRaises(ValueError): native.infer_grids(model, np.ones((2, 5, 4, 64, 64)))

    def test_native_scene_cache_covers_edges_and_has_no_fourth_band(self):
        import torch
        model = torch.nn.Conv2d(3, 1, 1).eval(); x = np.ones((3, 35, 67), np.float32)
        grids = native.scene_grids(model, x, parent_batch=2)
        self.assertEqual(grids['global'].shape, (6, 32, 32))
        policy = {'pixel_threshold': 0., 'anchor_threshold': 0., 'tta_threshold': 0., 'area_policy': {'cutoff': -1., 'protect_anchor': True}}
        masks = native.scene_masks(grids, (35, 67), 10., policy)
        self.assertTrue(all(v.all() for v in masks.values()))
        with self.assertRaises(ValueError): native.scene_grids(model, np.ones((4, 35, 67)))

    def test_bgr_dedup_checks_val_and_cross_region_and_label_conflicts(self):
        def row(g, i, h, label='a'):
            return {'group': g, 'index': i, 'sha256': h, 'label_sha256': label, 'legacy_selected': False}
        entries = [row(0, 0, 'val'), row(0, 1, 'cross'), row(1, 0, 'cross'), row(0, 2, 'conflict'), row(0, 3, 'conflict', 'b'), row(1, 1, 'good')]
        keep, removed = exp.parent.quarantine_train(entries, {'val'})
        self.assertEqual([r['sha256'] for r in keep], ['good']); self.assertEqual(len(removed), 5)


if __name__ == '__main__': unittest.main()
