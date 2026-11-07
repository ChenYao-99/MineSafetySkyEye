from pathlib import Path
import sys
import unittest
import numpy as np
sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parents[1]/'validation')]
import rgb_resolution as resolution
import rgb_resolution_study as study


class ResolutionTests(unittest.TestCase):
    def test_low_resolution_preprocessing_matches_frozen_whole_view(self):
        for n in (32, 128, 224):
            x = np.random.default_rng(n).normal(size=(3, n, n)).astype(np.float32); x[1] = 3
            old = study.exp.native.make_views(x)[0][0]
            np.testing.assert_array_equal(old, resolution.whole_view(x, 64))
            self.assertEqual(resolution.whole_view(x).shape, (3, 128, 128))
        with self.assertRaises(ValueError): resolution.whole_view(np.ones((4, 32, 32)))

    def test_output_and_batching_at128(self):
        import torch
        torch.set_num_threads(4); torch.manual_seed(1); m = torch.nn.Conv2d(3, 1, 1).eval()
        x = np.random.default_rng(1).random((3, 3, 128, 128)).astype(np.float16)
        a = resolution.predict(m, x, 1); b = resolution.predict(m, x, 3)
        np.testing.assert_allclose(a, b, atol=1e-7, rtol=0)
        self.assertEqual(a.shape, (3, 128, 128))

    def test_source_gate_does_not_trade_away_f1_for_coverage(self):
        def row(f1, cover, iou): return dict(f1=f1, small_cover50=cover, small_iou50=iou, background_fpr=.02, false_components_per_mpix=30.)
        limits=dict(min_macro_small_cover_gain=.02,min_macro_small_iou_gain=.02,max_macro_f1_loss=.005,max_region_f1_loss=.02,max_macro_fpr_increase=.002,max_region_fpr_increase=.01)
        a={'0':row(.6,.4,.3)}
        self.assertTrue(study.source_gate(a,{'0':row(.6,.43,.33)},limits)['passed'])
        self.assertFalse(study.source_gate(a,{'0':row(.58,.5,.4)},limits)['passed'])


if __name__ == '__main__': unittest.main()
