from pathlib import Path
import sys
import unittest
import numpy as np
sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parents[1]/'validation')]
from runtime_paths import configure_runtime_paths
configure_runtime_paths()
import rgb_final_training as exp
import rgb_native_detector as detector


class FinalRGBTests(unittest.TestCase):
    def test_all_seven_sources_have_exactly_three_examples(self):
        g = np.repeat(np.arange(7), 80); rng = np.random.default_rng(42)
        for _ in range(10):
            ids, k, flip = exp.sample(g, rng); self.assertEqual(np.bincount(g[ids], minlength=7).tolist(), [3]*7)

    def test_source_threshold_requires_seven_and_controls_equal_region_fpr(self):
        h = np.ones(1001, np.int64); hist = {g: h for g in range(7)}
        t, cal = exp.threshold(hist); self.assertLessEqual(cal['source_macro_fpr'], .02)
        self.assertGreaterEqual(t, .98)
        with self.assertRaises(ValueError): exp.threshold({g: h for g in range(6)})

    def test_standalone_inference_is_three_band_and_batch_invariant(self):
        import torch
        torch.manual_seed(7); model = torch.nn.Conv2d(3, 1, 1).eval(); d = detector.NativeRGBDetector(model, .5)
        x = np.random.default_rng(1).random((35, 67, 3)).astype(np.float32)
        a = d.detect_image(x, max_batch=2); b = d.detect_image(x, max_batch=5)
        self.assertEqual(a.shape, (35, 67)); np.testing.assert_allclose(a, b, atol=1e-7, rtol=0)
        with self.assertRaises(ValueError): d.detect_image(np.ones((35, 67, 4)))


if __name__ == '__main__': unittest.main()
