from pathlib import Path
import sys
import unittest
import numpy as np
sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parents[1]/'validation')]
import qualify_brumadinho_visible_core as q


class VisibleCoreTests(unittest.TestCase):
    def test_core_is_positive_only_with_strict_concurrence(self):
        mask = np.zeros((60, 60), bool); mask[5:55, 5:55] = True; valid = np.ones_like(mask)
        pre = np.full((60, 60, 4), .1, np.float32); pre[..., 3] = .5
        post = np.full_like(pre, .15); post[..., 3] = .16
        labels, weights, c = q.select_core(mask, mask, valid, pre, post)
        self.assertGreater(c['positive_auxiliary_pixels'], 0); self.assertFalse(np.any(labels == 0)); self.assertTrue(np.all(weights[labels == 255] == 0))
        valid[:] = False; labels, _, c = q.select_core(mask, mask, valid, pre, post); self.assertTrue(np.all(labels == 255))

    def test_unchanged_or_bright_pixels_are_never_promoted(self):
        mask = np.ones((60, 60), bool); pre = np.full((60, 60, 4), .1, np.float32); pre[..., 3] = .5
        labels, _, _ = q.select_core(mask, mask, mask, pre, pre); self.assertTrue(np.all(labels == 255))
        post = np.full_like(pre, .8); labels, _, _ = q.select_core(mask, mask, mask, pre, post); self.assertTrue(np.all(labels == 255))


if __name__ == '__main__': unittest.main()
