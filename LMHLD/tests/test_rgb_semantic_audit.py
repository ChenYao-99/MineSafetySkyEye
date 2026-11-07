from pathlib import Path
import sys
import unittest
import numpy as np
sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parents[1]/'validation')]
import rgb_semantic_audit as a


class SemanticAuditTests(unittest.TestCase):
    def test_only_explicit_reviewed_support_is_partitioned(self):
        normal = np.zeros((8, 8), np.uint8); normal[2:4, 2:4] = 1
        d = {'id': 'haul_road', 'polygon_pixel_xy': [[0, 0], [6, 0], [6, 6], [0, 6]], 'retained_pixels': 4}
        parts = a.normal_partitions(normal.shape, normal, np.ones_like(normal, bool), [d]); np.testing.assert_array_equal(parts[0][2], normal > 0)
        with self.assertRaises(ValueError): a.normal_partitions(normal.shape, normal, np.ones_like(normal, bool), [d, d])
        with self.assertRaises(ValueError): a.normal_partitions(normal.shape, normal, np.ones_like(normal, bool), [])

    def test_seed_common_errors_are_not_summed_as_independent_samples(self):
        x = np.ones((2, 2), bool); p = np.array([[1, 0], [0, 1]], bool); q = np.array([[1, 0], [1, 0]], bool)
        r = a.activation(x, [p, q]); self.assertEqual(r['pixels'], 4); self.assertEqual(r['both_activated'], 1); self.assertEqual(r['both_missed'], 1); self.assertEqual(r['disagreement'], 2)


if __name__ == '__main__': unittest.main()
