import sys
from pathlib import Path
import unittest
import numpy as np
sys.path[:0] = [str(Path(__file__).resolve().parents[1]/'validation')]
import rgb_native_new_region as exp


class AdmissionTests(unittest.TestCase):
    def test_admission_uses_bgr_disjointness_not_other_two_bands(self):
        x = np.random.default_rng(7).random((4, 5, 64, 64), dtype=np.float32)
        y = np.zeros((4, 1, 64, 64), np.uint8)
        x[1, :3] = x[0, :3]
        h = exp.exp.base.array_sha(x[2, :3])
        records, excluded = exp.admit(x, y, {h})
        self.assertEqual([r['index'] for r in records], [0, 3]); self.assertEqual(len(excluded), 2)

    def test_conflicting_duplicate_labels_stop_admission(self):
        x = np.ones((2, 5, 64, 64), np.float32); y = np.zeros((2, 1, 64, 64), np.uint8); y[1, 0, 4, 4] = 1
        with self.assertRaises(ValueError): exp.admit(x, y, set())


if __name__ == '__main__': unittest.main()
