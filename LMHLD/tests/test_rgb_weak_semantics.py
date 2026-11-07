from pathlib import Path
import sys
import unittest
import numpy as np
sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parents[1]/'validation')]
import rgb_weak_semantics as s
import mine_rgb_head as old


class WeakSemanticTests(unittest.TestCase):
    def fixture(self):
        rng = np.random.default_rng(3); supports = {}; full = {}; refs = {}
        for name in ('A', 'B'):
            embedding = rng.random((8, 8, 16)).astype(np.float16); tta = rng.random((8, 8)).astype(np.float32)
            labels = np.full((8, 8), 255, np.uint8); labels[1:3] = 1; normal = np.zeros((8, 8), np.uint8); normal[4:7] = 1
            valid = np.ones((8, 8), bool); idx = np.flatnonzero(((labels == 1) | (normal == 1)).ravel())
            full[name] = {'embedding': embedding, 'tta': tta}; refs[name] = (labels, normal, valid)
            reg = np.zeros((8, 8), np.uint16); reg[4] = 1; reg[5:7] = 2
            supports[name] = {'x': old.matrix(full[name])[idx], 'index': idx, 'label': (labels == 1).ravel()[idx].astype(np.uint8), 'region': reg.ravel()[idx], 'probability': tta.ravel()[idx]}
        return supports, full, refs

    def test_pixel_head_reproduces_previous_classifier_and_sparse_dense_scores(self):
        packs, full, refs = self.fixture(); a = s.fit(packs, ['A', 'B'], 'H'); b = old.fit_head(full, refs, ['A', 'B'], 'H')
        for key in ('mean', 'scale', 'coef', 'intercept'): np.testing.assert_allclose(a[key], b[key], atol=1e-9, rtol=0)
        for c in packs:
            expected = old.predict_head(b, full[c]).ravel()[packs[c]['index']]
            np.testing.assert_array_equal(s.predict(a, packs[c]['x']), expected)

    def test_region_balance_uses_same_pixels_and_equal_region_mass(self):
        packs, _, _ = self.fixture(); a = s.fit(packs, ['A', 'B'], 'H'); b = s.fit(packs, ['A', 'B'], 'H', 'region_balanced')
        self.assertEqual(a['samples'], b['samples'])
        for r in b['normal_region_masses']: self.assertAlmostEqual(r['unscaled_mass'], .125)
        with self.assertRaises(ValueError): s.fit(packs, ['A', 'B'], 'A')

    def test_support_with_unmarked_normal_or_unknown_is_rejected(self):
        packs, _, _ = self.fixture(); packs['A']['region'][:] = 0
        with self.assertRaises(ValueError): s.fit(packs, ['A', 'B'], 'H', 'region_balanced')
        packs['A']['label'][0] = 255
        with self.assertRaises(ValueError): s.fit(packs, ['A', 'B'], 'H')


if __name__ == '__main__': unittest.main()
