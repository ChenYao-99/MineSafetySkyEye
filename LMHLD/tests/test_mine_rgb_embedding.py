from pathlib import Path
import sys
import unittest
import numpy as np
sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parents[1]/'validation')]
from runtime_paths import configure_runtime_paths
configure_runtime_paths()
import mine_rgb_head as h


def reference():
    y = np.full((8, 8), 255, np.uint8); y[:3] = 1
    n = np.zeros((8, 8), np.uint8); n[4:7] = 1
    return y, n, np.ones((8, 8), bool)


def features():
    f = np.zeros((8, 8, 16), np.float16); f[:3, :, 0] = 1; f[4:7, :, 0] = -1
    return {'embedding': f, 'tta': np.full((8, 8), .5, np.float32)}


class WeakMineHeadTests(unittest.TestCase):
    def test_unknown_is_ignored_except_explicit_independent_normal(self):
        y, n, valid = reference(); pos, neg = h.supervision(y, n, valid)
        self.assertTrue(np.all(y[neg] == 255)); self.assertEqual(int(pos.sum()), 24); self.assertEqual(int(neg.sum()), 24)
        n[0, 0] = 1
        with self.assertRaises(ValueError): h.supervision(y, n, valid)

    def test_source_fit_rejects_held_case_and_samples_no_unknown_pixels(self):
        r = {c: reference() for c in ['a', 'b']}; f = {c: features() for c in r}
        model = h.fit_head(f, r, ['a', 'b'], 'held', cap=12)
        self.assertEqual(len(model['coef']), 17)
        for row in model['samples']:
            pos, neg = h.supervision(*r[row['case']]); scope = pos if row['label'] else neg
            self.assertTrue(scope.ravel()[row['indices']].all()); self.assertEqual(len(set(row['indices'])), len(row['indices']))
        p = h.predict_head(model, features()); self.assertGreater(float(p[:3].mean()), float(p[4:7].mean()))
        with self.assertRaises(ValueError): h.fit_head(f, r, ['a', 'b'], 'a')

    def test_negative_only_cutoff_never_uses_positive_scores(self):
        ids = ['a', 'b', 'c']; refs = {c: reference() for c in ids}; scores = {c: np.full((8, 8), .2, np.float32) for c in ids}
        rates = {c: .1 for c in ids}; first = h.calibrate(scores, refs, ids, 'held', rates, rates)
        for c in ids: scores[c][:3] = .999
        self.assertEqual(first, h.calibrate(scores, refs, ids, 'held', rates, rates))
        self.assertLessEqual(first['source_mean_normal_activation'], first['source_budget'])
        with self.assertRaises(ValueError): h.calibrate(scores, refs, ids, 'a', rates, rates)

    def test_original_forward_and_tta_inverse_features_agree(self):
        import torch
        import rgb_native_detector as runtime
        class Encoder(torch.nn.Conv2d):
            def forward(self, x): return [super().forward(x)]
        class Decoder(torch.nn.Module):
            def forward(self, f): return f[0]
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__(); self.encoder = Encoder(3, 16, 1); self.decoder = Decoder(); self.segmentation_head = torch.nn.Conv2d(16, 1, 1)
            def forward(self, x): return self.segmentation_head(self.decoder(self.encoder(x)))
        torch.manual_seed(7); model = Model().eval(); x = np.random.default_rng(7).random((2, 3, 64, 64), dtype=np.float32)
        p, single, f = h.forward_features(model, x, check_forward=True)
        np.testing.assert_allclose(p, runtime.predict_tta(model, x), atol=1e-7, rtol=0)
        np.testing.assert_allclose(single, p, atol=1e-7, rtol=0); self.assertEqual(f.shape, (2, 16, 64, 64))
        raw = np.random.default_rng(8).random((3, 35, 67), dtype=np.float32)
        for side in (32, 64):
            maps = h.scene_features(model, raw, parent_batch=2, native_side=side)
            self.assertEqual(maps['embedding'].shape, (35, 67, 16)); self.assertTrue(np.isfinite(maps['embedding']).all())

    def test_summary_comparisons_are_json_native(self):
        import json
        import mine_rgb_embedding_study as study
        metric = {'positive_core': {'rate': .2}, 'reviewed_normal': {'rate': .1}}
        rows = [{'held_mine': 'a', 'methods': {'unchanged_base': metric, 'selected': metric}}]
        values = study.native_means(rows, {'a': {'0.4': metric}})
        self.assertTrue(all(type(v) is float for v in values))
        json.dumps({'core_retained': values[1] >= values[0]-.01, 'normal_reduced': values[3] <= values[2]*.8})


if __name__ == '__main__': unittest.main()
