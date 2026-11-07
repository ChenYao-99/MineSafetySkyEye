from pathlib import Path
import sys
import unittest
import numpy as np
sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parents[1]/'validation')]
import rgb_detail_adaptation as a


class DetailAdaptationTests(unittest.TestCase):
    def test_sampling_excludes_held_and_rio_and_pairs_augmentation(self):
        groups = np.repeat(np.arange(7), 20); sources = [0, 1, 3, 4, 5]
        r1, r2 = np.random.default_rng(7), np.random.default_rng(7)
        for _ in range(10):
            x, k, f = a.draw(groups, sources, r1); y, kk, ff = a.draw(groups, sources, r2)
            np.testing.assert_array_equal(x, y); self.assertEqual((k, f), (kk, ff)); self.assertEqual(np.bincount(groups[x], minlength=7).tolist(), [3, 3, 0, 3, 3, 3, 0])

    def test_frozen_encoder_and_all_bn_stats_survive_decoder_update(self):
        import torch
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__(); self.encoder = torch.nn.Sequential(torch.nn.Conv2d(3, 4, 1), torch.nn.BatchNorm2d(4)); self.decoder = torch.nn.Sequential(torch.nn.Conv2d(4, 4, 1), torch.nn.BatchNorm2d(4)); self.segmentation_head = torch.nn.Conv2d(4, 1, 1)
            def forward(self, x): return self.segmentation_head(self.decoder(self.encoder(x)))
        m = Model(); opt = torch.optim.AdamW(a.freeze_representation(m), lr=.001); before = a.encoder_hash(m)
        bn = {k: v.clone() for k, v in m.state_dict().items() if 'running_' in k or 'num_batches' in k}
        old = m.segmentation_head.weight.detach().clone()
        m(torch.rand(2, 3, 8, 8)).square().mean().backward(); opt.step()
        self.assertEqual(before, a.encoder_hash(m)); self.assertTrue(all(p.grad is None for p in m.encoder.parameters())); self.assertFalse(torch.equal(old, m.segmentation_head.weight))
        for k, v in bn.items(): self.assertTrue(torch.equal(v, m.state_dict()[k]))
        self.assertTrue(all(not k.startswith('encoder.') for k in a.delta(m)))


if __name__ == '__main__': unittest.main()
