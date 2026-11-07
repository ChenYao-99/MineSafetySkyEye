from pathlib import Path
import sys
import unittest
sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parents[1]/'validation')]
import rgb_resolution_bn as b


class ResolutionBNTests(unittest.TestCase):
    def test_only_bn_buffers_change_and_are_exactly_reconstructable(self):
        import torch
        torch.set_num_threads(4); torch.manual_seed(1)
        m = torch.nn.Sequential(torch.nn.Conv2d(3, 4, 1), torch.nn.BatchNorm2d(4), torch.nn.Conv2d(4, 1, 1)).eval()
        original = {k: v.clone() for k, v in m.state_dict().items()}; before = b.parameter_hash(m)
        delta = b.recalibrate(m, [torch.rand(7, 3, 8, 8) for _ in range(3)])
        self.assertEqual(before, b.parameter_hash(m)); self.assertEqual(int(m[1].num_batches_tracked), 3)
        self.assertFalse(torch.equal(original['1.running_mean'], m[1].running_mean)); self.assertFalse(m.training); self.assertFalse(m[1].training)
        state = {k: v.clone() for k, v in m.state_dict().items()}; original.update(delta); m.load_state_dict(original)
        for k, v in state.items(): self.assertTrue(torch.equal(v, m.state_dict()[k]))
        self.assertTrue(all(p.grad is None for p in m.parameters()))


if __name__ == '__main__': unittest.main()
