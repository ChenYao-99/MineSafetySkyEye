import sys
from pathlib import Path
import unittest
sys.path[:0] = [str(Path(__file__).resolve().parents[1]/'validation')]
import rgb_bn_source_pilot as exp


class SourceGateTests(unittest.TestCase):
    def test_average_improvement_does_not_hide_a_source_loss(self):
        before = {'macro_f1': .65, 'regions': {'0': {'f1': .65}, '1': {'f1': .65}}}
        good = {'macro_f1': .67, 'regions': {'0': {'f1': .67}, '1': {'f1': .67}}}
        bad = {'macro_f1': .67, 'regions': {'0': {'f1': .61}, '1': {'f1': .73}}}
        self.assertTrue(exp.gate(before, good)['passed']); self.assertFalse(exp.gate(before, bad)['passed'])


if __name__ == '__main__': unittest.main()
