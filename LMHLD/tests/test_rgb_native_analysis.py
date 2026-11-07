from pathlib import Path
import sys
import unittest
sys.path[:0] = [str(Path(__file__).resolve().parents[1]/'validation')]
import rgb_native_analysis as exp


class ContrastTests(unittest.TestCase):
    def test_gains_do_not_pass_when_background_increases(self):
        limits = {'min_cover_gain': .02, 'min_iou_gain': .02, 'max_f1_loss': .005,
                  'max_macro_fpr_increase': .002, 'max_region_fpr_increase': .01,
                  'max_macro_fpr': .03, 'max_region_fpr': .05, 'max_candidate_ratio': 1.10}
        old = {'f1': .65, 'small_cover50': .5, 'small_iou50': .3, 'background_fpr': .02, 'false_components_per_mpix': 35.}
        new = {**old, 'small_cover50': .55, 'small_iou50': .35}
        self.assertTrue(exp.assess([(old, new)]*3, limits)['passed'])
        bad = {**new, 'background_fpr': .08}
        self.assertFalse(exp.assess([(old, bad)]*3, limits)['passed'])
        bad = {**new, 'f1': .60}
        self.assertFalse(exp.assess([(old, bad)]*3, limits)['checks']['f1_preserved'])


if __name__ == '__main__': unittest.main()
