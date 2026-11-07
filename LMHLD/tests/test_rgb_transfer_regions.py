from pathlib import Path
import sys
import unittest
sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parents[1]/'validation')]
import rgb_transfer_regions as r


class RegionTransferTests(unittest.TestCase):
    def test_factory_injection_restores_originals_on_failure(self):
        factory, loader = r.exp.new_model, r.exp.load_model
        stub = lambda *a: None
        with self.assertRaises(RuntimeError):
            with r.factories(stub):
                self.assertIs(r.exp.new_model, stub); self.assertIs(r.exp.load_model, r.load_model)
                raise RuntimeError('test')
        self.assertIs(r.exp.new_model, factory); self.assertIs(r.exp.load_model, loader)

    def test_matching_rejects_changed_parent_trace_and_budget(self):
        a = {k: 1 for k in ('parent_schedule_sha256', 'view_draw_sha256', 'visits_per_region', 'unique_seen_per_region', 'available_per_region', 'updates', 'original_four_input_random_init_sha256')}
        self.assertTrue(r.paired_training(a, dict(a)))
        for key in ('parent_schedule_sha256', 'updates'):
            b = dict(a); b[key] = 2; self.assertFalse(r.paired_training(a, b))

    def test_gain_cannot_hide_higher_fpr_or_region_collapse(self):
        def row(f1, fpr=.02): return dict(f1=f1, small_cover50=.5, small_iou50=.4, background_fpr=fpr, false_components_per_mpix=30.)
        p = {'backbone_gate': dict(min_macro_f1_gain_each_seed=.02, max_region_f1_loss=.03, max_macro_background_fpr_increase=.002, max_macro_small_coverage_loss=.01)}
        old = [row(.5)]*3
        self.assertTrue(r.backbone_gate([row(.53)]*3, old, p)['passed'])
        self.assertFalse(r.backbone_gate([row(.6, .03)]*3, old, p)['passed'])
        self.assertFalse(r.backbone_gate([row(.46), row(.6), row(.6)], old, p)['passed'])


if __name__ == '__main__': unittest.main()
