"""Scientific regressions: held-out exclusion, geometry and honest small metrics."""
import sys
from pathlib import Path
import unittest
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "validation"))
import cross_region_scale as exp


class ScaleProtocolTests(unittest.TestCase):
    def test_target_never_sampled_and_sources_balanced(self):
        groups = np.repeat(np.arange(6), 11)
        for held in range(6):
            rng = np.random.default_rng(42)
            for _ in range(10):
                ids = exp.sample_batch(groups, held, rng)
                self.assertNotIn(held, groups[ids])
                self.assertEqual(sorted(np.bincount(groups[ids], minlength=6)), [0, 4, 4, 4, 4, 4])

    def test_calibration_refuses_target_in_sources(self):
        with self.assertRaises(ValueError):
            exp.source_threshold({}, {}, [0, 1, 2, 3, 4, 5], held=0)

    def test_calibration_ignores_extreme_target_predictions(self):
        labels = {g: {"labels": [np.zeros((4, 4), np.uint8)]} for g in range(6)}
        predictions = {g: [np.full((4, 4), .25, np.float32)] for g in range(6)}
        a, _ = exp.source_threshold(predictions, labels, [1, 2, 3, 4, 5], 0)
        predictions[0] = [np.ones((4, 4), np.float32)]
        b, _ = exp.source_threshold(predictions, labels, [1, 2, 3, 4, 5], 0)
        self.assertEqual(a, b)
        self.assertGreater(a, .25)

    def test_small_means_physical_area_not_just_pixel_count(self):
        gt = np.zeros((32, 32), np.uint8); gt[8:14, 8:14] = 1
        self.assertEqual(exp.gt_components(gt, 1)["small"].sum(), 1)
        self.assertEqual(exp.gt_components(gt, 10)["small"].sum(), 0)

    def test_border_fragment_does_not_become_small_object(self):
        gt = np.zeros((32, 32), np.uint8); gt[0:2, 8:10] = 1; gt[8:10, 8:10] = 1
        self.assertEqual(exp.gt_components(gt, 10)["small"].sum(), 1)

    def test_one_pixel_hit_does_not_claim_half_coverage(self):
        gt = np.zeros((32, 32), np.uint8); gt[8:12, 8:12] = 1
        prob = np.zeros_like(gt, np.float32); prob[8, 8] = .9
        counts = exp.parent_counts(prob, gt, exp.gt_components(gt, 10), .5)
        self.assertEqual(counts["small_n"], 1)
        self.assertEqual(counts["small_any"], 1)
        self.assertEqual(counts["small_cover50"], 0)
        self.assertEqual(counts["small_iou50"], 0)

    def test_whole_image_positive_does_not_claim_good_object_iou(self):
        gt = np.zeros((32, 32), np.uint8); gt[8:12, 8:12] = 1
        counts = exp.parent_counts(np.ones_like(gt, np.float32), gt, exp.gt_components(gt, 10), .5)
        self.assertEqual(counts["small_cover50"], 1)
        self.assertEqual(counts["small_iou50"], 0)
        self.assertEqual(exp.aggregate_counts([counts])["background_fpr"], 1.)

    def test_downsample_keeps_fractional_foreground(self):
        image = np.random.default_rng(1).random((4, 128, 128), dtype=np.float32)
        gt = np.zeros((128, 128), np.uint8); gt[10, 10] = 1
        _, labels = exp.make_views(image, gt)
        self.assertGreater(labels[0].sum(), 0)
        self.assertLess(labels[0].max(), 1)

    def test_tta_inverse_preserves_asymmetric_position(self):
        import torch
        a = torch.arange(36).reshape(1, 6, 6)
        for k in (1, 2, 3):
            self.assertTrue(torch.equal(exp.inverse_tta(a.rot90(k, (-2, -1)), k), a))
        self.assertTrue(torch.equal(exp.inverse_tta(a.flip(-1), 4), a))


if __name__ == "__main__": unittest.main()
