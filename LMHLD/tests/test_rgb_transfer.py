from pathlib import Path
import sys
import unittest
import numpy as np
sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parents[1]/'validation')]
from runtime_paths import configure_runtime_paths
configure_runtime_paths()
import torch
import rgb_transfer_model as architecture
import rgb_transfer_study as study
torch.set_num_threads(4)


class TransferTests(unittest.TestCase):
    def test_rgb_normalization_reorders_bgr(self):
        m = architecture.RGBNormalizedEncoder(torch.nn.Identity())
        bgr = torch.tensor([.1, .2, .3]).reshape(1, 3, 1, 1)
        expected = (torch.tensor([.3, .2, .1])-torch.tensor([.485, .456, .406]))/torch.tensor([.229, .224, .225])
        torch.testing.assert_close(m(bgr).flatten(), expected)

    def test_decoder_is_matched_and_encoder_loads_official_state(self):
        torch.manual_seed(42); a = architecture.RGBTransferModel().eval(); before = a.encoder.core.conv1.weight.detach().clone()
        ah = architecture.decoder_hash(a); del a
        torch.manual_seed(42); b = architecture.RGBTransferModel(study.WEIGHTS).eval()
        self.assertEqual(ah, architecture.decoder_hash(b))
        state = torch.load(study.WEIGHTS, map_location='cpu', weights_only=True)
        self.assertTrue(torch.equal(b.encoder.core.conv1.weight, state['conv1.weight']))
        self.assertFalse(torch.equal(before, b.encoder.core.conv1.weight))
        x = torch.rand(2, 3, 64, 64)
        with torch.inference_mode():
            expected = b.segmentation_head(b.decoder(b.encoder(x)))
            torch.testing.assert_close(b(x), expected, rtol=0, atol=0)
        self.assertEqual(tuple(expected.shape), (2, 1, 64, 64))

    def test_both_five_view_methods_match_shared_benchmark_and_batching(self):
        class Counter(torch.nn.Conv2d):
            def forward(self, x):
                self.examples += len(x)
                return super().forward(x)
        torch.manual_seed(1); m = Counter(3, 1, 1).eval(); m.examples = 0
        x = np.random.default_rng(4).random((35, 67, 3)).astype(np.float32)
        native = study.exp.native.scene_grids(m, x.transpose(2, 0, 1), parent_batch=2)
        expected = study.stitch(native, x.shape[:2])
        for method in ('tta5', 'mean_multiscale5'):
            d = architecture.RGBTransferDetector(m, .7, method); m.examples = 0
            a = d.detect_image(x, max_batch=2)
            self.assertEqual(m.examples, 5*6)
            b = d.detect_image(x, max_batch=5)
            np.testing.assert_allclose(a, b, rtol=0, atol=1e-7)
            np.testing.assert_allclose(a, expected[method], rtol=0, atol=1e-7)
            with self.assertRaises(ValueError): d.detect_image(np.ones((35, 67, 4)))
            with self.assertRaises(ValueError): d.detect_image(x, max_batch=True)

    def test_pilot_gate_requires_gain_and_protects_every_source(self):
        def record(values):
            return {'methods': {'tta5': {'selection': {str(i): {'f1': v} for i, v in enumerate(values)}, 'selection_macro': {'f1': float(np.mean(values))}}}}
        a = record([.5]*7)
        self.assertTrue(study.pilot_gate(a, record([.53]*7))['passed'])
        self.assertFalse(study.pilot_gate(a, record([.51]*7))['passed'])
        self.assertFalse(study.pilot_gate(a, record([.46]+[.56]*6))['passed'])

    def test_packed_masks_validate_length_dtype_and_padding(self):
        x = np.random.default_rng(1).random((3, 5)) > .5
        np.testing.assert_array_equal(study.unpack_mask(np.packbits(x.ravel()), x.shape), x)
        with self.assertRaises(ValueError): study.unpack_mask(np.ones(3, np.uint8), x.shape)
        bad = np.packbits(x.ravel()); bad[-1] |= 1
        with self.assertRaises(ValueError): study.unpack_mask(bad, x.shape)

    def test_source_outcome_metrics_survive_threshold_metadata_attachment(self):
        row = {'f1': .6, 'small_cover50': .4, 'small_iou50': .2, 'background_fpr': .019, 'false_components_per_mpix': 5.}
        r = {'tta5': {'selection': {'0': dict(row)}, 'calibration': {'0': dict(row)}}}
        cal = {'source_macro_fpr': .019, 'source_fprs': [.019]*7, 'target_used': False}
        study.add_source_statistics(r, {'tta5': .7}, {'tta5': cal})
        self.assertEqual(r['tta5']['calibration']['0'], row)
        self.assertEqual(r['tta5']['source_calibration'], cal)
        self.assertEqual(r['tta5']['calibration_macro']['f1'], .6)


if __name__ == '__main__': unittest.main()
