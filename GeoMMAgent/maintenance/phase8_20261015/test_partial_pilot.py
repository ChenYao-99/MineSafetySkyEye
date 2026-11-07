"""Scientific invariants that a normal full-mask trainer would violate."""
import unittest
import numpy as np
import torch
from run_partial_pilot import masked_positive_loss, pixel_metrics


class PartialSupervisionTests(unittest.TestCase):
    def test_unknown_logits_never_contribute_loss_or_gradient(self):
        labels = torch.tensor([[[[1, 255], [255, 1]]]], dtype=torch.uint8)
        logits = torch.tensor([[[[.1, -50.], [50., -.2]]]], requires_grad=True)
        loss = masked_positive_loss(logits, labels)
        loss.backward()
        self.assertTrue(torch.all(logits.grad[labels == 255] == 0))
        self.assertTrue(torch.all(logits.grad[labels == 1] < 0))
        changed = logits.detach().clone()
        changed[labels == 255] *= -100
        self.assertEqual(loss.item(), masked_positive_loss(changed, labels).item())

    def test_no_implicit_new_background_or_all_unknown_sample(self):
        for labels in [torch.zeros(1, 1, 2, 2, dtype=torch.uint8), torch.full((1, 1, 2, 2), 255, dtype=torch.uint8)]:
            with self.assertRaises(ValueError):
                masked_positive_loss(torch.zeros(1, 1, 2, 2), labels)

    def test_unscored_background_predictions_cannot_create_precision(self):
        labels = np.array([1, 255, 255])
        known = labels == 1
        a = pixel_metrics(np.array([.8, 0., 0.]), labels, known, .4)
        b = pixel_metrics(np.array([.8, 1., 1.]), labels, known, .4)
        self.assertEqual(a, b)
        self.assertEqual(a['fp'], 0)
        self.assertIsNone(a['annotation_background_activation'])
        self.assertIsNone(a['precision_on_scored_reference'])
        self.assertIsNone(a['f1_on_scored_reference'])


if __name__ == '__main__':
    unittest.main()
