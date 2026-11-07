import unittest
import torch
from run_anchored_pilot import bernoulli_kl


class ConsistencyScopeTests(unittest.TestCase):
    def test_consistency_never_pushes_known_core_or_backpropagates_teacher(self):
        student = torch.tensor([[[[2., -2.], [1., -1.]]]], requires_grad=True)
        teacher = torch.zeros_like(student, requires_grad=True)
        unknown = torch.tensor([[[[False, True], [True, False]]]])
        loss = bernoulli_kl(student, teacher, unknown)
        loss.backward()
        self.assertTrue(torch.all(student.grad[~unknown] == 0))
        self.assertTrue(torch.all(student.grad[unknown] != 0))
        self.assertIsNone(teacher.grad)

    def test_equal_outputs_have_zero_preservation_penalty(self):
        logits = torch.tensor([-.9, 0., .9])
        self.assertAlmostEqual(float(bernoulli_kl(logits, logits)), 0., places=6)


if __name__ == '__main__':
    unittest.main()
