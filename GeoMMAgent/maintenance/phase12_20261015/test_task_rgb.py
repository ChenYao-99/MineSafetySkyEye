import unittest
from task_rgb import RGBTaskAdapter,task_loss,torch
from rgb_feature_residual import FrozenRGBFeatureModel,RGBFeatureResidual
from refine_legacy import LegacyTaskAdapter


class RGBContracts(unittest.TestCase):
    def test_identity_initialization_and_real_nir_cannot_enter(self):
        torch.manual_seed(42)
        model=RGBTaskAdapter()
        x=torch.rand(2,3,32,32)
        self.assertTrue(torch.equal(model(x),x[:,2:3]))
        with self.assertRaises(ValueError):model(torch.rand(2,4,32,32))

    def test_frozen_detector_still_passes_gradient_to_adapter(self):
        adapter=RGBTaskAdapter()
        frozen=torch.nn.Conv2d(4,1,1,bias=False).requires_grad_(False).eval()
        with torch.no_grad():frozen.weight.fill_(1)
        x=torch.rand(2,3,8,8)
        latent=adapter(x)
        logits=frozen(torch.cat([x,latent],1))
        loss,_=task_loss(logits,torch.ones_like(logits))
        loss.backward()
        self.assertIsNone(frozen.weight.grad)
        self.assertGreater(float(adapter.head.weight.grad.abs().sum()),0)

    def test_wrong_confident_teacher_has_no_distillation_gradient(self):
        student=torch.zeros(1,1,2,2,requires_grad=True)
        labels=torch.tensor([[[[1.,0.],[1.,0.]]]])
        wrong_teacher=torch.tensor([[[[-8.,8.],[-8.,8.]]]],requires_grad=True)
        supervised,_=task_loss(student,labels)
        augmented,kl=task_loss(student,labels,wrong_teacher)
        self.assertEqual(float(kl),0)
        self.assertEqual(float(augmented),float(supervised))
        augmented.backward()
        self.assertIsNone(wrong_teacher.grad)

    def test_bounded_latent_remains_nonphysical_proxy(self):
        model=RGBTaskAdapter()
        with torch.no_grad():model.head.bias.fill_(20)
        x=torch.rand(2,3,8,8)
        out=model(x)
        self.assertTrue(torch.all((out>=0)&(out<=1)))
        self.assertLessEqual(float((out-x[:,2:3]).abs().max()),.500001)

    def test_legacy_component_stays_frozen_in_training_and_initial_proxy_exact(self):
        legacy=torch.nn.Sequential(torch.nn.Conv2d(3,1,1),torch.nn.BatchNorm2d(1)).eval()
        adapter=LegacyTaskAdapter(legacy)
        x=torch.rand(3,3,8,8)
        expected=legacy(x).clamp(0,1)
        before=legacy[1].running_mean.clone()
        adapter.train()
        self.assertFalse(legacy.training)
        self.assertTrue(torch.equal(adapter(x),expected))
        self.assertTrue(torch.equal(legacy[1].running_mean,before))
        self.assertFalse(any(p.requires_grad for p in legacy.parameters()))

    def test_multiscale_rgb_validation_cannot_read_true_nir(self):
        import sys
        sys.path.insert(0,'D:/LMHLD')
        from detector import LandslideDetector
        torch.set_num_threads(4)
        base=LandslideDetector('D:/LMHLD/models_prior/real_ft_T1b_ours_seed42/best_model.pth',device='cpu').model
        head=RGBFeatureResidual().eval()
        with torch.no_grad():
            # Nonzero head, so equality cannot be explained by identity initialization.
            head.readout[-1].weight.fill_(.01)
            model=FrozenRGBFeatureModel(base,head)
            x=torch.rand(2,4,32,32)*255
            modified=x.clone();modified[:,3]=torch.rand_like(modified[:,3])*10000
            self.assertTrue(torch.equal(model(x),model(modified)))


if __name__=='__main__':unittest.main(verbosity=2)
