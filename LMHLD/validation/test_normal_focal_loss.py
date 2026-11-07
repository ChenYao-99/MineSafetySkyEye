# -*- coding: utf-8 -*-
import unittest
import torch
from normal_false_positive_loss import known_normal_focal


class NormalFocalTests(unittest.TestCase):
    def test_extra_gradient_only_applies_to_known_normal(self):
        logits=torch.tensor([[[[2.,2.,2.,2.]]]],requires_grad=True)
        y=torch.tensor([[[[0.,1.,255.,0.]]]]);w=torch.tensor([[[[.25,.6,0.,0.]]]])
        loss=known_normal_focal(logits,y,w);loss.backward()
        self.assertGreater(float(logits.grad[0,0,0,0]),0)
        self.assertTrue(torch.all(logits.grad[0,0,0,1:]==0))

    def test_unknown_padding_does_not_dilute_normal_penalty(self):
        a=torch.full((1,1,4,4),2.);y=torch.zeros_like(a);w=torch.full_like(a,.25)
        large=torch.zeros((1,1,32,32));yy=torch.full_like(large,255);ww=torch.zeros_like(large)
        large[:,:,:4,:4]=a;yy[:,:,:4,:4]=y;ww[:,:,:4,:4]=w
        self.assertAlmostEqual(float(known_normal_focal(a,y,w)),float(known_normal_focal(large,yy,ww)),places=7)

    def test_quality_discount_is_retained_and_no_normal_case_is_zero(self):
        x=torch.full((2,1,8,8),2.,requires_grad=True);y=torch.zeros_like(x);w=torch.ones_like(x)
        a=known_normal_focal(x,y,w);b=known_normal_focal(x,y,w*.25);self.assertAlmostEqual(float(a*.25),float(b),places=7)
        zero=known_normal_focal(x,torch.full_like(x,255),torch.zeros_like(x));zero.backward()
        self.assertEqual(float(zero),0);self.assertTrue(torch.all(x.grad==0))


if __name__=='__main__':unittest.main(verbosity=2)
