# -*- coding: utf-8 -*-
"""Known-normal-only focal cost; unknowns/positives have exactly zero extra gradient."""
import torch


def known_normal_focal(logits,labels,confidence,gamma=2.):
    normal=(labels==0).to(logits.dtype);weight=confidence.to(logits.dtype)*normal
    p=torch.sigmoid(logits)
    loss=torch.nn.functional.binary_cross_entropy_with_logits(logits,torch.zeros_like(logits),reduction='none')*p.pow(gamma)*weight
    return (loss.sum((1,2,3))/normal.sum((1,2,3)).clamp_min(1)).mean()
