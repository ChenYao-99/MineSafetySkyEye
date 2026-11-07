"""Task-trained latent fourth channel; never a measured or reconstructed NIR claim."""
import sys
from pathlib import Path

sys.path.insert(0, 'D:/LMHLD')
from runtime_paths import configure_runtime_paths
configure_runtime_paths()
sys.dont_write_bytecode = True
import torch
from torch import nn
import torch.nn.functional as F


class RGBTaskAdapter(nn.Module):
    """BGR in [0,1] -> bounded task latent, with exact red-copy initialization."""
    def __init__(self, width=16, max_residual=.5):
        super().__init__()
        self.max_residual = max_residual
        self.body = nn.Sequential(nn.Conv2d(3,width,3,padding=1),nn.GELU(),
            nn.Conv2d(width,width,3,padding=1),nn.GELU(),
            nn.Conv2d(width,width,3,padding=2,dilation=2),nn.GELU())
        self.head = nn.Conv2d(width,1,1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, bgr):
        if bgr.ndim != 4 or bgr.shape[1] != 3:
            raise ValueError('Adapter accepts BGR only; real NIR cannot enter this branch')
        delta = self.max_residual * torch.tanh(self.head(self.body(bgr)))
        return (bgr[:,2:3] + delta).clamp(0,1)


def task_loss(logits, labels, teacher_logits=None):
    loss = F.binary_cross_entropy_with_logits(logits,labels)
    probabilities = torch.sigmoid(logits)
    dice = 1-(2*(probabilities*labels).sum((2,3))+1)/(probabilities.sum((2,3))+labels.sum((2,3))+1)
    loss = loss + .5*dice.mean()
    distill = logits.new_zeros(())
    if teacher_logits is not None:
        teacher_logits = teacher_logits.detach()
        teacher_p = torch.sigmoid(teacher_logits)
        correct_confident = ((teacher_p>=.8)&(labels==1)) | ((teacher_p<=.2)&(labels==0))
        softened = torch.sigmoid(teacher_logits/2)
        entropy = -(softened*torch.log(softened.clamp_min(1e-7)) +
                    (1-softened)*torch.log((1-softened).clamp_min(1e-7)))
        kl = F.binary_cross_entropy_with_logits(logits/2,softened,reduction='none')-entropy
        if correct_confident.any():
            distill = kl[correct_confident].mean()*4
        loss = loss + .25*distill
    return loss, distill
