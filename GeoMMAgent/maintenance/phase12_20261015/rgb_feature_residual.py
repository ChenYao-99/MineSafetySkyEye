"""RGB-only residual prediction from frozen encoder/decoder features at five scales."""
from task_rgb import torch,nn,F


class RGBFeatureResidual(nn.Module):
    def __init__(self):
        super().__init__()
        self.projections=nn.ModuleList([nn.Conv2d(c,8,1) for c in [64,64,128,256,16]])
        self.readout=nn.Sequential(nn.Conv2d(44,16,3,padding=1),nn.GELU(),nn.Conv2d(16,1,1))
        nn.init.zeros_(self.readout[-1].weight)
        nn.init.zeros_(self.readout[-1].bias)

    def forward(self,features,decoded,base_logits,bgr255):
        target=decoded.shape[-2:]
        selected=[features[1],features[2],features[3],features[4],decoded]
        values=[bgr255/255,torch.sigmoid(base_logits)]
        for projection,feature in zip(self.projections,selected):
            projected=projection(feature)
            projected=F.group_norm(projected,4)
            values.append(F.interpolate(projected,size=target,mode='bilinear',align_corners=False))
        return self.readout(torch.cat(values,dim=1))


class FrozenRGBFeatureModel(nn.Module):
    def __init__(self,trunk,head):
        super().__init__()
        self.trunk=trunk.eval().requires_grad_(False)
        self.head=head

    def components(self,inputs):
        # Source true NIR may exist for teacher training, never enters RGB features.
        bgr=inputs[:,:3]
        proxy=torch.cat([bgr,bgr[:,2:3]],dim=1)
        with torch.no_grad():
            features=self.trunk.encoder(proxy)
            decoded=self.trunk.decoder(features)
            base=self.trunk.segmentation_head(decoded)
        residual=self.head(features,decoded,base,bgr)
        return base+residual,residual

    def forward(self,inputs):return self.components(inputs)[0]
