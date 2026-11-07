"""Experimental RGB-only probability fusion of existing registered components.

The fourth channel is a task proxy, never measured NIR. Explicit RGB input is
required. This module does not alter service selection or model registration.
"""
import hashlib
import json
import math
from pathlib import Path
import numpy as np
import torch
import segmentation_models_pytorch as smp
from detector import LandslideDetector,ModelNumericalError,ModelComponentError


class RGBCompletionFusionDetector(LandslideDetector):
    def __init__(self,current_weights,completion_weights,completion_weight=.15,**kwargs):
        if isinstance(completion_weight,bool) or not math.isfinite(completion_weight) or not 0<=completion_weight<=1:
            raise ValueError('Completion probability weight must be finite in [0,1]')
        super().__init__(current_weights,**kwargs)
        self.completion=smp.Unet(encoder_name='resnet18',encoder_weights=None,in_channels=3,classes=1)
        self.completion.load_state_dict(torch.load(completion_weights,map_location='cpu',weights_only=True))
        self.completion.to(self.device).eval().requires_grad_(False)
        self.completion_weight=float(completion_weight)

    @torch.no_grad()
    def predict_batch(self,patches):
        x=torch.as_tensor(np.ascontiguousarray(patches),dtype=torch.float32,device=self.device) if not torch.is_tensor(patches) else patches.to(self.device,dtype=torch.float32)
        if x.ndim!=4 or x.shape[1]!=4 or not torch.equal(x[:,3],x[:,2]):
            raise ValueError('RGB completion fusion requires an explicit BGR+copied-red proxy; actual NIR is not accepted')
        if not torch.isfinite(x).all() or float(x.min())<0 or float(x.max())>255.001:
            raise ModelNumericalError('RGB completion fusion requires finite normalized 0..255 patches')
        current=super().predict_batch(x)
        try:
            proxy=self.completion(x[:,:3]/255)
        except torch.cuda.OutOfMemoryError:
            raise
        except Exception as exc:
            raise ModelComponentError('RGB completion component failed; no silent fallback') from exc
        if proxy.shape!=(len(x),1,x.shape[2],x.shape[3]) or not torch.isfinite(proxy).all():
            raise ModelNumericalError('Invalid RGB completion shape/value; no screening result')
        mixed=x.clone();mixed[:,3:4]=proxy.clamp(0,1)*255
        completed=super().predict_batch(mixed)
        weight=self.completion_weight
        return (1-weight)*current+weight*completed

    def detect_image(self,image_hwc,stride=None,patch_norm=None,nir_synth_model=None,max_batch=None,*,input_kind=None):
        if input_kind!='rgb_proxy':
            raise ValueError('Specify input_kind=rgb_proxy for this experimental RGB branch')
        if patch_norm!='minmax' or nir_synth_model is not None:
            raise ValueError('RGB fusion requires minmax and its fixed internal completion component')
        return super().detect_image(image_hwc,stride=stride,patch_norm=patch_norm,max_batch=max_batch)


def load_rgb_completion_fusion(config_path,models_root,**kwargs):
    config=json.loads(Path(config_path).read_text(encoding='utf-8'))
    if config.get('schema_version')!=1 or config.get('input_kind')!='rgb_proxy':
        raise ValueError('Unsupported RGB completion fusion contract')
    numerical=config.get('numerical_protocol')
    if numerical!={'dtype':'float32','cuda_tf32':False,'cudnn_deterministic':True}:
        raise ValueError('Missing frozen FP32 numerical protocol')
    device=kwargs.get('device') or ('cuda' if torch.cuda.is_available() else 'cpu')
    if str(device).startswith('cuda') and (torch.backends.cuda.matmul.allow_tf32 or
            torch.backends.cudnn.allow_tf32 or not torch.backends.cudnn.deterministic):
        raise ValueError('Use the frozen FP32/no-TF32/deterministic CUDA protocol before loading this offline candidate')
    root=Path(models_root).resolve();paths=[]
    for key in ['current','completion']:
        row=config['components'][key]
        path=(root/row['path']).resolve()
        if not path.is_relative_to(root) or hashlib.sha256(path.read_bytes()).hexdigest()!=row['sha256']:
            raise ValueError('RGB completion component path/hash mismatch')
        paths.append(path)
    for key,value in [('patch_size',config['patch_size']),('threshold',config['threshold'])]:
        if key in kwargs and kwargs[key]!=value:
            raise ValueError('Caller changed frozen RGB fusion input/threshold contract')
        kwargs[key]=value
    return RGBCompletionFusionDetector(*paths,completion_weight=config['completion_weight'],**kwargs)
