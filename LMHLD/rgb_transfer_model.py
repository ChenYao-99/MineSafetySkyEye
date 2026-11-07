"""BGR-native segmentation with explicit ImageNet RGB normalization.

Initialization is supplied by the experiment, never downloaded by inference.
Random and pretrained encoders share the exact same normalization and decoder.
"""
from pathlib import Path
import json
import hashlib
import numpy as np
from runtime_paths import configure_runtime_paths
configure_runtime_paths()
import torch


class RGBNormalizedEncoder(torch.nn.Module):
    def __init__(self, core):
        super().__init__(); self.core = core
        self.register_buffer('rgb_mean', torch.tensor([.485, .456, .406]).reshape(1, 3, 1, 1))
        self.register_buffer('rgb_std', torch.tensor([.229, .224, .225]).reshape(1, 3, 1, 1))

    def forward(self, bgr):
        return self.core((bgr[:, [2, 1, 0]]-self.rgb_mean)/self.rgb_std)


class RGBTransferModel(torch.nn.Module):
    def __init__(self, pretrained_path=None):
        import segmentation_models_pytorch as smp
        super().__init__()
        net = smp.UnetPlusPlus(encoder_name='resnet34', encoder_weights=None, in_channels=3, classes=1)
        if pretrained_path is not None:
            state = torch.load(pretrained_path, map_location='cpu', weights_only=True)
            net.encoder.load_state_dict(dict(state), strict=True)
            if not torch.equal(net.encoder.conv1.weight, state['conv1.weight']):
                raise ValueError('Official encoder initialization did not load exactly')
        self.encoder = RGBNormalizedEncoder(net.encoder)
        self.decoder = net.decoder; self.segmentation_head = net.segmentation_head

    def forward(self, bgr):
        return self.segmentation_head(self.decoder(self.encoder(bgr)))


def decoder_hash(model):
    h = hashlib.sha256()
    for name in ('decoder', 'segmentation_head'):
        for key, a in getattr(model, name).state_dict().items(): h.update((name+key).encode()); h.update(a.detach().cpu().numpy().tobytes())
    return h.hexdigest()


class RGBTransferDetector:
    """Five-view source-selected inference, without labels or mine-specific rules."""
    def __init__(self, model, threshold, method='tta5'):
        from rgb_native_detector import NativeRGBDetector
        self.tta = NativeRGBDetector(model, threshold)
        if method not in ('tta5', 'mean_multiscale5'): raise ValueError('Unknown view method')
        self.model = self.tta.model; self.threshold = self.tta.threshold; self.method = method

    def detect_image(self, image_hwc, max_batch=64):
        if self.method == 'tta5': return self.tta.detect_image(image_hwc, max_batch)
        import cv2
        import rgb_native
        if isinstance(max_batch, bool) or not isinstance(max_batch, (int, np.integer)) or not 1 <= max_batch <= 256:
            raise ValueError('max_batch must be an integer in1..256')
        a = np.asarray(image_hwc)
        if a.ndim != 3 or a.shape[2] != 3 or min(a.shape[:2]) < 2 or not np.isfinite(a).all():
            raise ValueError('Finite BGR HWC with exactly3 bands required')
        h, w = a.shape[:2]; side = 32
        pad = np.pad(a, ((0, (-h) % side), (0, (-w) % side), (0, 0)), mode='reflect')
        positions = [(y, x) for y in range(0, h, side) for x in range(0, w, side)]
        result = np.empty((h, w), np.float32); device = next(self.model.parameters()).device
        with torch.inference_mode():
            for start in range(0, len(positions), max_batch):
                part = positions[start:start+max_batch]
                views = np.stack([rgb_native.make_views(pad[y:y+side, x:x+side].transpose(2, 0, 1))[0] for y, x in part])
                flat = views.reshape(-1, 3, 64, 64); predictions = []
                for j in range(0, len(flat), max_batch):
                    x = torch.as_tensor(np.ascontiguousarray(flat[j:j+max_batch]), device=device).float()
                    p = self.model(x).sigmoid()[:, 0]
                    if p.shape[1:] != (64, 64) or not torch.isfinite(p).all(): raise ValueError('Invalid transfer output')
                    predictions.append(p.cpu().numpy())
                grids = np.concatenate(predictions).reshape(-1, 5, 64, 64)
                for (y, x), grid in zip(part, grids):
                    whole = cv2.resize(grid[0], (side, side), interpolation=cv2.INTER_LINEAR)
                    local = np.empty((side, side), np.float32)
                    for q, (yy, xx) in enumerate(((0, 0), (0, 16), (16, 0), (16, 16))):
                        local[yy:yy+16, xx:xx+16] = cv2.resize(grid[q+1], (16, 16), interpolation=cv2.INTER_LINEAR)
                    prob = .5*(whole+local); hh, ww = min(side, h-y), min(side, w-x)
                    result[y:y+hh, x:x+ww] = prob[:hh, :ww]
        return result


def load_candidate(config_path, device=None):
    config_path = Path(config_path).resolve(); c = json.loads(config_path.read_text(encoding='utf-8'))
    if c.get('model_type') != 'UnetPlusPlus_resnet34_BGR_to_ImageNetRGB' or c.get('numerical_protocol') != {'dtype': 'float32', 'cuda_tf32': False, 'cudnn_deterministic': True}:
        raise ValueError('Unsupported transfer model/numerical contract')
    d = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    if str(d).startswith('cuda') and (torch.backends.cuda.matmul.allow_tf32 or torch.backends.cudnn.allow_tf32 or not torch.backends.cudnn.deterministic):
        raise ValueError('Use deterministic FP32/no-TF32 CUDA for this offline candidate')
    weight = (config_path.parent/c['weight']).resolve()
    if not weight.is_relative_to(config_path.parent) or hashlib.sha256(weight.read_bytes()).hexdigest() != c['weight_sha256']:
        raise ValueError('Transfer weights changed or escaped their directory')
    model = RGBTransferModel()
    model.load_state_dict(torch.load(weight, map_location='cpu', weights_only=True)); model.to(d)
    return RGBTransferDetector(model, c['threshold'], c['method'])
