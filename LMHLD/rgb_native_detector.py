"""Standalone native-BGR candidate inference, isolated from registered services."""
from pathlib import Path
import hashlib
import json
from runtime_paths import configure_runtime_paths
configure_runtime_paths()
import numpy as np
import torch
import cv2
import rgb_native


def predict_tta(model, parents, batch=64):
    a = np.asarray(parents)
    if a.ndim != 4 or a.shape[1:] != (3, 64, 64) or not len(a) or not np.isfinite(a).all() or a.min() < 0 or a.max() > 1:
        raise ValueError('Normalized three-channel64 parents required')
    if isinstance(batch, bool) or not isinstance(batch, int) or batch < 1: raise ValueError('Positive batch required')
    device = next(model.parameters()).device; result = []
    with torch.inference_mode():
        for start in range(0, len(a), batch):
            x = torch.as_tensor(np.ascontiguousarray(a[start:start+batch]), device=device).float(); total = None
            for k in range(5):
                z = x.rot90(k, (-2, -1)) if k in (1, 2, 3) else x.flip(-1) if k == 4 else x
                p = model(z).sigmoid()[:, 0]
                p = p.rot90(-k, (-2, -1)) if k in (1, 2, 3) else p.flip(-1) if k == 4 else p
                if p.shape[1:] != (64, 64) or not torch.isfinite(p).all(): raise ValueError('Invalid native RGB probability')
                total = p if total is None else total+p
            result.append((total/5).cpu().numpy())
    return np.concatenate(result)


class NativeRGBDetector:
    def __init__(self, model, threshold):
        if isinstance(threshold, bool) or not isinstance(threshold, (int, float)) or not np.isfinite(threshold) or not 0 < threshold <= 1.001:
            raise ValueError('Explicit source-calibrated threshold required')
        self.model = model.eval(); self.threshold = float(threshold)

    def detect_image(self, image_hwc, max_batch=64):
        if isinstance(max_batch, bool) or not isinstance(max_batch, (int, np.integer)) or not 1 <= max_batch <= 256:
            raise ValueError('max_batch must be an integer in1..256')
        a = np.asarray(image_hwc)
        if a.ndim != 3 or a.shape[2] != 3 or min(a.shape[:2]) < 2 or not np.isfinite(a).all():
            raise ValueError('Finite BGR HWC with exactly3 bands required; caller must handle NoData explicitly')
        h, w = a.shape[:2]; side = 32
        a = np.pad(a, ((0, (-h) % side), (0, (-w) % side), (0, 0)), mode='reflect')
        positions = [(y, x) for y in range(0, h, side) for x in range(0, w, side)]
        p = np.empty((h, w), np.float32)
        for start in range(0, len(positions), max_batch):
            part = positions[start:start+max_batch]
            v = np.stack([rgb_native.make_views(a[y:y+side, x:x+side].transpose(2, 0, 1))[0][0] for y, x in part])
            probs = predict_tta(self.model, v, batch=max_batch)
            for (y, x), prob in zip(part, probs):
                native = cv2.resize(prob, (side, side), interpolation=cv2.INTER_LINEAR)
                hh, ww = min(side, h-y), min(side, w-x); p[y:y+hh, x:x+ww] = native[:hh, :ww]
        return p


def load_candidate(config_path, device=None):
    import segmentation_models_pytorch as smp
    config_path = Path(config_path).resolve(); c = json.loads(config_path.read_text(encoding='utf-8'))
    if c.get('schema_version') != 1 or c.get('input') != 'BGR3_parent_minmax01_64_TTA5': raise ValueError('Unsupported RGB candidate contract')
    if c.get('numerical_protocol') != {'dtype': 'float32', 'cuda_tf32': False, 'cudnn_deterministic': True}:
        raise ValueError('Missing candidate numerical protocol')
    chosen_device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    if str(chosen_device).startswith('cuda') and (torch.backends.cuda.matmul.allow_tf32 or torch.backends.cudnn.allow_tf32 or not torch.backends.cudnn.deterministic):
        raise ValueError('Configure deterministic FP32/no-TF32 CUDA before loading this isolated offline candidate')
    weight = (config_path.parent/c['weight']).resolve()
    if not weight.is_relative_to(config_path.parent) or hashlib.sha256(weight.read_bytes()).hexdigest() != c['weight_sha256']:
        raise ValueError('Candidate weights changed or escaped their directory')
    model = smp.UnetPlusPlus(encoder_name='resnet34', encoder_weights=None, in_channels=3, classes=1)
    model.load_state_dict(torch.load(weight, weights_only=True, map_location='cpu')); model.to(chosen_device)
    return NativeRGBDetector(model, c['threshold'])
