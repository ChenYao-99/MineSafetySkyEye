"""Image-only, native-detail inference with an explicit spatial compute budget."""
import numpy as np
import cv2


def whole_view(image, side=128):
    a = np.asarray(image, np.float32)
    if a.ndim != 3 or a.shape[0] != 3 or a.shape[1] != a.shape[2] or a.shape[1] < 2 or not np.isfinite(a).all():
        raise ValueError('Finite square three-band BGR parent required')
    if side not in (64, 128): raise ValueError('Registered input size required')
    lo = a.min((1, 2), keepdims=True); span = a.max((1, 2), keepdims=True)-lo
    a = np.divide(a-lo, span, out=np.zeros_like(a), where=span > 0)
    inter = cv2.INTER_AREA if a.shape[-1] > side else cv2.INTER_LINEAR
    # Frozen RGB preprocessing resizes a four-channel BGR+copied-red buffer.
    # Preserve that OpenCV arithmetic path, then discard the copied channel.
    packed = np.concatenate([a, a[2:3]], axis=0)
    return cv2.resize(packed.transpose(1, 2, 0), (side, side), interpolation=inter).transpose(2, 0, 1)[:3].astype(np.float16)


def predict(model, parents, batch=16):
    import torch
    a = np.asarray(parents)
    if a.ndim != 4 or a.shape[1:] != (3, 128, 128) or not len(a) or not np.isfinite(a).all() or a.min() < 0 or a.max() > 1:
        raise ValueError('Normalized 128-square BGR batches required')
    if type(batch) is not int or batch < 1: raise ValueError('Positive batch required')
    result = []; device = next(model.parameters()).device
    with torch.inference_mode():
        for start in range(0, len(a), batch):
            x = torch.as_tensor(np.ascontiguousarray(a[start:start+batch]), device=device).float()
            p = model(x).sigmoid()[:, 0]
            if p.shape[1:] != (128, 128) or not torch.isfinite(p).all(): raise ValueError('Invalid resolution output')
            result.append(p.cpu().numpy())
    return np.concatenate(result)
