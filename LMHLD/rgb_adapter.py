"""Calibrated probability blend for the paired RGB adapter and existing detector."""
import hashlib
import json
from pathlib import Path

from detector import LandslideDetector


class GuardedRGBDetector(LandslideDetector):
    def __init__(self, current_weights, adapter_weights, adapter_weight, **kwargs):
        if not 0 <= adapter_weight <= 1:
            raise ValueError("adapter_weight must be between zero and one")
        super().__init__(current_weights, **kwargs)
        self.adapter = LandslideDetector(adapter_weights, device=self.device,
                                         threshold=self.threshold, patch_size=self.patch_size)
        self.adapter_weight = float(adapter_weight)

    def predict_batch(self, patches):
        current = super().predict_batch(patches)
        adapter = self.adapter.predict_batch(patches)
        return ((1-self.adapter_weight)*current+self.adapter_weight*adapter)


def load_guarded_adapter(config_path, models_root, **kwargs):
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    if config.get("schema_version") != 1 or not isinstance(config.get("components"), dict):
        raise ValueError("Unsupported RGB adapter config")
    if config.get("regression_passed") is not True:
        raise ValueError("RGB adapter has not passed its regression gates")
    if any(key not in config["components"] for key in ("current", "adapter")):
        raise ValueError("RGB adapter components are incomplete")
    root = Path(models_root).resolve()
    paths = []
    for key in ("current", "adapter"):
        component = config["components"][key]
        if (not isinstance(component, dict) or not isinstance(component.get("path"), str)
                or not isinstance(component.get("sha256"), str)):
            raise ValueError(f"Invalid RGB adapter component: {key}")
        path = (root/component["path"]).resolve()
        if not path.is_relative_to(root):
            raise ValueError("Adapter component must remain inside models_root")
        if not path.is_file():
            raise FileNotFoundError(f"Missing RGB adapter component: {key}")
        if hashlib.sha256(path.read_bytes()).hexdigest()!=component["sha256"]:
            raise ValueError(f"RGB adapter component hash mismatch: {key}")
        paths.append(path)
    detector = GuardedRGBDetector(*paths, adapter_weight=config["adapter_weight"], **kwargs)
    detector.validation_note = config["validation_note"]
    return detector
