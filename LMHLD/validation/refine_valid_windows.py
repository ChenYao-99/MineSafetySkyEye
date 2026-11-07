"""Pre-prediction quality correction: exclude partial padded windows explicitly."""
import hashlib
import json
from datetime import datetime
from pathlib import Path
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / "experiments" / "mining_validation_20261001"
OUT = STUDY / "evaluation"


def refine():
    if list(OUT.glob("results*.json")):
        raise RuntimeError("Test predictions already exist; this pre-prediction correction is not allowed")
    manifest_path = STUDY / "validation_cases.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("version") == 2:
        return
    case = manifest["cases"][0]
    path = STUDY / case["valid"]
    raw = path.read_bytes()
    valid = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_UNCHANGED)
    ref = cv2.imdecode(np.fromfile(STUDY / case["reference"], np.uint8), cv2.IMREAD_UNCHANGED)
    valid[(valid.shape[0]//32)*32:, :] = 0
    valid[:, (valid.shape[1]//32)*32:] = 0
    (STUDY / "validation_cases_pre_prediction_v1.json").write_bytes(manifest_path.read_bytes())
    path.with_name("las_cruces_valid_pre_prediction_v1.png").write_bytes(raw)
    ok, data = cv2.imencode(".png", valid)
    if not ok:
        raise ValueError("Unable to save corrected ignore mask")
    path.write_bytes(data.tobytes())
    case["sha256"]["valid"] = hashlib.sha256(path.read_bytes()).hexdigest()
    case["scored_positive_pixels"] = int(((ref > 0) & (valid > 0)).sum())
    case["scored_background_pixels"] = int(((ref == 0) & (valid > 0)).sum())
    case["excluded"] += "; all partial edge windows excluded because reflection padding can import annotation pixels"
    manifest.update(version=2, quality_corrected_at=datetime.now().isoformat(timespec="seconds"),
                    quality_correction="partial-window exclusion corrected before any test prediction; references and training unchanged")
    manifest_path.write_bytes((json.dumps(manifest, ensure_ascii=False, indent=2)+"\n").encode())
    plan_path = OUT / "evaluation_plan.json"
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    (OUT / "evaluation_plan_pre_prediction_v1.json").write_bytes(plan_path.read_bytes())
    plan.update(validation_cases_sha256=hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
                quality_corrected_at=manifest["quality_corrected_at"],
                quality_correction=manifest["quality_correction"])
    plan_path.write_bytes((json.dumps(plan, ensure_ascii=False, indent=2)+"\n").encode())
    image = cv2.imdecode(np.fromfile(STUDY / case["image"], np.uint8), cv2.IMREAD_COLOR)
    image[valid == 0] = (image[valid == 0]*.4).astype(np.uint8)
    cv2.drawContours(image, cv2.findContours(ref, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0], -1, (255, 200, 0), 1)
    ok, data = cv2.imencode(".png", image)
    if ok:
        (STUDY / "cases/las_cruces_reference_review.png").write_bytes(data.tobytes())
    print("Corrected clean-window counts:", case["scored_positive_pixels"], case["scored_background_pixels"])


if __name__ == "__main__":
    refine()
