"""Prepare source-linked mine cases without any model predictions or training labels."""
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / "experiments" / "mining_validation_20261001"
CASES = STUDY / "cases"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_image(path, image):
    ok, encoded = cv2.imencode(".png", image)
    if not ok:
        raise ValueError(f"Unable to encode {path}")
    path.write_bytes(encoded.tobytes())


def prepare():
    CASES.mkdir(parents=True, exist_ok=True)
    manifest = STUDY / "validation_cases.json"
    if manifest.exists():
        value = json.loads(manifest.read_text(encoding="utf-8"))
        for case in value["cases"]:
            for key, expected in case["sha256"].items():
                if sha(STUDY / case[key]) != expected:
                    raise ValueError("Frozen case changed")
        return value
    figure_path = STUDY / "inspection" / "las_cruces_figure3.jpg"
    figure = cv2.imdecode(np.fromfile(figure_path, np.uint8), cv2.IMREAD_COLOR)
    # Fixed crop of Figure 3b; raw pixels are retained, including the published red outline.
    crop = [30, 388, 503, 714]
    image = figure[crop[1]:crop[3], crop[0]:crop[2]]
    b, g, r = [channel.astype(np.int16) for channel in cv2.split(image)]
    red = ((r > 100) & (r-g > 25) & (r-b > 20)).astype(np.uint8)
    line = cv2.morphologyEx(red, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    contours, _ = cv2.findContours(line, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boundary = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(boundary)
    if not .1 * image.shape[0]*image.shape[1] < area < .6*image.shape[0]*image.shape[1]:
        raise ValueError("Published boundary cannot be recovered reliably")
    reference = np.zeros(image.shape[:2], np.uint8)
    cv2.drawContours(reference, [boundary], -1, 1, cv2.FILLED)
    artefacts = cv2.dilate(red, np.ones((5, 5), np.uint8))
    # Plot letter and scale-bar areas are excluded along with annotation-line contact.
    artefacts[:32, :32] = 1
    artefacts[282:, 315:] = 1
    valid = np.ones_like(reference)
    # Inference uses independent 32px windows. Excluding every touched window means
    # scored inputs never see the reference line or legend, avoiding label leakage.
    for y in range(0, len(valid), 32):
        for x in range(0, valid.shape[1], 32):
            part = artefacts[y:y+32, x:x+32]
            if part.shape != (32, 32) or part.any():
                valid[y:y+32, x:x+32] = 0
    positives = int((reference * valid).sum())
    negatives = int(((1-reference) * valid).sum())
    if positives < 1024 or negatives < 4096:
        raise ValueError("Insufficient unannotated, scored foreground/background pixels")
    outputs = {"image": CASES / "las_cruces_201901_figure_input.png",
               "reference": CASES / "las_cruces_201901_reference.png",
               "valid": CASES / "las_cruces_201901_valid.png"}
    save_image(outputs["image"], image)
    save_image(outputs["reference"], reference)
    save_image(outputs["valid"], valid)
    review = image.copy()
    review[valid == 0] = (review[valid == 0].astype(float) * .4).astype(np.uint8)
    cv2.drawContours(review, [boundary], -1, (255, 200, 0), 1)
    save_image(CASES / "las_cruces_reference_review.png", review)
    las = {"id": "las_cruces_201901", "mine": "Las Cruces, Spain", "event": "2019-01",
           "image_acquisition": "2019-01, according to Figure 3 caption", "evaluation_mode": "figure_reference",
           "source": "https://zenodo.org/records/4704955", "source_figure": "Figure 3b, PDF page 21 (1-based)",
           "source_license": "CC BY 4.0 (verified repository metadata)",
           "authors": "Lopez-Vinielles et al. (2020)", "figure_crop_xyxy": crop,
           "label_origin": "digitized red external landslide boundary published by research authors",
           "label_quality": "published-figure-derived reference, not original GIS or independently audited masks",
           "excluded": "all 32px windows touching annotation line or plot legend; border uncertainty excluded",
           "scored_positive_pixels": positives, "scored_background_pixels": negatives,
           "reference_area_ratio": float(reference.mean()), "image_shape": list(image.shape),
           "tile_size": 32, "use": "held-out evaluation only; no training or threshold selection",
           **{key: str(path.relative_to(STUDY)).replace("\\", "/") for key, path in outputs.items()},
           "sha256": {key: sha(path) for key, path in outputs.items()}}
    nasa_path = STUDY / "sources" / "bingham_post_20130502.jpg"
    nasa = {"id": "bingham_20130410", "mine": "Bingham Canyon, USA", "event": "2013-04-10",
            "image_acquisition": "2013-05-02", "evaluation_mode": "confirmed_event_without_pixel_truth",
            "source": "https://science.nasa.gov/earth/earth-observatory/sizing-up-the-landslide-at-bingham-canyon-mine-81364/",
            "credit": "NASA Earth Observatory / EO-1 ALI; 2011 context image from USDA NAIP",
            "label_quality": "event and acquisition dates confirmed; no pixel reference mask is asserted",
            "inventory_audit": "Utah public inventory bbox query returned zero features; not used as ground truth",
            "image": str(nasa_path.relative_to(STUDY)).replace("\\", "/"),
            "use": "qualitative case only; never report pixel F1/recall without a reference",
            "sha256": {"image": sha(nasa_path)}}
    value = {"version": 2, "frozen_before_prediction": True, "cases": [las, nasa],
             "independent_mine_events": 2, "scorable_figure_reference_cases": 1,
             "full_expert_audited_test_sets": 0, "no_claim_of_production_warning_validation": True}
    manifest.write_bytes((json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode())
    print(f"Frozen cases: Las Cruces scored foreground={positives}, background={negatives}; NASA event is qualitative only.")
    return value


if __name__ == "__main__":
    prepare()
