"""Evaluate frozen, held-out public cases after all preregistered training runs finish."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "cache" / "matplotlib"))
os.environ.setdefault("TORCH_HOME", str(ROOT / "cache" / "torch"))
os.environ["SMP_DISABLE_VERSION_CHECK"] = "1"

import argparse
import csv
import hashlib
import json
import sys
import time
from datetime import datetime

sys.path.insert(0, str(ROOT))
import cv2
import numpy as np
import torch
from detector import LandslideDetector
from finetune_negatives import norm_patch, DATA_ROOT, NEG_NPY

TRAIN = ROOT / "experiments" / "core_ablation_20261001"
STUDY = ROOT / "experiments" / "mining_validation_20261001"
OUT = STUDY / "evaluation"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_image(path):
    image = cv2.imdecode(np.fromfile(path, np.uint8), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f"Cannot decode {path}")
    return image


def write_json(path, value):
    path.write_bytes((json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode())


def metrics(prediction, target, valid=None):
    p, y = np.asarray(prediction, bool), np.asarray(target, bool)
    v = np.ones_like(y) if valid is None else np.asarray(valid, bool)
    tp, fp = int((p & y & v).sum()), int((p & ~y & v).sum())
    fn, tn = int((~p & y & v).sum()), int((~p & ~y & v).sum())
    return {"TP": tp, "FP": fp, "FN": fn, "TN": tn,
            "precision": tp / max(tp+fp, 1), "recall": tp / max(tp+fn, 1),
            "F1": 2*tp / max(2*tp+fp+fn, 1), "IoU": tp / max(tp+fp+fn, 1),
            "background_FPR": fp / max(fp+tn, 1)}


def register_plan():
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "evaluation_plan.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    plan = {"registered_at": datetime.now().isoformat(timespec="seconds"),
            "training_protocol_sha256": sha(TRAIN / "protocol.json"),
            "validation_cases_sha256": sha(STUDY / "validation_cases.json"),
            "primary_threshold": .5, "secondary_threshold": .4,
            "primary_comparison": "paired ours-minus-naive on identical source-figure clean windows",
            "primary_readouts": ["F1", "recall", "background_FPR"],
            "source_retention_gate": .85,
            "avoid_single_metric_selection": "report recall and FPR together; no winner chosen from FPR alone",
            "case_limit": "one published-figure reference case; no full original GIS test set",
            "nasa_limit": "Bingham event-confirmed image has no asserted pixel truth; never give its F1/recall",
            "baige_limit": "training scene; never counted as unseen mine validation",
            "all_models_complete_before_first_test_prediction": True,
            "no_threshold_or_training_adjustments_from_test": True}
    write_json(path, plan)
    return plan


def complete_records():
    protocol = json.loads((TRAIN / "protocol.json").read_text(encoding="utf-8"))
    digest = sha(TRAIN / "protocol.json")
    rows = []
    for seed in (42, 43, 44):
        for arm in ("control", "naive", "ours"):
            path = TRAIN / f"train_{arm}_seed{seed}.json"
            if not path.exists():
                return None
            try:
                record = json.loads(path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                # Training may be writing its progress snapshot at this instant.
                return None
            if record["completed_epochs"] != 30 or record["protocol_sha256"] != digest:
                return None
            if not Path(record["weights"]).is_file():
                return None
            rows.append(record)
    return rows


def predict_patches(detector, images, batch=128):
    return np.concatenate([detector.predict_batch(images[i:i+batch]) for i in range(0, len(images), batch)])


def prepare_background_holdout(images, indices):
    return norm_patch(np.asarray(images[indices]).astype(np.float32))


def evaluate(rows, plan, append=False):
    if sha(TRAIN / "protocol.json") != plan["training_protocol_sha256"] or sha(STUDY / "validation_cases.json") != plan["validation_cases_sha256"]:
        raise ValueError("A registered protocol or test case has changed")
    manifest = json.loads((STUDY / "validation_cases.json").read_text(encoding="utf-8"))
    cases = manifest["cases"]
    for case in cases:
        for key, expected in case["sha256"].items():
            if sha(STUDY / case[key]) != expected:
                raise ValueError(f"Frozen file changed for {case['id']}")
    region = Path(DATA_ROOT) / "Emilia-Romagna_Italy_32"
    x = norm_patch(np.load(region / "test_images.npy").astype(np.float32))
    y = np.load(region / "test_labels.npy")[:, 0] > 0
    split = json.loads((ROOT / "experiments/nir_train/neg_split_seed42.json").read_text(encoding="utf-8"))
    neg = prepare_background_holdout(np.load(NEG_NPY), split["holdout_idx"])
    protocol = json.loads((TRAIN / "protocol.json").read_text(encoding="utf-8"))
    previous = json.loads((OUT / "results.json").read_text(encoding="utf-8")) if append else None
    work = rows if append else [{"arm": "source_base", "seed": None, "weights": protocol["base_weights"]}] + rows
    all_results = previous["models"] if previous else []
    if append:
        (OUT / "results_pre_deployment_diagnostic.json").write_bytes((OUT / "results.json").read_bytes())
    for record in work:
        arm, seed = record["arm"], record["seed"]
        detector = LandslideDetector(record["weights"], threshold=.5, patch_size=32)
        source = metrics(predict_patches(detector, x) >= .5, y)
        copied = x.copy(); copied[:, 3] = copied[:, 2]
        rgb_source = metrics(predict_patches(detector, copied) >= .5, y)
        negative_probs = predict_patches(detector, neg)
        negative_stats = {"n": len(neg), "pixel_activation": float((negative_probs >= .5).mean()),
                          "block_activation": float(((negative_probs >= .5).sum(axis=(1, 2)) >= 10).mean()),
                          "preprocessing": "per-patch, per-channel minmax to [0,255]",
                          "block_definition": "at least 10 predicted pixels at threshold 0.5, matching historical reports",
                          "interpretation": "same-scene-pool pseudo-background patch holdout, not independently verified mine FPR"}
        row = {"arm": arm, "seed": seed, "weights": record["weights"], "weights_sha256": sha(Path(record["weights"])),
               "source_test": source, "source_RGB_proxy_test": rgb_source, "generic_background_holdout": negative_stats,
               "cases": {}}
        row["evaluation_role"] = record.get("evaluation_role", "preregistered_comparison")
        if record.get("deployment_threshold") is not None:
            row["deployment_threshold"] = record["deployment_threshold"]
        folder = OUT / f"{arm}_seed{seed}"
        folder.mkdir(exist_ok=True)
        for case in cases:
            image = read_image(STUDY / case["image"])
            if image.ndim != 3 or image.shape[2] != 3:
                raise ValueError("Cases must contain actual 3-channel RGB imagery")
            input4 = np.concatenate([image, image[..., 2:3]], axis=-1).astype(np.float32)
            probabilities = detector.detect_image(input4, patch_norm="minmax", max_batch=128)
            item = {"label_quality": case["label_quality"], "mean_probability": float(probabilities.mean()),
                    "whole_image_activation@0.5": float((probabilities >= .5).mean())}
            if case["evaluation_mode"] == "figure_reference":
                target, valid = read_image(STUDY / case["reference"]) > 0, read_image(STUDY / case["valid"]) > 0
                item["primary@0.5"] = metrics(probabilities >= .5, target, valid)
                item["secondary@0.4"] = metrics(probabilities >= .4, target, valid)
                # Additional discrimination on the same frozen clean windows is descriptive only.
                item["positive_mean_probability"] = float(probabilities[target & valid].mean())
                item["background_mean_probability"] = float(probabilities[~target & valid].mean())
                overlay = image.copy()
                p = probabilities >= .5
                tp, fp, fn = p & target & valid, p & ~target & valid, ~p & target & valid
                for selected, color in ((tp, (40, 170, 40)), (fp, (0, 200, 240)), (fn, (40, 40, 220))):
                    overlay[selected] = (.45*overlay[selected] + .55*np.asarray(color)).astype(np.uint8)
                overlay[~valid] = (.35*overlay[~valid]).astype(np.uint8)
            else:
                item["evaluation_note"] = "qualitative known-event response; no reference mask, no accuracy or recall asserted"
                overlay = image.copy()
                p = probabilities >= .5
                overlay[p] = (.5*overlay[p] + .5*np.array([30, 30, 230])).astype(np.uint8)
            ok, data = cv2.imencode(".png", overlay)
            if not ok:
                raise ValueError("Cannot encode review overlay")
            (folder / f"{case['id']}_review.png").write_bytes(data.tobytes())
            row["cases"][case["id"]] = item
        all_results.append(row)
        write_json(OUT / "results_partial.json", {"plan": plan, "models": all_results})
        print(f"evaluated {arm}/seed{seed}: source_F1={source['F1']:.4f}; Las F1={row['cases']['las_cruces_201901']['primary@0.5']['F1']:.4f}", flush=True)
        del detector
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    base_f1 = all_results[0]["source_test"]["F1"]
    summary = {}
    for arm in ("control", "naive", "ours"):
        group = [r for r in all_results if r["arm"] == arm]
        fields = {}
        for metric in ("F1", "recall", "precision", "background_FPR"):
            values = [r["cases"]["las_cruces_201901"]["primary@0.5"][metric] for r in group]
            fields[metric] = {"mean": float(np.mean(values)), "std": float(np.std(values, ddof=1)), "values": values}
        retention = [r["source_test"]["F1"] / base_f1 for r in group]
        fields["source_retention"] = {"mean": float(np.mean(retention)), "values": retention,
                                     "all_pass_0.85": all(v >= .85 for v in retention)}
        summary[arm] = fields
    paired = {}
    for metric in ("F1", "recall", "background_FPR"):
        delta = [summary["ours"][metric]["values"][i] - summary["naive"][metric]["values"][i] for i in range(3)]
        paired[metric] = {"values": delta, "mean": float(np.mean(delta)), "std": float(np.std(delta, ddof=1))}
    result = {"finished_at": datetime.now().isoformat(timespec="seconds"), "plan": plan,
              "models": all_results, "summary": summary, "paired_ours_minus_naive": paired,
              "limitations": ["one image-derived reference case, no full independently audited mine test set",
                              "interior clean-window metrics exclude boundary and plot artefacts",
                              "NASA Bingham results are qualitative only", "three seeds characterize training variation, not across-mine generalization",
                              "Baige is a training scene; its response is not used as unseen positive validation"]}
    if previous:
        result["auxiliary_metric_correction"] = previous.get("auxiliary_metric_correction")
        result["additional_deployment_diagnostic"] = "existing served model checked after the primary study; not included in paired-seed superiority claims"
    write_json(OUT / "results.json", result)
    with open(OUT / "metrics.csv", "w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["arm", "seed", "source_F1", "source_RGB_F1", "figure_F1@0.5", "figure_recall@0.5", "figure_background_FPR@0.5"])
        for r in all_results:
            m = r["cases"]["las_cruces_201901"]["primary@0.5"]
            writer.writerow([r["arm"], r["seed"], r["source_test"]["F1"], r["source_RGB_proxy_test"]["F1"], m["F1"], m["recall"], m["background_FPR"]])
    print(json.dumps({"summary": summary, "paired": paired}, ensure_ascii=False), flush=True)


def refresh_background_only():
    path = OUT / "results.json"
    result = json.loads(path.read_text(encoding="utf-8"))
    archive = OUT / "results_auxiliary_preprocessing_v1.json"
    if not archive.exists():
        archive.write_bytes(path.read_bytes())
    split = json.loads((ROOT / "experiments/nir_train/neg_split_seed42.json").read_text(encoding="utf-8"))
    negative = prepare_background_holdout(np.load(NEG_NPY), split["holdout_idx"])
    for row in result["models"]:
        if sha(Path(row["weights"])) != row["weights_sha256"]:
            raise ValueError("Evaluation checkpoint changed")
        detector = LandslideDetector(row["weights"], patch_size=32)
        probabilities = predict_patches(detector, negative)
        row["generic_background_holdout"] = {
            "n": len(negative), "pixel_activation": float((probabilities >= .5).mean()),
            "block_activation": float(((probabilities >= .5).sum(axis=(1, 2)) >= 10).mean()),
            "preprocessing": "per-patch, per-channel minmax to [0,255]",
            "block_definition": "at least 10 predicted pixels at threshold 0.5, matching historical reports",
            "interpretation": "same-scene-pool pseudo-background patch holdout, not independently verified mine FPR"}
        print(f"background correction {row['arm']}/seed{row['seed']}: {row['generic_background_holdout']['block_activation']:.4f}", flush=True)
        del detector
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    result["auxiliary_metric_correction"] = {
        "at": datetime.now().isoformat(timespec="seconds"),
        "reason": "initial auxiliary background evaluation omitted minmax; corrected preprocessing and historical >=10px block definition",
        "unaffected": "training, model selection, source test and all held-out mine-reference metrics"}
    write_json(path, result)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("--register-only", action="store_true")
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--refresh-background-only", action="store_true")
    parser.add_argument("--deployment-only", action="store_true")
    args = parser.parse_args()
    if args.refresh_background_only:
        refresh_background_only()
        sys.exit(0)
    plan = register_plan()
    if args.deployment_only:
        from detector_service import UPLOAD_MODELS, UPLOAD_THRESHOLDS
        entry = UPLOAD_MODELS["mine"]
        if entry[1:] != (32, "minmax", 1.0):
            raise ValueError("Served model has a different inference protocol")
        if any(row["arm"] == "deployed_mine" for row in json.loads((OUT / "results.json").read_text(encoding="utf-8"))["models"]):
            raise ValueError("Deployment diagnostic already exists")
        evaluate([{"arm": "deployed_mine", "seed": 42, "weights": entry[0],
                   "deployment_threshold": UPLOAD_THRESHOLDS["mine"],
                   "evaluation_role": "additional_existing_deployment_diagnostic"}], plan, append=True)
        sys.exit(0)
    if not args.register_only:
        rows = complete_records()
        deadline = time.monotonic() + 7200
        while rows is None and args.watch and time.monotonic() < deadline:
            time.sleep(10)
            rows = complete_records()
        if rows is None:
            raise RuntimeError("All nine frozen runs must finish before test prediction")
        evaluate(rows, plan)
