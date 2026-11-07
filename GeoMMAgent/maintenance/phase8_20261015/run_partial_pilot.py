"""Four fixed-budget fits with a paired source-replay control, entirely offline."""
from __future__ import annotations
import argparse
import gc
import json
import os
import random
import sys
import time
from pathlib import Path

from admit_partial_data import DET, GEO, HERE, OLD, OUT, read, write, sha, now
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import numpy as np
import rasterio
import torch
import torch.nn.functional as F
import segmentation_models_pytorch as smp
from detector import LandslideDetector


def masked_positive_loss(logits, labels):
    """Only independently supplied positives carry a gradient; 255 is unknown."""
    if logits.shape != labels.shape or not torch.all((labels == 1) | (labels == 255)):
        raise ValueError("New partial labels require shape agreement and the explicit 1/255 schema")
    known = labels == 1
    if not bool(known.any()):
        raise ValueError("An entirely unknown target is not a supervised sample")
    loss = F.binary_cross_entropy_with_logits(logits, torch.ones_like(logits), reduction="none")
    return loss[known].mean()


def validate_inputs(protocol):
    for record in protocol["protected_registries"] + [protocol["base_model"], protocol["training_pack"], protocol["source_phase7_qualification"]]:
        if sha(record["path"]) != record["sha256"]:
            raise ValueError("Frozen experiment dependency changed")
    if protocol["new_event_group_role"]["role"] != "development" or protocol["deployment_promotion_allowed"]:
        raise ValueError("This pilot cannot open a blind set or promote deployment")
    train_rows = {(row["region"], row["split"], row["index"]) for row in protocol["source_replay_rows"]}
    val_rows = {(row["region"], row["split"], row["index"]) for row in protocol["source_validation_rows"]}
    if train_rows & val_rows or any(row[1] != "train" for row in train_rows) or any(row[1] != "val" for row in val_rows):
        raise ValueError("Source split leakage")
    for mine in protocol["historical_mine_monitors"]:
        for key, path in mine["paths"].items():
            if sha(path) != mine["hashes"][key]:
                raise ValueError("Historical development reference changed")


def model_from_baseline(protocol, device):
    model = smp.UnetPlusPlus(encoder_name="resnet34", encoder_weights=None, in_channels=4, classes=1)
    model.load_state_dict(torch.load(protocol["base_model"]["path"], weights_only=True, map_location="cpu"))
    return model.to(device).eval()


def predict(model, x, device, batch=64):
    rows = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(x), batch):
            tensor = torch.as_tensor(np.ascontiguousarray(x[start:start+batch]), device=device, dtype=torch.float32)
            values = model(tensor)
            if not torch.isfinite(values).all():
                raise ValueError("Nonfinite inference logits")
            rows.append(torch.sigmoid(values).cpu().numpy())
    return np.concatenate(rows)


def pixel_metrics(prob, labels, known, threshold):
    p, y, valid = np.asarray(prob).reshape(-1), np.asarray(labels).reshape(-1), np.asarray(known).reshape(-1).astype(bool)
    if len(p) != len(y) or len(p) != len(valid) or not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
        raise ValueError("Invalid probability/reference grid")
    positive, background = valid & (y == 1), valid & (y == 0)
    hit = p >= threshold
    tp, fn, fp, tn = [int(a.sum()) for a in [positive & hit, positive & ~hit, background & hit, background & ~hit]]
    precision = (tp / (tp + fp) if tp + fp else 0.0) if background.any() else None
    recall = tp / (tp + fn) if tp + fn else None
    f1 = (2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0) if background.any() else None
    return {"tp": tp, "fn": fn, "fp": fp, "tn": tn, "precision_on_scored_reference": precision,
            "recall_on_scored_reference": recall, "f1_on_scored_reference": f1,
            "annotation_background_activation": fp / (fp + tn) if fp + tn else None,
            "unknown_pixels_scored": 0}


def evaluate(model, pack, protocol, device, output):
    output.mkdir(parents=True, exist_ok=True)
    source_prob = predict(model, pack["validation_x"], device)
    source_y = pack["validation_y"]
    result = {"source_validation": {}, "historical_seen_mine_monitors": {}, "target_training_fit_diagnostic": {}}
    for threshold in protocol["thresholds"]:
        key = str(threshold)
        result["source_validation"][key] = pixel_metrics(source_prob, source_y, np.ones_like(source_y, bool), threshold)
    for mine in protocol["historical_mine_monitors"]:
        ids = mine["indices"]
        x = np.asarray(np.load(mine["paths"]["x"], mmap_mode="r")[ids])
        y = np.asarray(np.load(mine["paths"]["y"], mmap_mode="r")[ids])
        weight = np.asarray(np.load(mine["paths"]["weight"], mmap_mode="r")[ids])
        probability = predict(model, x, device)
        result["historical_seen_mine_monitors"][mine["case_id"]] = {
            "role": mine["role"], "processed_patches": len(ids),
            "background_is_historical_annotation_not_verified_normal_operation": True,
            "thresholds": {str(t): pixel_metrics(probability, y, weight > 0, t) for t in protocol["thresholds"]}}
    folder = OLD / "imagery/TW_ZHONGHE2024_SOURCE_CANDIDATE"
    with rasterio.open(folder / "post_provider_BGRN_native10m.tif") as src:
        native = src.read().transpose(1, 2, 0)
    # Reuse the exact existing production sliding-window implementation.
    wrapper = object.__new__(LandslideDetector)
    wrapper.model, wrapper.device, wrapper.patch_size, wrapper.threshold = model, device, 32, protocol["primary_threshold"]
    probability = wrapper.detect_image(native, stride=32, patch_norm="minmax", max_batch=64)
    np.save(output / "zhonghe_probability.npy", probability)
    reference = read(OLD / "references/TW_ZHONGHE2024_PARTIAL/qualification.json")
    for kind, path in [("mine_core", reference["core_mask"]["path"]),
                       ("surrounding_auxiliary", reference["auxiliary_surrounding_collapse_reference"]["mask"])]:
        with rasterio.open(path) as src:
            labels = src.read(1)
        known = labels == 1
        result["target_training_fit_diagnostic"][kind] = {
            "known_positive_pixels": int(known.sum()),
            "mean_probability_on_known_core": float(probability[known].mean()),
            "thresholds": {str(t): {"known_positive_core_recall": float((probability[known] >= t).mean()),
                                    "at_least_one_core_pixel_detected": bool(np.any(probability[known] >= t))}
                           for t in protocol["thresholds"]},
            "used_in_training_so_not_independent_evaluation": True}
    result["target_training_fit_diagnostic"]["whole_window_activation_diagnostic"] = {
        str(t): float((probability >= t).mean()) for t in protocol["thresholds"]}
    result["target_training_fit_diagnostic"]["whole_window_activation_is_not_false_positive_rate"] = True
    result["independent_blind_test_completed"] = False
    write(output / "evaluation.json", result)
    return result


def train(protocol, pack, arm, seed, device):
    output = OUT / f"{arm}_seed{seed}"
    if (output / "result.json").exists():
        record = read(output / "result.json")
        if record["protocol_sha256"] != sha(OUT / "protocol.json"):
            raise ValueError("Prior fit belongs to another frozen protocol")
        return record
    output.mkdir(parents=True, exist_ok=True)
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    model = model_from_baseline(protocol, device)
    for parameter in model.encoder.parameters():
        parameter.requires_grad_(False)
    optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], **{
        key: value for key, value in protocol["optimizer"].items() if key != "kind"})
    rng = np.random.default_rng(seed)
    history = []
    started = time.perf_counter()
    for update in range(protocol["updates_per_fit"]):
        model.train()
        for module in model.modules():
            if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
                module.eval()
        ids = rng.integers(len(pack["source_x"]), size=protocol["batch_source"])
        target_id = int(rng.integers(len(pack["target_x"])))
        source_x, source_y = pack["source_x"][ids].copy(), pack["source_y"][ids].copy()
        target_x = pack["target_x"][target_id:target_id+1].copy()
        target_y = pack["target_y"][target_id:target_id+1, None].copy()
        # Same RNG and geometric augmentations in the paired arms.
        for axis in [-2, -1]:
            if rng.random() < .5:
                source_x, source_y = np.flip(source_x, axis), np.flip(source_y, axis)
            if rng.random() < .5:
                target_x, target_y = np.flip(target_x, axis), np.flip(target_y, axis)
        x = torch.as_tensor(np.ascontiguousarray(np.concatenate([source_x, target_x])), dtype=torch.float32, device=device)
        sy = torch.as_tensor(np.ascontiguousarray(source_y), dtype=torch.float32, device=device)
        ty = torch.as_tensor(np.ascontiguousarray(target_y), dtype=torch.uint8, device=device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        source_loss = F.binary_cross_entropy_with_logits(logits[:-1], sy)
        partial_loss = masked_positive_loss(logits[-1:], ty)
        coefficient = protocol["partial_loss_weight"] * float(pack["target_weight"][target_id]) if arm == "replay_plus_partial" else 0.0
        loss = source_loss + coefficient * partial_loss
        if not torch.isfinite(loss):
            raise ValueError("Nonfinite training loss; fit stopped")
        loss.backward()
        if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
            raise ValueError("Nonfinite training gradients; fit stopped")
        optimizer.step()
        if (update + 1) % 20 == 0:
            row = {"update": update + 1, "source_loss": float(source_loss.item()),
                   "partial_loss_before_weight": float(partial_loss.item()), "total_loss": float(loss.item())}
            history.append(row)
            print(arm, seed, json.dumps(row), flush=True)
    elapsed = time.perf_counter() - started
    weight_path = output / "last_model.pth"
    torch.save(model.state_dict(), weight_path)
    evaluation = evaluate(model.eval(), pack, protocol, device, output)
    record = {"arm": arm, "seed": seed, "completed_at": now(), "protocol_sha256": sha(OUT / "protocol.json"),
              "updates_completed": protocol["updates_per_fit"], "train_seconds": elapsed,
              "history": history, "weights": {"path": str(weight_path), "sha256": sha(weight_path), "bytes": weight_path.stat().st_size},
              "evaluation": evaluation, "independent_blind_test_completed": False}
    write(output / "result.json", record)
    del optimizer, model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return record


def summarize(protocol):
    results = [read(OUT / f"{arm}_seed{seed}" / "result.json") for seed in protocol["seeds"] for arm in protocol["arms"]]
    paired = []
    threshold = str(protocol["primary_threshold"])
    for seed in protocol["seeds"]:
        control = next(r for r in results if r["seed"] == seed and r["arm"] == "replay_control")
        partial = next(r for r in results if r["seed"] == seed and r["arm"] == "replay_plus_partial")
        a, b = control["evaluation"], partial["evaluation"]
        f1_delta = b["source_validation"][threshold]["f1_on_scored_reference"] - a["source_validation"][threshold]["f1_on_scored_reference"]
        core_delta = b["target_training_fit_diagnostic"]["mine_core"]["thresholds"][threshold]["known_positive_core_recall"] - a["target_training_fit_diagnostic"]["mine_core"]["thresholds"][threshold]["known_positive_core_recall"]
        old_mine_deltas = {mine: b["historical_seen_mine_monitors"][mine]["thresholds"][threshold]["recall_on_scored_reference"] - a["historical_seen_mine_monitors"][mine]["thresholds"][threshold]["recall_on_scored_reference"]
                           for mine in a["historical_seen_mine_monitors"]}
        activation_delta = b["target_training_fit_diagnostic"]["whole_window_activation_diagnostic"][threshold] - a["target_training_fit_diagnostic"]["whole_window_activation_diagnostic"][threshold]
        paired.append({"seed": seed, "source_f1_delta": f1_delta, "mine_training_core_recall_delta": core_delta,
                       "historical_seen_mine_recall_deltas": old_mine_deltas, "unscored_whole_window_activation_delta": activation_delta,
                       "limited_pilot_gate_passed": f1_delta >= -.02 and core_delta > 0 and min(old_mine_deltas.values()) >= -.05 and activation_delta <= .05})
    # This gate is only an offline prototype retention rule, never deployment.
    consistent = all(row["limited_pilot_gate_passed"] for row in paired)
    winner = max([r for r in results if r["arm"] == "replay_plus_partial"], key=lambda r: r["evaluation"]["source_validation"][threshold]["f1_on_scored_reference"]) if consistent else None
    removed = []
    for record in results:
        path = Path(record["weights"]["path"]).resolve()
        if not path.is_relative_to(OUT.resolve()) or path.name != "last_model.pth":
            raise ValueError("Invalid experiment-only checkpoint cleanup path")
        keep = winner is not None and record["weights"]["sha256"] == winner["weights"]["sha256"]
        if not keep and path.exists():
            if sha(path) != record["weights"]["sha256"]:
                raise ValueError("Checkpoint changed before cleanup")
            removed.append({"path": str(path), "bytes": path.stat().st_size, "sha256": record["weights"]["sha256"]})
            path.unlink()
    validate_inputs(protocol)
    summary = {"completed_at": now(), "status": "bounded_partial_supervision_pilot_complete",
               "fits_completed": len(results), "paired_results": paired,
               "consistent_limited_pilot_gain": consistent,
               "retained_offline_prototype": winner["weights"] if winner else None,
               "discarded_checkpoints": removed, "deployment_weight_changed": False,
               "new_cloud_generation_calls": 0, "independent_blind_tests_opened": 0,
               "known_target_pixels_are_training_fit_not_generalization": True,
               "new_complete_accident_boundary_claimed": False}
    write(OUT / "summary.json", summary)
    state = read(HERE / "state.json")
    state.update(status=summary["status"], completed_at=summary["completed_at"], new_training_completed=len(results),
                 updates_per_fit=protocol["updates_per_fit"], summary=str(OUT / "summary.json"),
                 default_weight_changed=False, retained_offline_prototype=summary["retained_offline_prototype"],
                 next_priority="More independent mining-positive/operation references; avoid another parameter sweep on this one development event")
    write(HERE / "state.json", state)
    print(json.dumps(summary, ensure_ascii=False), flush=True)


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--summarize", action="store_true")
    args = parser.parse_args()
    protocol = read(OUT / "protocol.json")
    validate_inputs(protocol)
    source_manifest_path = OUT / "execution_source_manifest.json"
    source_manifest = {path.name: sha(path) for path in [HERE / "run_partial_pilot.py", HERE / "admit_partial_data.py", HERE / "test_partial_pilot.py", DET / "detector.py"]}
    if source_manifest_path.exists():
        if read(source_manifest_path) != source_manifest:
            raise ValueError("Training/evaluation implementation changed after freezing")
    else:
        write(source_manifest_path, source_manifest)
    if args.summarize:
        summarize(protocol)
        return
    if not args.train:
        parser.error("select --train or --summarize")
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with np.load(protocol["training_pack"]["path"]) as loaded:
        pack = {key: loaded[key] for key in loaded.files}
    if not set(np.unique(pack["target_y"])).issubset({1, 255}) or np.any(pack["target_y"] == 0):
        raise ValueError("New partial labels were recoded as background")
    if not (OUT / "baseline/evaluation.json").exists():
        model = model_from_baseline(protocol, device)
        evaluate(model, pack, protocol, device, OUT / "baseline")
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
        print("Baseline evaluation complete; development monitors only", flush=True)
    for seed in protocol["seeds"]:
        for arm in protocol["arms"]:
            train(protocol, pack, arm, seed, device)
            state = read(HERE / "state.json")
            state.update(status="frozen_budget_training_in_progress", new_training_completed=len(list(OUT.glob("*/result.json"))))
            write(HERE / "state.json", state)
    summarize(protocol)


if __name__ == "__main__":
    main()
