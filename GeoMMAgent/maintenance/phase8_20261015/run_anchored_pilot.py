"""One causal follow-up: exact inference tiles and frozen-teacher preservation.

Teacher outputs are consistency targets, not new human labels or safe negatives.
The first four-run result and its frozen protocol remain unchanged.
"""
from __future__ import annotations
import gc
import json
import random
import sys
import time
from pathlib import Path

from run_partial_pilot import (torch, F, np, rasterio, model_from_baseline,
                               masked_positive_loss, evaluate, validate_inputs)
from admit_partial_data import DET, HERE, OLD, OUT, read, write, sha, now, normalize

RUN = OUT / "anchored_followup"


def bernoulli_kl(logits, teacher_logits, mask=None):
    probabilities = torch.sigmoid(teacher_logits.detach()).clamp(1e-6, 1 - 1e-6)
    ce = F.binary_cross_entropy_with_logits(logits, probabilities, reduction="none")
    entropy = -(probabilities * probabilities.log() + (1 - probabilities) * (1 - probabilities).log())
    divergence = ce - entropy
    if mask is not None:
        if mask.shape != divergence.shape or not bool(mask.any()):
            raise ValueError("Consistency mask must contain explicit unlabelled native pixels")
        return divergence[mask].mean()
    return divergence.mean()


def prepare(base, source_pack):
    RUN.mkdir(parents=True, exist_ok=True)
    protocol_path = RUN / "protocol.json"
    if protocol_path.exists():
        return read(protocol_path)
    folder = OLD / "imagery/TW_ZHONGHE2024_SOURCE_CANDIDATE"
    with rasterio.open(folder / "post_provider_BGRN_native10m.tif") as src:
        image = src.read()
    h, w = image.shape[1:]
    pad_h, pad_w = (-h) % 32, (-w) % 32
    image = np.pad(image, ((0, 0), (0, pad_h), (0, pad_w)), mode="reflect")
    union = np.zeros(image.shape[1:], bool)
    object_masks = []
    for obj in base["objects"]:
        mask = np.zeros((h, w), bool)
        for index, provenance in enumerate(base["target_crops"]):
            if provenance["source_row"] == obj["source_row"]:
                row, col = provenance["row"], provenance["col"]
                mask[row:row+32, col:col+32] |= source_pack["target_y"][index] == 1
        mask = np.pad(mask, ((0, pad_h), (0, pad_w)), constant_values=False)
        union |= mask
        object_masks.append((obj, mask))
    tx, ty, weight, unknown, rows = [], [], [], [], []
    native = np.zeros(image.shape[1:], bool)
    native[:h, :w] = True
    for obj, mask in object_masks:
        if not mask.any():
            continue
        coordinates = sorted({(int(r // 32 * 32), int(c // 32 * 32)) for r, c in np.argwhere(mask)})
        for row, col in coordinates:
            labels = np.full((32, 32), 255, np.uint8)
            labels[mask[row:row+32, col:col+32]] = 1
            tx.append(normalize(image[:, row:row+32, col:col+32]))
            ty.append(labels)
            weight.append(obj["confidence_weight"])
            unknown.append((~union & native)[row:row+32, col:col+32])
            rows.append({"source_row": obj["source_row"], "row": row, "col": col,
                         "known_pixels": int((labels == 1).sum()), "kind": obj["kind"]})
    pack = RUN / "aligned_target_pack.npz"
    np.savez_compressed(pack, x=np.stack(tx), y=np.stack(ty), weight=np.asarray(weight, np.float32), unknown=np.stack(unknown))
    protocol = dict(base)
    protocol.update(registered_at=now(), purpose="targeted_followup_for_observed_forgetting_and_tile_context_mismatch",
                    source_first_pilot_summary={"path": str(OUT / "summary.json"), "sha256": sha(OUT / "summary.json")},
                    arms=["teacher_control", "teacher_plus_partial"], updates_per_fit=200,
                    aligned_target_pack={"path": str(pack), "sha256": sha(pack), "tile_rows": rows},
                    source_supervised_replay_loss_used=False,
                    teacher_role="frozen baseline output preservation, not ground truth, not operational negative labels",
                    source_teacher_kl_weight=1.0, unknown_target_teacher_kl_weight=1.0,
                    target_geometric_augmentation=False,
                    target_core_unique_pixels_actually_used=int(union.sum()),
                    pixel_augmentation_does_not_change_independent_event_count=True,
                    preservation_gate={"source_f1_max_drop_vs_baseline": .01,
                                       "seen_mine_recall_max_drop_per_case_vs_baseline": .02,
                                       "mine_training_core_recall_min_gain_vs_paired_control": .2,
                                       "unknown_window_activation_max_increase_vs_baseline": .02},
                    checkpoint_selection="fixed update 200 only; predeclared preservation criteria; no threshold search")
    write(protocol_path, protocol)
    write(RUN / "execution_source_manifest.json", {p.name: sha(p) for p in [Path(__file__), HERE / "run_partial_pilot.py", HERE / "admit_partial_data.py", DET / "detector.py"]})
    print("Follow-up frozen:", len(tx), "exact native inference tiles; labels unchanged; four fits", flush=True)
    return protocol


def fit(protocol, source, targets, arm, seed, device):
    output = RUN / f"{arm}_seed{seed}"
    if (output / "result.json").exists():
        return read(output / "result.json")
    output.mkdir(parents=True, exist_ok=True)
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    teacher = model_from_baseline(protocol, device)
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    model = model_from_baseline(protocol, device)
    for parameter in model.encoder.parameters():
        parameter.requires_grad_(False)
    optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], **{
        key: value for key, value in protocol["optimizer"].items() if key != "kind"})
    rng = np.random.default_rng(seed)
    # Balance objects, so splitting one long object into tiles cannot increase
    # its scientific sample weight or drown out the small mine core.
    groups = sorted({row["source_row"] for row in protocol["aligned_target_pack"]["tile_rows"]})
    group_ids = {group: [i for i, row in enumerate(protocol["aligned_target_pack"]["tile_rows"]) if row["source_row"] == group] for group in groups}
    history = []
    start = time.perf_counter()
    for update in range(protocol["updates_per_fit"]):
        model.train()
        for module in model.modules():
            if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
                module.eval()
        ids = rng.integers(len(source["source_x"]), size=protocol["batch_source"])
        group = groups[int(rng.integers(len(groups)))]
        tid = group_ids[group][int(rng.integers(len(group_ids[group])))]
        sx = source["source_x"][ids].copy()
        for axis in [-2, -1]:
            if rng.random() < .5:
                sx = np.flip(sx, axis)
        x = torch.as_tensor(np.ascontiguousarray(np.concatenate([sx, targets["x"][tid:tid+1]])), device=device, dtype=torch.float32)
        labels = torch.as_tensor(targets["y"][tid:tid+1, None], device=device, dtype=torch.uint8)
        mask = torch.as_tensor(targets["unknown"][tid:tid+1, None], device=device, dtype=torch.bool)
        with torch.no_grad():
            teacher_logits = teacher(x)
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        source_kl = bernoulli_kl(logits[:-1], teacher_logits[:-1])
        unknown_kl = bernoulli_kl(logits[-1:], teacher_logits[-1:], mask)
        positive_loss = masked_positive_loss(logits[-1:], labels)
        coefficient = protocol["partial_loss_weight"] * float(targets["weight"][tid]) if arm == "teacher_plus_partial" else 0.0
        loss = source_kl + unknown_kl + coefficient * positive_loss
        if not torch.isfinite(loss):
            raise ValueError("Nonfinite follow-up loss")
        loss.backward()
        optimizer.step()
        if (update + 1) % 50 == 0:
            row = {"update": update + 1, "source_preservation_kl": float(source_kl.item()),
                   "unknown_context_preservation_kl": float(unknown_kl.item()), "positive_loss": float(positive_loss.item())}
            history.append(row)
            print(arm, seed, json.dumps(row), flush=True)
    del teacher
    evaluation = evaluate(model.eval(), source, protocol, device, output)
    weight_path = output / "last_model.pth"
    torch.save(model.state_dict(), weight_path)
    record = {"arm": arm, "seed": seed, "completed_at": now(), "updates_completed": protocol["updates_per_fit"],
              "protocol_sha256": sha(RUN / "protocol.json"), "seconds": time.perf_counter() - start,
              "history": history, "evaluation": evaluation,
              "weights": {"path": str(weight_path), "sha256": sha(weight_path), "bytes": weight_path.stat().st_size}}
    write(output / "result.json", record)
    del optimizer, model
    gc.collect()
    torch.cuda.empty_cache()
    return record


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    base = read(OUT / "protocol.json")
    validate_inputs(base)
    with np.load(base["training_pack"]["path"]) as data:
        source = {key: data[key] for key in data.files}
    protocol = prepare(base, source)
    expected_sources = read(RUN / "execution_source_manifest.json")
    for path in [Path(__file__), HERE / "run_partial_pilot.py", HERE / "admit_partial_data.py", DET / "detector.py"]:
        if sha(path) != expected_sources[path.name]:
            raise ValueError("Frozen follow-up implementation changed")
    with np.load(protocol["aligned_target_pack"]["path"]) as data:
        targets = {key: data[key] for key in data.files}
    if sha(protocol["aligned_target_pack"]["path"]) != protocol["aligned_target_pack"]["sha256"]:
        raise ValueError("Exact inference-tile pack changed")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results = [fit(protocol, source, targets, arm, seed, device) for seed in protocol["seeds"] for arm in protocol["arms"]]
    baseline = read(OUT / "baseline/evaluation.json")
    threshold = str(protocol["primary_threshold"])
    gates, pairs = protocol["preservation_gate"], []
    for seed in protocol["seeds"]:
        control = next(row for row in results if row["arm"] == "teacher_control" and row["seed"] == seed)["evaluation"]
        partial = next(row for row in results if row["arm"] == "teacher_plus_partial" and row["seed"] == seed)["evaluation"]
        f1_delta = partial["source_validation"][threshold]["f1_on_scored_reference"] - baseline["source_validation"][threshold]["f1_on_scored_reference"]
        old_deltas = {mine: partial["historical_seen_mine_monitors"][mine]["thresholds"][threshold]["recall_on_scored_reference"] - baseline["historical_seen_mine_monitors"][mine]["thresholds"][threshold]["recall_on_scored_reference"] for mine in baseline["historical_seen_mine_monitors"]}
        core_delta = partial["target_training_fit_diagnostic"]["mine_core"]["thresholds"][threshold]["known_positive_core_recall"] - control["target_training_fit_diagnostic"]["mine_core"]["thresholds"][threshold]["known_positive_core_recall"]
        activation_delta = partial["target_training_fit_diagnostic"]["whole_window_activation_diagnostic"][threshold] - baseline["target_training_fit_diagnostic"]["whole_window_activation_diagnostic"][threshold]
        passed = f1_delta >= -gates["source_f1_max_drop_vs_baseline"] and min(old_deltas.values()) >= -gates["seen_mine_recall_max_drop_per_case_vs_baseline"] and core_delta >= gates["mine_training_core_recall_min_gain_vs_paired_control"] and activation_delta <= gates["unknown_window_activation_max_increase_vs_baseline"]
        pairs.append({"seed": seed, "source_f1_delta_vs_baseline": f1_delta,
                      "historical_seen_mine_recall_deltas_vs_baseline": old_deltas,
                      "mine_training_core_recall_delta_vs_paired_control": core_delta,
                      "unscored_activation_delta_vs_baseline": activation_delta,
                      "limited_preservation_gate_passed": passed})
    passed = all(row["limited_preservation_gate_passed"] for row in pairs)
    candidates = [row for row in results if row["arm"] == "teacher_plus_partial"]
    winner = max(candidates, key=lambda row: row["evaluation"]["source_validation"][threshold]["f1_on_scored_reference"]) if passed else None
    removed = []
    for row in results:
        path = Path(row["weights"]["path"]).resolve()
        if not path.is_relative_to(RUN.resolve()) or path.name != "last_model.pth":
            raise ValueError("Follow-up cleanup escaped current experiment")
        if (winner is None or row["weights"]["sha256"] != winner["weights"]["sha256"]) and path.exists():
            if sha(path) != row["weights"]["sha256"]:
                raise ValueError("Follow-up checkpoint changed before cleanup")
            removed.append(row["weights"])
            path.unlink()
    validate_inputs(base)
    summary = {"completed_at": now(), "fits_completed": 4, "updates_per_fit": protocol["updates_per_fit"],
               "paired_results": pairs, "consistent_limited_preservation_gain": passed,
               "retained_offline_prototype": winner["weights"] if winner else None,
               "discarded_checkpoints": removed, "default_weight_changed": False,
               "new_independent_test_opened": False, "new_cloud_generation_calls": 0,
               "teacher_predictions_are_not_ground_truth": True,
               "new_case_scores_are_training_fit_only": True}
    write(RUN / "summary.json", summary)
    state = read(HERE / "state.json")
    state.update(status="partial_pilot_and_targeted_preservation_followup_complete", completed_at=summary["completed_at"],
                 new_training_completed=8, followup_summary=str(RUN / "summary.json"),
                 retained_offline_prototype=summary["retained_offline_prototype"])
    write(HERE / "state.json", state)
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
