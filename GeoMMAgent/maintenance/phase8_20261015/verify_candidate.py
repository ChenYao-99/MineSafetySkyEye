"""Verify the retained offline candidate and close this small development run."""
from __future__ import annotations
import ast
import json
import shutil
from pathlib import Path
from run_partial_pilot import torch, np, rasterio, LandslideDetector, validate_inputs
from admit_partial_data import DET, GEO, HERE, OLD, OUT, read, write, sha, now, normalize


def main():
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    base = read(OUT / "protocol.json")
    first = read(OUT / "summary.json")
    follow = read(OUT / "anchored_followup/summary.json")
    protocol = read(OUT / "anchored_followup/protocol.json")
    validate_inputs(base)
    candidate = follow["retained_offline_prototype"]
    if not follow["consistent_limited_preservation_gain"] or sha(candidate["path"]) != candidate["sha256"]:
        raise ValueError("Retained candidate identity/gate failed")
    with np.load(protocol["aligned_target_pack"]["path"]) as data:
        targets = {key: data[key] for key in data.files}
    folder = OLD / "imagery/TW_ZHONGHE2024_SOURCE_CANDIDATE"
    with rasterio.open(folder / "post_provider_BGRN_native10m.tif") as src:
        raw = src.read()
    h, w = raw.shape[1:]
    padded = np.pad(raw, ((0, 0), (0, (-h) % 32), (0, (-w) % 32)), mode="reflect")
    for index, row in enumerate(protocol["aligned_target_pack"]["tile_rows"]):
        r, c = row["row"], row["col"]
        expected = normalize(padded[:, r:r+32, c:c+32])
        if not np.array_equal(targets["x"][index], expected):
            raise ValueError("Training target differs from native deployment normalization/window")
    if set(np.unique(targets["y"])) != {1, 255}:
        raise ValueError("New target pack contains accidental background labels")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    detector = LandslideDetector(candidate["path"], device=device, threshold=.4, patch_size=32)
    probabilities = detector.detect_image(raw.transpose(1, 2, 0), stride=32, patch_norm="minmax", max_batch=64)
    saved = np.load(Path(candidate["path"]).parent / "zhonghe_probability.npy")
    difference = float(np.max(np.abs(probabilities - saved)))
    if difference > 2e-6:
        raise ValueError("Reloaded candidate does not reproduce retained inference")
    baseline_state = torch.load(base["base_model"]["path"], weights_only=True, map_location="cpu")
    candidate_state = torch.load(candidate["path"], weights_only=True, map_location="cpu")
    encoder_unchanged = all(torch.equal(candidate_state[key], value) for key, value in baseline_state.items() if key.startswith("encoder."))
    bn_unchanged = all(torch.equal(candidate_state[key], value) for key, value in baseline_state.items()
                       if key.endswith(("running_mean", "running_var", "num_batches_tracked")))
    if not encoder_unchanged or not bn_unchanged:
        raise ValueError("Frozen encoder or BatchNorm buffers changed")
    for path in HERE.glob("*.py"):
        ast.parse(path.read_text(encoding="utf-8"))
    for manifest_path in [OUT / "execution_source_manifest.json", OUT / "anchored_followup/execution_source_manifest.json"]:
        for name, digest in read(manifest_path).items():
            path = DET / name if name == "detector.py" else HERE / name
            if sha(path) != digest:
                raise ValueError("Frozen experiment source changed")
    discarded = first["discarded_checkpoints"] + follow["discarded_checkpoints"]
    if any(Path(record["path"]).exists() for record in discarded):
        raise ValueError("Discarded checkpoint unexpectedly remains")
    # Only generated code caches in this batch and the reused phase-7 reader.
    cleaned_files = cleaned_bytes = 0
    for allowed_root in [HERE.resolve(), (GEO / "maintenance/phase7_20261015/vendor").resolve()]:
        for path in sorted(allowed_root.rglob("__pycache__"), reverse=True):
            target = path.resolve()
            if not target.is_relative_to(allowed_root) or target.name != "__pycache__":
                raise ValueError("Unexpected cleanup target")
            files = [item for item in target.rglob("*") if item.is_file()]
            cleaned_files += len(files)
            cleaned_bytes += sum(item.stat().st_size for item in files)
            shutil.rmtree(target)
    validate_inputs(base)
    receipt = {
        "verified_at": now(), "retained_candidate": candidate,
        "native_reload_prediction_max_abs_difference": difference,
        "training_tiles_exactly_match_deployed_window_and_normalization": True,
        "encoder_parameters_and_buffers_unchanged": encoder_unchanged,
        "batchnorm_buffers_unchanged": bn_unchanged,
        "new_partial_pack_has_no_negative_labels": True,
        "scientific_invariant_tests_passed": 5,
        "default_model_and_old_protected_registries_unchanged": True,
        "independent_blind_tests_opened": 0, "cloud_generation_calls": 0,
        "fits_completed": 8, "optimizer_updates_total": 4 * 80 + 4 * 200,
        "failed_or_control_checkpoints_deleted": len(discarded),
        "checkpoint_bytes_deleted": sum(item["bytes"] for item in discarded),
        "bytecode_cache_files_deleted": cleaned_files, "bytecode_cache_bytes_deleted": cleaned_bytes,
        "runtime": {"torch": torch.__version__, "numpy": np.__version__, "rasterio": rasterio.__version__,
                    "device": device, "gpu": torch.cuda.get_device_name(0) if device == "cuda" else None},
    }
    write(OUT / "candidate_verification.json", receipt)
    write(OUT / "offline_candidate.json", {
        "artifact": candidate, "architecture": "UnetPlusPlus/resnet34, 4 BGRN channels",
        "input": base["input_protocol"], "patch_size": 32, "threshold": .4,
        "status": "offline_development_candidate_only", "deployment_promotion_allowed": False,
        "training_event_group": "TW_HUALIEN20240403", "new_event_independent_test_eligible": False,
        "known_mine_reference_objects": 1, "mine_supervised_core_pixels": 5,
        "surrounding_auxiliary_core_pixels": 95,
        "source_negative_labels": "existing LMHLD reference background, not mine operating-safety truth",
        "limitations": ["Core scores are training fit; not new accident-object recall or independent generalization",
                        "No complete accident boundary or verified normal-operation false-positive rate",
                        "No physical reflectance indices were used in this adaptation"],
        "frozen_protocol": str(OUT / "anchored_followup/protocol.json"),
        "verification": str(OUT / "candidate_verification.json"),
    })
    state = read(HERE / "state.json")
    state.update(status="completed_positive_only_adaptation_pilot_candidate_verified",
                 completed_at=receipt["verified_at"], new_training_completed=8,
                 training_budget={"first_pilot_fits": 4, "first_updates_each": 80,
                                  "followup_fits": 4, "followup_updates_each": 200,
                                  "total_optimizer_updates": 1120},
                 new_event_role="development;all TW_HUALIEN20240403 cases cannot be claimed as untouched blind data",
                 candidate_verification=str(OUT / "candidate_verification.json"),
                 default_weight_changed=False, new_cloud_generation_calls=0,
                 independent_blind_acceptance_completed=False,
                 native_development_prediction_runs_including_reload_check=10,
                 cleanup={"checkpoint_files_deleted": len(discarded), "checkpoint_bytes_deleted": receipt["checkpoint_bytes_deleted"],
                          "retained_candidate_weight_files": 1, "code_cache_files_deleted": cleaned_files},
                 next_priority="Extend independent event/mine references and normal-operation evidence before cross-mine or deployment claims; no threshold/LR sweep on this one trained event")
    state.pop("updates_per_fit", None)
    write(HERE / "state.json", state)
    old_state_path = GEO / "maintenance/phase7_20261015/state.json"
    old_state = read(old_state_path)
    old_state["phase8_handoff"] = {"state": str(HERE / "state.json"), "event_group": "TW_HUALIEN20240403",
                                   "new_use_role": "development", "fits_completed": 8,
                                   "original_qualification_snapshot_not_rewritten": True,
                                   "independent_blind_reuse_allowed": False}
    write(old_state_path, old_state)
    print(json.dumps(receipt, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")
    main()
