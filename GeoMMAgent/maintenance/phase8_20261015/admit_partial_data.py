"""Scope the new positive-only reference for a bounded development pilot.

This is a new use decision, not a rewrite of the phase-7 source qualification.
No complete accident masks, safe mine negatives or blind-test claims are made.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import sys
from pathlib import Path
from datetime import datetime, timezone, timedelta

DET = Path(r"D:\LMHLD")
GEO = Path(r"D:\GeoMMAgent")
HERE = Path(__file__).resolve().parent
OLD = DET / "datasets/public_reference_screen_20261015"
OUT = DET / "experiments/partial_reference_pilot_20261015"
sys.path.insert(0, str(DET))
from runtime_paths import configure_runtime_paths
configure_runtime_paths()
import numpy as np
import rasterio
from rasterio.features import rasterize
from rasterio.warp import transform_geom
from data_paths import resolve_data_root


def now():
    return datetime.now(timezone(timedelta(hours=8))).isoformat(timespec="seconds")


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalize(x):
    """Exactly the existing service's per-channel patch minmax protocol."""
    x = np.asarray(x, dtype=np.float32)
    mn, mx = x.min(axis=(-2, -1), keepdims=True), x.max(axis=(-2, -1), keepdims=True)
    return (x - mn) / (mx - mn + 1e-6) * 255.0


def check_development_extent(bounds):
    protected = read(DET / "datasets/input_reserve_registry.json")["cases"]
    canonical = read(DET / "datasets/mine_benchmark_20261002/mine_case_registry.json")["cases"]
    for name, case in {**canonical, **protected}.items():
        if case.get("role") != "reserved_final_test":
            continue
        other = case.get("protected_geographic_extent_wgs84")
        if other and min(bounds[2], other[2]) > max(bounds[0], other[0]) and min(bounds[3], other[3]) > max(bounds[1], other[1]):
            raise ValueError(f"New development window overlaps reserved geography: {name}")


def balanced_indices(labels, per_class, seed):
    rng = np.random.default_rng(seed)
    positive = np.asarray([bool(np.any(row == 1)) for row in labels])
    selected = []
    for flag in [False, True]:
        pool = np.flatnonzero(positive == flag)
        if len(pool) < per_class:
            raise ValueError("Original source split has insufficient class support")
        selected.extend(rng.choice(pool, per_class, replace=False).tolist())
    rng.shuffle(selected)
    return selected


def prepare():
    OUT.mkdir(parents=True, exist_ok=True)
    protocol_path = OUT / "protocol.json"
    if protocol_path.exists():
        raise ValueError("Protocol already frozen; use its existing pack rather than reselecting data")
    # Source-only labels: 255 always means unknown in this new partial reference.
    reference_path = OLD / "references/TW_ZHONGHE2024_PARTIAL/qualification.json"
    reference = read(reference_path)
    if reference["complete_mine_accident_extent_verified"] or reference["core_mask"]["negative_pixels"] != 0:
        raise ValueError("Unexpected partial-reference scope")
    folder = OLD / "imagery/TW_ZHONGHE2024_SOURCE_CANDIDATE"
    selection = read(folder / "acquisition_plan.json")
    check_development_extent(selection["bbox_wgs84"])
    acquisition = read(folder / "acquisition.json")
    for record in acquisition["epochs"].values():
        for name, digest in record["files_sha256"].items():
            if sha(folder / name) != digest:
                raise ValueError("Original native image/quality asset changed")
    with rasterio.open(folder / "post_provider_BGRN_native10m.tif") as src:
        image, grid, crs = src.read(), src.transform, src.crs
    known_quality = np.ones(image.shape[1:], bool)
    for epoch in ["pre", "post"]:
        with rasterio.open(folder / f"{epoch}_SCL_nearest_on10m.tif") as src:
            known_quality &= ~np.isin(src.read(1), [0, 1, 3, 7, 8, 9, 10, 11])
        with rasterio.open(folder / f"{epoch}_provider_BGRN_native10m.tif") as src:
            known_quality &= np.all(src.read() > 0, axis=0)
    # Keep the original confidence bands; do not widen the scarce mine core.
    core = reference["core_mask"]
    if sha(core["path"]) != core["sha256"]:
        raise ValueError("Positive core changed")
    with rasterio.open(core["path"]) as src:
        labels = src.read(1)
        if (src.transform, src.crs, src.shape) != (grid, crs, image.shape[1:]):
            raise ValueError("Partial core and native image differ in grid")
    if not set(np.unique(labels)).issubset({1, 255}):
        raise ValueError("Unknown partial labels must not become background")
    objects = [{"source_row": reference["source_row_index_zero_based"], "kind": "mine_object",
                "confidence_weight": 1.0, "erosion_m": 10, "mask": labels == 1}]
    sys.path.insert(0, str(GEO / "maintenance/phase7_20261015/vendor"))
    from shapely.geometry import shape, mapping
    auxiliary_path = OLD / "references/TW_ZHONGHE2024_PARTIAL/surrounding_collapse_auxiliary_geometry.geojson"
    for feature in read(auxiliary_path)["features"]:
        geom = shape(transform_geom("EPSG:4326", crs, feature["geometry"])).buffer(-20)
        mask = rasterize([(mapping(geom), 1)], out_shape=image.shape[1:], transform=grid, dtype="uint8") > 0 if not geom.is_empty else np.zeros(image.shape[1:], bool)
        mask &= known_quality
        objects.append({"source_row": feature["properties"]["source_row_index_zero_based"],
                        "kind": "surrounding_auxiliary", "confidence_weight": 0.25,
                        "erosion_m": 20, "mask": mask})
    rng = np.random.default_rng(20261015)
    tx, ty, tw, target_provenance = [], [], [], []
    minmax_errors = []
    object_rows = []
    for obj in objects:
        positions = np.argwhere(obj["mask"] & known_quality)
        object_rows.append({key: value for key, value in obj.items() if key != "mask"} | {"unique_positive_pixels": len(positions)})
        if not len(positions):
            continue
        for repeat in range(4):
            anchor = positions[int(rng.integers(len(positions)))]
            row = int(np.clip(anchor[0] - 16 + rng.integers(-6, 7), 0, image.shape[1] - 32))
            col = int(np.clip(anchor[1] - 16 + rng.integers(-6, 7), 0, image.shape[2] - 32))
            raw = image[:, row:row+32, col:col+32].astype(np.float32)
            if not np.all(raw > 0):
                raise ValueError("Selected training patch contains provider nodata")
            patch_mask = obj["mask"][row:row+32, col:col+32] & known_quality[row:row+32, col:col+32]
            if not patch_mask.any():
                raise ValueError("Positive training crop lost its source core")
            # Any per-channel affine calibration preserves minmax shape, apart
            # from the existing epsilon. No offset/clipping is applied to input.
            normalized = normalize(raw)
            alt = normalize(raw * np.float32(0.0001) - np.float32(0.1))
            minmax_errors.append(float(np.max(np.abs(normalized - alt))))
            partial = np.full((32, 32), 255, np.uint8)
            partial[patch_mask] = 1
            tx.append(normalized)
            ty.append(partial)
            tw.append(obj["confidence_weight"])
            target_provenance.append({"row": row, "col": col, "shape": [32, 32], "source_row": obj["source_row"],
                                      "kind": obj["kind"], "repeat_is_augmentation_not_independent_sample": True})
    if max(minmax_errors) > 0.05:
        raise ValueError("Unexpected calibration sensitivity in the selected minmax input branch")
    # Original LMHLD train and validation remain separate; no test file is read.
    data_root = Path(resolve_data_root())
    sx, sy, vx, vy, source_rows, val_rows, source_files = [], [], [], [], [], [], []
    for region_index, region in enumerate(["Emilia-Romagna_Italy_32", "Gorkha_Nepal_32"]):
        for split, count in [("train", 64), ("val", None)]:
            xp, yp = data_root / region / f"{split}_images.npy", data_root / region / f"{split}_labels.npy"
            x, y = np.load(xp, mmap_mode="r"), np.load(yp, mmap_mode="r")
            if set(np.unique(y)) != {0.0, 1.0} or x.shape[1:] != (4, 32, 32):
                raise ValueError("Original labelled source schema differs from the frozen plan")
            ids = balanced_indices(y, count, 20261015 + region_index * 10) if split == "train" else list(range(len(y)))
            x_selected = normalize(x[ids])
            y_selected = np.asarray(y[ids], dtype=np.float32).reshape(-1, 1, 32, 32)
            if split == "train":
                sx.append(x_selected); sy.append(y_selected)
                source_rows.extend({"region": region, "split": split, "index": index} for index in ids)
            else:
                vx.append(x_selected); vy.append(y_selected)
                val_rows.extend({"region": region, "split": split, "index": index} for index in ids)
            source_files.extend({"path": str(path), "sha256": sha(path)} for path in [xp, yp])
    pack = OUT / "training_pack.npz"
    np.savez_compressed(pack, source_x=np.concatenate(sx), source_y=np.concatenate(sy),
                        validation_x=np.concatenate(vx), validation_y=np.concatenate(vy),
                        target_x=np.stack(tx), target_y=np.stack(ty), target_weight=np.asarray(tw, np.float32))
    # Previously reused, explicitly development-only mine patches are regression
    # monitors. They are never described as a new independent cross-mine test.
    from validation.mine_data_protocol import assert_development_cases
    old_mines = ["Amyntaio", "LasCruces", "Hpakant"]
    assert_development_cases(old_mines)
    historical_monitor = []
    old_root = DET / "experiments/three_mine_context_20261003"
    for mine in old_mines:
        paths = {key: old_root / f"{mine}_size32_{key}.npy" for key in ["x", "y", "weight"]}
        y, weight = np.load(paths["y"], mmap_mode="r"), np.load(paths["weight"], mmap_mode="r")
        support = np.asarray([np.any((a == 1) & (b > 0)) for a, b in zip(y, weight)])
        positives = np.flatnonzero(support)
        others = np.flatnonzero(~support)
        ids = np.concatenate([positives[:32], others[:32]]).tolist()
        historical_monitor.append({"case_id": mine, "role": "previously_seen_development_regression", "indices": ids,
                                   "paths": {key: str(path) for key, path in paths.items()},
                                   "hashes": {key: sha(path) for key, path in paths.items()}})
    model_path = DET / "models_prior/real_ft_T1b_ours_seed42/best_model.pth"
    model_sha = sha(model_path)
    if model_sha != "0d8d8a0540419d4409951ed91e69499fac206b68303f3feec08a32f0841cebef":
        raise ValueError("Deployment baseline differs from the frozen source")
    protocol = {
        "registered_at": now(), "purpose": "new_partial_supervision_development_pilot_not_independent_acceptance",
        "new_event_group_role": {"event_group": "TW_HUALIEN20240403", "role": "development",
                                 "all_dates_objects_and_nearby_cases_stay_together": True,
                                 "future_independent_blind_use_allowed": False},
        "source_phase7_qualification": {"path": str(reference_path), "sha256": sha(reference_path),
                                       "original_unassigned_snapshot_preserved": True},
        "new_training_use_decision": "Only this derived, hash-frozen positive-only pack is admitted; no general source or final-test unlock",
        "partial_labels": {"1": "published positive core", "255": "unknown, zero loss", "0": "not supplied"},
        "objects": object_rows, "unique_mine_positive_pixels": int((labels == 1).sum()),
        "unique_auxiliary_positive_pixels": reference["auxiliary_surrounding_collapse_reference"]["joint_clear_positive_pixels_by_erosion_m"]["20"],
        "independent_event_groups": 1, "duplicates_and_augmentations_increase_independent_n": False,
        "target_crops": target_provenance,
        "source_replay_rows": source_rows, "source_validation_rows": val_rows,
        "source_files": source_files, "original_test_files_read": 0,
        "source_geography_overlap": "Known source split origins retained; no claim of geographic independence from opaque pretrained/source patches",
        "input_protocol": {"bands": ["B02", "B03", "B04", "B08"], "patch_size": 32,
                           "normalization": "exact deployed per-patch per-channel minmax to 0..255, on unaltered provider DN",
                           "new_reference_physical_indices_or_prior_loss": False,
                           "physical_reflectance_semantics_still_pending": True,
                           "affine_calibration_max_abs_normalized_difference": max(minmax_errors),
                           "no_calibrated_value_clipping_or_proxy_NIR": True},
        "training_pack": {"path": str(pack), "sha256": sha(pack), "source_patches": sum(len(a) for a in sx),
                          "validation_patches": sum(len(a) for a in vx), "augmented_target_crops": len(tx)},
        "base_model": {"path": str(model_path), "sha256": model_sha},
        "arms": ["replay_control", "replay_plus_partial"], "seeds": [42, 43], "updates_per_fit": 80,
        "batch_source": 16, "batch_target": 1,
        "optimizer": {"kind": "SGD", "lr": 0.0001, "momentum": 0.9, "weight_decay": 0.0001},
        "partial_loss_weight": 0.1, "encoder_frozen": True, "batchnorm_running_statistics_frozen": True,
        "paired_budget": "Identical source minibatches, augmentations, target forwards and optimiser update counts",
        "checkpoint_selection": "fixed last update only; no threshold or epoch search",
        "thresholds": [0.4, 0.5], "primary_threshold": 0.4,
        "historical_mine_monitors": historical_monitor,
        "evaluation_limits": ["New target reference is used in training, so its scores are training fit diagnostics",
                              "Surrounding objects are auxiliary, not additional mining accidents",
                              "Unknown-area activation is not a mine false-positive rate",
                              "Previously seen mine/source validation scores are regression monitors, not blind acceptance"],
        "deployment_promotion_allowed": False, "no_new_independent_test_opened": True,
        "protected_registries": [{"path": str(path), "sha256": sha(path)} for path in [
            DET / "datasets/input_reserve_registry.json", DET / "datasets/mine_benchmark_20261002/mine_case_registry.json"]],
    }
    write(protocol_path, protocol)
    write(HERE / "state.json", {"client_work_date": "2026-10-15", "status": "data_admitted_for_frozen_positive_only_development_pilot",
                               "protocol": str(protocol_path), "protocol_sha256": sha(protocol_path),
                               "new_training_completed": 0, "new_cloud_generation_calls": 0,
                               "independent_blind_acceptance_completed": False, "default_weight_changed": False})
    print(json.dumps({"admitted": True, "source_patches": protocol["training_pack"]["source_patches"],
                      "validation_patches": protocol["training_pack"]["validation_patches"], "target_crops": len(tx),
                      "objects": object_rows, "minmax_affine_error": max(minmax_errors), "fits_planned": 4,
                      "source_and_target_test_claims": False}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare", action="store_true")
    args = parser.parse_args()
    if args.prepare:
        prepare()
    else:
        parser.error("select --prepare")
