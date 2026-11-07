"""Reproducible, source-calibrated leave-one-region-out scale experiment.

Offline research only. No registered model, service, official test file, mine
image, provider API or original dataset is modified by this runner.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone, timedelta
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from runtime_paths import configure_runtime_paths
configure_runtime_paths()

import cv2
import numpy as np

REGIONS = [
    ("Emilia-Romagna_Italy_32", 10.0),
    ("Gorkha_Nepal_32", 10.0),
    ("Hokkaido_Japan_128", 3.0),
    ("Jiuzhaigou_China_128", 1.0),
    ("Wenchuan_China_224", 0.8),
    ("Taiwan_China_224", 3.0),
]
SIDE = 64
METHODS = ("full", "tta5", "multiscale5")


def now():
    return datetime.now(timezone(timedelta(hours=8))).isoformat(timespec="seconds")


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def array_sha(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def write(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                    allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def freeze_protocol(out):
    """Register hypotheses/budgets before inspecting any new model outputs."""
    out = Path(out).resolve()
    if out.drive.upper() != "D:" or not out.is_relative_to(ROOT / "experiments"):
        raise ValueError("Experiment output must stay inside D:/LMHLD/experiments")
    out.mkdir(parents=True, exist_ok=True)
    path = out / "protocol.json"
    if path.exists():
        protocol = read(path)
        if protocol["runner_sha256"] != sha(__file__):
            raise ValueError("Frozen runner changed: use a new experiment directory")
        for asset in protocol["protected_assets"]:
            if sha(asset["path"]) != asset["sha256"]:
                raise ValueError("Protected default/registry changed")
        return protocol
    protected = [ROOT / "model_registry.json",
                 ROOT / "models_prior/real_ft_T1b_ours_seed42/best_model.pth",
                 ROOT / "datasets/input_reserve_registry.json",
                 ROOT / "datasets/mine_benchmark_20261002/mine_case_registry.json"]
    protocol = {
        "registered_at": now(), "runner_sha256": sha(__file__),
        "question": "Does scale variation improve small mapped-component detection on a held-out region without increased annotated-background false positives?",
        "stage": "bounded mechanistic pilot, not a new blind acceptance or production replacement",
        "regions": [{"name": name, "nominal_gsd_m": gsd} for name, gsd in REGIONS],
        "gsd_source": "Liu et al. IEEE TGRS 2025 DOI 10.1109/TGRS.2025.3619062 Table II; nominal image resolution, no per-patch geotransform",
        "folds": [{"held_out": h, "sources": [r for r, _ in REGIONS if r != h]}
                  for h, _ in REGIONS],
        "initialization": "random from scratch, encoder_weights=None; never load existing LMHLD/mine-trained weights",
        "architecture": "segmentation_models_pytorch UnetPlusPlus/resnet34, four actual BGRN channels, one output, identical in both arms",
        "training_arms": ["fixed", "scale_jitter"], "seeds": [42, 43],
        "updates": 1200, "batch": 20, "examples_per_source_per_step": 4,
        "train_parent_cap": 512, "val_parent_cap": 256,
        "val_parent_caps": {"Gorkha_Nepal_32": 794},
        "data_only_amendment": "Gorkha sampled validation has only 8 interior <=2500m2 components; use all 794 original val parents (26 components). Decision based only on label counts, before inspecting scores. One pilot fit completed before termination; preserved separately and excluded from this protocol. sampling_amendment.json records provenance.",
        "selection_seed": 20261019, "network_side": SIDE,
        "optimizer": {"name": "AdamW", "lr": .001, "weight_decay": .0001},
        "schedule": "cosine 0.001 to 0.00001; fixed final update, no checkpoint selection",
        "loss": "BCE with fractional foreground occupancy + 0.5 per-image soft Dice; no pseudo NIR or prior",
        "normalization": "once per original parent, per-band minmax [0,1], then crop/resize; constant bands zero",
        "training_pairing": "identical initialization, source parent schedule, steps, tensor sizes and paired flips/rotations; only crop-view choice differs",
        "fixed_arm": "entire original parent resampled to 64x64",
        "scale_jitter_arm": "50% whole parent, 50% one uniformly chosen native quadrant resampled to 64x64",
        "label_resampling": "INTER_AREA for shrink, INTER_LINEAR for enlargement; fractional occupancy retained, no hard threshold in training",
        "inference_methods": {
            "full": "one whole-parent forward",
            "tta5": "five whole-parent forwards: identity, rotations 90/180/270, horizontal flip; inverse transform then probability mean",
            "multiscale5": "one whole-parent plus four native quadrants, all 64x64; resize probabilities to original label grid; 50:50 whole/local mean at each pixel",
        },
        "compute_control": "tta5 and multiscale5 have exactly five same-shaped network forwards per parent; training updates/examples/shapes identical; record timings",
        "calibration": "for each fold/seed/arm/method, lowest threshold on 0.001 grid with equal-region source-val mean annotated-background FPR <= 0.02; held-out labels and predictions excluded",
        "secondary_threshold": .5,
        "primary_comparison": "scale_jitter_multiscale5 vs fixed_tta5",
        "ablation_comparisons": ["fixed_multiscale5 vs fixed_tta5", "scale_jitter_full vs fixed_full", "scale_jitter_multiscale5 vs scale_jitter_tta5"],
        "primary_small_definition": "8-connected GT components <=2500 m2 using nominal GSD, exclude components touching original parent border; primary hit requires >=50% native GT pixels recovered",
        "secondary_small_definition": "<=25 native pixels, same border exclusion; do not equate physical sizes across sensors",
        "other_metrics": "native-grid pixel precision/recall/F1/IoU/FPR; any-overlap small hit; one-to-one component IoU>=0.5 small hit; unmatched predicted components >=2px with zero GT overlap per megapixel",
        "stability_gate": {
            "mean_small_cover50_gain_each_seed": .02,
            "regions_with_positive_gain_in_both_seeds": 4,
            "minimum_small_components_each_region": 10,
            "macro_fpr_increase_each_seed_max": .002,
            "any_region_fpr_increase_each_seed_max": .01,
            "macro_f1_change_each_seed_min": -.005,
            "candidate_macro_fpr_each_seed_max": .03,
            "candidate_any_region_fpr_each_seed_max": .05,
        },
        "leakage_audit": "hash every selected raw train/val parent; remove train parents identical to any selected validation image or shared by multiple regions; no tile reshuffling",
        "validation_history": "historically exposed original validation partitions; region held out from each NEW fit/calibration, not globally untouched test data",
        "limitations": [
            "Original parent tiles lack location/event object IDs: near-duplicate or adjacency cannot be fully audited; exact input duplicates are audited.",
            "Mapped connected components are not guaranteed unique complete landslide instances; border components excluded only from object metrics.",
            "Scale/region/sensor covary; this pilot tests relative image scale, not a certified physical-resolution-adaptive algorithm.",
            "Fixed small training budget from scratch is not an architecture ceiling or a full converged benchmark.",
            "Dataset-labeled background is not a verified normal mining operation or field safety reference.",
            "No mine recall claim, no accident prediction or pre-event warning validation.",
        ],
        "official_test_files_opened": 0, "new_mine_predictions": 0,
        "external_generation_calls": 0, "downloads": 0,
        "production_change_allowed": False,
        "retention": "no fit checkpoints retained; retain exact row manifest, results, source code and runtime receipts for replay",
        "protected_assets": [{"path": str(p), "sha256": sha(p)} for p in protected],
        "paper_source": "https://doi.org/10.1109/TGRS.2025.3619062",
        "dataset_source": "https://doi.org/10.5281/zenodo.11424987",
        "novelty_claim": "standard multi-scale augmentation/inference controls; hypothesis test only, no original-method claim",
    }
    write(path, protocol)
    return protocol


def make_views(image, label=None):
    image = np.asarray(image, np.float32)
    if image.shape[0] != 4 or not np.isfinite(image).all():
        raise ValueError("Expected finite actual four-band CHW parent")
    lo = image.min((1, 2), keepdims=True)
    span = image.max((1, 2), keepdims=True) - lo
    image = np.divide(image - lo, span, out=np.zeros_like(image), where=span > 0)
    p = image.shape[-1]
    if image.shape[-2] != p or p % 2:
        raise ValueError("Even square native parents required")
    boxes = [(0, 0, p, p)] + [(x, y, x + p//2, y + p//2)
                            for y in (0, p//2) for x in (0, p//2)]
    xs, ys = [], []
    for x0, y0, x1, y1 in boxes:
        interpolation = cv2.INTER_AREA if x1-x0 > SIDE else cv2.INTER_LINEAR
        x = cv2.resize(image[:, y0:y1, x0:x1].transpose(1, 2, 0),
                       (SIDE, SIDE), interpolation=interpolation).transpose(2, 0, 1)
        xs.append(x)
        if label is not None:
            y = cv2.resize(np.asarray(label, np.float32).reshape(p, p)[y0:y1, x0:x1],
                           (SIDE, SIDE), interpolation=interpolation)
            ys.append(y[None])
    return np.asarray(xs, np.float16), (np.asarray(ys, np.float16) if label is not None else None)


def prepare_data(out, protocol):
    from data_paths import resolve_data_root
    root = Path(resolve_data_root("native"))
    records, val_data, train_entries, hashes = [], {}, [], {}
    for g, (region, gsd) in enumerate(REGIONS):
        folder = root / region
        record = {"region": region, "gsd_m": gsd, "splits": {}}
        for s, split in enumerate(("train", "val")):
            xp, yp = folder / f"{split}_images.npy", folder / f"{split}_labels.npy"
            x, y = np.load(xp, mmap_mode="r"), np.load(yp, mmap_mode="r")
            if x.shape[1] != 4 or y.shape != (len(x), 1, x.shape[2], x.shape[3]):
                raise ValueError("Source image/label layout mismatch")
            cap = protocol[f"{split}_parent_cap"]
            if split == "val": cap = protocol.get("val_parent_caps", {}).get(region, cap)
            rng = np.random.default_rng(protocol["selection_seed"] + 101*g + s)
            ids = np.sort(rng.choice(len(x), min(cap, len(x)), replace=False))
            row_hashes, label_hashes, views, labels = [], [], [], []
            for idx in ids:
                raw, gt = np.asarray(x[idx]), np.asarray(y[idx])
                if not np.isfinite(gt).all() or not set(np.unique(gt)).issubset({0., 1.}):
                    raise ValueError("Ground truth must be finite binary")
                digest = array_sha(raw)
                row_hashes.append(digest); label_hashes.append(array_sha(gt))
                hashes.setdefault(digest, []).append((g, split, int(idx)))
                if split == "val":
                    views.append(make_views(raw)[0]); labels.append(gt[0].astype(np.uint8))
                else:
                    train_entries.append((g, int(idx), digest))
            record["splits"][split] = {
                "image_path": str(xp), "label_path": str(yp),
                "image_shape": list(x.shape), "indices": ids.tolist(),
                "raw_parent_sha256": row_hashes, "label_parent_sha256": label_hashes,
            }
            if split == "val":
                val_data[g] = {"views": np.stack(views), "labels": labels,
                               "indices": ids.tolist(), "gsd": gsd,
                               "components": [gt_components(y, gsd) for y in labels]}
        records.append(record)
        print("AUDIT", region, "selected train", len(record["splits"]["train"]["indices"]),
              "val", len(val_data[g]["labels"]), flush=True)
    blocked = {h for h, rows in hashes.items()
               if any(s == "val" for _, s, _ in rows) or len({g for g, _, _ in rows}) > 1}
    xs, ys, groups, kept, dropped = [], [], [], [], []
    pools = {}
    for g, idx, digest in train_entries:
        if digest in blocked:
            dropped.append({"region": REGIONS[g][0], "index": idx, "sha256": digest})
            continue
        if g not in pools:
            folder = root / REGIONS[g][0]
            pools[g] = (np.load(folder / "train_images.npy", mmap_mode="r"),
                        np.load(folder / "train_labels.npy", mmap_mode="r"))
        x, y = pools[g]
        xx, yy = make_views(x[idx], y[idx])
        xs.append(xx); ys.append(yy); groups.append(g); kept.append((g, idx, digest))
    manifest = {"registered_at": now(), "regions": records,
                "dropped_train_parents": dropped,
                "kept_train": [{"region": REGIONS[g][0], "index": idx, "sha256": h}
                               for g, idx, h in kept],
                "selected_exact_duplicate_groups": [rows for rows in hashes.values() if len(rows) > 1],
                "test_files_opened": 0,
                "no_tiled_dataset_used": True,
                "regional_train_counts": dict(Counter(REGIONS[g][0] for g in groups)),
                "val_small_interior_components": {REGIONS[g][0]: sum(int(c["small"].sum()) for c in d["components"])
                                                   for g, d in val_data.items()}}
    path = Path(out) / "data_manifest.json"
    if path.exists():
        old = read(path)
        # Timestamp is the only replay-dependent field.
        if {k: v for k, v in old.items() if k != "registered_at"} != {
                k: v for k, v in manifest.items() if k != "registered_at"}:
            raise ValueError("Frozen source rows or pixels changed")
    else:
        write(path, manifest)
    return np.stack(xs), np.stack(ys), np.asarray(groups), val_data, manifest


def gt_components(gt, gsd):
    n, lab, stats, _ = cv2.connectedComponentsWithStats(np.asarray(gt, np.uint8), 8)
    areas = stats[:, cv2.CC_STAT_AREA].astype(np.int64)
    border_ids = np.unique(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]]))
    interior = np.ones(n, bool); interior[0] = False; interior[border_ids] = False
    return {"labels": lab, "areas": areas,
            "small": interior & (areas * gsd * gsd <= 2500.0 + 1e-8),
            "small25": interior & (areas <= 25), "interior": interior}


def sample_batch(groups, held, rng, per_source=4):
    groups = np.asarray(groups)
    ids = np.concatenate([rng.choice(np.flatnonzero(groups == g), per_source, replace=True)
                          for g in range(len(REGIONS)) if g != held])
    rng.shuffle(ids)
    if np.any(groups[ids] == held):
        raise AssertionError("Held-out region entered gradient batch")
    return ids


def inverse_tta(tensor, kind):
    if kind in (1, 2, 3):
        return tensor.rot90(-kind, (-2, -1))
    if kind == 4:
        return tensor.flip(-1)
    return tensor


def train_model(x, y, groups, held, seed, arm, protocol):
    import torch
    import torch.nn.functional as F
    import segmentation_models_pytorch as smp
    fold_seed = seed + held * 100
    torch.manual_seed(fold_seed); torch.cuda.manual_seed_all(fold_seed)
    model = smp.UnetPlusPlus(encoder_name="resnet34", encoder_weights=None,
                            in_channels=4, classes=1).cuda()
    init_hash = hashlib.sha256()
    for value in model.state_dict().values():
        init_hash.update(value.detach().cpu().numpy().tobytes())
    opt = torch.optim.AdamW(model.parameters(), **{
        k: v for k, v in protocol["optimizer"].items() if k != "name"})
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, protocol["updates"], eta_min=.00001)
    rng = np.random.default_rng(fold_seed)
    losses, visits = [], np.zeros(6, np.int64)
    schedule_hash = hashlib.sha256()
    torch.cuda.synchronize(); started = time.perf_counter()
    model.train()
    for step in range(protocol["updates"]):
        ids = sample_batch(groups, held, rng, protocol["examples_per_source_per_step"])
        proposed_views = np.where(rng.random(len(ids)) < .5, 0, rng.integers(1, 5, len(ids)))
        views = proposed_views if arm == "scale_jitter" else np.zeros(len(ids), np.int64)
        k, flip = int(rng.integers(0, 4)), bool(rng.integers(0, 2))
        schedule_hash.update(np.asarray([*ids, k, int(flip)], np.int64).tobytes())
        visits += np.bincount(groups[ids], minlength=6)
        a = x[ids, views].float().rot90(k, (-2, -1))
        b = y[ids, views].float().rot90(k, (-2, -1))
        if flip: a, b = a.flip(-1), b.flip(-1)
        opt.zero_grad(set_to_none=True)
        logits = model(a)
        p = logits.sigmoid()
        dice = (1 - (2 * (p*b).sum((2, 3)) + 1e-6) /
                (p.sum((2, 3)) + b.sum((2, 3)) + 1e-6)).mean()
        loss = F.binary_cross_entropy_with_logits(logits, b) + .5*dice
        if not torch.isfinite(loss):
            raise ValueError("Non-finite training loss; do not interpret failed training")
        loss.backward(); opt.step(); scheduler.step()
        losses.append(float(loss.detach()))
        if (step + 1) % 300 == 0:
            print("TRAIN", REGIONS[held][0], seed, arm, step+1,
                  "loss100", round(float(np.mean(losses[-100:])), 5), flush=True)
    torch.cuda.synchronize()
    if visits[held] != 0:
        raise AssertionError("Target region used during training")
    return model.eval(), {
        "seconds": time.perf_counter()-started, "initial_state_sha256": init_hash.hexdigest(),
        "parent_schedule_sha256": schedule_hash.hexdigest(), "visits_per_region": visits.tolist(),
        "mean_loss_first100": float(np.mean(losses[:100])),
        "mean_loss_last100": float(np.mean(losses[-100:])),
        "loss_every100": [float(np.mean(losses[i:i+100])) for i in range(0, len(losses), 100)],
        "final_lr": scheduler.get_last_lr()[0], "updates": len(losses),
    }


def predict_views(model, data, batch=96):
    import torch
    views = data["views"]
    def run(array, kind=0):
        outputs = []
        with torch.inference_mode():
            for start in range(0, len(array), batch):
                a = torch.from_numpy(np.ascontiguousarray(array[start:start+batch])).cuda().float()
                if kind in (1, 2, 3): a = a.rot90(kind, (-2, -1))
                elif kind == 4: a = a.flip(-1)
                p = inverse_tta(model(a).sigmoid()[:, 0], kind)
                if not torch.isfinite(p).all(): raise ValueError("Non-finite predictions")
                outputs.append(p.cpu().numpy())
        return np.concatenate(outputs)
    torch.cuda.synchronize(); started = time.perf_counter()
    full = run(views[:, 0])
    full_seconds = time.perf_counter()-started
    started = time.perf_counter()
    tta = full.copy()
    for k in (1, 2, 3, 4): tta += run(views[:, 0], k)
    tta /= 5
    tta_seconds = time.perf_counter()-started + full_seconds
    started = time.perf_counter()
    quads = run(views[:, 1:].reshape(-1, 4, SIDE, SIDE)).reshape(-1, 4, SIDE, SIDE)
    multiscale_seconds = time.perf_counter()-started + full_seconds
    result = {k: [] for k in METHODS}
    for i, gt in enumerate(data["labels"]):
        side = len(gt); half = side//2
        fp = cv2.resize(full[i], (side, side), interpolation=cv2.INTER_LINEAR)
        tp = cv2.resize(tta[i], (side, side), interpolation=cv2.INTER_LINEAR)
        local = np.empty_like(fp)
        for q in range(4):
            y0, x0 = (q//2)*half, (q%2)*half
            local[y0:y0+half, x0:x0+half] = cv2.resize(
                quads[i, q], (half, half), interpolation=cv2.INTER_LINEAR)
        result["full"].append(fp)
        result["tta5"].append(tp)
        result["multiscale5"].append(.5*(fp+local))
    return result, {"full": full_seconds, "tta5": tta_seconds,
                    "multiscale5": multiscale_seconds,
                    "parents": len(full), "forwards_per_parent": [1, 5, 5]}


def source_threshold(probabilities, val_data, source_groups, held, budget=.02):
    if held in source_groups or set(source_groups) != set(range(6)) - {held}:
        raise ValueError("Calibration must use exactly five source regions")
    fprs = []
    for g in source_groups:
        histogram = np.zeros(1001, np.int64)
        for p, gt in zip(probabilities[g], val_data[g]["labels"]):
            # Float64 avoids rounding a just-below-grid probability up to the
            # next histogram bin; use the same comparison in parent_counts.
            bins = np.minimum((p[gt == 0].astype(np.float64) * 1000).astype(np.int32), 1000)
            histogram += np.bincount(bins, minlength=1001)
        if histogram.sum() == 0: raise ValueError("Source calibration has no background")
        tail = np.cumsum(histogram[::-1])[::-1] / histogram.sum()
        fprs.append(np.concatenate([tail, [0.]]))
    macro = np.mean(fprs, axis=0)
    index = int(np.flatnonzero(macro <= budget)[0])
    threshold = index / 1000
    return threshold, {"source_regions": [REGIONS[g][0] for g in source_groups],
                       "source_macro_fpr": float(macro[index]),
                       "source_region_fprs": [float(v[index]) for v in fprs],
                       "held_out_in_calibration": False}


def parent_counts(prob, gt, components, threshold):
    pred = np.asarray(prob, np.float64) >= threshold
    truth = gt.astype(bool)
    tp = int((pred & truth).sum()); fp = int((pred & ~truth).sum())
    fn = int((~pred & truth).sum()); tn = int((~pred & ~truth).sum())
    lab, areas = components["labels"], components["areas"]
    hits = np.bincount(lab[pred], minlength=len(areas))
    small, small25 = components["small"], components["small25"]
    covered50 = hits >= .5*areas
    n_pred, plab, pstats, _ = cv2.connectedComponentsWithStats(pred.astype(np.uint8), 8)
    pa = pstats[:, cv2.CC_STAT_AREA]
    overlap = np.bincount(plab[truth], minlength=n_pred)
    false_components = int(((pa >= 2) & (overlap == 0) & (np.arange(n_pred) > 0)).sum())
    # Greedy one-to-one matches in descending IoU; includes all GT in matching,
    # then counts the predefined interior small subset. No size filtering of predictions.
    keys, nums = np.unique(lab.astype(np.int64)*n_pred + plab, return_counts=True)
    gi, pi = keys//n_pred, keys % n_pred
    valid = (gi > 0) & (pi > 0)
    gi, pi, nums = gi[valid], pi[valid], nums[valid]
    ious = nums / np.maximum(areas[gi] + pa[pi] - nums, 1)
    used_g, used_p, matched = set(), set(), np.zeros(len(areas), bool)
    for j in np.argsort(-ious):
        if ious[j] < .5: break
        a, b = int(gi[j]), int(pi[j])
        if a not in used_g and b not in used_p:
            used_g.add(a); used_p.add(b); matched[a] = True
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "small_n": int(small.sum()), "small_cover50": int((small & covered50).sum()),
            "small_any": int((small & (hits > 0)).sum()),
            "small_iou50": int((small & matched).sum()),
            "small25_n": int(small25.sum()),
            "small25_cover50": int((small25 & covered50).sum()),
            "false_components_ge2px": false_components, "pixels": int(gt.size)}


def aggregate_counts(rows):
    total = {k: sum(r[k] for r in rows) for k in rows[0]}
    tp, fp, fn, tn = (total[k] for k in ("tp", "fp", "fn", "tn"))
    result = {"counts": total, "precision": tp/max(tp+fp, 1), "recall": tp/max(tp+fn, 1),
              "f1": 2*tp/max(2*tp+fp+fn, 1), "iou": tp/max(tp+fp+fn, 1),
              "background_fpr": fp/max(fp+tn, 1),
              "false_components_per_mpix": total["false_components_ge2px"]*1e6/total["pixels"]}
    for key in ("small_cover50", "small_any", "small_iou50"):
        result[key] = total[key]/total["small_n"] if total["small_n"] else None
    result["small25_cover50"] = total["small25_cover50"]/total["small25_n"] if total["small25_n"] else None
    return result


def evaluate_fit(model, val_data, held):
    all_probs, timings = {}, {}
    for g, data in val_data.items():
        all_probs[g], timings[REGIONS[g][0]] = predict_views(model, data)
    results = {}
    sources = [g for g in range(6) if g != held]
    for method in METHODS:
        probs = {g: v[method] for g, v in all_probs.items()}
        threshold, calibration = source_threshold(probs, val_data, sources, held)
        data = val_data[held]
        rows = [parent_counts(p, gt, c, threshold) for p, gt, c in
                zip(probs[held], data["labels"], data["components"])]
        fixed_rows = [parent_counts(p, gt, c, .5) for p, gt, c in
                      zip(probs[held], data["labels"], data["components"])]
        results[method] = {"threshold": threshold, "calibration": calibration,
                           "held_out": aggregate_counts(rows),
                           "held_out_at_fixed_05": aggregate_counts(fixed_rows),
                           "per_parent_counts": rows}
    return results, timings


def summarize(out, protocol):
    rows = [read(p) for p in sorted(Path(out).glob("fold_*/seed*/result_*.json"))]
    expected = 6*len(protocol["seeds"])*len(protocol["training_arms"])
    lookup = {(r["held_index"], r["seed"], r["arm"]): r for r in rows}
    if len(lookup) != expected:
        raise ValueError(f"Incomplete fits: {len(lookup)}/{expected}")
    comparisons = {
        "primary_joint_vs_compute_control": (("scale_jitter", "multiscale5"), ("fixed", "tta5")),
        "inference_only": (("fixed", "multiscale5"), ("fixed", "tta5")),
        "training_only": (("scale_jitter", "full"), ("fixed", "full")),
        "inference_with_jitter": (("scale_jitter", "multiscale5"), ("scale_jitter", "tta5")),
    }
    output = {}
    metrics = ("small_cover50", "small_any", "small_iou50", "small25_cover50", "f1", "background_fpr")
    for name, (candidate, base) in comparisons.items():
        by_seed, regional = [], []
        for seed in protocol["seeds"]:
            candidate_values, base_values, diffs = [], [], []
            for g, (region, _) in enumerate(REGIONS):
                a = lookup[g, seed, candidate[0]]["methods"][candidate[1]]["held_out"]
                b = lookup[g, seed, base[0]]["methods"][base[1]]["held_out"]
                delta = {k: a[k]-b[k] if a[k] is not None and b[k] is not None else None for k in metrics}
                regional.append({"region": region, "seed": seed,
                                 "candidate": {k: a[k] for k in metrics},
                                 "baseline": {k: b[k] for k in metrics}, "delta": delta,
                                 "small_n": a["counts"]["small_n"]})
                candidate_values.append(a); base_values.append(b); diffs.append(delta)
            by_seed.append({"seed": seed,
                "candidate_macro": {k: float(np.mean([v[k] for v in candidate_values if v[k] is not None])) for k in metrics},
                "baseline_macro": {k: float(np.mean([v[k] for v in base_values if v[k] is not None])) for k in metrics},
                "macro_delta": {k: float(np.mean([v[k] for v in diffs if v[k] is not None])) for k in metrics},
                "candidate_worst_region_fpr": max(v["background_fpr"] for v in candidate_values),
                "worst_region_fpr_increase": max(v["background_fpr"] for v in diffs),
                "positive_small_regions": sum(v["small_cover50"] is not None and v["small_cover50"] > 0 for v in diffs)})
        output[name] = {"candidate": candidate, "baseline": base, "by_seed": by_seed, "regions": regional}
    primary = output["primary_joint_vs_compute_control"]
    both_positive = sum(all(next(r for r in primary["regions"] if r["region"] == region and r["seed"] == seed)["delta"]["small_cover50"] is not None and
                            next(r for r in primary["regions"] if r["region"] == region and r["seed"] == seed)["delta"]["small_cover50"] > 0
                            for seed in protocol["seeds"]) for region, _ in REGIONS)
    gate = protocol["stability_gate"]
    checks = {
        "small_gain_each_seed": all(r["macro_delta"]["small_cover50"] >= gate["mean_small_cover50_gain_each_seed"] for r in primary["by_seed"]),
        "at_least_four_regions_positive_both_seeds": both_positive >= gate["regions_with_positive_gain_in_both_seeds"],
        "adequate_small_objects": all(r["small_n"] >= gate["minimum_small_components_each_region"] for r in primary["regions"]),
        "macro_fpr_increase_bounded": all(r["macro_delta"]["background_fpr"] <= gate["macro_fpr_increase_each_seed_max"] for r in primary["by_seed"]),
        "regional_fpr_increase_bounded": all(r["worst_region_fpr_increase"] <= gate["any_region_fpr_increase_each_seed_max"] for r in primary["by_seed"]),
        "macro_f1_preserved": all(r["macro_delta"]["f1"] >= gate["macro_f1_change_each_seed_min"] for r in primary["by_seed"]),
        "candidate_macro_fpr_bounded": all(r["candidate_macro"]["background_fpr"] <= gate["candidate_macro_fpr_each_seed_max"] for r in primary["by_seed"]),
        "candidate_worst_fpr_bounded": all(r["candidate_worst_region_fpr"] <= gate["candidate_any_region_fpr_each_seed_max"] for r in primary["by_seed"]),
    }
    paired_ok = all(lookup[g, s, "fixed"]["training"][k] == lookup[g, s, "scale_jitter"]["training"][k]
                    for g in range(6) for s in protocol["seeds"]
                    for k in ("initial_state_sha256", "parent_schedule_sha256", "visits_per_region"))
    if not paired_ok: raise ValueError("Training pairing failed")
    protected_ok = all(sha(p["path"]) == p["sha256"] for p in protocol["protected_assets"])
    if not protected_ok: raise ValueError("Protected asset changed")
    result = {"completed_at": now(), "status": "completed", "fits": len(rows),
              "optimizer_updates": sum(r["training"]["updates"] for r in rows),
              "total_fit_seconds": sum(r["training"]["seconds"] for r in rows),
              "comparisons": output, "stability_checks": checks,
              "stability_gate_passed": all(checks.values()), "regions_positive_both_seeds": both_positive,
              "initialization_and_schedule_pairing_verified": paired_ok,
              "protected_assets_unchanged": protected_ok,
              "new_official_test_predictions": 0, "new_mine_predictions": 0,
              "external_calls": 0, "production_changed": False,
              "warning": "Historically exposed validation. LOO refers to NEW gradient/calibration exclusion, not untouched independent acceptance. No field or mining safety FPR claim."}
    write(Path(out) / "results.json", result)
    write(Path(out) / "state.json", {k: v for k, v in result.items() if k != "comparisons"})
    return result


def run(out):
    import torch
    protocol = freeze_protocol(out)
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if not torch.cuda.is_available(): raise RuntimeError("CUDA unavailable")
    x, y, groups, val_data, manifest = prepare_data(out, protocol)
    x, y = torch.from_numpy(x).cuda(), torch.from_numpy(y).cuda()
    write(Path(out) / "runtime.json", {"torch": torch.__version__, "cuda": torch.version.cuda,
          "device": torch.cuda.get_device_name(0), "threads": torch.get_num_threads(),
          "deterministic_cudnn": True, "tf32": False, "cache_drive": "D:",
          "train_view_bytes": x.numel()*x.element_size()+y.numel()*y.element_size()})
    completed = 0
    for held, (region, _) in enumerate(REGIONS):
        for seed in protocol["seeds"]:
            for arm in protocol["training_arms"]:
                directory = Path(out) / f"fold_{held}_{region}" / f"seed{seed}"
                directory.mkdir(parents=True, exist_ok=True)
                destination = directory / f"result_{arm}.json"
                if destination.exists():
                    completed += 1
                    continue
                write(Path(out) / "state.json", {"status": "running", "held_out": region,
                      "seed": seed, "arm": arm, "fits_completed": completed,
                      "total_fits": 24, "production_changed": False, "updated_at": now()})
                model, training = train_model(x, y, groups, held, seed, arm, protocol)
                methods, timings = evaluate_fit(model, val_data, held)
                write(destination, {"held_out": region, "held_index": held, "seed": seed,
                      "arm": arm, "completed_at": now(), "training": training,
                      "methods": methods, "prediction_timings": timings,
                      "model_weights_saved": False})
                del model
                torch.cuda.empty_cache()
                completed += 1
                print("FIT_COMPLETED", completed, "/24", region, seed, arm, flush=True)
    result = summarize(out, protocol)
    print("COMPLETE", json.dumps({"fits": result["fits"],
          "stability_gate_passed": result["stability_gate_passed"],
          "checks": result["stability_checks"]}), flush=True)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--stage", choices=("register", "run", "summarize"), default="run")
    args = parser.parse_args()
    if args.stage == "register": freeze_protocol(args.output)
    elif args.stage == "summarize": summarize(args.output, freeze_protocol(args.output))
    else: run(args.output)
