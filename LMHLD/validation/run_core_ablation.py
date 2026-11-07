"""Frozen-data, paired-seed comparison; external mine positives never select weights."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("TORCH_HOME", str(ROOT / "cache" / "torch"))
os.environ.setdefault("HF_HOME", str(ROOT / "cache" / "huggingface"))
os.environ["SMP_DISABLE_VERSION_CHECK"] = "1"

import argparse
import hashlib
import json
import math
import random
import sys
import time
from datetime import datetime

sys.path.insert(0, str(ROOT))
import numpy as np
import torch
import segmentation_models_pytorch as smp
from finetune_negatives import build_dataset, norm_patch, DS, LMHLD_REGIONS

OUT = ROOT / "experiments" / "core_ablation_20261001"
MODEL_OUT = ROOT / "models_prior" / "core_ablation_20261001"
BASE = ROOT / "models_prior" / "Emilia-Romagna_Italy_32_upload_seed42" / "best_model.pth"
NEG_DIR = ROOT / "experiments" / "mining_neg"
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def file_hash(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    path.write_bytes((json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode())


def source_validation(n_each=200):
    """Use source val files, never source train patches or any mining-event reference."""
    pos, neg, pos_records, neg_records = [], [], [], []
    rng = np.random.default_rng(1001)
    for root, region in LMHLD_REGIONS:
        paths = [Path(root) / region / (f"val_{kind}.npy") for kind in ("images", "labels")]
        # The released dataset sometimes uses 'validation' as the split name.
        if not all(p.is_file() for p in paths):
            paths = [Path(root) / region / (f"validation_{kind}.npy") for kind in ("images", "labels")]
        if not all(p.is_file() for p in paths):
            raise FileNotFoundError(f"No genuine validation files for {region}")
        x, y = [np.load(p, mmap_mode="r") for p in paths]
        cp, cn = [], []
        for index in rng.permutation(len(x)):
            mask = np.asarray(y[index, 0]) > 0
            ratio = float(mask.mean())
            if ratio >= .1 and len(cp) < n_each // len(LMHLD_REGIONS):
                cp.append((np.asarray(x[index]), mask, int(index)))
            elif ratio <= .01 and len(cn) < n_each // len(LMHLD_REGIONS):
                cn.append((np.asarray(x[index]), mask, int(index)))
            if len(cp) + len(cn) >= 2 * n_each // len(LMHLD_REGIONS):
                break
        for group, label in ((cp, "positive"), (cn, "background")):
            for image, mask, index in group:
                (pos if label == "positive" else neg).append((image, mask))
                (pos_records if label == "positive" else neg_records).append(
                    {"region": region, "split": paths[0].stem.split("_")[0], "index": index, "label": label})
    pairs = pos + neg
    if len(pos) < 150 or len(neg) < 150:
        raise ValueError("Insufficient source validation classes")
    return norm_patch(np.stack([p[0] for p in pairs]).astype(np.float32)), np.stack([p[1] for p in pairs]).astype(np.float32), pos_records + neg_records


def prepare():
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = OUT / "protocol.json"
    if manifest.exists():
        prior = json.loads(manifest.read_text(encoding="utf-8"))
        for name, digest in prior["frozen_files"].items():
            if file_hash(OUT / name) != digest:
                raise ValueError(f"Frozen dataset changed: {name}")
        return prior
    split = json.loads((ROOT / "experiments/nir_train/neg_split_seed42.json").read_text(encoding="utf-8"))
    train_neg = np.asarray(split["train_idx"], dtype=np.int64)
    common, labels, composition, selected, origins = build_dataset(seed=42, neg_idx=train_neg, return_provenance=True)
    assert np.array_equal(np.sort(selected), train_neg)
    ours = np.load(NEG_DIR / "negatives_t1_b_ours.npy")
    naive = np.load(NEG_DIR / "negatives_t1_b_naive.npy")
    if len(ours) != 2000 or ours.shape != naive.shape or not np.array_equal(ours[:600], naive[:600]):
        raise ValueError("Negative pools do not implement the declared 600+1400 paired design")
    ours, naive = norm_patch(ours.astype(np.float32)), norm_patch(naive.astype(np.float32))
    # Keep all weak Baige positives in training. They are never independent test cases.
    generic_indices = [i for i, origin in enumerate(origins) if origin["kind"] == "generic_negative"]
    rng = np.random.default_rng(1001)
    val_generic = set(rng.choice(generic_indices, 200, replace=False).tolist())
    keep = np.array([i for i in range(len(common)) if i not in val_generic])
    val_indices = sorted(val_generic)
    vx, vy, val_records = source_validation()
    vx = np.concatenate([vx, common[val_indices]])
    vy = np.concatenate([vy, labels[val_indices]])
    files = {"common_x.npy": common[keep], "common_y.npy": labels[keep],
             "ours_extra.npy": ours, "naive_extra.npy": naive, "val_x.npy": vx, "val_y.npy": vy}
    for name, array in files.items():
        np.save(OUT / name, array)
    protocol = {"created_at": datetime.now().isoformat(timespec="seconds"),
                "data_seed": 42, "training_seeds": [42, 43, 44],
                "base_weights": str(BASE), "base_sha256": file_hash(BASE),
                "primary_pair": ["naive", "ours"], "reference_arm": "control",
                "common_train_n": len(keep), "validation_n": len(vx),
                "common_composition_before_internal_holdout": composition,
                "extra_negative_n": 2000, "arid_n": 600, "second_negative_n": 1400,
                "only_paired_difference": "industrial-facility vs mine-pit second negative pool",
                "selection": "minimum internal validation loss, external test never evaluated during training",
                "validation_label_policy": "original source validation masks, including small positives; generic background rows are pseudo-negative",
                "optimizer": "SGD lr=0.0005 momentum=0.9 weight_decay=0.0001",
                "epochs": 30, "batch": 16, "nir_dropout": .5,
                "schedule": "ExponentialLR gamma=0.96",
                "control_budget": "same optimizer update count, sampling its smaller common set with replacement",
                "primary_test_threshold": .5, "secondary_display_threshold": .4,
                "evaluation_limit": "event-reference and published-figure subsets are not full expert-verified benchmarks",
                "baige_status": "training weak-positive scene, not independent unseen positive validation",
                "common_train_provenance": [origins[i] for i in keep],
                "internal_validation_provenance": val_records + [origins[i] for i in val_indices],
                "frozen_files": {name: file_hash(OUT / name) for name in files}}
    write_json(manifest, protocol)
    print(f"Frozen data: train_common={len(keep)}, pair_extra=2000, validation={len(vx)}", flush=True)
    return protocol


def validation_loss(model, images, labels, batch=64):
    total = 0
    model.eval()
    with torch.no_grad():
        for start in range(0, len(images), batch):
            x = torch.as_tensor(np.ascontiguousarray(images[start:start+batch]), device=DEV)
            y = torch.as_tensor(np.ascontiguousarray(labels[start:start+batch]), device=DEV).unsqueeze(1)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(model(x), y)
            total += float(loss) * len(x)
    return total / len(images)


def train(arm, seed, protocol, epochs):
    folder = MODEL_OUT / f"{arm}_seed{seed}"
    result_path = OUT / f"train_{arm}_seed{seed}.json"
    if result_path.exists() and (folder / "best_model.pth").exists():
        result = json.loads(result_path.read_text(encoding="utf-8"))
        if result.get("completed_epochs") == epochs and result.get("protocol_sha256") == file_hash(OUT / "protocol.json"):
            print(f"Reusing complete run {arm}/seed{seed}", flush=True)
            return
        raise ValueError("Output name is occupied by a different/incomplete run; choose another study directory")
    folder.mkdir(parents=True, exist_ok=True)
    common = np.load(OUT / "common_x.npy")
    labels = np.load(OUT / "common_y.npy")
    if arm == "control":
        images, targets = common, labels
    else:
        extra = np.load(OUT / f"{arm}_extra.npy")
        images = np.concatenate([common, extra])
        targets = np.concatenate([labels, np.zeros((len(extra), 32, 32), np.float32)])
    vx, vy = np.load(OUT / "val_x.npy"), np.load(OUT / "val_y.npy")
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if DEV == "cuda":
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    model = smp.UnetPlusPlus(encoder_name="resnet34", encoder_weights=None, in_channels=4, classes=1).to(DEV)
    model.load_state_dict(torch.load(BASE, map_location=DEV, weights_only=True))
    dataset = DS(images, targets, nir_dropout=.5, train=True, seed=seed)
    optimizer = torch.optim.SGD(model.parameters(), lr=.0005, momentum=.9, weight_decay=.0001)
    schedule = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=.96)
    batch = 16
    updates = (len(common) + 2000) // batch
    best, best_epoch = float("inf"), 0
    history, start = [], time.time()
    protocol_hash = file_hash(OUT / "protocol.json")
    for epoch in range(1, epochs + 1):
        rng = np.random.default_rng(seed * 10000 + epoch)
        indices = (rng.permutation(len(images))[:updates*batch] if len(images) >= updates*batch
                   else rng.choice(len(images), updates*batch, replace=True))
        model.train(); running = 0
        for offset in range(0, len(indices), batch):
            pairs = [dataset[int(index)] for index in indices[offset:offset+batch]]
            x = torch.stack([pair[0] for pair in pairs]).to(DEV)
            y = torch.stack([pair[1] for pair in pairs]).to(DEV).unsqueeze(1)
            optimizer.zero_grad(set_to_none=True)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(model(x), y)
            loss.backward(); optimizer.step(); running += float(loss)
        schedule.step()
        val = validation_loss(model, vx, vy)
        if val < best:
            best, best_epoch = val, epoch
            torch.save(model.state_dict(), folder / "best_model.pth")
        row = {"epoch": epoch, "train_loss": running / updates, "val_loss": val,
               "elapsed_seconds": round(time.time()-start, 2)}
        history.append(row)
        record = {"arm": arm, "seed": seed, "completed_epochs": epoch, "requested_epochs": epochs,
                  "protocol_sha256": protocol_hash, "base_sha256": protocol["base_sha256"],
                  "train_n": len(images), "updates_per_epoch": updates, "device": DEV,
                  "best_epoch": best_epoch, "best_val_loss": best, "history": history,
                  "weights": str(folder / "best_model.pth")}
        write_json(result_path, record)
        if epoch % 5 == 0 or epoch == 1 or epoch == epochs:
            print(f"{arm}/seed{seed}: epoch {epoch}/{epochs}, val={val:.5f}, best={best:.5f}, elapsed={time.time()-start:.0f}s", flush=True)
    del model
    if DEV == "cuda":
        torch.cuda.empty_cache()


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--rebuild-before-training", action="store_true")
    parser.add_argument("--arms", nargs="+", choices=["control", "naive", "ours"], default=["control", "naive", "ours"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--epochs", type=int, default=30)
    args = parser.parse_args()
    if args.rebuild_before_training:
        if MODEL_OUT.exists() and any(MODEL_OUT.rglob("*.pth")):
            raise ValueError("Cannot rebuild a protocol after training weights exist")
        previous = OUT / "protocol.json"
        if previous.exists():
            previous.rename(OUT / ("protocol_pretraining_" + str(int(time.time())) + ".json"))
    protocol = prepare()
    if not args.prepare_only:
        for seed in args.seeds:
            for arm in args.arms:
                train(arm, seed, protocol, args.epochs)
