"""Acquire fixed-date native pixel windows; no prediction or training.

Uses the project's existing range-length/ETag-validated COG reader. Provider DN
values are preserved. Physical reflectance interpretation stays blocked until
the provider's offset convention is resolved separately.
"""
from __future__ import annotations

import hashlib
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, r"D:\LMHLD")
from runtime_paths import configure_runtime_paths
configure_runtime_paths()

import numpy as np
import rasterio
from rasterio.transform import Affine, array_bounds
from rasterio.windows import from_bounds
from rasterio.warp import transform_bounds, reproject, Resampling
from verified_cog_window import read_native_cog_window

from public_reference_screen import ROOT, write_json, acquire, now
from screen_hualien import protected_overlap, box

CASES = {
    "xincheng": ("TW_XINCHENG2024_SOURCE_CANDIDATE", [121.603, 24.125, 121.645, 24.164]),
    "zhonghe": ("TW_ZHONGHE2024_SOURCE_CANDIDATE", [121.679, 24.231, 121.710, 24.255]),
}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", choices=CASES, default="xincheng")
    args = parser.parse_args()
    case_id, bbox = CASES[args.case]
    output = ROOT / "imagery" / case_id
    if protected_overlap(box(*bbox)):
        raise ValueError("new source window overlaps previous protected geography")
    output.mkdir(parents=True, exist_ok=True)
    scenes = []
    for epoch, date in [("pre", "20240214"), ("post", "20240429")]:
        catalogue = ROOT / "sources" / f"xincheng_{date}_catalogue.json"
        source = json.loads(catalogue.read_text(encoding="utf-8"))
        if len(source["features"]) != 1:
            raise ValueError("fixed-date catalogue is ambiguous; review before acquisition")
        scenes.append((epoch, source["features"][0], sha(catalogue)))
    plan = {
        "bbox_wgs84": bbox, "scene_selection": "source-reference fixed dates, not model outputs",
        "scenes": [{"epoch": epoch, "id": scene["id"], "catalogue_sha256": digest}
                   for epoch, scene, digest in scenes],
        "model_predictions": 0, "classification_allowed": False,
        "multispectral_resampling_allowed": False,
        "quality_scl_excluded": [0, 1, 3, 7, 8, 9, 10, 11],
        "source_group": "TW_HUALIEN20240403",
    }
    plan_path = output / "acquisition_plan.json"
    if plan_path.exists() and json.loads(plan_path.read_text(encoding="utf-8")) != plan:
        raise ValueError("frozen source selection changed")
    write_json(plan_path, plan)
    ledger_path = output / "acquisition.json"
    ledger = json.loads(ledger_path.read_text(encoding="utf-8")) if ledger_path.exists() else {"epochs": {}}
    native_grid = native_shape = native_crs = None
    for epoch, scene, _ in scenes:
        if epoch in ledger["epochs"]:
            for name, digest in ledger["epochs"][epoch]["files_sha256"].items():
                if sha(output / name) != digest:
                    raise ValueError("retained native window changed")
            with rasterio.open(output / f"{epoch}_provider_BGRN_native10m.tif") as src:
                native_grid, native_shape, native_crs = src.transform, src.shape, src.crs
            continue
        blue = scene["assets"]["blue"]
        crs = f"EPSG:{scene['properties']['proj:epsg']}"
        grid = Affine(*blue["proj:transform"][:6])
        if native_grid is None:
            window = from_bounds(*transform_bounds("EPSG:4326", crs, *bbox), transform=grid).round_offsets().round_lengths()
            native_grid = grid * Affine.translation(window.col_off, window.row_off)
            native_shape = (int(window.height), int(window.width))
            native_crs = crs
            if max(native_shape) > 600:
                raise ValueError("source window budget exceeded")
        else:
            if crs != native_crs:
                raise ValueError("cross-CRS multispectral warping is not allowed")
            window = from_bounds(*array_bounds(*native_shape, native_grid), transform=grid).round_offsets().round_lengths()
            if not (grid * Affine.translation(window.col_off, window.row_off)).almost_equals(native_grid):
                raise ValueError("native dates do not share the selected grid")
        bands = []
        transport = []
        if min(window.row_off, window.col_off) < 0 or window.row_off + window.height > blue["proj:shape"][0] or window.col_off + window.width > blue["proj:shape"][1]:
            raise ValueError("selected native scene does not fully cover this new case")
        for band in ["blue", "green", "red", "nir"]:
            asset = scene["assets"][band]
            raw, evidence = read_native_cog_window(
                asset["href"], int(window.row_off), int(window.col_off), *native_shape,
                ROOT / "_range_cache", asset["proj:shape"], asset["proj:transform"])
            bands.append(raw)
            transport.append({"band": band, "advertised_raster_metadata": asset["raster:bands"], **evidence})
            print(epoch, band, raw.shape, "native pixels retained; no model", flush=True)
        profile = dict(driver="GTiff", height=native_shape[0], width=native_shape[1], count=4,
                       dtype="uint16", transform=native_grid, crs=native_crs, nodata=0, compress="deflate")
        image_path = output / f"{epoch}_provider_BGRN_native10m.tif"
        with rasterio.open(image_path, "w", **profile) as dst:
            dst.write(np.stack(bands))
            dst.descriptions = ("B02", "B03", "B04", "B08")
            dst.update_tags(acquisition_datetime=scene["properties"]["datetime"],
                            provider_pixels_unaltered="true", physical_reflectance_interpretation="pending_offset_audit",
                            classification_allowed="false", source_scene_id=scene["id"])
        scl_asset = scene["assets"]["scl"]
        with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR", GDAL_HTTP_TIMEOUT="30",
                          GDAL_HTTP_MAX_RETRY="1", GDAL_HTTP_MULTIRANGE="SERIAL",
                          CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tif"):
            with rasterio.open(scl_asset["href"]) as src:
                scl_window = from_bounds(*array_bounds(*native_shape, native_grid), transform=src.transform)
                # Cover the 10 m window without shifting the original 20 m grid.
                c0, r0 = int(np.floor(scl_window.col_off)), int(np.floor(scl_window.row_off))
                c1 = int(np.ceil(scl_window.col_off + scl_window.width))
                r1 = int(np.ceil(scl_window.row_off + scl_window.height))
                from rasterio.windows import Window
                scl_window = Window(c0, r0, c1 - c0, r1 - r0)
                scl = src.read(1, window=scl_window)
                scl_grid, scl_crs = src.window_transform(scl_window), src.crs
        scl_original = output / f"{epoch}_SCL_native20m.tif"
        with rasterio.open(scl_original, "w", **{**profile, "count": 1, "dtype": "uint8",
                          "height": scl.shape[0], "width": scl.shape[1], "transform": scl_grid}) as dst:
            dst.write(scl, 1)
        aligned = np.zeros(native_shape, dtype=np.uint8)
        reproject(scl, aligned, src_transform=scl_grid, src_crs=scl_crs,
                  dst_transform=native_grid, dst_crs=native_crs, resampling=Resampling.nearest)
        scl_aligned = output / f"{epoch}_SCL_nearest_on10m.tif"
        with rasterio.open(scl_aligned, "w", **{**profile, "count": 1, "dtype": "uint8"}) as dst:
            dst.write(aligned, 1)
        granule_name = f"xincheng_{epoch}_granule.xml"
        receipt = acquire(granule_name, scene["assets"]["granule_metadata"]["href"], direct=True)
        if not receipt["ok"]:
            raise ValueError("original sensing metadata not acquired")
        clear = ~np.isin(aligned, plan["quality_scl_excluded"]) & np.all(np.stack(bands) > 0, axis=0)
        ledger["epochs"][epoch] = {
            "scene_id": scene["id"], "datetime": scene["properties"]["datetime"],
            "native_shape": list(native_shape), "scl_counts": {str(k): int(v) for k, v in zip(*np.unique(aligned, return_counts=True))},
            "clear_fraction": float(clear.mean()), "transport": transport,
            "source_boa_offset_applied_flag": scene["properties"].get("earthsearch:boa_offset_applied"),
            "physical_reflectance_eligible": False,
            "files_sha256": {p.name: sha(p) for p in [image_path, scl_original, scl_aligned]},
        }
        ledger.update(updated_at=now(), new_model_predictions=0, labels_admitted=False,
                      physical_indices_computed=False, multispectral_warping=False)
        write_json(ledger_path, ledger)
        print(epoch, "clear fraction", float(clear.mean()), flush=True)
    print(json.dumps({"completed": True, "epochs": {key: {k: v for k, v in row.items()
                   if k in {"scene_id", "clear_fraction", "native_shape"}} for key, row in ledger["epochs"].items()},
                   "model_predictions": 0}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
