"""Native fixed-date source QA only; no predictions, labels or remote code execution."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from xml.etree import ElementTree as ET

sys.path.insert(0, r"D:\LMHLD")
from runtime_paths import configure_runtime_paths
configure_runtime_paths()
sys.path.insert(0, r"D:\GeoMMAgent\maintenance\phase7_20261015")

import numpy as np
import rasterio
from rasterio.features import rasterize
from rasterio.transform import Affine, array_bounds
from rasterio.windows import from_bounds, Window
from rasterio.warp import transform_bounds, transform_geom, reproject, Resampling
from verified_cog_window import read_native_cog_window
from screen_hualien import protected_overlap
from shapely.geometry import shape, box, mapping
from expand_event_references import DATA, HERE, read, write, sha, acquire, now


def history_from_original_api(way_id=536939324, snapshot="2023-07-13T00:00:00Z", prefix="dayu"):
    # A current full response also certifies the historical geometry only when
    # every participating node and the way already had this version before then.
    url = f"https://api.openstreetmap.org/api/0.6/way/{way_id}/full"
    record = acquire(f"{prefix}_osm_full.xml", url, direct=False)
    if not record["ok"]:
        record = acquire(f"{prefix}_osm_full.xml", url, direct=True)
    report = {"original_response": record, "snapshot_date": snapshot,
              "historical_geometry_verified": False, "accident_truth": False}
    if record["ok"]:
        root = ET.parse(record["path"]).getroot()
        nodes = {row.attrib["id"]: row.attrib for row in root.findall("node")}
        way = next(row for row in root.findall("way") if row.attrib["id"] == str(way_id))
        refs = [row.attrib["ref"] for row in way.findall("nd")]
        objects = [way.attrib] + [nodes[ref] for ref in refs]
        verified = all(row.get("timestamp", "9999") < report["snapshot_date"] for row in objects)
        report.update(historical_geometry_verified=verified,
                      way_version=way.attrib["version"], way_timestamp=way.attrib["timestamp"],
                      node_count=len(nodes), latest_node_timestamp=max(row["timestamp"] for row in nodes.values()),
                      verification_rule="way and every referenced node version timestamp before snapshot; no later edit",
                      licence="OpenStreetMap contributors ODbL; land-use hint, not certified mine bounds")
        geometry = {"type": "Polygon", "coordinates": [[(float(nodes[ref]["lon"]), float(nodes[ref]["lat"])) for ref in refs]]}
        write(DATA / "qualification" / f"{prefix}_osm_geometry.geojson", {"type": "Feature", "geometry": geometry,
              "properties": {"historical_geometry_verified": verified, "accident_truth": False}})
    write(DATA / "qualification" / f"{prefix}_history_api_qualification.json", report)
    print(json.dumps({f"{prefix}_historical_geometry_verified": report["historical_geometry_verified"]}), flush=True)


def acquire_pair(plan_filename="dayu_source_selection.json"):
    plan = read(DATA / "qualification" / plan_filename)
    prefix = plan.get("source_prefix", "dayu")
    output = DATA / "imagery" / plan["case_id"]
    output.mkdir(parents=True, exist_ok=True)
    bbox = plan["bbox_wgs84"]
    if protected_overlap(box(*bbox)):
        raise ValueError("Window overlaps protected geography")
    for row in read(HERE / "protocol.json")["previously_observed_windows"]:
        if box(*bbox).intersects(box(*row["bounds"])):
            raise ValueError("Window overlaps previously consumed development imagery")
    sources = []
    for epoch, date in zip(["pre", "post"], plan["source_dates"]):
        path = DATA / "sources" / f"{prefix}_{date.replace('-', '')}_catalogue.json"
        source = read(path)
        if len(source["features"]) != 1:
            raise ValueError("Ambiguous exact-date scene; review before acquisition")
        sources.append((epoch, source["features"][0], sha(path)))
    acquisition_plan = {**plan, "scene_ids": [row[1]["id"] for row in sources],
                        "catalogue_sha256": [row[2] for row in sources],
                        "quality_scl_excluded": [0, 1, 3, 7, 8, 9, 10, 11],
                        "resampling_multispectral": False, "physical_indices_allowed": False}
    frozen_path = output / "acquisition_plan.json"
    if frozen_path.exists() and read(frozen_path) != acquisition_plan:
        raise ValueError("Frozen source acquisition plan changed")
    write(frozen_path, acquisition_plan)
    ledger_path = output / "acquisition.json"
    ledger = read(ledger_path) if ledger_path.exists() else {"epochs": {}}
    native_grid = native_shape = native_crs = None
    for epoch, scene, _ in sources:
        if epoch in ledger["epochs"]:
            for name, digest in ledger["epochs"][epoch]["files_sha256"].items():
                if sha(output / name) != digest:
                    raise ValueError("Retained native window changed")
            with rasterio.open(output / f"{epoch}_provider_BGRN_native10m.tif") as src:
                native_grid, native_shape, native_crs = src.transform, src.shape, src.crs
            continue
        crs = f"EPSG:{scene['properties']['proj:epsg']}"
        asset = scene["assets"]["blue"]
        grid = Affine(*asset["proj:transform"][:6])
        if native_grid is None:
            window = from_bounds(*transform_bounds("EPSG:4326", crs, *bbox), transform=grid).round_offsets().round_lengths()
            native_grid = grid * Affine.translation(window.col_off, window.row_off)
            native_shape, native_crs = (int(window.height), int(window.width)), crs
            if max(native_shape) > 600:
                raise ValueError("Source window exceeds 600-pixel budget")
        else:
            if crs != native_crs:
                raise ValueError("Cross-CRS source warping blocked")
            window = from_bounds(*array_bounds(*native_shape, native_grid), transform=grid).round_offsets().round_lengths()
            if not (grid * Affine.translation(window.col_off, window.row_off)).almost_equals(native_grid):
                raise ValueError("Native date grids differ")
        bands, transports = [], []
        for name in ["blue", "green", "red", "nir"]:
            asset = scene["assets"][name]
            raw, receipt = read_native_cog_window(asset["href"], int(window.row_off), int(window.col_off),
                                                 *native_shape, DATA / "_range_cache",
                                                 asset["proj:shape"], asset["proj:transform"])
            bands.append(raw)
            transports.append({"band": name, "provider_raster_metadata": asset["raster:bands"], **receipt})
            print(epoch, name, raw.shape, "native; no prediction", flush=True)
        profile = dict(driver="GTiff", height=native_shape[0], width=native_shape[1], count=4,
                       dtype="uint16", transform=native_grid, crs=native_crs, nodata=0, compress="deflate")
        image_path = output / f"{epoch}_provider_BGRN_native10m.tif"
        with rasterio.open(image_path, "w", **profile) as dst:
            dst.write(np.stack(bands))
            dst.descriptions = ("B02", "B03", "B04", "B08")
            dst.update_tags(acquisition_datetime=scene["properties"]["datetime"], source_scene_id=scene["id"],
                            provider_pixels_unaltered="true", physical_indices_allowed="false", classification_allowed="false")
        with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR", GDAL_HTTP_TIMEOUT="30",
                          GDAL_HTTP_MAX_RETRY="1", GDAL_HTTP_MULTIRANGE="SERIAL",
                          CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tif"):
            with rasterio.open(scene["assets"]["scl"]["href"]) as src:
                win = from_bounds(*array_bounds(*native_shape, native_grid), transform=src.transform)
                c0, r0 = int(np.floor(win.col_off)), int(np.floor(win.row_off))
                c1, r1 = int(np.ceil(win.col_off + win.width)), int(np.ceil(win.row_off + win.height))
                win = Window(c0, r0, c1 - c0, r1 - r0)
                scl, scl_grid, scl_crs = src.read(1, window=win), src.window_transform(win), src.crs
        scl_path = output / f"{epoch}_SCL_native20m.tif"
        with rasterio.open(scl_path, "w", **{**profile, "count": 1, "dtype": "uint8", "height": scl.shape[0],
                          "width": scl.shape[1], "transform": scl_grid}) as dst:
            dst.write(scl, 1)
        aligned = np.zeros(native_shape, dtype=np.uint8)
        reproject(scl, aligned, src_transform=scl_grid, src_crs=scl_crs, dst_transform=native_grid,
                  dst_crs=native_crs, resampling=Resampling.nearest)
        aligned_path = output / f"{epoch}_SCL_nearest_on10m.tif"
        with rasterio.open(aligned_path, "w", **{**profile, "count": 1, "dtype": "uint8"}) as dst:
            dst.write(aligned, 1)
        xml_name = f"{prefix}_{scene['properties']['datetime'][:10].replace('-', '')}_granule.xml"
        xml_receipt = acquire(xml_name, scene["assets"]["granule_metadata"]["href"], direct=True)
        if not xml_receipt["ok"]:
            raise ValueError("Original granule metadata missing")
        clear = ~np.isin(aligned, acquisition_plan["quality_scl_excluded"]) & np.all(np.stack(bands) > 0, axis=0)
        ledger["epochs"][epoch] = {"scene_id": scene["id"], "datetime": scene["properties"]["datetime"],
                                   "granule_metadata_receipt": xml_receipt,
                                   "native_shape": list(native_shape), "clear_fraction": float(clear.mean()),
                                   "scl_counts": {str(k): int(v) for k, v in zip(*np.unique(aligned, return_counts=True))},
                                   "source_boa_offset_applied_flag": scene["properties"].get("earthsearch:boa_offset_applied"),
                                   "files_sha256": {p.name: sha(p) for p in [image_path, scl_path, aligned_path]},
                                   "transport": transports}
        ledger.update(updated_at=now(), predictions=0, training=0, labels_admitted=False, physical_indices_computed=False)
        write(ledger_path, ledger)
    audit_granule_bindings(plan_filename)
    qualify(output, acquisition_plan)


def audit_granule_bindings(plan_filename):
    """Verify date-specific original metadata, including earlier acquisition runs."""
    plan = read(DATA / "qualification" / plan_filename)
    output = DATA / "imagery" / plan["case_id"]
    ledger = read(output / "acquisition.json")
    prefix = plan.get("source_prefix", "dayu")
    for epoch, date in zip(["pre", "post"], plan["source_dates"]):
        scene = read(DATA / "sources" / f"{prefix}_{date.replace('-', '')}_catalogue.json")["features"][0]
        if scene["id"] != ledger["epochs"][epoch]["scene_id"]:
            raise ValueError("Acquired source scene identity differs")
        name = f"{prefix}_{date.replace('-', '')}_granule.xml"
        receipt = acquire(name, scene["assets"]["granule_metadata"]["href"], direct=True)
        if not receipt["ok"]:
            raise ValueError("Original date-specific metadata not acquired")
        root = ET.parse(receipt["path"]).getroot()
        times = [item.text for item in root.iter() if item.tag.split('}')[-1] == "SENSING_TIME"]
        if len(times) != 1 or not times[0].startswith(date):
            raise ValueError("Original sensing time disagrees with selected date")
        ledger["epochs"][epoch]["granule_metadata_receipt"] = receipt
        ledger["epochs"][epoch]["granule_sensing_time_verified"] = times[0]
    write(output / "acquisition.json", ledger)
    print(json.dumps({"date_specific_granule_metadata_verified": plan["case_id"]}), flush=True)


def qualify(output, plan):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    arrays, clear_maps = {}, {}
    for epoch in ["pre", "post"]:
        with rasterio.open(output / f"{epoch}_provider_BGRN_native10m.tif") as src:
            arrays[epoch], transform, crs = src.read(), src.transform, src.crs
            extent = array_bounds(src.height, src.width, src.transform)
            dims = src.shape
        with rasterio.open(output / f"{epoch}_SCL_nearest_on10m.tif") as src:
            clear_maps[epoch] = ~np.isin(src.read(1), plan["quality_scl_excluded"]) & np.all(arrays[epoch] > 0, axis=0)
    joint = clear_maps["pre"] & clear_maps["post"]
    objects = read(DATA / "qualification" / plan.get("discovery_file", "mining_event_discovery.geojson"))["features"]
    selected_row = plan["source_row_zero_based"]
    objects = [feature for feature in objects if feature["properties"]["source_row_zero_based"] == selected_row]
    footprint_path = DATA / "qualification" / f"{plan.get('source_prefix', 'dayu')}_osm_geometry.geojson"
    footprints = read(footprint_path) if footprint_path.exists() else None
    records, geometries = [], []
    for feature in objects:
        geom = shape(transform_geom("EPSG:4326", crs, feature["geometry"]))
        geometries.append(geom)
        record = {"source_row_zero_based": feature["properties"]["source_row_zero_based"],
                  "source_polygon_area_m2": feature["properties"]["reference_polygon_area_m2"],
                  "source_dates": plan["source_dates"], "mask_is_truth": False, "core_sensitivity": []}
        for erosion in [0, 10, 20]:
            core = geom.buffer(-erosion) if erosion else geom
            mask = rasterize([(mapping(core), 1)], out_shape=dims, transform=transform, fill=0, dtype="uint8").astype(bool) if not core.is_empty else np.zeros(dims, bool)
            count = int(mask.sum())
            record["core_sensitivity"].append({"erosion_m": erosion, "pixels": count,
                                               "joint_clear_pixels": int((mask & joint).sum()),
                                               "joint_clear_fraction": float(joint[mask].mean()) if count else None})
        records.append(record)
    pixel_values = np.concatenate([arrays[epoch][[2, 1, 0]][:, clear_maps[epoch]] for epoch in ["pre", "post"]], axis=1)
    lo, hi = np.percentile(pixel_values, [2, 98], axis=1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 6), constrained_layout=True)
    for axis, epoch, date in zip(axes, ["pre", "post"], plan["source_dates"]):
        rgb = arrays[epoch][[2, 1, 0]].transpose(1, 2, 0)
        rgb = np.clip((rgb - lo) / np.maximum(hi - lo, 1), 0, 1)
        axis.imshow(rgb, extent=[extent[0], extent[2], extent[1], extent[3]])
        for geom, record in zip(geometries, records):
            axis.plot(*geom.exterior.xy, color="yellow", linewidth=1.5)
            axis.text(geom.centroid.x, geom.centroid.y, str(record["source_row_zero_based"]), color="yellow", fontsize=8)
        if footprints:
            mine = shape(transform_geom("EPSG:4326", crs, footprints["geometry"]))
            axis.plot(*mine.exterior.xy, color="cyan", linewidth=1.1)
        axis.set_title(f"{date} native Sentinel-2 RGB | no model output")
        axis.set_xlabel("UTM 51N easting (m)")
        axis.set_ylabel("Northing (m)")
        axis.ticklabel_format(style="plain", useOffset=False)
        axis.set_xlim(extent[0], extent[2])
        axis.set_ylim(extent[1], extent[3])
    fig.suptitle(f"{plan['case_id']} source qualification\nFull accident extent and mine-slope membership remain unverified", fontsize=12)
    fig.legend(handles=[Line2D([], [], color="yellow", label=f"Original official collapse polygon (row {selected_row})"),
                        Line2D([], [], color="cyan", label="OSM quarry footprint (not accident truth)")], loc="outside lower center", ncol=2)
    panel = output / "source_qualification.png"
    fig.savefig(panel, dpi=180)
    plt.close(fig)
    report = {"completed_at": now(), "native_shape": list(dims), "joint_clear_fraction": float(joint.mean()),
              "objects": records, "visual_qa_panel": str(panel), "visual_qa_sha256": sha(panel),
              "original_source_properties_preserved": True, "protected_geography_overlap": [],
              "source_dates_are_observation_interval_not_exact_failure_time": True,
              "original_reference_dates": plan.get("original_reference_dates", plan["source_dates"]),
              "original_reference_sensors": plan.get("original_reference_sensors", ["Sentinel-2", "Sentinel-2"]),
              "source_imagery_is_substitute_not_original_high_resolution": plan.get("source_imagery_is_substitute_not_original_high_resolution", False),
              "legacy_datum_verification_pending": plan.get("legacy_datum_verification_pending", False),
              "failure_in_mining_slope_context_confirmed": False, "full_accident_extent_confirmed": False,
              "training_allowed": False, "predictions_allowed": False, "final_blind_test_allowed": False,
              "admission_decision": "pending_visual_and_independent_mine_context_review"}
    write(output / "qualification.json", report)
    print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    history_from_original_api()
    acquire_pair()
