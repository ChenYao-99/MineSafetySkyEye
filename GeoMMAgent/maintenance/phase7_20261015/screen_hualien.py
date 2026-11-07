"""Qualify original references around a published mine location, without AI.

The search window is NOT a mine boundary. Every selected polygon retains its
entire source geometry; proximity alone does not admit a mining accident label.
"""
from __future__ import annotations

import hashlib
import io
import json
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from zipfile import ZipFile

sys.path.insert(0, str(Path(__file__).parent / "vendor"))
import shapefile
from shapely.geometry import shape, mapping, Point, box
from rasterio.crs import CRS
from rasterio.warp import transform, transform_geom

from public_reference_screen import ROOT, write_json, now

PROTOCOL = {
    "registered_before_model_predictions": True,
    "case_id": "TW_XINCHENG2024_SOURCE_CANDIDATE",
    "mine_name": "新城山礦場",
    "location_wgs84": [121.623746, 24.144416],
    "location_source": "https://esg.acc.com.tw/upload/2025/r12_files/2024%E6%B0%B8%E7%BA%8C%E5%A0%B1%E5%91%8A%E6%9B%B8.pdf",
    "location_source_context": "运营方2024永续报告GRI矿业披露；不是事故或作业面范围",
    "search_radius_m": 2000,
    "event_date_local": "2024-04-03",
    "event_timezone": "Asia/Taipei",
    "official_event_name": "0403花蓮地震",
    "polygon_eligibility": "requires original provenance, source geometry validity, temporal match, mine-face identity, imagery QA and licence",
    "all_same_earthquake_candidates_share_event_group": "TW_HUALIEN20240403",
    "split_assigned": False,
    "model_predictions": 0,
}


def vector_rows(filename):
    path = ROOT / "sources" / filename
    receipt = json.loads((ROOT / "receipts" / (filename + ".json")).read_text(encoding="utf-8"))
    if hashlib.sha256(path.read_bytes()).hexdigest() != receipt["sha256"]:
        raise ValueError("source receipt mismatch")
    with ZipFile(path) as archive:
        shp_names = [x for x in archive.namelist() if x.lower().endswith(".shp")]
        if len(shp_names) != 1:
            raise ValueError("select exactly one known original layer")
        base = shp_names[0][:-4]
        encoding = archive.read(base + ".cpg").decode("ascii").strip()
        crs = CRS.from_wkt(archive.read(base + ".prj").decode("utf-8-sig"))
        reader = shapefile.Reader(shp=io.BytesIO(archive.read(base + ".shp")),
                                 shx=io.BytesIO(archive.read(base + ".shx")),
                                 dbf=io.BytesIO(archive.read(base + ".dbf")), encoding=encoding)
        for index, row in enumerate(reader.iterShapeRecords()):
            native = row.shape.__geo_interface__
            geographic = transform_geom(crs, "EPSG:4326", native, precision=-1)
            yield index, row.record.as_dict(), shape(geographic), native, crs


def valid_date(value):
    try:
        return datetime.strptime(value, "%Y%m%d").date()
    except (ValueError, TypeError):
        return None


def protected_overlap(geom):
    registry = json.loads(Path(r"D:\LMHLD\datasets\input_reserve_registry.json").read_text(encoding="utf-8-sig"))
    extents = [(key, value["protected_geographic_extent_wgs84"]) for key, value in registry["cases"].items()]
    prior = json.loads(Path(r"D:\LMHLD\validation\reserved_geographic_protection_20261005.json").read_text(encoding="utf-8-sig"))
    extents += [(x["case_id"], x["bounds_wgs84"]) for x in prior["cases"]]
    return [case for case, extent in extents if geom.intersects(box(*extent))]


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    write_json(ROOT / "qualification" / "xincheng_search_protocol.json", PROTOCOL)
    lon, lat = PROTOCOL["location_wgs84"]
    x, y = transform("EPSG:4326", "EPSG:3826", [lon], [lat])
    search_native = Point(x[0], y[0]).buffer(PROTOCOL["search_radius_m"])
    search = shape(transform_geom("EPSG:3826", "EPSG:4326", mapping(search_native)))
    overlaps = protected_overlap(search)
    if overlaps:
        raise ValueError(f"search overlaps existing reserved extents: {overlaps}")
    features = []
    summaries = []
    for filename in ["hualien_author_inventory.zip", "taiwan_2024_official_inventory.zip"]:
        counts = Counter()
        selected = []
        seen = set()
        for index, properties, geom, native, crs in vector_rows(filename):
            counts["records"] += 1
            is_valid = geom.is_valid and not geom.is_empty
            counts["valid_geometry" if is_valid else "invalid_geometry"] += 1
            digest = hashlib.sha256(geom.wkb).hexdigest()
            if digest in seen:
                counts["duplicate_geometry"] += 1
            seen.add(digest)
            official = "Events" in properties
            same_event = not official or properties["Events"] == PROTOCOL["official_event_name"]
            counts["target_event"] += int(same_event)
            if official:
                before, after = valid_date(properties["BeforeDate"]), valid_date(properties["AfterDate"])
                temporal_ok = bool(before and after and before < after)
                counts["ordered_image_dates" if temporal_ok else "invalid_or_unordered_image_dates"] += 1
            else:
                before = after = None
                temporal_ok = None
            if not same_event or not is_valid or not geom.intersects(search):
                continue
            projected = shape(transform_geom("EPSG:4326", "EPSG:3826", mapping(geom)))
            row = {
                "source_archive": filename, "source_row_index_zero_based": index,
                "original_properties": properties, "geometry_sha256_after_wgs84_transform": digest,
                "geometry_valid": is_valid, "full_source_geometry_preserved": True,
                "polygon_area_m2_epsg3826": projected.area,
                "distance_from_published_mine_location_m": projected.distance(Point(x[0], y[0])),
                "reference_dates_bracket_event": bool(before and after and before < datetime(2024, 4, 3).date() <= after) if official else None,
                "mine_face_identity_verified": False, "matched_original_imagery_acquired": False,
                "accident_label_admitted": False, "split": "qualification_only_unassigned",
                "event_group": PROTOCOL["all_same_earthquake_candidates_share_event_group"],
            }
            selected.append(row)
            features.append({"type": "Feature", "geometry": mapping(geom), "properties": row})
        summaries.append({"archive": filename, "counts": dict(counts), "near_mine_target_event_polygons": selected})
    report = {
        "completed_at": now(), "protocol": PROTOCOL, "sources": summaries,
        "new_qualified_mine_accident_cases": 0,
        "new_qualified_full_mine_accident_boundaries": 0,
        "pending_gates": ["确认原始多边形与具体矿场作业面的对应，排除邻近自然山坡",
                          "匹配原始影像及云/覆盖/配准质量",
                          "作者仓库未显式提供数据许可，官方数据可追溯至开放条款",
                          "独立参考的边界覆盖范围与漏绘情况；未标注区域保持未知",
                          "按矿区和事件去重并在首次预测前分配开发或保留角色"],
        "new_model_predictions": 0,
    }
    write_json(ROOT / "qualification" / "xincheng_candidate_screen.json", report)
    write_json(ROOT / "qualification" / "xincheng_source_polygons.geojson", {"type": "FeatureCollection", "features": features})
    print(json.dumps({"sources": [{"archive": s["archive"], "counts": s["counts"],
                                 "near_mine_polygons": len(s["near_mine_target_event_polygons"])}
                                for s in summaries]}, ensure_ascii=False), flush=True)


def review_overlay(case="xincheng"):
    # No model result is involved; this is a plot of original references only.
    sys.path.insert(0, r"D:\LMHLD")
    from runtime_paths import configure_runtime_paths
    configure_runtime_paths()
    import numpy as np
    import rasterio
    from rasterio.features import rasterize
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from xml.etree import ElementTree

    case_id = PROTOCOL["case_id"] if case == "xincheng" else "TW_ZHONGHE2024_SOURCE_CANDIDATE"
    folder = ROOT / "imagery" / case_id
    ledger = json.loads((folder / "acquisition.json").read_text(encoding="utf-8"))
    arrays, masks, transforms, crss, bounds = [], [], [], [], []
    sensing = {}
    for epoch in ["pre", "post"]:
        record = ledger["epochs"][epoch]
        for name, digest in record["files_sha256"].items():
            if hashlib.sha256((folder / name).read_bytes()).hexdigest() != digest:
                raise ValueError("acquired source window changed")
        xml = ElementTree.parse(ROOT / "sources" / f"xincheng_{epoch}_granule.xml")
        times = [node.text.strip() for node in xml.iter() if node.tag.rsplit("}", 1)[-1] == "SENSING_TIME"]
        if len(times) != 1:
            raise ValueError("original sensing time missing or ambiguous")
        original_time = datetime.fromisoformat(times[0].replace("Z", "+00:00"))
        catalogue_time = datetime.fromisoformat(record["datetime"].replace("Z", "+00:00"))
        # This catalogue truncates XML microseconds to milliseconds. Preserve
        # both timestamps; do not invent a precision the catalogue lacks.
        if original_time.replace(microsecond=original_time.microsecond // 1000 * 1000) != catalogue_time:
            raise ValueError("original sensing time disagrees with selected catalogue")
        tile_ids = [node.text.strip() for node in xml.iter() if node.tag.rsplit("}", 1)[-1] == "TILE_ID"]
        selected = json.loads((ROOT / "sources" / ("xincheng_20240214_catalogue.json" if epoch == "pre" else "xincheng_20240429_catalogue.json")).read_text(encoding="utf-8"))["features"][0]
        if tile_ids != [selected["properties"]["s2:granule_id"]]:
            raise ValueError("original granule identity differs from selected scene")
        sensing[epoch] = {"xml": times[0], "catalogue": record["datetime"],
                          "match_at_millisecond_precision": True,
                          "xml_minus_catalogue_microseconds": round((original_time - catalogue_time).total_seconds() * 1000000)}
        with rasterio.open(folder / f"{epoch}_provider_BGRN_native10m.tif") as src:
            data = src.read()
            arrays.append(data)
            transforms.append(src.transform)
            crss.append(src.crs)
            bounds.append(src.bounds)
        with rasterio.open(folder / f"{epoch}_SCL_nearest_on10m.tif") as src:
            scl = src.read(1)
        masks.append(~np.isin(scl, [0, 1, 3, 7, 8, 9, 10, 11]) & np.all(data > 0, axis=0))
    if crss[0] != crss[1] or transforms[0] != transforms[1] or arrays[0].shape != arrays[1].shape:
        raise ValueError("paired native grids differ")
    if case == "xincheng":
        geo = json.loads((ROOT / "qualification" / "xincheng_source_polygons.geojson").read_text(encoding="utf-8"))
        footprint = None
    else:
        geo = json.loads((ROOT / "qualification" / "historical_quarry_reference_polygons.geojson").read_text(encoding="utf-8"))
        geo["features"] = [feature for feature in geo["features"] if any(row["osm_way_id"] == 511661858 for row in feature["properties"]["mine_matches"])]
        footprints = json.loads((ROOT / "qualification" / "historical_quarry_footprints.geojson").read_text(encoding="utf-8"))
        footprint = shape(transform_geom("EPSG:4326", crss[0], next(feature["geometry"] for feature in footprints["features"] if feature["properties"]["osm_way_id"] == 511661858)))
    geometries = []
    rows = []
    for feature in geo["features"]:
        geom = shape(transform_geom("EPSG:4326", crss[0], feature["geometry"]))
        geometries.append((geom, feature["properties"]))
        mask = rasterize([(mapping(geom), 1)], out_shape=arrays[0].shape[1:], transform=transforms[0], dtype="uint8") > 0
        rows.append({"archive": feature["properties"]["source_archive"],
                     "row_index": feature["properties"]["source_row_index_zero_based"],
                     "native10m_reference_pixels": int(mask.sum()),
                     "joint_clear_reference_pixels": int((mask & masks[0] & masks[1]).sum()),
                     "mine_face_identity_verified": False, "accident_label_admitted": False})
    values = np.concatenate([a[[2, 1, 0]][:, m] for a, m in zip(arrays, masks)], axis=1)
    lo, hi = np.percentile(values, [1, 99.8], axis=1)
    rgb = [np.clip((a[[2, 1, 0]].transpose(1, 2, 0) - lo) / (hi - lo), 0, 1) ** 0.8 for a in arrays]
    fig, axes = plt.subplots(1, 3, figsize=(16, 6), constrained_layout=True)
    extent = [bounds[0].left, bounds[0].right, bounds[0].bottom, bounds[0].top]
    for ax, image, title in zip(axes, [rgb[0], rgb[1], rgb[1]],
                               ["2024-02-14 | provider pixels", "2024-04-29 | provider pixels", "Original reference outlines | NOT model output"]):
        ax.imshow(image, extent=extent)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("Easting (m), EPSG:32651")
        ax.ticklabel_format(style="plain", useOffset=False)
        from matplotlib.ticker import MaxNLocator
        ax.xaxis.set_major_locator(MaxNLocator(4))
    axes[0].set_ylabel("Northing (m)")
    for geom, properties in geometries:
        color = "#ffda3a" if properties["source_archive"].startswith("hualien_author") else "#40dfff"
        pieces = [geom] if geom.geom_type == "Polygon" else geom.geoms
        for polygon in pieces:
            x, y = polygon.exterior.xy
            axes[2].plot(x, y, color=color, linewidth=0.9)
        if properties["source_archive"].startswith("hualien_author"):
            axes[2].text(geom.centroid.x, geom.centroid.y, f"A{properties['source_row_index_zero_based']}", color="yellow", fontsize=8)
        elif case == "zhonghe":
            axes[2].text(geom.centroid.x, geom.centroid.y, f"G{properties['source_row_index_zero_based']}", color="#ffda3a", fontsize=8)
    if footprint is None:
        x, y = transform("EPSG:4326", crss[0], [PROTOCOL["location_wgs84"][0]], [PROTOCOL["location_wgs84"][1]])
        axes[2].plot(x, y, "+", color="magenta", markersize=12)
    else:
        axes[2].plot(*footprint.exterior.xy, color="magenta", linewidth=1.2)
    legend = [Line2D([], [], color="#40dfff", label="Official 2024 inventory")]
    if any(properties["source_archive"].startswith("hualien_author") for _, properties in geometries):
        legend.append(Line2D([], [], color="#ffda3a", label="Chen et al. 2025 inventory"))
    legend.append(Line2D([], [], marker="+" if footprint is None else None, color="magenta", linestyle="None" if footprint is None else "-", label="Published mine location (not boundary)" if footprint is None else "Historic OSM working footprint (auxiliary)"))
    axes[2].legend(handles=legend,
                   fontsize=8, loc="lower left")
    fig.suptitle(case.title() + " public-source qualification | 10 m | Shared 1-99.8% display stretch, gamma 0.8", fontsize=12)
    output = ROOT / "qualification" / f"{case}_source_overlay.png"
    fig.savefig(output, dpi=160)
    plt.close(fig)
    report = {"completed_at": now(), "sensing_times_verified_against_original_xml": sensing,
              "native_grids_identical": True, "joint_clear_fraction": float((masks[0] & masks[1]).mean()),
              "reference_objects": rows, "outside_reference_is_unknown": True,
              "physical_indices_computed": False, "new_model_predictions": 0,
              "mine_reference_admission_performed_by_this_QA": False,
              "complete_mine_accident_cases_qualified": 0, "overlay": str(output)}
    write_json(ROOT / "qualification" / f"{case}_image_reference_qa.json", report)
    print(json.dumps({key: value for key, value in report.items() if key != "reference_objects"}, ensure_ascii=False))


def mine_intersections():
    """Original event polygons vs dated auxiliary footprints; no truth admission."""
    from shapely.geometry import Polygon
    from rasterio.warp import transform_bounds
    historic_path = ROOT / "sources" / "east_taiwan_osm_quarry_20240401.json"
    historic = json.loads(historic_path.read_text(encoding="utf-8"))
    mines = []
    for element in historic["elements"]:
        coordinates = [(p["lon"], p["lat"]) for p in element["geometry"]]
        if coordinates[0] != coordinates[-1]:
            raise ValueError("auxiliary quarry way is not a closed polygon")
        geom = Polygon(coordinates)
        if not geom.is_valid:
            raise ValueError("auxiliary quarry polygon is invalid")
        if protected_overlap(geom):
            raise ValueError("new candidate overlaps existing protected geography")
        native = shape(transform_geom("EPSG:4326", "EPSG:3826", mapping(geom)))
        mines.append({"osm_way_id": element["id"], "tags": element.get("tags", {}),
                      "geometry": geom, "native": native, "search": native.buffer(100), "matches": []})
    archive_path = ROOT / "sources" / "taiwan_2024_official_inventory.zip"
    source_receipt = json.loads((ROOT / "receipts" / (archive_path.name + ".json")).read_text(encoding="utf-8"))
    if hashlib.sha256(archive_path.read_bytes()).hexdigest() != source_receipt["sha256"]:
        raise ValueError("official source changed")
    features = []
    with ZipFile(archive_path) as archive:
        base = next(name[:-4] for name in archive.namelist() if name.endswith(".shp"))
        crs = CRS.from_wkt(archive.read(base + ".prj").decode("utf-8-sig"))
        if crs.to_epsg() != 3826:
            raise ValueError("review original source CRS before geometric intersection")
        reader = shapefile.Reader(shp=io.BytesIO(archive.read(base + ".shp")),
                                 shx=io.BytesIO(archive.read(base + ".shx")),
                                 dbf=io.BytesIO(archive.read(base + ".dbf")), encoding="utf-8")
        for index, row in enumerate(reader.iterShapeRecords()):
            properties = row.record.as_dict()
            if properties["Events"] != PROTOCOL["official_event_name"]:
                continue
            candidates = [mine for mine in mines if box(*row.shape.bbox).intersects(box(*mine["search"].bounds))]
            if not candidates:
                continue
            native = shape(row.shape.__geo_interface__)
            if not native.is_valid or native.is_empty:
                continue
            matched = []
            for mine in candidates:
                if not native.intersects(mine["search"]):
                    continue
                overlap = native.intersection(mine["native"]).area
                item = {"source_row_index_zero_based": index, "source_properties": properties,
                        "source_polygon_area_m2": native.area,
                        "overlap_historic_quarry_m2": overlap,
                        "overlap_fraction_of_source_polygon": overlap / native.area,
                        "relation": "strict_intersection" if overlap > 0 else "within_100m_only",
                        "accident_label_admitted": False}
                mine["matches"].append(item)
                matched.append({"osm_way_id": mine["osm_way_id"], **item})
            if matched:
                features.append({"type": "Feature", "geometry": transform_geom(crs, "EPSG:4326", mapping(native)),
                                 "properties": {"source_archive": archive_path.name, "source_row_index_zero_based": index,
                                                "original_properties": properties, "mine_matches": matched,
                                                "full_original_polygon_preserved": True,
                                                "accident_label_admitted": False, "event_group": "TW_HUALIEN20240403"}})
    clean_mines = [{key: value for key, value in mine.items() if key not in {"native", "geometry", "search"}} for mine in mines]
    report = {"completed_at": now(), "historic_snapshot_requested": "2024-04-01T00:00:00Z",
              "historic_source_sha256": hashlib.sha256(historic_path.read_bytes()).hexdigest(),
              "official_source_sha256": source_receipt["sha256"], "mines": clean_mines,
              "new_mine_accident_labels_admitted": 0, "new_model_predictions": 0,
              "osm_role": "dated volunteered quarry working-footprint hint; not certified mine permit or accident truth",
              "event_groups": 1, "independent_mine_count_not_inferred_from_polygon_count": True,
              "split_assigned": False}
    write_json(ROOT / "qualification" / "historical_quarry_intersections.json", report)
    write_json(ROOT / "qualification" / "historical_quarry_reference_polygons.geojson", {"type": "FeatureCollection", "features": features})
    write_json(ROOT / "qualification" / "historical_quarry_footprints.geojson", {"type": "FeatureCollection", "features": [
        {"type": "Feature", "geometry": mapping(mine["geometry"]), "properties": {"osm_way_id": mine["osm_way_id"], **mine["tags"]}} for mine in mines]})
    print(json.dumps({"mines": [{"osm_way_id": mine["osm_way_id"], "name": mine["tags"].get("name"),
                                 "strict_intersection_polygons": sum(row["relation"] == "strict_intersection" for row in mine["matches"]),
                                 "nearby_only_polygons": sum(row["relation"] != "strict_intersection" for row in mine["matches"])} for mine in mines],
                      "new_mine_accident_labels_admitted": 0}, ensure_ascii=False))


def partial_reference():
    """Admit a published object reference, without declaring a complete event."""
    import numpy as np
    import rasterio
    from rasterio.features import rasterize

    case_id = "TW_ZHONGHE2024_SOURCE_CANDIDATE"
    folder = ROOT / "imagery" / case_id
    qa = json.loads((ROOT / "qualification" / "zhonghe_image_reference_qa.json").read_text(encoding="utf-8"))
    if not qa["native_grids_identical"] or qa["new_model_predictions"] != 0:
        raise ValueError("source image qualification not complete")
    rescue_source = ROOT / "sources" / "hualien_official_zhonghe_rescue_record.pdf"
    rescue = json.loads((ROOT / "qualification" / "rescue_report_evidence.json").read_text(encoding="utf-8"))
    if hashlib.sha256(rescue_source.read_bytes()).hexdigest() != rescue["source"]["sha256"]:
        raise ValueError("documentary evidence changed")
    if not any("中和礦場" in "".join(row["text"].split()) and "崩塌" in row["text"] for row in rescue["relevant_pages"]):
        raise ValueError("independent record does not support mine slope collapse")
    original = json.loads((ROOT / "qualification" / "historical_quarry_reference_polygons.geojson").read_text(encoding="utf-8"))
    chosen = []
    for feature in original["features"]:
        matches = [row for row in feature["properties"]["mine_matches"] if row["osm_way_id"] == 511661858]
        properties = feature["properties"]["original_properties"]
        exact_dates = properties["BeforeDate"] == "20240214" and properties["AfterDate"] == "20240429"
        exact_sensors = properties["BeforeImag"] == "Sentinel-2" and properties["AfterImage"] == "Sentinel-2"
        if exact_dates and exact_sensors and any(row["overlap_fraction_of_source_polygon"] >= 0.5 for row in matches):
            chosen.append(feature)
    if len(chosen) != 1:
        raise ValueError("source-only object eligibility is ambiguous; review before admission")
    with rasterio.open(folder / "post_provider_BGRN_native10m.tif") as src:
        profile = src.profile.copy()
        native_shape, grid, crs = src.shape, src.transform, src.crs
    clear = np.ones(native_shape, dtype=bool)
    for epoch in ["pre", "post"]:
        with rasterio.open(folder / f"{epoch}_SCL_nearest_on10m.tif") as src:
            clear &= ~np.isin(src.read(1), [0, 1, 3, 7, 8, 9, 10, 11])
        with rasterio.open(folder / f"{epoch}_provider_BGRN_native10m.tif") as src:
            clear &= np.all(src.read() > 0, axis=0)
    geom = shape(transform_geom("EPSG:4326", crs, chosen[0]["geometry"]))
    sensitivity = {}
    output = ROOT / "references" / "TW_ZHONGHE2024_PARTIAL"
    output.mkdir(parents=True, exist_ok=True)
    for erosion in [0, 10, 20, 30]:
        inner = geom.buffer(-erosion) if erosion else geom
        mask = rasterize([(mapping(inner), 1)], out_shape=native_shape, transform=grid, dtype="uint8") > 0 if not inner.is_empty else np.zeros(native_shape, bool)
        sensitivity[str(erosion)] = {"native10m_pixels": int(mask.sum()), "joint_clear_pixels": int((mask & clear).sum())}
        if erosion == 10:
            tri_state = np.full(native_shape, 255, dtype=np.uint8)
            tri_state[mask & clear] = 1
            mask_path = output / "published_object_core10m_unknown_elsewhere.tif"
            with rasterio.open(mask_path, "w", **{**profile, "count": 1, "dtype": "uint8", "nodata": 255}) as dst:
                dst.write(tri_state, 1)
                dst.update_tags(role="qualification_partial_reference_split_unassigned",
                                legend="1=published positive core;255=unknown;0=negative (none supplied)",
                                reference_row="1486 zero-based", boundary_erosion_m="10",
                                erosion_is_not_certified_geolocation_error="true", training_allowed="false")
    write_json(output / "published_object_original_geometry.geojson", {"type": "FeatureCollection", "features": chosen})
    report = {
        "completed_at": now(), "case_id": "TW_ZHONGHE2024_PARTIAL", "label_level": "B_published_object_reference",
        "source_row_index_zero_based": chosen[0]["properties"]["source_row_index_zero_based"],
        "independently_mapped_collapse_object_reference": True, "reference_object_count": 1,
        "source_reference_area_m2": chosen[0]["properties"]["original_properties"]["Area_ha"] * 10000,
        "mine_context": "named earthquake disaster mine; overlap with dated OSM working-footprint hint and source imagery visually inspected",
        "documentary_source": {"path": str(rescue_source), "sha256": rescue["source"]["sha256"], "pdf_pages_one_based": [2, 3]},
        "published_label_source": "https://data.gov.tw/dataset/173374",
        "published_label_licence": "Taiwan Open Government Data Licence, version 1.0 (source catalogue)",
        "auxiliary_geometry_licence": "OpenStreetMap ODbL; copyright OpenStreetMap contributors",
        "geometric_reference": "original independent remotely interpreted object; not a new field survey",
        "subtype": "collapse/mass movement; rockfall versus sliding subtype not independently established",
        "complete_mine_accident_extent_verified": False,
        "absolute_ground_registration_error_certified": False,
        "erosion_sensitivity_m": sensitivity,
        "core_mask": {"path": str(mask_path), "sha256": hashlib.sha256(mask_path.read_bytes()).hexdigest(),
                      "legend": {"0": "verified negative (zero supplied)", "1": "published positive core", "255": "unknown"},
                      "positive_pixels": int((tri_state == 1).sum()), "negative_pixels": int((tri_state == 0).sum()),
                      "unknown_pixels": int((tri_state == 255).sum()),
                      "uncertainty": "10 m erosion is a declared sensitivity choice, not a proof of subpixel boundary accuracy"},
        "event_group": "TW_HUALIEN20240403", "split": "unassigned_blocked",
        "training_allowed": False, "final_blind_test_allowed": False,
        "pending_for_training_or_evaluation": ["冻结用途及按矿区/事件隔离的数据划分",
            "核对全部训练来源的地理重叠；LMHLD作者说明台湾来源为2018，与本2024事件年份不同，但全量源图块地理独立尚未证明",
            "COG辐射偏移约定核查；本轮不计算物理指数",
            "明确对象检出或部分监督用途；不按完整事故边界评分"],
        "new_detector_predictions": 0, "new_fits": 0,
    }
    if report["core_mask"]["negative_pixels"] != 0:
        raise ValueError("unknown reference area was incorrectly turned into negative labels")
    # Independently mapped surrounding collapses have a useful, distinct role:
    # auxiliary landslide context. They are not additional mining accidents.
    auxiliary = []
    for feature in original["features"]:
        matches = [row for row in feature["properties"]["mine_matches"] if row["osm_way_id"] == 511661858]
        props = feature["properties"]["original_properties"]
        if (matches and all(row["relation"] == "within_100m_only" for row in matches)
                and props["BeforeDate"] == "20240214" and props["AfterDate"] == "20240429"):
            auxiliary.append(feature)
    auxiliary_counts = {}
    for erosion in [10, 20, 30]:
        union_mask = np.zeros(native_shape, bool)
        for feature in auxiliary:
            inner = shape(transform_geom("EPSG:4326", crs, feature["geometry"])).buffer(-erosion)
            if not inner.is_empty:
                union_mask |= rasterize([(mapping(inner), 1)], out_shape=native_shape, transform=grid, dtype="uint8") > 0
        auxiliary_counts[str(erosion)] = int((union_mask & clear).sum())
        if erosion == 20:
            mask_path_aux = output / "surrounding_collapse_auxiliary_core20m_unknown_elsewhere.tif"
            values = np.full(native_shape, 255, dtype=np.uint8)
            values[union_mask & clear] = 1
            with rasterio.open(mask_path_aux, "w", **{**profile, "count": 1, "dtype": "uint8", "nodata": 255}) as dst:
                dst.write(values, 1)
                dst.update_tags(role="auxiliary_surrounding_landslide_reference_not_mine_accident_truth",
                                training_allowed="false", legend="1=published positive core;255=unknown;no negatives")
    write_json(output / "surrounding_collapse_auxiliary_geometry.geojson", {"type": "FeatureCollection", "features": auxiliary})
    report["auxiliary_surrounding_collapse_reference"] = {
        "role": "mapped surrounding mass-movement context; not counted as mine accident objects",
        "object_count": len(auxiliary), "source_rows": [f["properties"]["source_row_index_zero_based"] for f in auxiliary],
        "joint_clear_positive_pixels_by_erosion_m": auxiliary_counts,
        "mask": str(mask_path_aux), "mask_sha256": hashlib.sha256(mask_path_aux.read_bytes()).hexdigest(),
        "negative_pixels": 0, "training_allowed": False,
    }
    write_json(output / "qualification.json", report)
    print(json.dumps({key: report[key] for key in ["case_id", "label_level", "reference_object_count", "erosion_sensitivity_m", "auxiliary_surrounding_collapse_reference", "training_allowed"]}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    if sys.argv[1:] == ["--review-overlay"]:
        review_overlay()
    elif sys.argv[1:] == ["--review-zhonghe"]:
        review_overlay("zhonghe")
    elif sys.argv[1:] == ["--mine-intersections"]:
        mine_intersections()
    elif sys.argv[1:] == ["--partial-reference"]:
        partial_reference()
    elif not sys.argv[1:]:
        main()
    else:
        raise SystemExit("usage: screen_hualien.py [--review-overlay|--mine-intersections]")
