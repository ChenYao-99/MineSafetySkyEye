"""Expand independent public event leads before any additional model fitting.

New masks remain qualification-only. Mine/land-use overlap is a discovery signal,
not proof of a mining accident or a normal-operation negative.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import io
import json
import sys
from collections import Counter
from datetime import datetime, timezone, timedelta
from pathlib import Path
from urllib.parse import quote, urlencode
from zipfile import ZipFile

GEO = Path(r"D:\GeoMMAgent")
DET = Path(r"D:\LMHLD")
HERE = Path(__file__).resolve().parent
DATA = DET / "datasets/event_expansion_20261015"
SPEC = importlib.util.spec_from_file_location("public_source_reader", GEO / "maintenance/phase7_20261015/public_reference_screen.py")
ACQ = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ACQ)
ACQ.ROOT = DATA
sys.path.insert(0, str(GEO / "maintenance/phase7_20261015/vendor"))
import shapefile
from shapely.geometry import Polygon, box, shape, mapping
from shapely.strtree import STRtree
from rasterio.crs import CRS
from rasterio.warp import transform_geom


def now():
    return datetime.now(timezone(timedelta(hours=8))).isoformat(timespec="seconds")


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    ACQ.write_json(path, value)


def acquire(name, url, cap=4, direct=False):
    receipt_path = DATA / "receipts" / (name + ".json")
    if receipt_path.exists():
        receipt = read(receipt_path)
        if receipt.get("ok") and sha(receipt["path"]) == receipt["sha256"]:
            if receipt["requested_url"] != url:
                raise ValueError("Retained source name already belongs to a different URL; use a date-specific name")
            return receipt
    record = ACQ.acquire(name, url, cap * 1024 * 1024, direct=direct)
    print(json.dumps({key: record.get(key) for key in ["name", "ok", "bytes", "error"]}, ensure_ascii=False), flush=True)
    return record


def initialise():
    DATA.mkdir(parents=True, exist_ok=True)
    path = HERE / "protocol.json"
    if path.exists():
        return
    candidate = read(DET / "experiments/partial_reference_pilot_20261015/offline_candidate.json")
    prior = DET / "datasets/public_reference_screen_20261015"
    known_footprints = read(prior / "qualification/historical_quarry_footprints.geojson")
    seen_windows = []
    for case in ["TW_XINCHENG2024_SOURCE_CANDIDATE", "TW_ZHONGHE2024_SOURCE_CANDIDATE"]:
        selected = read(prior / "imagery" / case / "acquisition_plan.json")
        seen_windows.append({"case_id": case, "bounds": selected["bbox_wgs84"]})
    protocol = {
        "registered_at": now(), "objective": "补不同事件、不同矿区的独立参考及作业证据，先资格核查后定用途",
        "data_root": str(DATA), "new_model_predictions_allowed": False, "new_training_allowed": False,
        "default_model": {"path": str(DET / "models_prior/real_ft_T1b_ours_seed42/best_model.pth"),
                          "sha256": "0d8d8a0540419d4409951ed91e69499fac206b68303f3feec08a32f0841cebef"},
        "candidate_model": candidate["artifact"],
        "protected_registry_hashes": [{"path": str(p), "sha256": sha(p)} for p in [
            DET / "datasets/input_reserve_registry.json", DET / "datasets/mine_benchmark_20261002/mine_case_registry.json"]],
        "prior_development_event_groups": ["TW_HUALIEN20240403"],
        "previously_qualified_mine_way_ids": [feature["properties"]["osm_way_id"] for feature in known_footprints["features"]],
        "previously_observed_windows": seen_windows,
        "new_source_years": [2023, 2025],
        "selection_without_model_scores": True,
        "candidate_rules": ["原始独立多边形、坐标系、日期及来源保留",
            "矿业地物相交只用于发现；区分邻近自然山坡、采场边坡、采掘变化与范围争议",
            "无编目记录不代表无事故；不得生成安全负例",
            "旧矿区/旧观察窗口与不同年份同矿不得冒充新跨矿盲测",
            "按事件和矿区分组；预测前再冻结具体用途及验收协议",
            "未经核查的类别和缺失日期保持未知，不因条数不足而放宽"],
        "download_caps_mib": {"metadata": 4, "each_inventory_archive": 96, "osm_geometry": 8},
        "outbound": "仅公开来源读取；无外部生成模型调用和项目资料外发",
    }
    write(path, protocol)
    write(HERE / "state.json", {"client_work_date": "2026-10-15", "status": "new_event_and_operation_reference_qualification_in_progress",
                               "protocol": str(path), "data_root": str(DATA), "new_training": 0,
                               "new_model_predictions": 0, "new_cloud_generation_calls": 0,
                               "original_plan_alignment": "继续数据资格与不同事件扩展；未提前开展多尺度或盲测",
                               "user_inputs_required_now": []})


def fetch_inventories():
    initialise()
    record = acquire("official_2023_inventory.zip", "https://data.moa.gov.tw/GetOpenDataFile.aspx?id=J04&FileType=SHP&RID=39301", 96)
    if not record["ok"]:
        raise ValueError("2023 original source not acquired")
    record = acquire("official_2025_index.json", "https://data.moa.gov.tw/api/FileToJson.ashx?DataId=J88&DataTable=OpenDataList")
    if not record["ok"]:
        raise ValueError("2025 source index not acquired")
    index = read(record["path"])
    if len(index) != 1 or not index[0]["FileUrl"].startswith("https://data.moa.gov.tw/"):
        raise ValueError("Review ambiguous source version before downloading")
    record = acquire("official_2025_inventory.zip", index[0]["FileUrl"], 96)
    if not record["ok"]:
        raise ValueError("2025 original source not acquired")


def fetch_quarries():
    initialise()
    merged = DATA / "qualification/quarry_discovery_merged.json"
    alternative = DATA / "receipts/quarry_discovery_alternative.json.json"
    if merged.exists() and alternative.exists():
        receipt = read(alternative)
        if receipt.get("ok") and sha(receipt["path"]) == receipt["sha256"]:
            saved = read(merged)
            expected = read(receipt["path"])
            if (not expected.get("remark") and
                {row["id"] for row in saved["elements"]} == {row["id"] for row in expected["elements"]}):
                print(json.dumps({"reused_verified_discovery": len(saved["elements"])}), flush=True)
                return
    boxes = {"sw": (21.8, 119.8, 24.1, 121.0), "nw": (24.1, 119.8, 25.5, 121.0),
             "se": (21.8, 121.0, 24.1, 122.2), "ne": (24.1, 121.0, 25.5, 122.2)}
    elements, coverage = {}, []
    for key, extent in boxes.items():
        query = '[out:json][timeout:25];way["landuse"="quarry"](' + ','.join(map(str, extent)) + ');out geom;'
        url = "https://overpass-api.de/api/interpreter?data=" + quote(query, safe="()")
        record = acquire(f"quarry_discovery_{key}.json", url, 8, direct=True)
        if record["ok"]:
            body = read(record["path"])
            if body.get("remark"):
                raise ValueError("Partial Overpass result cannot be treated as complete coverage")
            for element in body["elements"]:
                elements[element["id"]] = element
        coverage.append({"box": key, "extent_south_west_north_east": extent, "acquired": record["ok"]})
    if not elements:
        query = '[out:json][timeout:25];way["landuse"="quarry"](21.8,119.8,25.5,122.2);out geom;'
        record = acquire("quarry_discovery_alternative.json", "https://overpass.private.coffee/api/interpreter?data=" + quote(query, safe="()"), 8, direct=True)
        if record["ok"]:
            body = read(record["path"])
            if not body.get("remark"):
                elements.update({row["id"]: row for row in body["elements"]})
                coverage = [{"box": "whole_selected_island", "acquired": True}]
    if not elements:
        fallback = DET / "datasets/public_reference_screen_20261015/sources/east_taiwan_osm_quarry_candidates.json"
        elements = {row["id"]: row for row in read(fallback)["elements"]}
        coverage.append({"box": "previous_verified_east_taiwan_subset", "acquired": True,
                         "bounds_south_west_north_east": [23.8, 121.3, 24.5, 121.9],
                         "source_path": str(fallback), "source_sha256": sha(fallback),
                         "not_whole_island_coverage": True})
    write(DATA / "qualification/quarry_discovery_merged.json", {
        "elements": sorted(elements.values(), key=lambda row: row["id"]), "coverage": coverage,
        "coverage_complete_for_selected_boxes": all(row["acquired"] for row in coverage),
        "provider": "OpenStreetMap contributors via Overpass", "licence": "ODbL",
        "mine_identity_or_accident_truth": False})
    print(json.dumps({"discovery_quarry_ways": len(elements), "coverage": coverage}, ensure_ascii=False), flush=True)


def fetch_history_and_catalogues():
    """Qualify the selected 2023 lead without altering the discovery rule."""
    discovery = read(DATA / "qualification/mining_event_discovery.geojson")
    ids = sorted({match["quarry_way_id"] for feature in discovery["features"]
                  for match in feature["properties"]["mine_matches"]})
    query = ('[out:json][timeout:25][date:"2023-07-13T00:00:00Z"];(' +
             ''.join(f'way({way_id});' for way_id in ids) + ');out meta geom;')
    receipt = acquire("quarry_candidates_20230713.json",
                      "https://overpass.private.coffee/api/interpreter?data=" + quote(query, safe="()"), 8, direct=True)
    body = read(receipt["path"]) if receipt["ok"] else {"elements": []}
    if body.get("remark"):
        raise ValueError("Partial historic result")
    footprints = {row["id"]: row for row in body["elements"]}
    rows = []
    for feature in discovery["features"]:
        props = feature["properties"]
        native = shape(transform_geom("EPSG:4326", "EPSG:3826", feature["geometry"]))
        for match in props["mine_matches"]:
            historic = footprints.get(match["quarry_way_id"])
            record = {"source_year": props["source_year"], "source_row_zero_based": props["source_row_zero_based"],
                      "way_id": match["quarry_way_id"], "name": match["quarry_tags"].get("name"),
                      "historical_footprint_found": bool(historic), "mining_accident_label_admitted": False}
            if historic:
                geom = Polygon([(point["lon"], point["lat"]) for point in historic["geometry"]])
                if not geom.is_valid:
                    raise ValueError("Invalid historical footprint")
                mine = shape(transform_geom("EPSG:4326", "EPSG:3826", mapping(geom)))
                record.update(historic_version=historic.get("version"), historic_timestamp=historic.get("timestamp"),
                              historic_tags=historic.get("tags"), historic_overlap_m2=native.intersection(mine).area,
                              historic_overlap_fraction=native.intersection(mine).area / native.area,
                              distance_m=native.distance(mine), historic_bounds_wgs84=list(geom.bounds))
            rows.append(record)
    write(DATA / "qualification/historical_relations.json", {
        "source": receipt, "requested_snapshot": "2023-07-13T00:00:00Z", "records": rows,
        "role": "OSM historical land-use hints only; not certified accident boundaries", "admitted": 0})
    # Exact dates and 600 m context are selected from the original source, before
    # any model scores. The entire original reference object remains intact.
    lead = next(feature for feature in discovery["features"] if feature["properties"]["source_row_zero_based"] == 217)
    native = shape(transform_geom("EPSG:4326", "EPSG:3826", lead["geometry"]))
    bbox = list(shape(transform_geom("EPSG:3826", "EPSG:4326", mapping(native.buffer(600)))).bounds)
    plan = {"case_id": "TW_DAYU2023_SOURCE_CANDIDATE", "bbox_wgs84": bbox,
            "source_year": 2023, "source_row_zero_based": 217, "event_group": "TW_HAIKUI2023",
            "selection": "highest nonzero current quarry intersection from frozen discovery; no model scores",
            "source_dates": ["2023-07-14", "2023-09-12"], "label_scope": "qualification_only",
            "reference_interval_is_not_exact_failure_date": True,
            "mining_accident_truth_confirmed": False, "predictions_allowed": False, "training_allowed": False}
    path = DATA / "qualification/dayu_source_selection.json"
    if path.exists() and read(path) != plan:
        raise ValueError("Frozen source plan differs")
    write(path, plan)
    for date in plan["source_dates"]:
        params = {"collections": "sentinel-2-l2a", "bbox": ','.join(map(str, bbox)),
                  "datetime": f"{date}T00:00:00Z/{date}T23:59:59Z", "limit": 30}
        acquire(f"dayu_{date.replace('-', '')}_catalogue.json",
                "https://earth-search.aws.element84.com/v1/search?" + urlencode(params), direct=True)
    # Read the author's public link as source data only. No remote JS execution.
    acquire("operation_author_public_script_page.html",
            "https://code.earthengine.google.com/aa655a2049776d31835d5739fdd18d00", direct=True)
    print(json.dumps({"historical_relations": rows, "source_plan": plan}, ensure_ascii=False), flush=True)


def date_value(value):
    if value is None:
        return None
    try:
        if isinstance(value, (int, float)):
            value = str(int(value)) if float(value).is_integer() else ""
        value = str(value).strip().replace("/", "").replace("-", "")
        return datetime.strptime(value, "%Y%m%d").date()
    except (ValueError, OverflowError):
        return None


def fetch_earlier_extension():
    sources = {
        2020: {"dataset": "https://data.gov.tw/dataset/152085", "url": "https://data.moa.gov.tw/GetOpenDataFile.aspx?FileType=SHP&RID=26714&id=I24"},
        2021: {"dataset": "https://data.gov.tw/dataset/159096", "url": "https://data.moa.gov.tw/GetOpenDataFile.aspx?FileType=SHP&RID=13299&id=I29"},
        2022: {"dataset": "https://data.gov.tw/dataset/160497", "url": "https://data.moa.gov.tw/GetOpenDataFile.aspx?FileType=SHP&RID=16179&id=I41"},
    }
    path = HERE / "source_extension_protocol.json"
    if not path.exists():
        write(path, {"registered_at": now(), "parent_protocol_sha256": sha(HERE / "protocol.json"),
                     "reason": "2023/2025 strict discovery produced no fresh priority objects; broaden event years rather than relax overlap or use scores",
                     "sources": sources, "mine_discovery_sha256": sha(DATA / "qualification/quarry_discovery_merged.json"),
                     "unchanged_discovery_rules": {"priority_overlap_fraction": .5, "nearby_distance_m": 100},
                     "max_archives": 3, "per_archive_cap_mib": 96, "selection_uses_model_scores": False,
                     "training_allowed": False, "predictions_allowed": False})
    for year, source in sources.items():
        record = acquire(f"official_{year}_inventory.zip", source["url"], 96)
        if not record["ok"]:
            raise ValueError(f"Earlier source {year} unavailable")
    inspect([2020, 2021, 2022])
    join_references([2020, 2021, 2022], "earlier_mining_event_discovery")


def join_references(years=None, prefix="mining_event_discovery"):
    protocol = read(HERE / "protocol.json")
    discovery = read(DATA / "qualification/quarry_discovery_merged.json")
    mines = []
    for element in discovery["elements"]:
        coordinates = [(row["lon"], row["lat"]) for row in element.get("geometry", [])]
        if len(coordinates) < 4 or coordinates[0] != coordinates[-1]:
            continue
        geom = Polygon(coordinates)
        if not geom.is_valid or geom.is_empty:
            continue
        native = shape(transform_geom("EPSG:4326", "EPSG:3826", mapping(geom)))
        reasons = []
        if element["id"] in protocol["previously_qualified_mine_way_ids"]:
            reasons.append("previously_qualified_mine_group")
        for window in protocol["previously_observed_windows"]:
            if geom.intersects(box(*window["bounds"])):
                reasons.append("previously_observed_window:" + window["case_id"])
        mines.append({"id": element["id"], "tags": element.get("tags", {}), "geographic": geom,
                      "native": native, "search": native.buffer(100), "prior_development_exclusions": reasons})
    spatial_index = STRtree([mine["search"] for mine in mines])
    features, source_summaries = [], []
    for year in years or [2023, 2025]:
        path = DATA / "sources" / f"official_{year}_inventory.zip"
        receipt = read(DATA / "receipts" / (path.name + ".json"))
        if sha(path) != receipt["sha256"]:
            raise ValueError("Source archive changed")
        counts, events = Counter(), Counter()
        with ZipFile(path) as archive:
            base = next(name[:-4] for name in archive.namelist() if name.lower().endswith(".shp"))
            original_wkt = archive.read(base + ".prj").decode("utf-8-sig")
            crs = CRS.from_wkt(original_wkt)
            legacy_datum_pending = False
            if crs.to_epsg() != 3826 and crs != CRS.from_epsg(3826):
                # Reviewed legacy files declare the same TM2 metric projection
                # but omit a recognized datum. Permit coarse discovery only;
                # do not silently certify these as pixel-aligned EPSG:3826.
                parameters = crs.to_dict()
                expected = {"proj": "tmerc", "lat_0": 0, "lon_0": 121, "k": .9999,
                            "x_0": 250000, "y_0": 0, "ellps": "GRS80", "units": "m"}
                if year not in [2020, 2022] or any(parameters.get(key) != value for key, value in expected.items()):
                    raise ValueError("Source CRS requires review")
                legacy_datum_pending = True
            reader = shapefile.Reader(shp=io.BytesIO(archive.read(base + ".shp")),
                                     shx=io.BytesIO(archive.read(base + ".shx")),
                                     dbf=io.BytesIO(archive.read(base + ".dbf")),
                                     encoding=archive.read(base + ".cpg").decode("ascii").strip())
            for index, row in enumerate(reader.iterShapeRecords()):
                counts["original_objects"] += 1
                properties = row.record.as_dict()
                events[str(properties.get("Events", properties.get("Event_s_")))] += 1
                before, after = date_value(properties.get("BeforeDate")), date_value(properties.get("AfterDate"))
                ordered = bool(before and after and before < after)
                counts["ordered_dates" if ordered else "missing_or_unordered_dates"] += 1
                candidates = [mines[int(i)] for i in spatial_index.query(box(*row.shape.bbox))]
                if not candidates:
                    continue
                native = shape(row.shape.__geo_interface__)
                if not native.is_valid or native.is_empty:
                    counts["invalid_geometry_near_mine"] += 1
                    continue
                matches = []
                for mine in candidates:
                    if not native.intersects(mine["search"]):
                        continue
                    overlap = native.intersection(mine["native"]).area
                    fraction = overlap / native.area
                    matches.append({"quarry_way_id": mine["id"], "quarry_tags": mine["tags"],
                                    "overlap_m2": overlap, "overlap_fraction": fraction,
                                    "relation": "strict_intersection" if overlap > 0 else "within_100m_only",
                                    "prior_development_exclusions": mine["prior_development_exclusions"],
                                    "historical_footprint_verified": False,
                                    "source_only_priority_candidate": fraction >= .5 and ordered and not mine["prior_development_exclusions"]})
                if not matches:
                    continue
                counts["objects_with_mine_or_nearby_relation"] += 1
                counts["objects_with_fresh_priority_relation"] += int(any(match["source_only_priority_candidate"] for match in matches))
                geographic = transform_geom(crs, "EPSG:4326", mapping(native))
                features.append({"type": "Feature", "geometry": geographic, "properties": {
                    "source_year": year, "source_archive": path.name, "source_row_zero_based": index,
                    "original_properties": properties, "reference_polygon_area_m2": native.area,
                    "before_date": before.isoformat() if before else None,
                    "after_date": after.isoformat() if after else None,
                    "reference_interval_days": (after - before).days if ordered else None,
                    "legacy_datum_verification_pending": legacy_datum_pending,
                    "mine_matches": matches, "mine_accident_reference_admitted": False,
                    "training_allowed": False, "prediction_allowed": False,
                    "absence_of_other_polygons_is_not_negative_evidence": True}})
        source_summaries.append({"year": year, "counts": dict(counts), "original_event_counts": dict(events),
                                 "crs_original_wkt": original_wkt, "crs_epsg": crs.to_epsg(),
                                 "legacy_datum_verification_pending": legacy_datum_pending})
    write(DATA / "qualification" / f"{prefix}.geojson", {"type": "FeatureCollection", "features": features})
    write(DATA / "qualification/quarry_discovery_polygons.geojson", {"type": "FeatureCollection", "features": [
        {"type": "Feature", "geometry": mapping(mine["geographic"]), "properties": {"way_id": mine["id"], **mine["tags"], "prior_exclusions": mine["prior_development_exclusions"]}} for mine in mines]})
    report = {"completed_at": now(), "sources": source_summaries,
              "quarry_discovery_coverage": discovery["coverage"],
              "current_footprints_only_historical_verification_pending": True,
              "new_mine_accident_reference_cases_admitted": 0,
              "new_model_predictions": 0, "new_training": 0,
              "priority_records": [{"source_year": feature["properties"]["source_year"],
                                    "source_row": feature["properties"]["source_row_zero_based"],
                                    "original_properties": feature["properties"]["original_properties"],
                                    "reference_area_m2": feature["properties"]["reference_polygon_area_m2"],
                                    "mine_matches": [row for row in feature["properties"]["mine_matches"] if row["source_only_priority_candidate"]]}
                                   for feature in features if any(row["source_only_priority_candidate"] for row in feature["properties"]["mine_matches"])]}
    write(DATA / "qualification" / f"{prefix}.json", report)
    print(json.dumps({"sources": source_summaries, "priority_records": report["priority_records"]}, ensure_ascii=False), flush=True)


def inspect(years=None):
    initialise()
    for filename in [f"official_{year}_inventory.zip" for year in years or [2023, 2025]]:
        archive_path = DATA / "sources" / filename
        receipt = read(DATA / "receipts" / (filename + ".json"))
        if sha(archive_path) != receipt["sha256"]:
            raise ValueError("Original inventory identity changed")
        with ZipFile(archive_path) as archive:
            expanded = sum(row.file_size for row in archive.infolist())
            if expanded > 512 * 1024 * 1024:
                raise ValueError("Expanded inventory exceeds the selected budget")
            shp_names = [name for name in archive.namelist() if name.lower().endswith(".shp")]
            layers = []
            for shp in shp_names:
                base = shp[:-4]
                encoding = archive.read(base + ".cpg").decode("ascii").strip()
                crs = CRS.from_wkt(archive.read(base + ".prj").decode("utf-8-sig"))
                reader = shapefile.Reader(shp=io.BytesIO(archive.read(base + ".shp")),
                                         shx=io.BytesIO(archive.read(base + ".shx")),
                                         dbf=io.BytesIO(archive.read(base + ".dbf")), encoding=encoding)
                attributes = {name: Counter() for name in [field[0] for field in reader.fields[1:]]
                              if any(tag in name.lower() for tag in ["event", "date", "source", "imag"])}
                examples = []
                for row in reader.iterRecords():
                    props = row.as_dict()
                    for name in attributes:
                        attributes[name][str(props[name])] += 1
                    if len(examples) < 1:
                        examples.append(props)
                layers.append({"layer": shp, "objects": len(reader), "crs_epsg": crs.to_epsg(),
                               "shape_type": reader.shapeTypeName, "fields": reader.fields[1:],
                               "attribute_counts": {name: dict(values) for name, values in attributes.items()},
                               "examples": examples})
            report = {"original_archive": receipt, "expanded_bytes_if_fully_extracted": expanded,
                      "layers": layers, "mining_accident_labels_admitted": 0}
        write(DATA / "qualification" / (filename + ".json"), report)
        print(json.dumps({"file": filename, "layers": [{key: value for key, value in row.items() if key in {"layer", "objects", "crs_epsg", "shape_type", "examples"}} for row in layers]}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["inventories", "quarries", "inspect", "join", "history", "earlier"])
    arguments = parser.parse_args()
    {"inventories": fetch_inventories, "quarries": fetch_quarries, "inspect": inspect, "join": join_references,
     "history": fetch_history_and_catalogues, "earlier": fetch_earlier_extension}[arguments.command]()
