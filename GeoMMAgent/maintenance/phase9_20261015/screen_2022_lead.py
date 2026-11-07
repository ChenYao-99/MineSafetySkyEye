"""Bounded 2022 mining-context source qualification, never detector evaluation."""
from datetime import datetime
from urllib.parse import urlencode
from expand_event_references import DATA, read, write, acquire, sha
from qualify_dayu_pair import history_from_original_api, acquire_pair
from shapely.geometry import shape, mapping
from rasterio.warp import transform_geom


def main():
    feature = next(row for row in read(DATA / "qualification/earlier_mining_event_discovery.geojson")["features"]
                   if row["properties"]["source_year"] == 2022 and row["properties"]["source_row_zero_based"] == 46)
    geom = shape(transform_geom("EPSG:4326", "EPSG:3826", feature["geometry"]))
    bbox = list(shape(transform_geom("EPSG:3826", "EPSG:4326", mapping(geom.buffer(600)))).bounds)
    path = DATA / "qualification/guohua2022_search_plan.json"
    policy = {"bbox_wgs84": bbox, "source_row_zero_based": 46,
              "source_discovery_sha256": sha(DATA / "qualification/earlier_mining_event_discovery.geojson"),
              "original_reference_dates": ["2022-07-13", "2022-09-21"],
              "original_reference_sensors": ["SPOT-6/7", "Pleiades"],
              "windows": {"pre": ["2022-07-01", "2022-07-13"], "post": ["2022-09-21", "2022-10-01"]},
              "selection_rule": "closest date bracketing original reference interval, covering full bbox; unique tile required; no model scores",
              "source_imagery_is_substitute_not_original_high_resolution": True,
              "legacy_datum_verification_pending": True,
              "predictions_allowed": False, "training_allowed": False}
    if path.exists() and read(path) != policy:
        raise ValueError("Search policy changed")
    write(path, policy)
    history_from_original_api(526217895, "2022-07-01T00:00:00Z", "guohua2022")
    scenes = []
    for epoch, (start, end) in policy["windows"].items():
        params = {"collections": "sentinel-2-l2a", "bbox": ','.join(map(str, bbox)),
                  "datetime": f"{start}T00:00:00Z/{end}T23:59:59Z", "limit": 30}
        record = acquire(f"guohua2022_{epoch}_search.json",
                         "https://earth-search.aws.element84.com/v1/search?" + urlencode(params), direct=True)
        if not record["ok"]:
            raise ValueError("Source catalogue unavailable")
        source = read(record["path"])
        # This catalogue emits a cursor link even on a complete short page.
        # Require the provider's explicit total and returned count to agree.
        matched = source.get("numberMatched", source.get("context", {}).get("matched"))
        if not isinstance(matched, int) or matched != len(source["features"]):
            raise ValueError("Catalogue pagination requires review")
        window = shape({"type": "Polygon", "coordinates": [[[bbox[0], bbox[1]], [bbox[2], bbox[1]],
                          [bbox[2], bbox[3]], [bbox[0], bbox[3]], [bbox[0], bbox[1]]]]})
        features = [row for row in source["features"] if shape(row["geometry"]).covers(window)]
        features.sort(key=lambda row: row["properties"]["datetime"], reverse=epoch == "pre")
        if not features:
            raise ValueError("No complete-coverage substitute scene")
        selected = features[0]
        date = selected["properties"]["datetime"][:10]
        if sum(row["properties"]["datetime"][:10] == date for row in features) != 1:
            raise ValueError("Closest-date scene is ambiguous")
        write(DATA / "sources" / f"guohua2022_{date.replace('-', '')}_catalogue.json",
              {"type": "FeatureCollection", "features": [selected], "links": [],
               "derived_from_receipt": str(DATA / "receipts" / (record["name"] + ".json")),
               "original_search_sha256": record["sha256"], "selection_rule": policy["selection_rule"]})
        scenes.append(selected)
    plan = {"case_id": "TW_GUOHUA2022_SOURCE_CANDIDATE", "bbox_wgs84": bbox, "source_year": 2022,
            "source_row_zero_based": 46, "source_prefix": "guohua2022",
            "discovery_file": "earlier_mining_event_discovery.geojson", "event_group": "TW_TAITUNG20220918",
            "selection": policy["selection_rule"], "source_dates": [row["properties"]["datetime"][:10] for row in scenes],
            "original_reference_dates": policy["original_reference_dates"],
            "original_reference_sensors": policy["original_reference_sensors"],
            "source_imagery_is_substitute_not_original_high_resolution": True,
            "legacy_datum_verification_pending": True, "label_scope": "qualification_only",
            "mining_accident_truth_confirmed": False, "predictions_allowed": False, "training_allowed": False}
    plan_path = DATA / "qualification/guohua2022_source_selection.json"
    if plan_path.exists() and read(plan_path) != plan:
        raise ValueError("Frozen selected scene changed")
    write(plan_path, plan)
    acquire_pair(plan_path.name)


def quality_fallback():
    """Use one remaining post scene in the already frozen date window."""
    old_plan = read(DATA / "qualification/guohua2022_source_selection.json")
    old_root = DATA / "imagery" / old_plan["case_id"]
    report = DATA / "qualification/guohua2022_rejected_pair.json"
    previous = read(old_root / "qualification.json") if old_root.exists() else read(report)["qualification"]
    if previous["joint_clear_fraction"] != 0:
        raise ValueError("This fallback is only for the documented cloud-rejected pair")
    if not report.exists():
        write(report, {"decision": "rejected_for_100_percent_post_cloud; no model used",
                       "acquisition_plan": read(old_root / "acquisition_plan.json"),
                       "acquisition": read(old_root / "acquisition.json"), "qualification": previous,
                       "subsequent_rule": "one remaining post scene in original date window; no search expansion"})
    source = read(DATA / "sources/guohua2022_post_search.json")
    rows = [row for row in source["features"] if row["properties"]["datetime"][:10] != old_plan["source_dates"][1]]
    if len(rows) != 1:
        raise ValueError("Exactly one remaining metadata-qualified date expected")
    selected = rows[0]
    date = selected["properties"]["datetime"][:10]
    plan = {**old_plan, "case_id": "TW_GUOHUA2022_CLEAR_SOURCE_CANDIDATE",
            "source_dates": [old_plan["source_dates"][0], date],
            "selection": "remaining post date in original frozen window after exact closest date rejected for 100% cloud; no model scores",
            "parent_source_plan_sha256": sha(DATA / "qualification/guohua2022_source_selection.json"),
            "rejected_pair_receipt_sha256": sha(report)}
    path = DATA / "qualification/guohua2022_clear_source_selection.json"
    if path.exists() and read(path) != plan:
        raise ValueError("Quality fallback plan changed")
    write(path, plan)
    write(DATA / "sources" / f"guohua2022_{date.replace('-', '')}_catalogue.json",
          {"type": "FeatureCollection", "features": [selected], "links": [],
           "derived_from_search_sha256": sha(DATA / "sources/guohua2022_post_search.json"),
           "selection_rule": plan["selection"]})
    acquire_pair(path.name)


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")
    if len(sys.argv) == 2 and sys.argv[1] == "quality-fallback":
        quality_fallback()
    else:
        main()
