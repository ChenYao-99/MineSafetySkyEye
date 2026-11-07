"""Nominal CRS clarification and scoped plan context; never creates safety labels."""
from __future__ import annotations
import io
import json
import sys
from pathlib import Path
from zipfile import ZipFile

from qualify_mine_context import DATA, OLD, HERE, read, write, sha, now
sys.path.insert(0, str(HERE.parent / "phase7_20261015/vendor"))
import shapefile
from shapely.geometry import Polygon, MultiLineString, mapping, shape
from rasterio.crs import CRS
from rasterio.warp import transform_geom


def clarify_nominal_crs():
    archive_path = OLD / "sources/official_2022_inventory.zip"
    receipt = read(OLD / "receipts/official_2022_inventory.zip.json")
    if sha(archive_path) != receipt["sha256"]:
        raise ValueError("Original inventory changed")
    with ZipFile(archive_path) as archive:
        base = next(name[:-4] for name in archive.namelist() if name.lower().endswith(".shp"))
        wkt = archive.read(base + ".prj").decode("utf-8-sig")
        original = CRS.from_wkt(wkt)
        expected = {"proj": "tmerc", "lat_0": 0, "lon_0": 121, "k": .9999,
                    "x_0": 250000, "y_0": 0, "ellps": "GRS80", "units": "m"}
        if not wkt.startswith('PROJCS["TM2_TWD97"') or any(original.to_dict().get(key) != value for key, value in expected.items()):
            raise ValueError("Nominal TWD97 interpretation lacks the exact source name/parameters")
        reader = shapefile.Reader(shp=io.BytesIO(archive.read(base + ".shp")),
                    shx=io.BytesIO(archive.read(base + ".shx")), dbf=io.BytesIO(archive.read(base + ".dbf")),
                    encoding=archive.read(base + ".cpg").decode("ascii").strip())
        native = reader.shape(46).__geo_interface__
        existing = shape(transform_geom(original, "EPSG:32651", native))
        explicit = shape(transform_geom("EPSG:3826", "EPSG:32651", native))
        difference = existing.hausdorff_distance(explicit)
    nlsc = DATA / "sources/nlsc_official_coordinate_system.html"
    if not nlsc.exists():
        raise ValueError("Official datum explanation missing")
    record = {"completed_at": now(), "source_archive": str(archive_path), "source_archive_sha256": receipt["sha256"],
              "original_wkt_preserved": wkt, "original_authority_auto_recognition": original.to_epsg(),
              "source_declares_nominal_twd97": True, "projection_parameters_checked": expected,
              "nominal_crs_interpretation": "EPSG:3826 (TWD97 / TM2 zone 121)",
              "interpretation_basis": "Original CRS explicitly named TM2_TWD97, exact TM2 zone 121 / GRS80 parameters, and NLSC official datum definition",
              "official_datum_source": "https://www.nlsc.gov.tw/en/cp.aspx?Create=1&n=2122",
              "official_datum_source_sha256": sha(nlsc),
              "nominal_definition_clarified": True,
              "difference_from_previous_coarse_transform_m_for_selected_object": difference,
              "transform_equivalence_is_not_absolute_registration_accuracy": True,
              "survey_epoch_and_pixel_registration_independently_certified": False,
              "source_coordinates_modified": False, "old_qualification_snapshots_modified": False,
              "training_allowed": False,
              "remaining_blockers": ["official mine/working-face identity", "original high-resolution reference or equivalent spatial validation", "only 5 diagnostic core pixels at 10m erosion"]}
    if difference > .01:
        raise ValueError("Nominal reinterpretation materially changes the existing coarse location")
    write(DATA / "qualification/guohua2022_nominal_crs_clarification.json", record)
    print(json.dumps({"nominal_crs_definition_clarified": True, "coarse_transform_difference_m": difference,
                      "absolute_registration_certified": False}), flush=True)


def qualify_plan():
    source = read(DATA / "qualification/xincheng_plan_native_vectors.json")
    polygon = Polygon(source["linearly_connected_plan_outline_xy"])
    if not polygon.is_valid or polygon.is_empty:
        raise ValueError("Cannot silently repair an interpreted planned-work outline")
    if not 10000 < polygon.area < 2e6:
        raise ValueError("Planned-work area outside the expected source map budget")
    role = {"mine_group": "TW_XINCHENG", "source_pdf": source["source_pdf"], "source_pdf_sha256": source["source_pdf_sha256"],
            "pdf_page_one_based": 47, "source_drawing_date": source["drawing_date"], "plan_year": 2025,
            "semantic_role": source["semantic_role"], "plan_outline_area_m2": polygon.area,
            "max_interpolated_dash_gap_m": source["max_interpolated_dash_gap_m"],
            "internal_pdf_control_residual_max_m": max(row["residual_m"] for row in source["independent_control_symbol_checks"]),
            "actual_completed_work_confirmed": False, "safe_negative_label": False, "accident_label": False,
            "training_allowed": False, "use_as_2024_working_face_truth_allowed": False}
    linework = MultiLineString(source["original_planned_boundary_dash_segments_xy"])
    features = [{"type": "Feature", "geometry": transform_geom("EPSG:3826", "EPSG:4326", mapping(geom)),
                 "properties": {**role, "geometry_role": name}}
                for name, geom in [("source_planned_boundary_dash_linework", linework),
                                   ("interpreted_connected_planned_outline", polygon)]]
    output = DATA / "qualification/xincheng_2025_planned_work_context.geojson"
    write(output, {"type": "FeatureCollection", "features": features,
                  "no_plan_polygon_is_accepted_as_negative_or_accident_truth": True})
    old_path = OLD.parent / "public_reference_screen_20261015/qualification/historical_quarry_reference_polygons.geojson"
    old = read(old_path)
    comparisons = []
    for row in old["features"]:
        matches = [match for match in row["properties"]["mine_matches"] if match["osm_way_id"] == 107457823]
        if not matches:
            continue
        original = shape(transform_geom("EPSG:4326", "EPSG:3826", row["geometry"]))
        overlap = original.intersection(polygon).area
        comparisons.append({"source_row_zero_based": row["properties"]["source_row_index_zero_based"],
                            "previous_osm_relation": matches[0]["relation"], "source_object_area_m2": original.area,
                            "overlap_with_2025_planned_outline_m2": overlap,
                            "overlap_fraction": overlap / original.area,
                            "different_year_comparison_is_not_2024_failure_or_safety_truth": True})
    write(DATA / "qualification/plan_scope_diagnostic.json", {"completed_at": now(), "context": role,
          "old_2024_source_objects": comparisons, "scope": "diagnostic scope distinction only; 2025 plan vs 2024 event",
          "does_not_resolve_2024_disputed_labels_by_itself": True,
          "no_existing_source_geometry_modified": True, "new_pixel_labels": 0, "model_predictions": 0})
    print(json.dumps({"plan_context_area_m2": polygon.area, "old_scope_objects_compared": len(comparisons),
                      "strictly_intersect_plan": sum(row["overlap_with_2025_planned_outline_m2"] > 0 for row in comparisons),
                      "new_pixel_labels": 0}), flush=True)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    clarify_nominal_crs()
    qualify_plan()
