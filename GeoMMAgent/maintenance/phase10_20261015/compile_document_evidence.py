"""Read original PDF figures and vector linework; never infer safety masks."""
from __future__ import annotations
import io
import json
import math
import re
import sys
from pathlib import Path

sys.path.insert(0, "D:/LMHLD")
from runtime_paths import configure_runtime_paths
configure_runtime_paths()
sys.dont_write_bytecode = True
import fitz
import numpy as np
from PIL import Image
from qualify_mine_context import DATA, read, write, sha, now


def compile_field_references():
    observations = []
    for date in ["20190410", "20190418"]:
        source = DATA / "sources" / f"operator_earthquake_inspection_{date}.pdf"
        receipt = read(DATA / "receipts" / (source.name + ".json"))
        if sha(source) != receipt["sha256"]:
            raise ValueError("Original inspection report changed")
        doc = fitz.open(source)
        text = ''.join(doc[0].get_text().split())
        if "無發生崩坍及落石" not in text:
            raise ValueError("Expected explicit scoped observation not in original report")
        page = doc[1]
        photos = []
        for entry in page.get_images(full=True):
            xref, _, width, height = entry[:4]
            if width < 300 or height < 150:
                continue
            rectangles = page.get_image_rects(xref)
            if len(rectangles) != 1:
                raise ValueError("Ambiguous original photograph placement")
            photos.append((rectangles[0].y0, xref, width, height))
        photos.sort()
        if len(photos) != 2:
            raise ValueError("The inspected page must contain exactly the two reviewed photographs")
        for figure, ((_, xref, width, height), location) in enumerate(zip(photos, ["final_wall", "north_slope"]), 1):
            original = doc.extract_image(xref)
            output = DATA / "field_references" / f"XC_{date}_{location}.{original['ext']}"
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_bytes(original["image"])
            with Image.open(io.BytesIO(original["image"])) as image:
                exif = image.getexif()
                metadata = {str(key): str(exif[key]) for key in [306, 36867, 36868, 34853] if key in exif}
            observations.append({"id": f"XC_{date}_{location}", "mine_group": "TW_XINCHENG",
                "event_group": "TW_HUALIEN201904_EARTHQUAKE_SEQUENCE", "report_date": f"{date[:4]}-{date[4:6]}-{date[6:]}",
                "inspection_location_as_named": location, "source_report": str(source),
                "source_report_sha256": receipt["sha256"], "pdf_page_one_based": 2, "figure": figure,
                "source_claim": "operator reports no observed collapse/rockfall at the specifically inspected face",
                "claim_is_operator_report_not_independent_model_validation": True,
                "image": str(output), "image_sha256": sha(output), "native_dimensions": [width, height],
                "original_embedded_image_bytes_unmodified": True, "original_exif": metadata,
                "view_type": "ground_oblique; not an orthorectified satellite image", "georeferenced": False,
                "pixel_labels_admitted": False, "training_allowed": False, "blind_test_allowed": False,
                "allowed_use": "dated/scoped source evidence and visual reference for later registration; development mine only"})
    if len({row["image_sha256"] for row in observations}) != 4:
        raise ValueError("Repeated photographs must not be counted as new dated observations")
    result = {"completed_at": now(), "observations": observations, "dated_reports": 2,
              "original_photographs": 4, "physical_inspected_face_types": 2, "independent_mine_groups": 1,
              "ground_photographs_are_not_4_independent_mines": True,
              "normal_reference_source_evidence_available": True, "verified_satellite_negative_pixels": 0}
    write(DATA / "qualification/normal_field_reference_pack.json", result)
    print(json.dumps({key: result[key] for key in ["dated_reports", "original_photographs", "physical_inspected_face_types", "independent_mine_groups", "verified_satellite_negative_pixels"]}), flush=True)


def fit_axis(pairs):
    x = np.array([pair[0] for pair in pairs], dtype=float)
    y = np.array([pair[1] for pair in pairs], dtype=float)
    slope, intercept = np.linalg.lstsq(np.column_stack([x, np.ones_like(x)]), y, rcond=None)[0]
    return float(slope), float(intercept), [float(value) for value in (slope * x + intercept - y)]


def compile_plan_vectors():
    source = DATA / "sources/xincheng_2025_annual_work_plan.pdf"
    receipt = read(DATA / "receipts" / (source.name + ".json"))
    if sha(source) != receipt["sha256"]:
        raise ValueError("Original public work plan changed")
    doc = fitz.open(source)
    page = doc[46]
    drawings = page.get_drawings()
    words = page.get_text("words")
    grid_lines = {"easting": [], "northing": []}
    for drawing in drawings:
        color = drawing.get("color")
        if color is None or any(abs(a - b) > .002 for a, b in zip(color, [0, .43137, .86667])):
            continue
        if len(drawing["items"]) != 1 or drawing["items"][0][0] != "l":
            continue
        _, start, end = drawing["items"][0]
        if abs(start.y - end.y) < 1e-4 and abs(start.x - end.x) > 400:
            grid_lines["easting"].append(float(start.y))
        if abs(start.x - end.x) < 1e-4 and abs(start.y - end.y) > 400:
            grid_lines["northing"].append(float(start.x))
    pairs = {"easting": [], "northing": []}
    for x0, y0, x1, y1, text, *_ in words:
        if text in ["312000", "313000", "314000"]:
            axis, position = "easting", (y0 + y1) / 2
        elif text in ["2670000", "2671000", "2672000"]:
            axis, position = "northing", (x0 + x1) / 2
        else:
            continue
        coord = min(grid_lines[axis], key=lambda value: abs(value - position))
        if abs(coord - position) > 8:
            raise ValueError("Grid label is not close to its original vector line")
        pairs[axis].append((coord, int(text)))
    if any(len(set(values)) != 3 for values in pairs.values()):
        raise ValueError("Require three labelled original grid lines per axis")
    easting = fit_axis(pairs["easting"])
    northing = fit_axis(pairs["northing"])
    def coordinate(point):
        return [easting[0] * point[1] + easting[1], northing[0] * point[0] + northing[1]]
    # These separately surveyed control symbols are not used to fit the grid.
    checks = []
    for name, expected in [("0132", [314085.700, 2671288.170]), ("Newlab", [313618.407, 2670841.534])]:
        predicted_pdf = [(expected[1] - northing[1]) / northing[0], (expected[0] - easting[1]) / easting[0]]
        symbols = [drawing for drawing in drawings if drawing["type"] == "s" and len(drawing["items"]) == 4
                   and abs(drawing["rect"].width - 10.62) < .04 and abs(drawing["rect"].height - 10.62) < .04]
        center = lambda drawing: [(drawing["rect"].x0 + drawing["rect"].x1) / 2,
                                  (drawing["rect"].y0 + drawing["rect"].y1) / 2]
        symbol = min(symbols, key=lambda drawing: math.dist(center(drawing), predicted_pdf))
        observed = coordinate(center(symbol))
        residual = math.dist(observed, expected)
        if residual > 2:
            raise ValueError("Independent PDF control-point check exceeds 2 m")
        checks.append({"point": name, "published_xy": expected, "symbol_xy_recovered": observed, "residual_m": residual})
    segments = []
    for drawing in drawings:
        if drawing.get("color") != (1., 0., 0.) or abs((drawing.get("width") or 0) - 2.28) > .01:
            continue
        if drawing["rect"].y0 >= 850 or (drawing["rect"].width == 0 and drawing["rect"].height == 0):
            continue  # exclude legend symbols and the zero-length closure marker
        points = []
        for item in drawing["items"]:
            if item[0] != "l":
                raise ValueError("Unreviewed nonlinear planned boundary primitive")
            if not points:
                points.append(list(item[1]))
            if math.dist(points[-1], item[1]) > .001:
                raise ValueError("Discontinuous primitive within a dash")
            points.append(list(item[2]))
        segments.append([coordinate(point) for point in points])
    if len(segments) < 20:
        raise ValueError("Insufficient original planned boundary segments")
    gaps = [math.dist(segments[i][-1], segments[(i+1) % len(segments)][0]) for i in range(len(segments))]
    if max(gaps) > 40:
        raise ValueError("Dash connection would introduce an excessive inferred gap")
    outline = [point for segment in segments for point in segment]
    outline.append(outline[0])
    result = {"completed_at": now(), "source_pdf": str(source), "source_pdf_sha256": receipt["sha256"],
              "pdf_page_one_based": 47, "drawing_date": "2025-04-11", "plan_year": 2025,
              "mine_name": "亚洲水泥新城山矿场", "mine_number": "A01705",
              "source_coordinate_system": "TM2-TWD97 zone 121; EPSG:3826 nominal interpretation",
              "pdf_rotation": page.rotation, "coordinate_space": "unrotated original PDF coordinates",
              "easting_from_pdf_y": easting[:2], "northing_from_pdf_x": northing[:2],
              "original_labelled_grid_pairs": pairs, "grid_residuals_m": {"easting": easting[2], "northing": northing[2]},
              "independent_control_symbol_checks": checks, "absolute_survey_accuracy_certified": False,
              "original_planned_boundary_dash_segments_xy": segments,
              "linearly_connected_plan_outline_xy": outline, "max_interpolated_dash_gap_m": max(gaps),
              "outline_is_interpreted_from_printed_dashes_not_a_surveyed_polygon_file": True,
              "original_line_width_m": 2.28 * (abs(easting[0]) + abs(northing[0])) / 2,
              "semantic_role": "2025 planned excavation context only; not proof of executed works or safe pixels",
              "may_be_applied_as_2024_working_face_truth": False,
              "accident_label": False, "safe_negative_label": False, "training_allowed": False}
    write(DATA / "qualification/xincheng_plan_native_vectors.json", result)
    print(json.dumps({"plan_boundary_dash_segments": len(segments), "max_gap_m": max(gaps),
                      "control_residuals_m": [row["residual_m"] for row in checks], "semantic_role": result["semantic_role"]}), flush=True)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    compile_field_references()
    compile_plan_vectors()
