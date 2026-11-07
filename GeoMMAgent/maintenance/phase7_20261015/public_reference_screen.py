"""Bounded public-data discovery; no detector calls or protected-label reads.

All downloaded content and receipts stay under D:\\LMHLD. Remote responses are
untrusted data, never code to execute. Run --probe before selecting any dataset.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import sys
import re
import sqlite3
import shutil
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zipfile import ZipFile

import requests

ROOT = Path(r"D:\LMHLD\datasets\public_reference_screen_20261015")
PROBES = {
    "uglc_zenodo_metadata": "https://zenodo.org/api/records/18643456",
    "uglc_ncdc_page": "https://www.ncdc.ac.cn/portal/metadata/04c5a9df-a452-463d-9ba7-f468528d4c8f",
    "uglc_polygon_readme": "https://raw.githubusercontent.com/UnibaGEO/UGLC_polygonal/master_poly/README.md",
}


def now():
    return datetime.now(timezone(timedelta(hours=8))).isoformat(timespec="seconds")


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def validate_format(path, name):
    with path.open("rb") as stream:
        header = stream.read(1024)
    suffix = Path(name).suffix.lower()
    expected = {".zip": b"PK\x03\x04", ".png": b"\x89PNG\r\n\x1a\n", ".gpkg": b"SQLite format 3\x00"}
    if suffix in expected and not header.startswith(expected[suffix]):
        raise ValueError(f"{suffix} file signature mismatch")
    if suffix == ".pdf" and b"%PDF-" not in header:
        raise ValueError("PDF file signature mismatch (HTML/error response is not a PDF)")
    if suffix == ".json" or name in {"uglc_zenodo_metadata", "utah_inventory_metadata"}:
        json.loads(path.read_text(encoding="utf-8-sig"))
    if suffix == ".html" or name == "uglc_ncdc_page":
        if b"<html" not in header.lower() and b"<!doctype html" not in header.lower():
            raise ValueError("expected HTML source document")
    if suffix == ".xml":
        from xml.etree import ElementTree
        ElementTree.parse(path)


def acquire(name, url, limit=4 * 1024 * 1024, direct=False):
    target = ROOT / "sources" / name
    target.parent.mkdir(parents=True, exist_ok=True)
    receipt_path = ROOT / "receipts" / (name + ".json")
    old = json.loads(receipt_path.read_text(encoding="utf-8")) if receipt_path.exists() else None
    receipt = {"name": name, "requested_url": url, "started_at": now(), "limit_bytes": limit,
               "connection": "direct_no_ambient_credentials" if direct else "system_network"}
    if old:
        receipt["previous_attempts"] = old.pop("previous_attempts", []) + [old]
    part = target.with_name(target.name + ".part")
    try:
        session = requests.Session()
        if direct:
            session.trust_env = False
        with session.get(url, timeout=(12, 30), stream=True,
                          headers={"User-Agent": "PublicResearchDataQualification/1.0"}) as response:
            receipt.update(status_code=response.status_code, resolved_url=response.url,
                           content_type=response.headers.get("Content-Type"),
                           content_length=response.headers.get("Content-Length"))
            response.raise_for_status()
            size = int(response.headers.get("Content-Length", "0"))
            if size > limit:
                raise ValueError(f"declared body {size} exceeds selected limit {limit}")
            count = 0
            digest = hashlib.sha256()
            with part.open("wb") as stream:
                for block in response.iter_content(65536):
                    count += len(block)
                    if count > limit:
                        raise ValueError("stream exceeds selected limit")
                    digest.update(block)
                    stream.write(block)
            validate_format(part, name)
            part.replace(target)
            receipt.update(ok=True, path=str(target), bytes=count, sha256=digest.hexdigest())
    except (requests.RequestException, ValueError, OSError) as exc:
        part.unlink(missing_ok=True)
        receipt.update(ok=False, error=f"{type(exc).__name__}: {exc}")
    receipt["completed_at"] = now()
    write_json(receipt_path, receipt)
    return receipt


def screen_point_tiles():
    keyword = re.compile(r"\b(?:min(?:e|es|ing)|quarr(?:y|ies)|open[- ]pit|pitwall|excavat\w*)\b", re.I)
    reports = []
    for path in sorted((ROOT / "sources").glob("UGLC_point_tile_*.gpkg")):
        receipt = json.loads((ROOT / "receipts" / (path.name + ".json")).read_text(encoding="utf-8"))
        if not receipt.get("publisher_checksum_verified") or hashlib.sha256(path.read_bytes()).hexdigest() != receipt["sha256"]:
            raise ValueError("point tile integrity not verified")
        with sqlite3.connect(path.as_uri() + "?mode=ro", uri=True) as connection:
            connection.row_factory = sqlite3.Row
            layer, *extent = connection.execute("select table_name,min_x,min_y,max_x,max_y from gpkg_contents where data_type='features'").fetchone()
            if not re.fullmatch(r"[A-Za-z0-9_]+", layer):
                raise ValueError("unexpected layer identifier")
            candidates = []
            count = 0
            for item in connection.execute('select * from "' + layer + '"'):
                count += 1
                row = {key: item[key] for key in item.keys() if key != "geom"}
                text = " ".join(str(row.get(key) or "") for key in ["NOTES", "LINK", "PHYSICAL FACTORS"])
                hits = sorted(set(match.group().lower() for match in keyword.finditer(text)))
                if hits or "anthropic" in str(row.get("PHYSICAL FACTORS")):
                    candidates.append({"original_attributes": row, "keyword_hits": hits,
                                       "mine_identity_confirmed": False, "geometry_role": "event_point_only",
                                       "accident_pixel_label_admitted": False})
        reports.append({"file": path.name, "bounds_wgs84": extent, "records": count,
                        "mining_or_anthropic_leads": candidates})
    report = {"completed_at": now(), "sources": reports, "new_mine_accident_boundaries": 0,
              "point_records_not_polygon_truth": True, "new_model_predictions": 0,
              "licence": "UGLC original publication/Zenodo CC BY-NC 4.0; original source conditions also apply"}
    write_json(ROOT / "qualification" / "uglc_point_lead_screen.json", report)
    print(json.dumps({"sources": [{"file": row["file"], "records": row["records"],
                                   "mining_or_anthropic_leads": len(row["mining_or_anthropic_leads"])}
                                  for row in reports], "new_mine_accident_boundaries": 0}, ensure_ascii=False), flush=True)


def finalize_batch():
    """Validate retained artifacts and remove only this batch's scratch files."""
    receipts = []
    removed_files = 0
    removed_bytes = 0
    verification_path = ROOT / "qualification" / "batch_verification.json"
    previous = json.loads(verification_path.read_text(encoding="utf-8")) if verification_path.exists() else {}
    for receipt_path in sorted((ROOT / "receipts").glob("*.json")):
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        if receipt.get("ok"):
            path = Path(receipt["path"]).resolve()
            if path.parent != (ROOT / "sources").resolve():
                raise ValueError("unexpected source path outside current batch")
            if hashlib.sha256(path.read_bytes()).hexdigest() != receipt["sha256"] or path.stat().st_size != receipt["bytes"]:
                raise ValueError("retained source hash/size mismatch")
            try:
                validate_format(path, receipt["name"])
                receipt["format_verified"] = True
            except (ValueError, UnicodeError) as exc:
                # The government server returned an HTML 404 page with HTTP
                # status 200. Never let that masquerade as an acquired PDF.
                if path.name != "gsmma_2024_mine_earthquake_report.pdf":
                    raise
                receipt.update(ok=False, transport_ok=True, format_verified=False,
                               error=str(exc), invalid_payload_removed=True)
                removed_bytes += path.stat().st_size
                removed_files += 1
                path.unlink()
            write_json(receipt_path, receipt)
        receipts.append({key: receipt.get(key) for key in ["name", "ok", "bytes", "sha256", "format_verified", "error"]})
    reference = json.loads((ROOT / "references" / "TW_ZHONGHE2024_PARTIAL" / "qualification.json").read_text(encoding="utf-8"))
    if reference["core_mask"]["negative_pixels"] != 0 or reference["training_allowed"] or reference["final_blind_test_allowed"]:
        raise ValueError("partial-reference use/unknown-area contract changed")
    for row in [reference["core_mask"], reference["auxiliary_surrounding_collapse_reference"]]:
        path = Path(row.get("path", row.get("mask")))
        expected = row.get("sha256", row.get("mask_sha256"))
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError("partial reference mask changed")
    registry = Path(r"D:\LMHLD\datasets\input_reserve_registry.json")
    baseline = json.loads(Path(r"D:\GeoMMAgent\maintenance\phase1_20261013\baseline.json").read_text(encoding="utf-8"))
    old_registry = baseline["reserve_registry"]
    registry_ok = hashlib.sha256(registry.read_bytes()).hexdigest() == old_registry["sha256"]
    model = Path(r"D:\LMHLD\models_prior\real_ft_T1b_ours_seed42\best_model.pth")
    model_ok = hashlib.sha256(model.read_bytes()).hexdigest() == "0d8d8a0540419d4409951ed91e69499fac206b68303f3feec08a32f0841cebef"
    if not registry_ok or not model_ok:
        raise ValueError("protected registry/default model differs from prior frozen state")
    cache = (ROOT / "_range_cache").resolve()
    transport_path = ROOT / "qualification" / "source_transport_manifest.json"
    if cache.exists():
        if cache.parent != ROOT.resolve() or cache.name != "_range_cache":
            raise ValueError("unexpected recursive cleanup target")
        manifests = [json.loads(path.read_text(encoding="utf-8")) for path in sorted(cache.glob("*/manifest.json"))]
        write_json(transport_path, {"verified_original_range_receipts": manifests,
                                   "decoded_native_windows_retained": True, "range_blocks_are_ephemeral": True})
        files = [path for path in cache.rglob("*") if path.is_file()]
        removed_files += len(files)
        removed_bytes += sum(path.stat().st_size for path in files)
        shutil.rmtree(cache)
    for ledger_path in (ROOT / "imagery").glob("*/acquisition.json"):
        ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
        for epoch in ledger["epochs"].values():
            for row in epoch["transport"]:
                if "cache_manifest" in row:
                    row["temporary_cache_manifest_at_acquisition"] = row.pop("cache_manifest")
                row["retained_transport_receipt"] = str(transport_path)
        ledger["temporary_range_cache_cleaned"] = True
        write_json(ledger_path, ledger)
    # These were visual/text probes, not accepted references. Keep source PDFs
    # and the relevant operator/rescue pages, discard irrelevant review pages.
    scratch_names = ["operator_report_page_12.png", "operator_report_page_20.png",
                     "lmhld_report_page_14.png", "lmhld_report_page_15.png", "lmhld_report_evidence.json",
                     "operator_report_text.tmp.txt"]
    for name in scratch_names:
        path = (ROOT / "qualification" / name).resolve()
        if path.parent != (ROOT / "qualification").resolve():
            raise ValueError("scratch path outside current qualification folder")
        if path.is_file():
            removed_bytes += path.stat().st_size
            removed_files += 1
            path.unlink()
    stage_root = Path(__file__).resolve().parent
    for candidate in sorted(stage_root.rglob("__pycache__"), reverse=True):
        candidate = candidate.resolve()
        if not candidate.is_relative_to(stage_root) or candidate.name != "__pycache__":
            raise ValueError("unexpected code cache cleanup target")
        files = [path for path in candidate.rglob("*") if path.is_file()]
        removed_files += len(files)
        removed_bytes += sum(path.stat().st_size for path in files)
        shutil.rmtree(candidate)
    report = {"completed_at": now(), "retained_source_receipts": receipts,
              "default_weight_unchanged": model_ok, "old_reserve_registry_unchanged": registry_ok,
              "partial_reference_unknown_area_contract_verified": True,
              "new_detector_predictions": 0, "new_fits": 0, "new_cloud_generation_calls": 0,
              "cleanup_removed_files": removed_files + previous.get("cleanup_removed_files", 0),
              "cleanup_removed_bytes": removed_bytes + previous.get("cleanup_removed_bytes", 0),
              "cleanup_this_run_files": removed_files, "cleanup_this_run_bytes": removed_bytes,
              "retained_data_bytes": sum(path.stat().st_size for path in ROOT.rglob("*") if path.is_file()),
              "partial_downloads_remaining": len(list(ROOT.rglob("*.part")))}
    write_json(verification_path, report)
    print(json.dumps({key: value for key, value in report.items() if key != "retained_source_receipts"}, ensure_ascii=False), flush=True)


def inspect_vectors(name):
    # Isolated reader dependency; production environments are not modified.
    sys.path.insert(0, str(Path(__file__).parent / "vendor"))
    import shapefile
    from rasterio.crs import CRS
    from rasterio.warp import transform_bounds

    path = ROOT / "sources" / name
    receipt = json.loads((ROOT / "receipts" / (name + ".json")).read_text(encoding="utf-8"))
    if hashlib.sha256(path.read_bytes()).hexdigest() != receipt.get("sha256"):
        raise ValueError("download receipt mismatch")
    report = {"source": receipt, "layers": [], "qualification": "geometry_and_attribute_inspection_only",
              "mine_accident_admitted": False, "model_predictions": 0}
    with ZipFile(path) as archive:
        members = archive.infolist()
        if sum(x.file_size for x in members) > 512 * 1024 * 1024:
            raise ValueError("archive expanded size exceeds inspection budget")
        report["archive_members"] = [{"name": x.filename, "bytes": x.file_size} for x in members]
        names = {x.filename.casefold(): x.filename for x in members}
        for member in members:
            if not member.filename.lower().endswith(".shp"):
                continue
            base = member.filename[:-4]
            def body(suffix):
                entry = names.get((base + suffix).casefold())
                return archive.read(entry) if entry else None
            encoding = (body(".cpg") or b"utf-8").decode("ascii").strip()
            encoding = {"65001": "utf-8", "950": "cp950"}.get(encoding, encoding)
            prj = (body(".prj") or b"").decode("utf-8-sig")
            crs = CRS.from_wkt(prj) if prj else None
            reader = shapefile.Reader(shp=io.BytesIO(body(".shp")),
                                     shx=io.BytesIO(body(".shx")),
                                     dbf=io.BytesIO(body(".dbf")), encoding=encoding)
            fields = [x[0] for x in reader.fields[1:]]
            counts = {field: Counter() for field in fields
                      if any(x in field.lower() for x in ("date", "event", "source", "image", "imag"))}
            examples = []
            for row in reader.iterRecords():
                values = row.as_dict()
                for key in counts:
                    counts[key][str(values[key])] += 1
                if len(examples) < 2:
                    examples.append(values)
            layer = dict(name=member.filename, features=len(reader), shape_type=reader.shapeTypeName,
                         fields=reader.fields[1:], encoding=encoding, crs_wkt=prj,
                         crs_epsg=crs.to_epsg() if crs else None, bounds_native=list(reader.bbox),
                         bounds_wgs84=list(transform_bounds(crs, "EPSG:4326", *reader.bbox)) if crs else None,
                         examples=examples,
                         attribute_counts={key: dict(value.most_common()) for key, value in counts.items()})
            report["layers"].append(layer)
    write_json(ROOT / "qualification" / (name + ".json"), report)
    print(json.dumps({"source": name, "layers": [{key: value for key, value in layer.items()
                        if key in {"name", "features", "shape_type", "crs_epsg", "bounds_wgs84", "fields", "examples"}}
                      for layer in report["layers"]]}, ensure_ascii=False, default=str), flush=True)


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe", action="store_true")
    parser.add_argument("--name")
    parser.add_argument("--url")
    parser.add_argument("--limit-mib", type=int, default=4)
    parser.add_argument("--inspect-vectors")
    parser.add_argument("--direct", action="store_true")
    parser.add_argument("--zenodo-file")
    parser.add_argument("--screen-point-tiles", action="store_true")
    parser.add_argument("--finalize-batch", action="store_true")
    args = parser.parse_args()
    if args.finalize_batch:
        finalize_batch()
    elif args.screen_point_tiles:
        screen_point_tiles()
    elif args.zenodo_file:
        metadata_path = ROOT / "sources" / "uglc_zenodo_metadata"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        files = [row for row in metadata["files"] if row["key"] == args.zenodo_file]
        if len(files) != 1 or Path(args.zenodo_file).name != args.zenodo_file:
            parser.error("select an exact published filename")
        entry = files[0]
        if entry["size"] > args.limit_mib * 1024 * 1024:
            parser.error("published file exceeds selected download cap")
        receipt = acquire(args.zenodo_file, entry["links"]["self"], args.limit_mib * 1024 * 1024, direct=True)
        if receipt["ok"]:
            algorithm, expected = entry["checksum"].split(":", 1)
            actual = hashlib.new(algorithm, Path(receipt["path"]).read_bytes()).hexdigest()
            receipt.update(publisher_checksum=entry["checksum"], publisher_checksum_verified=actual == expected,
                           source_record_sha256=hashlib.sha256(metadata_path.read_bytes()).hexdigest())
            if actual != expected:
                receipt.update(ok=False, error="publisher checksum mismatch; file not eligible")
            write_json(ROOT / "receipts" / (args.zenodo_file + ".json"), receipt)
        print(json.dumps(receipt, ensure_ascii=False), flush=True)
    elif args.inspect_vectors:
        if Path(args.inspect_vectors).name != args.inspect_vectors:
            parser.error("inspection name must be a filename")
        inspect_vectors(args.inspect_vectors)
    elif args.probe:
        for name, url in PROBES.items():
            print(json.dumps(acquire(name, url, direct=args.direct), ensure_ascii=False), flush=True)
    elif args.name and args.url:
        if Path(args.name).name != args.name or args.name in {".", ".."}:
            parser.error("name must be a filename")
        if not args.url.startswith("https://") or not 1 <= args.limit_mib <= 1024:
            parser.error("HTTPS and a 1..1024 MiB explicit cap are required")
        print(json.dumps(acquire(args.name, args.url, args.limit_mib * 1024 * 1024, direct=args.direct), ensure_ascii=False))
    else:
        parser.error("select --probe or explicit --name/--url")


if __name__ == "__main__":
    main()
