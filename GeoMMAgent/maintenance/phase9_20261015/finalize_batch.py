"""Close a source-qualification batch and remove its validated scratch data."""
from pathlib import Path
import shutil
import xml.etree.ElementTree as ET
import rasterio
from expand_event_references import DATA, HERE, ACQ, read, write, sha, now


def main():
    protocol = read(HERE / "protocol.json")
    decisions = []
    notes = {
        "TW_DAYU2023_SOURCE_CANDIDATE": {
            "event_group": "TW_HAIKUI2023", "way_id": 536939324,
            "historic_context_verified": True, "working_face_membership": "unresolved",
            "visual_review": "Original polygon includes an already exposed pre-event area; post exposure changes do not by themselves distinguish mining operations from failure at 10 m.",
            "missing": ["independent accident/mine-face attribution", "larger reliable reference support", "scoped verified normal-operation examples"],
            "decision": "source_reference_retained_unadmitted; no full mine accident truth"},
        "TW_GUOHUA2022_CLEAR_SOURCE_CANDIDATE": {
            "event_group": "TW_TAITUNG20220918", "way_id": 526217895,
            "historic_context_verified": True, "working_face_membership": "plausible_not_certified",
            "visual_review": "A new bare patch is visible after the event near the original official object and historic quarry edge. It is consistent with the reference but not independent proof of full failure extent or operational status.",
            "missing": ["official mine name/working-face mapping", "legacy CRS datum clarification", "original SPOT/Pleiades reference imagery or equivalent independent check", "larger reliable reference support", "scoped verified normal-operation examples"],
            "decision": "promising_independent_event_source_candidate; qualification_only"},
    }
    for case_id, note in notes.items():
        root = DATA / "imagery" / case_id
        report = read(root / "qualification.json")
        plan = read(root / "acquisition_plan.json")
        ledger = read(root / "acquisition.json")
        signature = None
        for epoch in ["pre", "post"]:
            row = ledger["epochs"][epoch]
            for name, digest in row["files_sha256"].items():
                if sha(root / name) != digest:
                    raise ValueError("Native raster changed")
            with rasterio.open(root / f"{epoch}_provider_BGRN_native10m.tif") as src:
                this = (src.crs, src.transform, src.shape, src.count, src.dtypes)
            if signature is not None and this != signature:
                raise ValueError("Selected native images do not share original grid")
            signature = this
            metadata = row["granule_metadata_receipt"]
            if sha(metadata["path"]) != metadata["sha256"]:
                raise ValueError("Sensing metadata changed")
            times = [node.text for node in ET.parse(metadata["path"]).getroot().iter()
                     if node.tag.split('}')[-1] == "SENSING_TIME"]
            if len(times) != 1 or times[0][:10] != row["datetime"][:10]:
                raise ValueError("Original sensing time mismatch")
        if sha(report["visual_qa_panel"]) != report["visual_qa_sha256"]:
            raise ValueError("Source QA panel changed")
        report.update(visual_review_completed=True, visual_review_scope="source-only manual inspection; no detector outputs",
                      admission_decision=note["decision"], missing_evidence=note["missing"],
                      provisional_role="unpredicted candidate; formal split not assigned",
                      training_allowed=False, predictions_allowed=False, final_blind_test_allowed=False)
        write(root / "qualification.json", report)
        decisions.append({"case_id": case_id, **note, "qualification": str(root / "qualification.json"),
                          "source_dates": plan["source_dates"], "original_reference_dates": report["original_reference_dates"],
                          "joint_clear_fraction": report["joint_clear_fraction"],
                          "source_pixel_support": report["objects"][0]["core_sensitivity"],
                          "model_scores_viewed": False, "training_allowed": False, "final_blind_test_allowed": False})
    operation = {
        "source_url": "https://tech.ardswc.gov.tw/EPaper/Home/EPaper?PaperID=78b8b3e4-5c52-4f96-a2b0-735981390952",
        "retained_original_source": r"D:\LMHLD\datasets\public_reference_screen_20261015\sources\ardswc_mining_confounder.html",
        "role": "published operation-caused SAR false-positive example; site/phenomenon context only",
        "exact_optical_epochs_and_pixel_geometry_confirmed": False,
        "public_author_script_access": "connection timeout; no script retrieved or executed",
        "verified_normal_operation_negative_pixels": 0,
        "training_allowed": False, "false_positive_rate_allowed": False,
    }
    write(DATA / "qualification/role_decisions.json", {"completed_at": now(), "candidates": decisions,
          "normal_operation_reference": operation,
          "different_event_and_mine_candidates_are_not_yet_a_passed_blind_test": True,
          "new_complete_mine_accident_boundaries": 0, "new_verified_operation_negative_pixels": 0})
    models = []
    for row in [protocol["default_model"], protocol["candidate_model"], *protocol["protected_registry_hashes"]]:
        if sha(row["path"]) != row["sha256"]:
            raise ValueError("Frozen model or protected registry changed")
        models.append({"path": row["path"], "sha256": row["sha256"], "unchanged": True})
    receipts = []
    for path in sorted((DATA / "receipts").glob("*.json")):
        row = read(path)
        if row.get("ok"):
            source = Path(row["path"])
            if source.resolve().parent != (DATA / "sources").resolve():
                raise ValueError("Unexpected retained source location")
            if source.stat().st_size != row["bytes"] or sha(source) != row["sha256"]:
                raise ValueError("Original source integrity mismatch")
            ACQ.validate_format(source, row["name"])
        receipts.append({key: row.get(key) for key in ["name", "ok", "bytes", "sha256", "error"]})
    removed = []
    cache = (DATA / "_range_cache").resolve()
    if cache.exists():
        if cache.parent != DATA.resolve() or cache.name != "_range_cache":
            raise ValueError("Unsafe cache cleanup target")
        manifests = [read(path) for path in sorted(cache.glob("*/manifest.json"))]
        write(DATA / "qualification/source_transport_manifest.json", {"original_range_receipts": manifests,
              "native_decoded_windows_retained": True, "ranges_are_temporary": True})
        files = [path for path in cache.rglob("*") if path.is_file()]
        removed.append({"target": str(cache), "files": len(files), "bytes": sum(path.stat().st_size for path in files)})
        shutil.rmtree(cache)
    # The rejected pair and its exact failure/transport evidence are preserved
    # in JSON; full-cloud pixels and duplicate pre imagery have no future use.
    rejected = (DATA / "imagery/TW_GUOHUA2022_SOURCE_CANDIDATE").resolve()
    if rejected.exists():
        if rejected.parent != (DATA / "imagery").resolve() or rejected.name != "TW_GUOHUA2022_SOURCE_CANDIDATE":
            raise ValueError("Unsafe cloud-rejected pair cleanup target")
        receipt = read(DATA / "qualification/guohua2022_rejected_pair.json")
        for epoch in receipt["acquisition"]["epochs"].values():
            for name, digest in epoch["files_sha256"].items():
                if sha(rejected / name) != digest:
                    raise ValueError("Rejected source changed before cleanup")
        files = [path for path in rejected.rglob("*") if path.is_file()]
        removed.append({"target": str(rejected), "files": len(files), "bytes": sum(path.stat().st_size for path in files)})
        shutil.rmtree(rejected)
    for case_id in notes:
        path = DATA / "imagery" / case_id / "acquisition.json"
        ledger = read(path)
        for epoch in ledger["epochs"].values():
            for transport in epoch["transport"]:
                if "cache_manifest" in transport:
                    transport["temporary_cache_manifest_at_acquisition"] = transport.pop("cache_manifest")
                transport["retained_range_receipts"] = str(DATA / "qualification/source_transport_manifest.json")
        ledger["temporary_range_cache_cleaned"] = True
        write(path, ledger)
    for directory in [HERE / "__pycache__", HERE.parent / "phase7_20261015/__pycache__"]:
        directory = directory.resolve()
        if directory.exists():
            if directory.name != "__pycache__" or directory.parent not in [HERE.resolve(), (HERE.parent / "phase7_20261015").resolve()]:
                raise ValueError("Unsafe code cache target")
            files = [path for path in directory.rglob("*") if path.is_file()]
            removed.append({"target": str(directory), "files": len(files), "bytes": sum(path.stat().st_size for path in files)})
            shutil.rmtree(directory)
    discovery = [read(DATA / "qualification" / name) for name in ["mining_event_discovery.json", "earlier_mining_event_discovery.json"]]
    inventories = sorted([{"year": row["year"], **row["counts"]} for item in discovery for row in item["sources"]], key=lambda row: row["year"])
    files = [path for path in DATA.rglob("*") if path.is_file()]
    verification = {"completed_at": now(), "original_source_receipts": receipts,
                    "frozen_models_and_protections": models,
                    "verified_native_pairs": list(notes), "inventory_counts": inventories,
                    "original_inventory_objects": sum(row["original_objects"] for row in inventories),
                    "unique_candidates_selected": len(notes), "new_verified_mine_accident_labels": 0,
                    "new_verified_normal_operation_negative_pixels": 0,
                    "new_training": 0, "new_model_predictions": 0, "new_cloud_generation_calls": 0,
                    "formal_blind_test_completed": False, "cleanup": removed,
                    "source_code_sha256": {path.name: sha(path) for path in sorted(HERE.glob("*.py"))},
                    "retained_data_bytes_before_this_receipt": sum(path.stat().st_size for path in files)}
    write(DATA / "qualification/batch_verification.json", verification)
    state = read(HERE / "state.json")
    state.update(status="completed_bounded_independent_source_qualification; unresolved_labels_remain_blocked",
                 completed_at=now(), readme=str(HERE / "README.md"), verification=str(DATA / "qualification/batch_verification.json"),
                 role_decisions=str(DATA / "qualification/role_decisions.json"), inventory_objects=verification["original_inventory_objects"],
                 different_event_source_candidates=2, new_verified_mine_accident_labels=0,
                 new_verified_operation_negative_pixels=0, final_blind_test_completed=False,
                 default_weight_changed=False, next_training_allowed=False,
                 next_priority="Resolve the promising 2022 quarry identity/reference CRS and higher-resolution source availability, add larger reliable positive objects and scoped normal-operation evidence, then freeze roles before any prediction. Do not repeat fitting on the existing few core pixels.",
                 retained_data_mib=verification["retained_data_bytes_before_this_receipt"] / 1024**2,
                 cleanup_bytes=sum(row["bytes"] for row in removed))
    write(HERE / "state.json", state)
    previous_path = HERE.parent / "phase8_20261015/state.json"
    previous = read(previous_path)
    previous["phase9_handoff"] = {"state": str(HERE / "state.json"), "readme": str(HERE / "README.md"),
                                 "summary": "Different-event source qualification completed; no new accepted labels or model claims."}
    write(previous_path, previous)
    print(json_dump({"status": state["status"], "objects": verification["original_inventory_objects"],
                     "source_candidates": len(notes), "native_pairs": 2, "accepted_mine_accident_labels": 0,
                     "retained_data_mib": state["retained_data_mib"], "cleaned_mib": state["cleanup_bytes"] / 1024**2}), flush=True)


def json_dump(value):
    import json
    return json.dumps(value, ensure_ascii=False)


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")
    main()
