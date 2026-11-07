"""Public mine identity / source qualification. Never runs remote code or models."""
from __future__ import annotations
import importlib.util
import json
import sys
from pathlib import Path
from html.parser import HTMLParser
from urllib.parse import urljoin

sys.path.insert(0, r"D:\LMHLD")
from runtime_paths import configure_runtime_paths
configure_runtime_paths()
sys.dont_write_bytecode = True

HERE = Path(__file__).resolve().parent
DATA = Path(r"D:\LMHLD\datasets\mine_context_qualification_20261015")
OLD = Path(r"D:\LMHLD\datasets\event_expansion_20261015")
SPEC = importlib.util.spec_from_file_location("source_transport", HERE.parent / "phase7_20261015/public_reference_screen.py")
ACQ = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ACQ)
ACQ.ROOT = DATA
write = ACQ.write_json
now = ACQ.now


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def sha(path):
    import hashlib
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def acquire(name, url, cap=4, direct=False):
    path = DATA / "receipts" / (name + ".json")
    if path.exists():
        record = read(path)
        if record.get("ok"):
            if record["requested_url"] != url:
                raise ValueError("Source name collision across URLs")
            if sha(record["path"]) != record["sha256"]:
                raise ValueError("Source integrity mismatch")
            return record
    attempts = sum(1 + len(read(path).get("previous_attempts", [])) for path in (DATA / "receipts").glob("*.json"))
    if attempts >= 48:
        raise ValueError("This bounded source batch has reached 48 requests")
    record = ACQ.acquire(name, url, cap * 1024 * 1024, direct=direct)
    print(json.dumps({key: record.get(key) for key in ["name", "ok", "bytes", "error"]}, ensure_ascii=False), flush=True)
    return record


def initialise():
    if (HERE / "protocol.json").exists():
        return
    previous = read(HERE.parent / "phase9_20261015/protocol.json")
    write(HERE / "protocol.json", {"registered_at": now(), "objective": "核矿权/作业面身份与原图条件；补较大事故对象和独立正常作业依据",
          "parent_state": str(HERE.parent / "phase9_20261015/state.json"),
          "data_root": str(DATA), "default_model": previous["default_model"],
          "candidate_model": previous["candidate_model"], "protected_registry_hashes": previous["protected_registry_hashes"],
          "initial_candidates": ["TW_GUOHUA2022_CLEAR_SOURCE_CANDIDATE", "TW_DAYU2023_SOURCE_CANDIDATE"],
          "predictions_allowed": False, "training_allowed": False,
          "role_assignment_required_before_predictions": True,
          "rules": ["公开矿权范围不等于实际采掘面，不等于事故边界", "当前状态不自动推定2022/2023年的作业状态",
                    "公开查看不等于原始商业影像使用许可", "缺失编目不能充当无事故或正常作业负例",
                    "更换来源，不围绕几个核心像素重复调参", "源文件和既有验收/训练快照不改写"],
          "max_public_get_requests": 48, "max_document_mib": 24, "max_metadata_mib": 4,
          "cloud_model_calls": 0, "network_scope": "Only public source GETs; no messages, secrets, project code or private imagery sent"})
    write(HERE / "state.json", {"client_work_date": "2026-10-15", "status": "source_identity_and_reference_qualification_in_progress",
          "data_root": str(DATA), "protocol": str(HERE / "protocol.json"), "new_training": 0,
          "new_model_predictions": 0, "new_cloud_generation_calls": 0, "user_input_required": []})


class Links(HTMLParser):
    def __init__(self):
        super().__init__()
        self.scripts, self.links = [], []
    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == "script" and attrs.get("src"):
            self.scripts.append(attrs["src"])
        if tag == "a" and attrs.get("href"):
            self.links.append(attrs["href"])


def probe():
    initialise()
    sources = {
        "mining_public_portal.html": "https://mlms.gsmma.gov.tw/mimspublic",
        "official_2022_dataset.html": "https://data.moa.gov.tw/open_detail.aspx?id=I41",
        "official_mining_open_data.html": "https://www.gsmma.gov.tw/nss/p/G000",
    }
    for name, url in sources.items():
        record = acquire(name, url)
        if record["ok"] and name == "mining_public_portal.html":
            parser = Links()
            parser.feed(Path(record["path"]).read_text(encoding="utf-8-sig"))
            links = {"source_sha256": record["sha256"], "script_urls": [urljoin(url, value) for value in parser.scripts],
                     "links": [urljoin(url, value) for value in parser.links], "remote_code_execution": False}
            write(DATA / "qualification/mining_public_entrypoints.json", links)
            print(json.dumps(links, ensure_ascii=False), flush=True)


def portal_sources():
    known = read(DATA / "qualification/mining_public_entrypoints.json")["script_urls"]
    selected = [url for url in known if any(url.endswith(path) for path in ["geosense/Repository.js", "geosense/env.js", "Code/CodeSvc.js", "Index/IndexSvc.js", "Pages/Index.js"])]
    for index, url in enumerate(selected):
        acquire(f"portal_source_{index}_{url.split('/')[-1]}", url)


def public_mine_query():
    """Identical read-only query used on initial load of the public site."""
    import requests
    url = "https://mlms.gsmma.gov.tw/MIMS/api/PublicIndex/GetMiningType"
    payload = {"County": "", "Status": "", "MiningType": "", "KickOff": ""}
    name = "official_current_mines.json"
    path = DATA / "receipts" / (name + ".json")
    amendment = HERE / "public_query_addendum.json"
    if not amendment.exists():
        write(amendment, {"registered_at": now(), "parent_protocol_sha256": sha(HERE / "protocol.json"),
              "reason": "Official public homepage performs its read-only mine search via POST rather than GET",
              "url": url, "method": "POST", "body": payload,
              "source_scripts": ["portal_source_1_env.js", "portal_source_3_IndexSvc.js", "portal_source_4_Index.js"],
              "max_calls": 1, "changes_external_state": False, "account_credentials": False,
              "outbound_project_data": False, "included_in_total_48_request_budget": True})
    if path.exists():
        receipt = read(path)
        if not receipt.get("ok"):
            raise ValueError("The one public query attempt failed; do not retry or access authenticated APIs")
        if receipt["requested_url"] != url or receipt["request_body"] != payload or sha(receipt["path"]) != receipt["sha256"]:
            raise ValueError("Retained public query identity mismatch")
    else:
        target = DATA / "sources" / name
        part = target.with_suffix(".json.part")
        receipt = {"name": name, "requested_url": url, "method": "POST", "request_body": payload,
                   "started_at": now(), "limit_bytes": 4 * 1024**2, "read_only_public_homepage_query": True}
        try:
            with requests.Session() as session:
                with session.post(url, json=payload, timeout=(12, 30), stream=True, allow_redirects=False,
                                  headers={"User-Agent": "PublicResearchDataQualification/1.0"}) as response:
                    receipt.update(status_code=response.status_code, content_type=response.headers.get("Content-Type"))
                    if response.status_code != 200:
                        raise ValueError("Public query not available without authentication or redirect")
                    count = 0
                    with part.open("wb") as stream:
                        for block in response.iter_content(65536):
                            count += len(block)
                            if count > receipt["limit_bytes"]:
                                raise ValueError("Public query size cap exceeded")
                            stream.write(block)
                    body = read(part)
                    if not isinstance(body.get("Data"), list):
                        raise ValueError("Unexpected public response schema")
                    part.replace(target)
                    receipt.update(ok=True, path=str(target), bytes=count, sha256=sha(target))
        except (requests.RequestException, ValueError, OSError) as exc:
            part.unlink(missing_ok=True)
            receipt.update(ok=False, error=f"{type(exc).__name__}: {exc}")
        receipt["completed_at"] = now()
        write(path, receipt)
    if not receipt.get("ok"):
        print(json.dumps(receipt, ensure_ascii=False), flush=True)
        return
    rows = read(receipt["path"])["Data"]
    hits = [row for row in rows if any(word in json.dumps(row, ensure_ascii=False) for word in ["國華", "大裕"])]
    write(DATA / "qualification/official_mine_identity_leads.json", {"source": receipt, "original_rows": len(rows),
          "selected_keyword_rows": hits, "current_record_is_not_historic_operation_proof": True})
    print(json.dumps({"original_rows": len(rows), "keyword_rows": hits}, ensure_ascii=False), flush=True)


def identity_documents():
    acquire("official_historical_mine_study.pdf", "https://ws.ndc.gov.tw/001/administrator/10/relfile/5680/4428/0010888.pdf", 24)
    acquire("public_management_documents.html", "https://mlms.gsmma.gov.tw/mimspublic/Information/InformationDownload")
    path = DATA / "sources/public_management_documents.html"
    if path.exists():
        parser = Links()
        parser.feed(path.read_text(encoding="utf-8-sig"))
        urls = [urljoin("https://mlms.gsmma.gov.tw/mimspublic/Information/InformationDownload", value) for value in parser.scripts]
        write(DATA / "qualification/public_management_entrypoints.json", {"script_urls": urls, "source_sha256": sha(path)})
        print(json.dumps({"management_scripts": [url for url in urls if any(key in url for key in ["Information", "Pages/", "service/"])]}, ensure_ascii=False), flush=True)


def management_sources():
    urls = read(DATA / "qualification/public_management_entrypoints.json")["script_urls"]
    selected = [url for url in urls if any(url.endswith(value) for value in ["Publish/PublishSvc.js", "File/FileSvc.js", "Information/informationDownload.js"])]
    for index, url in enumerate(selected):
        acquire(f"management_source_{index}_{url.split('/')[-1]}", url)


def additional_metadata():
    acquire("official_mine_document_params.json", "https://mlms.gsmma.gov.tw/MIMS/api/File/GetMiningAreaParams")
    acquire("global_mining_2022_metadata.html", "https://doi.pangaea.de/10.1594/PANGAEA.942325?format=html")
    acquire("global_mining_2020_metadata.html", "https://doi.pangaea.de/10.1594/PANGAEA.910894?format=html")
    acquire("global_mining_tang_metadata.json", "https://zenodo.org/api/records/5091250", direct=True)


def global_geometry():
    metadata = DATA / "sources/global_mining_2022_metadata.html"
    text = metadata.read_text(encoding="utf-8-sig")
    url = "https://download.pangaea.de/dataset/942325/files/global_mining_polygons_v2.gpkg"
    if url not in text or "https://creativecommons.org/licenses/by-sa/4.0/" not in text:
        raise ValueError("Publisher download / licence not confirmed")
    path = HERE / "global_mine_discovery_protocol.json"
    if not path.exists():
        write(path, {"registered_at": now(), "metadata_sha256": sha(metadata), "source_doi": "10.1594/PANGAEA.942325",
              "licence": "CC BY-SA 4.0", "max_download_mib": 24, "download_only_original_polygons_not_global_rasters": True,
              "purpose": "Use an independent mining-footprint source to expand discovery beyond sparse OSM ways",
              "not_accident_or_mine_permit_truth": True,
              "source_years": [2020, 2021, 2022, 2023, 2025], "discovery_region_wgs84": [119.8, 21.8, 122.2, 25.5],
              "priority_rules": {"minimum_reference_area_m2": 30000, "minimum_20m_core_area_m2": 10000,
                                 "minimum_mining_footprint_overlap_fraction": .5,
                                 "exclude_old_mine_groups_and_protected_geography": True},
              "rules_are_for_source_priority_not_label_admission": True,
              "no_model_scores_used": True, "training_allowed": False, "predictions_allowed": False})
    record = acquire("global_mining_polygons_v2.gpkg", url, 24)
    if not record["ok"]:
        return
    import sqlite3
    with sqlite3.connect(Path(record["path"]).as_uri() + "?mode=ro", uri=True) as connection:
        layers = connection.execute("select table_name,srs_id,min_x,min_y,max_x,max_y from gpkg_contents where data_type='features'").fetchall()
        info = []
        for layer in layers:
            table = layer[0]
            if not table.replace('_', '').isalnum():
                raise ValueError("Unexpected layer name")
            columns = connection.execute(f'pragma table_info("{table}")').fetchall()
            count = connection.execute(f'select count(*) from "{table}"').fetchone()[0]
            info.append({"table": table, "srs_id": layer[1], "bounds": layer[2:], "columns": columns, "objects": count})
        write(DATA / "qualification/global_mining_structure.json", {"source": record, "layers": info})
        print(json.dumps({"layers": info}, ensure_ascii=False), flush=True)


def scope_sources():
    acquire("nlsc_official_coordinate_system.html", "https://www.nlsc.gov.tw/en/cp.aspx?Create=1&n=2122")
    acquire("operation_author_public_script_page.html", "https://code.earthengine.google.com/aa655a2049776d31835d5739fdd18d00")


def public_document_queries():
    import requests
    url = "https://mlms.gsmma.gov.tw/MIMS/api/File/GetPublishCheckFile"
    empty = {"DateB": "", "DateE": "", "Type": "", "County": "", "MiningArea": "", "KeyWord": ""}
    queries = {
        "guohua_public_document_list.json": {**empty, "KeyWord": "國華"},
        "xincheng_public_document_list.json": {**empty, "MiningArea": "9ECC7C8F-51CF-43D7-9AF0-D50F8C8A9D61"},
    }
    manifest = HERE / "public_document_query_addendum.json"
    if not manifest.exists():
        write(manifest, {"registered_at": now(), "parent_protocol_sha256": sha(HERE / "protocol.json"),
              "method": "POST", "url": url, "queries": queries, "max_calls": 2,
              "source": "FileSvc.GetFileData + informationDownload.js, directly used by public document page",
              "read_only_public_query": True, "credentials": False, "included_in_total_48_request_budget": True})
    for name, payload in queries.items():
        receipt_path = DATA / "receipts" / (name + ".json")
        if receipt_path.exists():
            continue
        record = {"name": name, "requested_url": url, "method": "POST", "request_body": payload,
                  "started_at": now(), "read_only_public_query": True, "limit_bytes": 4 * 1024**2}
        target = DATA / "sources" / name
        part = target.with_suffix(".json.part")
        try:
            with requests.Session() as session:
                with session.post(url, json=payload, timeout=(12, 30), stream=True, allow_redirects=False,
                                  headers={"User-Agent": "PublicResearchDataQualification/1.0"}) as response:
                    record.update(status_code=response.status_code, content_type=response.headers.get("Content-Type"))
                    if response.status_code != 200:
                        raise ValueError("Public document query needs authentication or is unavailable")
                    count = 0
                    with part.open("wb") as stream:
                        for block in response.iter_content(65536):
                            count += len(block)
                            if count > record["limit_bytes"]:
                                raise ValueError("Public document response too large")
                            stream.write(block)
                    body = read(part)
                    if not isinstance(body.get("Data"), list):
                        raise ValueError("Unexpected public file response schema")
                    part.replace(target)
                    record.update(ok=True, bytes=count, path=str(target), sha256=sha(target))
        except (requests.RequestException, ValueError, OSError) as exc:
            part.unlink(missing_ok=True)
            record.update(ok=False, error=f"{type(exc).__name__}: {exc}")
        record["completed_at"] = now()
        write(receipt_path, record)
        print(json.dumps({key: record.get(key) for key in ["name", "ok", "error", "bytes"]}, ensure_ascii=False), flush=True)
        if record.get("ok"):
            rows = read(target)["Data"]
            print(json.dumps({"files": rows}, ensure_ascii=False), flush=True)


def mine_scope_documents():
    rows = read(DATA / "sources/xincheng_public_document_list.json")["Data"]
    selected = {
        "official_2024_joint_inspection_program.pdf": "466e67df-1531-42e5-8281-5fbfda05504d",
        "xincheng_2025_annual_work_plan.pdf": "C7168806-EF8E-4A19-9A43-D3A7BCA28F79",
    }
    for name, file_id in selected.items():
        original = next(row for row in rows if row["ID"] == file_id)
        record = acquire(name, f"https://mlms.gsmma.gov.tw/MIMS/api/File/DownloadFileById/{file_id}/MiningJoinCheckFiles", 24)
        if record["ok"]:
            write(DATA / "qualification" / (name + ".json"), {"file_record": original, "download_receipt": record,
                  "historic_working_face_or_accident_truth_admitted": False})


def monitoring_sources():
    record = acquire("operator_public_monitoring.html", "https://quarry.acchl.com.tw/")
    if not record["ok"]:
        return
    parser = Links()
    parser.feed(Path(record["path"]).read_text(encoding="utf-8-sig"))
    links = [urljoin("https://quarry.acchl.com.tw/", value) for value in parser.links]
    pdfs = [value for value in links if '.pdf' in value.lower()]
    write(DATA / "qualification/operator_monitoring_entrypoints.json", {"source_sha256": record["sha256"],
          "pdf_urls": sorted(set(pdfs)), "site_links": sorted(set(links)), "source_is_operator_publication": True})
    print(json.dumps({"pdf_urls": sorted(set(pdfs))}, ensure_ascii=False), flush=True)


def inspection_documents():
    from urllib.parse import quote
    original = read(DATA / "qualification/operator_monitoring_entrypoints.json")["pdf_urls"]
    selected = {"operator_monitoring_status_20250121.pdf": "新城山礦場邊坡監測辦理情形-2025-01-21.pdf",
                "operator_earthquake_inspection_20190418.pdf": "108.4.18.1301 地震對亞泥礦區及部落社區影響情形.pdf",
                "operator_earthquake_inspection_20190410.pdf": "108.4.10凌晨至清晨三次地震對亞泥礦區及部落社區影響情形.pdf"}
    for name, suffix in selected.items():
        url = next(value for value in original if value.endswith(suffix)).replace('\\', '/')
        receipt = acquire(name, quote(url, safe=':/?=&%'), 24)
        if receipt["ok"]:
            write(DATA / "qualification" / (name + ".json"), {"source_link_text": suffix,
                  "source_page_sha256": sha(DATA / "sources/operator_public_monitoring.html"),
                  "download": receipt, "independent_field_validation_of_our_model": False,
                  "pixel_labels_admitted": False})


def official_development_accident():
    path = HERE / "official_accident_source_addendum.json"
    url = "https://www.mem.gov.cn/gk/zfxxgkpt/fdzdgknr/202308/P020230829535806813294.pdf"
    if not path.exists():
        write(path, {"registered_at": now(), "case": "CN_XJ2023", "role": "existing development mine; never a new blind mine",
              "url": url, "source": "State Council accident investigation group; published by MEM",
              "purpose": "Check official extent/chronology against existing reference assumptions; not derive a pixel polygon from an area number",
              "max_source_mib": 24, "new_model_predictions_allowed": False, "new_training_allowed": False})
    acquire("xinjing_2023_official_accident_investigation.pdf", url, 24)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    {"probe": probe, "portal-sources": portal_sources, "public-mine-query": public_mine_query,
     "identity-documents": identity_documents, "management-sources": management_sources,
     "additional-metadata": additional_metadata, "global-geometry": global_geometry,
     "scope-sources": scope_sources, "public-document-queries": public_document_queries,
     "mine-scope-documents": mine_scope_documents, "monitoring-sources": monitoring_sources,
     "inspection-documents": inspection_documents, "official-development-accident": official_development_accident}[sys.argv[1] if len(sys.argv) > 1 else "probe"]()
