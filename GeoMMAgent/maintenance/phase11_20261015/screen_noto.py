"""Original official high-resolution event sources; qualification before models."""
from __future__ import annotations
import argparse
import importlib.util
import io
import json
import sys
from pathlib import Path
from urllib.parse import quote, urljoin
from zipfile import ZipFile
from html.parser import HTMLParser

sys.path.insert(0, "D:/LMHLD")
from runtime_paths import configure_runtime_paths
configure_runtime_paths()
sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
DATA = Path("D:/LMHLD/datasets/noto_reference_screen_20261015")
SPEC = importlib.util.spec_from_file_location("phase11_source_transport", HERE.parent / "phase7_20261015/public_reference_screen.py")
ACQ = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ACQ)
ACQ.ROOT = DATA
write, now = ACQ.write_json, ACQ.now


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def sha(path):
    import hashlib
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def acquire(name, url, cap=4, direct=False):
    path = DATA / "receipts" / (name + ".json")
    if path.exists():
        record = read(path)
        if record.get('payload_retained') is False:
            raise ValueError('Source payload intentionally discarded; receipt retained; no implicit reacquisition')
        if record.get("ok"):
            if record["requested_url"] != url or sha(record["path"]) != record["sha256"]:
                raise ValueError("Retained source identity changed")
            return record
        if record.get("do_not_retry"):
            return record
    attempts = sum(1 + len(read(path).get("previous_attempts", [])) for path in (DATA / "receipts").glob("*.json"))
    amendment = HERE / 'vector_tile_transport_addendum.json'
    limit = 40 + (read(amendment)['additional_metadata_request_budget'] if amendment.exists() else 0)
    if attempts >= limit:
        raise ValueError("Source discovery request budget reached")
    record = ACQ.acquire(name, url, cap * 1024**2, direct=direct)
    print(json.dumps({key: record.get(key) for key in ["name", "ok", "bytes", "error"]}, ensure_ascii=False), flush=True)
    return record


class Links(HTMLParser):
    def __init__(self):
        super().__init__()
        self.links = []
    def handle_starttag(self, tag, attributes):
        attributes = dict(attributes)
        if tag == 'a' and attributes.get('href'):
            self.links.append(attributes['href'])


def initialise():
    path = HERE / "protocol.json"
    if path.exists():
        return
    old = read(HERE.parent / "phase10_20261015/protocol.json")
    write(path, {"registered_at": now(), "objective": "以另一未核查事件的官方原始多边形和高分正射影像补足可靠像素支持",
          "source_event": "JP_NOTO20240101", "data_root": str(DATA),
          "default_model": old["default_model"], "candidate_model": old["candidate_model"],
          "protected_registry_hashes": old["protected_registry_hashes"],
          "predictions_allowed": False, "training_allowed": False,
          "current_role": "qualification_only; event/mine grouping and training-exposure audit before role assignment",
          "source_request_budget": 40, "metadata_cap_mib": 4, "each_vector_source_cap_mib": 64,
          "initial_inventory_years": [2024], "discovery_bounds_wgs84": [136.6, 36.8, 137.5, 37.9],
          "source_priority_rules": {"minimum_area_m2": 10000, "minimum_20m_eroded_area_m2": 2500,
                                    "minimum_quarry_overlap_fraction": .5, "nearby_discovery_distance_m": 100},
          "rules": ["保留官方源几何、时相和调查方法，滑坡/堆积对象与采场身份分别核验",
                    "矿区邻近自然滑坡不能冒充矿山事故", "未确认调查覆盖与类别前，不把空白区变成安全负例",
                    "预训练历史未核完不声称完全未见；旧保护地区和登记保持不变",
                    "先看来源质量，不根据模型输出选样本或改阈值", "公开正射瓦片不冒充未重采样的原传感器数据"],
          "outbound": "public data GET only; zero external generation calls; no project secrets, private imagery or code sent"})
    write(HERE / "state.json", {"status": "official_noto_reference_source_qualification_in_progress",
          "data_root": str(DATA), "protocol": str(path), "new_training": 0, "new_predictions": 0,
          "new_cloud_generation_calls": 0, "user_input_required": []})


def sources():
    initialise()
    acquire("gsi_event_source.html", "https://web1.gsi.go.jp/BOUSAI/20240101_noto_earthquake.html")
    acquire("gsi_dem_inventory_source.html", "https://gisstar.gsi.go.jp/2024notolandslide/")
    page = DATA / "sources/gsi_event_source.html"
    parser = Links()
    parser.feed(page.read_text(encoding="utf-8-sig"))
    links = [urljoin("https://web1.gsi.go.jp/BOUSAI/20240101_noto_earthquake.html", value) for value in parser.links]
    inventory_links = [url for url in links if url.lower().endswith('.zip') and 'earthquake/noto_' in url]
    if len(inventory_links) != 5:
        raise ValueError("Review ambiguous official event inventory version")
    write(DATA / "qualification/event_source_links.json", {"source_sha256": sha(page), "inventory_links": inventory_links,
          "source_page_discloses_interpreted_polygons_and_possible_debris_removal": True,
          "inventory_missing_object_is_not_safe_background": True})
    for url in inventory_links:
        acquire(url.rsplit('/', 1)[-1], url, 8)
    acquire("20240101noto_landslidepolygon.gpkg", "https://gisstar.gsi.go.jp/2024notolandslide/20240101noto_landslidepolygon.gpkg", 64)


def quarries():
    initialise()
    bbox = read(HERE / "protocol.json")["discovery_bounds_wgs84"]
    query = '[out:json][timeout:25];way["landuse"="quarry"](' + ','.join(map(str, [bbox[1], bbox[0], bbox[3], bbox[2]])) + ');out meta geom;'
    acquire("noto_quarry_discovery.json", "https://overpass.private.coffee/api/interpreter?data=" + quote(query, safe="()"), 8, direct=True)


def alternative_transport():
    # Use the previously successful geometry-only query, and an unchanged
    # certificate-verified direct connection for one legacy-host source file.
    query = '[out:json][timeout:25];way["landuse"="quarry"](36.8,136.6,37.9,137.5);out geom;'
    acquire("noto_quarry_geom_discovery.json", "https://overpass.private.coffee/api/interpreter?data=" + quote(query, safe="()"), 8, direct=True)
    source = read(DATA / "qualification/event_source_links.json")["inventory_links"][0]
    acquire(source.rsplit('/', 1)[-1], source, 8, direct=True)


def main_overpass_node():
    query = '[out:json][timeout:25];way["landuse"="quarry"](36.8,136.6,37.9,137.5);out geom;'
    acquire("noto_quarry_main_node.json", "https://overpass-api.de/api/interpreter?data=" + quote(query, safe="()"), 8, direct=True)


def documented_global_alternative():
    query = '[out:json][timeout:25];way["landuse"="quarry"](36.8,136.6,37.9,137.5);out geom;'
    acquire("noto_quarry_global_alternative.json", "https://maps.mail.ru/osm/tools/overpass/api/interpreter?data=" + quote(query, safe="()"), 8)


def system_route_query():
    import requests
    query = '[out:json][timeout:25];way["landuse"="quarry"](36.8,136.6,37.9,137.5);out geom;'
    url = "https://overpass.private.coffee/api/interpreter?data=" + quote(query, safe="()")
    name = "noto_quarry_system_route.json"
    path = DATA / "receipts" / (name + ".json")
    if path.exists():
        return
    record = {"name": name, "requested_url": url, "started_at": now(), "connection": "system_network",
              "limit_bytes": 8 * 1024**2, "distinct_from_failed_direct_route": True}
    target = DATA / "sources" / name
    try:
        with requests.Session() as session:
            with session.get(url, timeout=(12, 30), stream=True,
                             headers={"User-Agent": "PublicResearchDataQualification/1.0"}) as response:
                record.update(status_code=response.status_code, content_type=response.headers.get('Content-Type'))
                if response.status_code != 200:
                    snippet = next(response.iter_content(1024), b'').decode('utf-8', errors='replace')[:700]
                    record.update(error_body_excerpt=snippet, ok=False, do_not_retry=True)
                else:
                    content = bytearray()
                    for block in response.iter_content(65536):
                        content.extend(block)
                        if len(content) > record["limit_bytes"]:
                            raise ValueError("Discovery response over budget")
                    body = json.loads(content)
                    if not isinstance(body.get("elements"), list) or body.get("remark"):
                        raise ValueError("Partial/unexpected quarry query")
                    target.write_bytes(content)
                    record.update(ok=True, bytes=len(content), path=str(target), sha256=sha(target))
    except (requests.RequestException, ValueError, OSError) as exc:
        record.update(ok=False, do_not_retry=True, error=f"{type(exc).__name__}: {exc}")
    record['completed_at'] = now()
    write(path, record)
    print(json.dumps({key: record.get(key) for key in ['ok', 'bytes', 'status_code', 'error', 'error_body_excerpt']}, ensure_ascii=False), flush=True)


def map_definitions():
    acquire("gsi_layers_directory.json", "https://api.github.com/repos/gsi-cyberjapan/gsimaps/contents/layers_txt?ref=gh-pages")
    path = DATA / "sources/gsi_layers_directory.json"
    if path.exists():
        rows = read(path)
        print(json.dumps({"files": [{key: row.get(key) for key in ['name', 'type', 'download_url']} for row in rows]}, ensure_ascii=False), flush=True)


def selected_map_definitions():
    rows = read(DATA / "sources/gsi_layers_directory.json")
    names = ['layers_20240102noto.txt', 'layers_topic_20240102noto.txt']
    for name in names:
        item = next(row for row in rows if row['name'] == name)
        record = acquire(name, item['download_url'])
        if record['ok']:
            body = read(record['path'])
            selected = []
            def walk(value):
                if isinstance(value, dict):
                    if value.get('id') and any(word in str(value.get('id')) for word in ['houkai', '0102do', '0111do']):
                        selected.append(value)
                    for item in value.values():
                        walk(item)
                elif isinstance(value, list):
                    for item in value:
                        walk(item)
            walk(body)
            write(DATA / 'qualification' / (name + '.selected.json'), {'source_sha256': record['sha256'], 'selected_layers': selected})
            print(json.dumps({'definition': name, 'selected_layers': selected}, ensure_ascii=False), flush=True)


def alternate_data_sources():
    acquire('openfreemap_style.json', 'https://tiles.openfreemap.org/styles/liberty')
    style_path = DATA / 'sources/openfreemap_style.json'
    if style_path.exists():
        style = read(style_path)
        vector_sources = {key: value for key, value in style['sources'].items() if value.get('type') == 'vector'}
        print(json.dumps({'vector_sources': vector_sources}, ensure_ascii=False), flush=True)
        if len(vector_sources) != 1:
            raise ValueError('Ambiguous vector provider metadata')
        source = next(iter(vector_sources.values()))
        if not source['url'].startswith('https://tiles.openfreemap.org/'):
            raise ValueError('Unexpected provider')
        acquire('openfreemap_tilejson.json', source['url'])
    # These zoom-2 layers cover all of Japan in one source tile. Source features
    # remain intact within the tile; no guessed SSL downgrade or manual mask.
    definitions = read(DATA / 'sources/layers_20240102noto.txt')
    flat = []
    def walk(value):
        if isinstance(value, dict):
            if value.get('url') and value.get('maxNativeZoom') == 2:
                flat.append(value)
            for item in value.values():
                walk(item)
        elif isinstance(value, list):
            for item in value:
                walk(item)
    walk(definitions)
    for layer in flat:
        if layer['id'].endswith(('hokai_all', 'handoku_area', 'cloud_poly')):
            acquire(layer['id'] + '.geojson', layer['url'].format(z=2, x=3, y=1), 8)
    acquire('mapbox_vector_tile_package.json', 'https://pypi.org/pypi/mapbox-vector-tile/2.2.0/json')
    acquire('pyclipper_package.json', 'https://pypi.org/pypi/pyclipper/1.3.0.post6/json')
    acquire('protobuf_package.json', 'https://pypi.org/pypi/protobuf/6.31.1/json')


def geological_mine_leads():
    acquire('gsj_wajima_2019_geology_explanation.pdf', 'https://www.gsj.jp/data/200KGM/PDF/GSJ_MAP_G200_NJ5310_2019_D.pdf', 24)
    acquire('shibata1981_noto_quarry_samples.pdf', 'https://www.jstage.jst.go.jp/article/ganko1941/76/7/76_7_248/_pdf/-char/ja/', 24)


def official_locality_hints():
    from urllib.parse import urlencode
    localities = {'wajima_niigyo_geocoder.json': '石川県輪島市三井町仁行',
                  'suzu_kashihara_geocoder.json': '石川県珠洲市宝立町柏原'}
    for name, locality in localities.items():
        record = acquire(name, 'https://msearch.gsi.go.jp/address-search/AddressSearch?' + urlencode({'q': locality}))
        if record['ok']:
            print(json.dumps({'name': name, 'locality_results_are_not_quarry_coordinates': read(record['path'])}, ensure_ascii=False), flush=True)


def inspect():
    import sqlite3
    for path in sorted((DATA / "sources").glob("*.zip")):
        with ZipFile(path) as archive:
            layers = []
            if sum(row.file_size for row in archive.infolist()) > 64 * 1024**2:
                raise ValueError("Unexpected expanded vector size")
            for name in archive.namelist():
                if name.lower().endswith(('.geojson', '.json')):
                    body = json.loads(archive.read(name).decode("utf-8-sig"))
                    rows = body.get("features", [])
                    layers.append({"member": name, "objects": len(rows), "crs": body.get("crs"),
                                   "example_properties": rows[0].get("properties") if rows else None,
                                   "geometry_types": sorted({row['geometry']['type'] for row in rows})})
            write(DATA / "qualification" / (path.name + ".structure.json"), {"archive": str(path), "sha256": sha(path), "layers": layers})
            print(json.dumps({"archive": path.name, "layers": layers}, ensure_ascii=False), flush=True)
    path = DATA / "sources/20240101noto_landslidepolygon.gpkg"
    if path.exists():
        with sqlite3.connect(path.as_uri() + "?mode=ro", uri=True) as connection:
            layers = connection.execute("select table_name,srs_id from gpkg_contents where data_type='features'").fetchall()
            summary = []
            for table, srs in layers:
                if not table.replace('_', '').isalnum():
                    raise ValueError("Unexpected table identifier")
                columns = connection.execute(f'pragma table_info("{table}")').fetchall()
                count = connection.execute(f'select count(*) from "{table}"').fetchone()[0]
                example = connection.execute(f'select * from "{table}" limit 1').fetchone()
                summary.append({"table": table, "srs_id": srs, "columns": columns, "objects": count,
                                "example_attributes": {columns[i][1]: value for i, value in enumerate(example) if not isinstance(value, bytes)}})
            write(DATA / "qualification/dem_inventory_structure.json", {"source_sha256": sha(path), "layers": summary})
            print(json.dumps({"dem_layers": summary}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=['sources', 'quarries', 'inspect', 'alternative-transport', 'main-overpass-node', 'global-alternative', 'system-route', 'map-definitions', 'selected-map-definitions', 'alternate-data-sources', 'geological-mine-leads', 'official-locality-hints'])
    command = parser.parse_args().command
    {'sources': sources, 'quarries': quarries, 'inspect': inspect, 'alternative-transport': alternative_transport,
     'main-overpass-node': main_overpass_node, 'global-alternative': documented_global_alternative,
     'system-route': system_route_query, 'map-definitions': map_definitions,
     'selected-map-definitions': selected_map_definitions, 'alternate-data-sources': alternate_data_sources,
     'geological-mine-leads': geological_mine_leads, 'official-locality-hints': official_locality_hints}[command]()
