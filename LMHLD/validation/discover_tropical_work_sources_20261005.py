"""Bounded official mining-change discovery, before optical or model selection."""
from pathlib import Path
from datetime import datetime, timezone, timedelta
import json
import sys

ROOT = Path('D:/LMHLD')
sys.path.insert(0, str(ROOT))
from runtime_paths import configure_runtime_paths
configure_runtime_paths()
from model_registry import file_sha256
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

OUT = ROOT / 'datasets/tropical_work_sources_20261005'
SLATS = 'https://spatial-gis.information.qld.gov.au/arcgis/rest/services/Environment/SLATS/MapServer/255'
MINES = 'https://spatial-gis.information.qld.gov.au/arcgis/rest/services/GeoscientificInformation/MiningResources/MapServer/14'
BBOX = [141, -17.5, 146.5, -10]


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def run():
    OUT.mkdir(exist_ok=True)
    plan = {'registered_at': datetime.now(timezone(timedelta(hours=8))).isoformat(),
            'bbox_wgs84': BBOX, 'change_source': SLATS, 'identity_source': MINES,
            'class_filter': "class = 'Clearing - Mine'",
            'source_nominal_period': 'August2023 toAugust2024',
            'selection': 'all official mine clearing features and operating mine points inside fixed tropical north Queensland region;no optical or model score selection',
            'max_change_feature_count': 2000, 'model_predictions': 0,
            'annual_change_not_exact_daily_cause_or_safety': True,
            'development_only_no_reserved_cases': True}
    path = OUT / 'discovery_plan.json'
    if path.exists():
        old = json.loads(path.read_text(encoding='utf-8'))
        assert all(old[key] == value for key, value in plan.items() if key != 'registered_at')
    else:
        write(path, plan)
    session = requests.Session()
    session.mount('https://', HTTPAdapter(max_retries=Retry(
        total=3, backoff_factor=1, status_forcelist=(429, 500, 502, 503, 504))))

    def obtain(name, url, params):
        target = OUT / name
        if target.exists():
            return json.loads(target.read_text(encoding='utf-8'))
        response = session.get(url, params=params, timeout=(10, 45))
        response.raise_for_status()
        assert len(response.content) < 25 * 1024**2
        value = response.json()
        if value.get('error'):
            raise ValueError(value['error'])
        write(target, value)
        return value

    extent = json.dumps({'xmin': BBOX[0], 'ymin': BBOX[1], 'xmax': BBOX[2],
                         'ymax': BBOX[3], 'spatialReference': {'wkid': 4326}})
    counts = {}
    for name, url, where in [('change', SLATS, "class = 'Clearing - Mine'"),
                             ('mine_identity', MINES, '1=1')]:
        metadata = obtain(name + '_metadata.json', url, {'f': 'json'})
        args = {'where': where, 'geometry': extent, 'geometryType': 'esriGeometryEnvelope',
                'inSR': 4326, 'spatialRel': 'esriSpatialRelIntersects', 'f': 'json',
                'returnIdsOnly': 'true'}
        identifiers = obtain(name + '_IDs.json', url + '/query', args).get('objectIds') or []
        assert len(identifiers) <= 2000
        args.update(f='geojson', returnIdsOnly='false', returnGeometry='true',
                    outFields='*', outSR=4326, resultRecordCount=2000)
        collection = obtain(name + '_original.geojson', url + '/query', args)
        assert not collection.get('exceededTransferLimit')
        field = metadata.get('objectIdField', 'objectid')
        assert {feature['properties'][field] for feature in collection['features']} == set(identifiers)
        counts[name] = len(identifiers)
    write(OUT / 'acquisition.json', {
        'acquired_at': datetime.now(timezone(timedelta(hours=8))).isoformat(),
        'counts': counts, 'discovery_plan_sha256': file_sha256(path),
        'assets_sha256': {p.name: file_sha256(p) for p in OUT.iterdir()
                         if p.is_file() and p.name != 'acquisition.json'},
        'model_predictions': 0, 'accepted_pixel_labels': 0,
    })
    print('Official source counts:', counts, flush=True)
    for feature in json.loads((OUT / 'mine_identity_original.geojson').read_text(encoding='utf-8'))['features']:
        print(feature['properties'], feature['geometry'], flush=True)


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    run()
