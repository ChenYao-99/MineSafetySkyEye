"""Original official optical/DEM references vs current quarry discovery hints."""
import json
import sqlite3
import struct
import sys
from collections import Counter
from pathlib import Path

from screen_noto import DATA, HERE, read, write, sha, now
sys.path.insert(0, str(HERE.parent / 'phase7_20261015/vendor'))
sys.path.insert(0, str(HERE.parent / 'phase7_20261015'))
from shapely.geometry import shape, mapping, box
from shapely.wkb import loads
from shapely.strtree import STRtree
from rasterio.warp import transform_geom
from screen_hualien import protected_overlap


def gpkg_geometry(blob):
    if blob[:2] != b'GP' or blob[2] != 0:
        raise ValueError('Unexpected GeoPackage geometry header')
    flags = blob[3]
    if struct.unpack('<i' if flags & 1 else '>i', blob[4:8])[0] != 6675:
        raise ValueError('Source geometry CRS changed')
    size = {0: 0, 1: 32, 2: 48, 3: 48, 4: 64}[(flags >> 1) & 7]
    return loads(blob[8 + size:])


def main():
    mines = read(DATA / 'qualification/noto_quarry_discovery.geojson')['features']
    for mine in mines:
        geom = shape(mine['geometry'])
        mine['native'] = shape(transform_geom('EPSG:4326', 'EPSG:6675', mine['geometry']))
        mine['search'] = mine['native'].buffer(100)
        mine['protected_overlaps'] = protected_overlap(geom)
    tree = STRtree([mine['search'] for mine in mines])
    features, summary = [], []
    rule = read(HERE / 'protocol.json')['source_priority_rules']
    def collect(source, identifier, code, native, original_props):
        if not native.is_valid or native.is_empty:
            return False, False
        matches = []
        for i in tree.query(native):
            mine = mines[int(i)]
            if not native.intersects(mine['search']):
                continue
            overlap = native.intersection(mine['native']).area
            fraction = overlap / native.area
            eroded = native.buffer(-20).area
            priority = (native.area >= rule['minimum_area_m2'] and eroded >= rule['minimum_20m_eroded_area_m2'] and
                        fraction >= rule['minimum_quarry_overlap_fraction'] and not mine['protected_overlaps'] and code not in [5, 6])
            matches.append({'discovery_id': mine['properties']['discovery_id'], 'provider_feature_id': mine['properties'].get('provider_feature_id'),
                            'overlap_m2': overlap, 'overlap_fraction': fraction, 'distance_m': native.distance(mine['native']),
                            'source_priority': priority, 'protected_overlaps': mine['protected_overlaps'],
                            'historical_working_face_verified': False})
        if not matches:
            return False, False
        features.append({'type': 'Feature', 'geometry': transform_geom('EPSG:6675', 'EPSG:4326', mapping(native)),
                         'properties': {'source_family': source, 'source_id': identifier, 'source_code': code,
                                        'original_properties': original_props, 'source_area_m2': native.area,
                                        'core_20m_area_m2': native.buffer(-20).area, 'matches': matches,
                                        'mine_accident_label_admitted': False, 'training_allowed': False,
                                        'model_predictions_allowed': False}})
        return True, any(row['source_priority'] for row in matches)
    for path in sorted((DATA / 'sources').glob('*_hokai_all.geojson')):
        body = read(path)
        count = priority = invalid = 0
        for index, row in enumerate(body['features']):
            geom = shape(transform_geom('EPSG:4326', 'EPSG:6675', row['geometry']))
            if not geom.is_valid or geom.is_empty:
                invalid += 1
                continue
            nearby, eligible = collect('official_optical:' + path.stem, index, None, geom, row.get('properties'))
            count += int(nearby)
            priority += int(eligible)
        summary.append({'source': path.name, 'original_objects': len(body['features']), 'invalid_geometries': invalid,
                        'mine_or_100m_nearby_objects': count, 'large_priority_objects': priority})
    source = DATA / 'sources/20240101noto_landslidepolygon.gpkg'
    receipt = read(DATA / 'receipts/20240101noto_landslidepolygon.gpkg.json')
    if sha(source) != receipt['sha256']:
        raise ValueError('Original DEM interpretation changed')
    with sqlite3.connect(source.as_uri() + '?mode=ro', uri=True) as connection:
        selected, code_counts = {}, Counter()
        for mine in mines:
            west, south, east, north = mine['search'].bounds
            for fid, code, blob in connection.execute('select p.fid,p.code,p.geom from landslidepolygon p join rtree_landslidepolygon_geom r on p.fid=r.id where r.maxx>=? and r.minx<=? and r.maxy>=? and r.miny<=?', [west, east, south, north]):
                selected[fid] = (code, blob)
        count = priority = invalid = 0
        for fid, (code, blob) in selected.items():
            native = gpkg_geometry(blob)
            if not native.is_valid or native.is_empty:
                invalid += 1
                continue
            nearby, eligible = collect('official_dem_2025', fid, code, native, {'fid': fid, 'code': code})
            count += int(nearby)
            priority += int(eligible)
            if nearby:
                code_counts[code] += 1
        summary.append({'source': source.name, 'original_objects': 68844, 'invalid_geometries_near_mines': invalid,
                        'mine_or_100m_nearby_objects': count, 'large_priority_objects': priority,
                        'nearby_classification_code_counts': dict(code_counts)})
    write(DATA / 'qualification/mine_reference_relations.geojson', {'type': 'FeatureCollection', 'features': features})
    priorities = [row['properties'] for row in features if any(match['source_priority'] for match in row['properties']['matches'])]
    result = {'completed_at': now(), 'quarry_discovery_groups': len(mines), 'sources': summary,
              'large_source_priority_records': priorities, 'new_mine_accident_labels_admitted': 0,
              'mine_identity_and_source_imagery_verification_pending': True, 'new_model_predictions': 0, 'new_training': 0}
    write(DATA / 'qualification/mine_reference_discovery.json', result)
    print(json.dumps({'quarries': [{'id': mine['properties']['discovery_id'], 'provider_feature_id': mine['properties'].get('provider_feature_id'),
                                  'area_m2': mine['properties']['native_area_m2'], 'bounds': list(shape(mine['geometry']).bounds)} for mine in mines],
                      'sources': summary, 'priorities': priorities}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    main()
