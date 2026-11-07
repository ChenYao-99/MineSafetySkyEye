"""Bounded OSM-derived quarry discovery, not historical or accident truth."""
from __future__ import annotations
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

from screen_noto import DATA, HERE, read, write, sha, now
sys.path.insert(0, str(HERE / 'vendor'))
sys.path.insert(0, str(HERE.parent / 'phase7_20261015/vendor'))
import requests
import mapbox_vector_tile
from shapely.geometry import shape, mapping, box
from shapely.ops import transform, unary_union
from rasterio.warp import transform_geom

HALF_WORLD = 20037508.342789244


def tile_index(lon, lat, zoom):
    n = 2**zoom
    return int(math.floor((lon + 180) / 360 * n)), int(math.floor((1 - math.asinh(math.tan(math.radians(lat))) / math.pi) / 2 * n))


def native_geometry(geometry, x, y, zoom, extent):
    scale = 2 * HALF_WORLD / (2**zoom * extent)
    def convert(xs, ys, zs=None):
        try:
            return ([-HALF_WORLD + (x * extent + v) * scale for v in xs],
                    [HALF_WORLD - (y * extent + v) * scale for v in ys])
        except TypeError:
            return -HALF_WORLD + (x * extent + xs) * scale, HALF_WORLD - (y * extent + ys) * scale
    return transform(convert, shape(geometry))


def main():
    amendment = read(HERE / 'vector_tile_transport_addendum.json')
    meta = read(DATA / 'sources/openfreemap_tilejson.json')
    if len(meta['tiles']) != 1:
        raise ValueError('Ambiguous tile endpoint')
    template = meta['tiles'][0]
    if not template.startswith('https://tiles.openfreemap.org/planet/'):
        raise ValueError('Unexpected vector provider')
    west, south, east, north = read(HERE / 'protocol.json')['discovery_bounds_wgs84']
    zoom = 11
    x0, y0 = tile_index(west, north, zoom)
    x1, y1 = tile_index(east, south, zoom)
    tiles = [(x, y) for y in range(y0, y1 + 1) for x in range(x0, x1 + 1)]
    if len(tiles) > amendment['max_vector_tiles']:
        raise ValueError('Selected tile grid exceeds registered budget')
    plan = {'registered_at': now(), 'zoom': zoom, 'tile_grid': tiles, 'provider_template': template,
            'metadata_sha256': sha(DATA / 'sources/openfreemap_tilejson.json'),
            'source_coordinate_quantization_projected_m': 2 * HALF_WORLD / (2**zoom * 4096),
            'purpose': 'quarry class discovery only; simplified/clipped current OSM-derived geometry',
            'historical_footprint_verified': False, 'source_selection_uses_model_scores': False}
    path = DATA / 'qualification/quarry_tile_plan.json'
    if path.exists():
        old = read(path)
        if any(old[key] != plan[key] for key in ['zoom', 'provider_template', 'metadata_sha256']):
            raise ValueError('Frozen vector source changed')
        plan = old
    else:
        write(path, plan)
    folder = DATA / 'sources/quarry_vector_tiles'
    receipts = DATA / 'receipts/quarry_vector_tiles'
    folder.mkdir(parents=True, exist_ok=True)
    receipts.mkdir(parents=True, exist_ok=True)
    total, failures, parts, properties = 0, [], defaultdict(list), {}
    class_counts = Counter()
    session = requests.Session()
    session.headers['User-Agent'] = amendment['unique_user_agent']
    for ordinal, (x, y) in enumerate(tiles, 1):
        name = f'{zoom}_{x}_{y}.pbf'
        target, record_path = folder / name, receipts / (name + '.json')
        record = read(record_path) if record_path.exists() else None
        if record and record.get('ok'):
            if sha(target) != record['sha256']:
                raise ValueError('Retained vector tile changed')
        elif record:
            failures.append(name)
            continue
        else:
            url = template.format(z=zoom, x=x, y=y)
            record = {'name': name, 'url': url, 'started_at': now(), 'max_bytes': 2 * 1024**2}
            part = target.with_suffix('.pbf.part')
            try:
                with session.get(url, stream=True, timeout=(12, 30)) as response:
                    record.update(status_code=response.status_code, content_encoding=response.headers.get('Content-Encoding'))
                    response.raise_for_status()
                    count = 0
                    with part.open('wb') as stream:
                        for block in response.iter_content(65536):
                            count += len(block)
                            if count > record['max_bytes'] or total + count > amendment['max_vector_transfer_bytes']:
                                raise ValueError('Offline provider download budget reached')
                            stream.write(block)
                    mapbox_vector_tile.decode(part.read_bytes(), default_options={'y_coord_down': True})
                    part.replace(target)
                    record.update(ok=True, path=str(target), bytes=count, sha256=sha(target))
            except (requests.RequestException, ValueError, OSError) as exc:
                part.unlink(missing_ok=True)
                record.update(ok=False, error=f'{type(exc).__name__}: {exc}')
                failures.append(name)
            record['completed_at'] = now()
            write(record_path, record)
        if not record.get('ok'):
            continue
        total += record['bytes']
        if total > amendment['max_vector_transfer_bytes']:
            raise ValueError('Aggregate offline vector budget reached')
        body = mapbox_vector_tile.decode(target.read_bytes(), default_options={'y_coord_down': True})
        for layer_name, layer in body.items():
            if layer_name not in ['landuse', 'landcover']:
                continue
            for feature in layer['features']:
                props = feature['properties']
                class_counts[(layer_name, props.get('class'))] += 1
                if props.get('class') != 'quarry' and props.get('subclass') != 'quarry':
                    continue
                if feature['geometry']['type'] not in ['Polygon', 'MultiPolygon']:
                    continue
                geom = native_geometry(feature['geometry'], x, y, zoom, layer['extent'])
                if not geom.is_valid or geom.is_empty:
                    raise ValueError('Invalid decoded quarry geometry; no silent repair')
                key = f"{layer_name}:{feature.get('id')}" if feature.get('id') else f"unidentified:{name}:{len(parts)}"
                parts[key].append(geom)
                properties[key] = {'provider_feature_id': feature.get('id'), 'original_properties': props,
                                   'source_layer': layer_name, 'source_is_not_certified_mine_identity': True}
        if ordinal % 8 == 0 or ordinal == len(tiles):
            print(json.dumps({'tiles_processed': ordinal, 'tiles_total': len(tiles), 'bytes': total,
                              'quarry_feature_groups': len(parts), 'failed_tiles': len(failures)}), flush=True)
    session.close()
    features = []
    for key, geometries in parts.items():
        geom = unary_union(geometries)
        geographic = transform_geom('EPSG:3857', 'EPSG:4326', mapping(geom))
        native = shape(transform_geom('EPSG:3857', 'EPSG:6675', mapping(geom)))
        features.append({'type': 'Feature', 'geometry': geographic, 'properties': {
            'discovery_id': key, **properties[key], 'native_area_m2': native.area,
            'tile_fragments': len(geometries), 'historical_footprint_verified': False,
            'pixel_or_accident_truth': False}})
    outside = [row['properties']['discovery_id'] for row in features
               if not shape(row['geometry']).intersects(box(west, south, east, north))]
    for row in features:
        row['properties']['intersects_registered_discovery_bounds'] = row['properties']['discovery_id'] not in outside
    output = DATA / 'qualification/noto_quarry_discovery.geojson'
    write(output, {'type': 'FeatureCollection', 'features': features,
          'attribution': 'OpenFreeMap / OpenMapTiles / OpenStreetMap contributors, ODbL',
          'provider_snapshot_url': template, 'current_simplified_land_use_only': True})
    summary = {'completed_at': now(), 'selected_tiles': len(tiles), 'failed_tiles': failures, 'bytes': total,
               'quarry_feature_groups': len(features), 'class_counts': [{'layer': key[0], 'class': key[1], 'count': value} for key, value in class_counts.items()],
               'groups_intersecting_registered_bounds': len(features) - len(outside),
               'tile_margin_groups_outside_registered_bounds': outside,
               'coverage_for_selected_grid_complete': not failures, 'coverage_of_all_actual_quarries_not_established': True,
               'new_training': 0, 'new_predictions': 0}
    write(DATA / 'qualification/quarry_tile_discovery.json', summary)
    print(json.dumps({key: summary[key] for key in ['selected_tiles', 'failed_tiles', 'bytes', 'quarry_feature_groups']}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    main()
