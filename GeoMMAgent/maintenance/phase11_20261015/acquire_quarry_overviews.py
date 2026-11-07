"""Date-fixed released aerial tiles around published historic quarry leads."""
import json
import math
import sys
from pathlib import Path

from screen_noto import DATA, HERE, read, write, sha, now
sys.path.insert(0, str(HERE.parent / 'phase7_20261015/vendor'))
from shapely.geometry import shape, mapping, Point, box
from rasterio.warp import transform, transform_geom
from rasterio.transform import Affine
import numpy as np
import rasterio
from PIL import Image
import requests

HALF = 20037508.342789244
LEADS = [
    ('NT10_KANDADANI', 'Wajima Kandadani quarry', 37, 21.52, 136, 56.47, '20240102noto_wazimanaka_0102do'),
    ('NT11_MOTOGI', 'Noto Motogi quarry', 37, 16.25, 137, 1.46, '20240102noto_anamizu_0111do'),
    ('NT03_YOTSUYA', 'Yanagida Yotsuya quarry', 37, 21.38, 137, 4.44, '20240102noto_wazimahigashi_0102do'),
    ('NT06_SAIHOJI', 'West of Suzu Saihoji quarry', 37, 23.73, 137, 12.75, '20240102noto_suzu_0102do'),
    ('NT13_GANMON', 'Togi Ganmon quarry', 37, 5.67, 136, 44.02, '20240102noto_wazimanishi_0117do'),
]


def main(round_id='historical'):
    definitions = read(DATA / 'sources/layers_20240102noto.txt')
    layers = {}
    def walk(value):
        if isinstance(value, dict):
            if value.get('id') and value.get('url'):
                layers[value['id']] = value
            for item in value.values():
                walk(item)
        elif isinstance(value, list):
            for item in value:
                walk(item)
    walk(definitions)
    plans = []
    zoom = 15
    specifications = LEADS
    if round_id == 'documented':
        specifications = []
        for prefix, filename, name, january_layer in [
            ('NIIGYO', 'wajima_niigyo_geocoder.json', '2019 reported andesite quarry locality', '20240102_noto_wazimanaka_0105do'),
            ('KASHIHARA', 'suzu_kashihara_geocoder.json', '2019 reported dacite quarry locality', '20240102_noto_suzu_0105do')]:
            longitude, latitude = read(DATA / 'sources' / filename)[0]['geometry']['coordinates']
            for suffix, layer_id in [('JAN05', january_layer), ('APR_CONTEXT', '20240102noto_0405_0426do')]:
                specifications.append((prefix + '_' + suffix, name, int(latitude), (latitude-int(latitude))*60,
                                       int(longitude), (longitude-int(longitude))*60, layer_id))
    for identifier, name, latd, latm, lond, lonm, layer_id in specifications:
        lon, lat = lond + lonm / 60, latd + latm / 60
        # Original 1981 coordinate datum is unspecified. Keep both hypotheses
        # as search hints; never designate either as a measured mine position.
        tokyo_lon, tokyo_lat = transform('EPSG:4301', 'EPSG:4326', [lon], [lat])
        x = int((lon + 180) / 360 * 2**zoom)
        y = int((1 - math.asinh(math.tan(math.radians(lat))) / math.pi) / 2 * 2**zoom)
        layer = layers[layer_id]
        plans.append({'id': identifier, 'name': name, 'original_published_dms': [latd, latm, lond, lonm] if round_id == 'historical' else None,
                      'numeric_as_wgs84_search_hint': [lon, lat], 'tokyo_datum_hypothesis_wgs84': [tokyo_lon[0], tokyo_lat[0]],
                      'source_coordinate_datum_confirmed': False, 'historic_source_year': 1981 if round_id == 'historical' else 2019,
                      'source_pdf': str(DATA / 'sources' / ('shibata1981_noto_quarry_samples.pdf' if round_id == 'historical' else 'gsj_wajima_2019_geology_explanation.pdf')),
                      'source_page': 3 if round_id == 'historical' else 6,
                      'search_location_type': 'historic quarry sampling point, datum unresolved' if round_id == 'historical' else 'official locality search point, not quarry coordinates',
                      'layer_id': layer_id, 'layer_title': layer['title'], 'provider_url_template': layer['url'],
                      'zoom': zoom, 'tiles': [[a, b] for b in range(y-1, y+2) for a in range(x-1, x+2)]})
    path = HERE / ('quarry_aerial_overview_protocol.json' if round_id == 'historical' else 'documented_quarry_overview_protocol.json')
    protocol = {'registered_at': now(), 'source_pdf_sha256': sha(DATA / 'sources' / ('shibata1981_noto_quarry_samples.pdf' if round_id == 'historical' else 'gsj_wajima_2019_geology_explanation.pdf')),
                'official_layer_definition_sha256': sha(DATA / 'sources/layers_20240102noto.txt'),
                'purpose': 'Verify present physical quarry context around historic published points; not assume mining accidents',
                'planned_cases': plans, 'max_aerial_tile_requests': 45 if round_id == 'historical' else 36, 'max_download_bytes': 16 * 1024**2,
                'april_context_cannot_be_used_as_january_pixel_truth': True,
                'provider_published_tiles_are_not_unresampled_sensor_images': True,
                'classification_or_training_allowed': False, 'model_predictions': 0}
    if path.exists():
        old = read(path)
        if old['planned_cases'] != plans:
            raise ValueError('Frozen overview selection changed')
        protocol = old
    else:
        write(path, protocol)
    # Output receipt selection does not alter the original frozen acquisition.
    protocol['summary_filename'] = 'historic_quarry_overviews.json' if round_id == 'historical' else 'documented_quarry_overviews.json'
    acquire_plans(protocol)


def acquire_plans(protocol):
    plans = protocol['planned_cases']
    if sum(len(plan['tiles']) for plan in plans) > protocol['max_aerial_tile_requests']:
        raise ValueError('Selected tile count exceeds registered request budget')
    cleanup_path = DATA / 'qualification/cleanup_manifest.json'
    if cleanup_path.exists():
        removed = set(read(cleanup_path).get('discarded_imagery_case_ids', []))
        if any(plan['id'] in removed for plan in plans):
            raise ValueError('Discarded qualification trial; archived receipt retained, no implicit reacquisition')
    session = requests.Session()
    session.headers['User-Agent'] = 'GeoMMAgentNotoReference/1.0'
    total = 0
    reports = []
    reusable = {}
    for path in (DATA / 'imagery_overviews').glob('*/_tiles/*.png.json'):
        record = read(path)
        if record.get('ok') and Path(record['path']).exists():
            reusable[record['url']] = record
    for plan in plans:
        zoom = plan['zoom']
        side = int(math.sqrt(len(plan['tiles'])))
        if side * side != len(plan['tiles']) or side % 2 != 1:
            raise ValueError('Only an odd square source tile grid is supported')
        pixels = side * 256
        root = DATA / 'imagery_overviews' / plan['id']
        tiles_root = root / '_tiles'
        tiles_root.mkdir(parents=True, exist_ok=True)
        mosaic = np.zeros((pixels, pixels, 3), np.uint8)
        validity = np.zeros((pixels, pixels), np.uint8)
        ledger = []
        for index, (x, y) in enumerate(plan['tiles']):
            filename = f'{zoom}_{x}_{y}.png'
            target, receipt_path = tiles_root / filename, tiles_root / (filename + '.json')
            receipt = read(receipt_path) if receipt_path.exists() else None
            url = plan['provider_url_template'].format(z=zoom, x=x, y=y)
            if receipt is None and url in reusable:
                import os
                import shutil
                previous = reusable[url]
                old_path = Path(previous['path']).resolve()
                if not old_path.is_relative_to(DATA.resolve()) or sha(old_path) != previous['sha256']:
                    raise ValueError('Reusable released tile identity changed')
                try:
                    os.link(old_path, target)
                except OSError:
                    shutil.copyfile(old_path, target)
                receipt = {**previous, 'path': str(target), 'reused_verified_original_tile': True,
                           'reuse_source_path': str(old_path)}
                write(receipt_path, receipt)
            if receipt and receipt.get('ok'):
                if sha(target) != receipt['sha256']:
                    raise ValueError('Original released aerial tile changed')
            elif receipt is None:
                url = plan['provider_url_template'].format(z=zoom, x=x, y=y)
                receipt = {'url': url, 'started_at': now(), 'limit_bytes': 1024**2}
                part = target.with_suffix('.png.part')
                try:
                    with session.get(url, stream=True, timeout=(12, 30)) as response:
                        receipt['status_code'] = response.status_code
                        response.raise_for_status()
                        count = 0
                        with part.open('wb') as stream:
                            for block in response.iter_content(65536):
                                count += len(block)
                                if count > receipt['limit_bytes'] or total + count > protocol['max_download_bytes']:
                                    raise ValueError('Aerial overview transfer budget exceeded')
                                stream.write(block)
                        with Image.open(part) as image:
                            if image.size != (256, 256):
                                raise ValueError('Unexpected provider tile dimensions')
                            image.verify()
                        part.replace(target)
                        receipt.update(ok=True, path=str(target), bytes=count, sha256=sha(target))
                except (requests.RequestException, ValueError, OSError) as exc:
                    part.unlink(missing_ok=True)
                    receipt.update(ok=False, error=f'{type(exc).__name__}: {exc}')
                receipt['completed_at'] = now()
                write(receipt_path, receipt)
            ledger.append(receipt)
            if not receipt.get('ok'):
                continue
            total += receipt['bytes']
            with Image.open(target) as image:
                rgba = np.asarray(image.convert('RGBA'))
            row, col = divmod(index, side)
            mosaic[row*256:(row+1)*256, col*256:(col+1)*256] = rgba[:, :, :3]
            validity[row*256:(row+1)*256, col*256:(col+1)*256] = (rgba[:, :, 3] > 0).astype(np.uint8)
        x0, y0 = plan['tiles'][0]
        tile_width = 2 * HALF / 2**zoom
        grid = Affine(tile_width / 256, 0, -HALF + x0 * tile_width, 0, -tile_width / 256, HALF - y0 * tile_width)
        image_path = root / 'released_aerial_RGB_3857.tif'
        profile = dict(driver='GTiff', height=pixels, width=pixels, count=3, dtype='uint8', crs='EPSG:3857', transform=grid, compress='deflate')
        with rasterio.open(image_path, 'w', **profile) as dst:
            dst.write(mosaic.transpose(2, 0, 1))
            dst.write_mask(validity * 255)
            dst.update_tags(source_layer=plan['layer_id'], role='source_qualification_only', imagery_date_from_layer_title=plan['layer_title'])
        preview = root / 'overview.png'
        Image.fromarray(mosaic).save(preview)
        report = {'completed_at': now(), 'plan': plan, 'valid_source_pixel_fraction': float(validity.mean()),
                  'source_tiles': ledger, 'released_RGB': str(image_path), 'released_RGB_sha256': sha(image_path),
                  'preview': str(preview), 'preview_sha256': sha(preview),
                  'current_quarry_identity_confirmed': False, 'mine_accident_reference_admitted': False,
                  'training_allowed': False, 'model_predictions': 0}
        write(root / 'qualification.json', report)
        reports.append({'id': plan['id'], 'valid_source_pixel_fraction': report['valid_source_pixel_fraction'],
                        'failed_tiles': sum(not row.get('ok') for row in ledger)})
        print(json.dumps(reports[-1]), flush=True)
    session.close()
    write(DATA / 'qualification' / protocol['summary_filename'],
          {'reports': reports, 'source_bytes': total, 'model_predictions': 0})


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    main(sys.argv[1] if len(sys.argv) > 1 else 'historical')
