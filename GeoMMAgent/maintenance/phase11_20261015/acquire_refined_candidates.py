"""Released high-resolution imagery around visibly benched quarry candidates."""
import json
import math
import sys
from screen_noto import DATA, HERE, read, write, now, sha
from acquire_quarry_overviews import acquire_plans


def main():
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
    specs = [
        ('NIIGYO_QUARRY_JAN05', 136.971, 37.334, '20240102_noto_wazimanaka_0105do', ['2024-01-05', '2024-01-05']),
        ('NIIGYO_QUARRY_APR_CONTEXT', 136.971, 37.334, '20240102noto_0405_0426do', ['2024-04-05', '2024-04-26']),
        ('KASHIHARA_CANDIDATE_JAN02', 137.1925, 37.391, '20240102noto_wazimahigashi_0102do', ['2024-01-02', '2024-01-02']),
        ('KASHIHARA_CANDIDATE_APR_CONTEXT', 137.1925, 37.391, '20240102noto_0405_0426do', ['2024-04-05', '2024-04-26']),
    ]
    plans, zoom = [], 17
    for name, lon, lat, layer_id, period in specs:
        x = int((lon + 180) / 360 * 2**zoom)
        y = int((1 - math.asinh(math.tan(math.radians(lat))) / math.pi) / 2 * 2**zoom)
        layer = layers[layer_id]
        plans.append({'id': name, 'name': name, 'numeric_as_wgs84_search_hint': [lon, lat],
                      'source_coordinate_datum_confirmed': False, 'mine_survey_position_confirmed': False,
                      'source_pdf': str(DATA / 'sources/gsj_wajima_2019_geology_explanation.pdf'), 'source_page': 6,
                      'search_location_type': 'georeferenced aerial visual lead near reported quarry locality; not official mine coordinates',
                      'source_of_search_center': 'NIIGYO/KASHIHARA_APR_CONTEXT source-only visual review; no model scores',
                      'tokyo_datum_hypothesis_wgs84': None, 'historical_datum_hypothesis_not_applicable': True,
                      'layer_id': layer_id, 'layer_title': layer['title'], 'capture_date_interval': period,
                      'provider_url_template': layer['url'], 'zoom': zoom,
                      'tiles': [[a, b] for b in range(y-2, y+3) for a in range(x-2, x+3)]})
    path = HERE / 'refined_quarry_imagery_protocol.json'
    protocol = {'registered_at': now(), 'planned_cases': plans, 'max_aerial_tile_requests': 100,
                'max_download_bytes': 32 * 1024**2, 'summary_filename': 'refined_quarry_imagery.json',
                'source_definition_sha256': sha(DATA / 'sources/layers_20240102noto.txt'),
                'april_context_is_not_january_truth': True, 'mine_identity_and_accident_admission_pending': True,
                'training_allowed': False, 'model_predictions': 0, 'cloud_generation_calls': 0}
    if path.exists():
        old = read(path)
        if old['planned_cases'] != plans:
            raise ValueError('Source-quality based selection changed')
        protocol = old
    else:
        write(path, protocol)
    acquire_plans(protocol)


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    main()
