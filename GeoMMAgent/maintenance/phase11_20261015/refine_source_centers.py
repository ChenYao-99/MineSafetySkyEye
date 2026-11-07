"""Second source-only localization: affine-tied quarry feature, not guessed GPS."""
import math
import sys
from screen_noto import DATA, HERE, read, write, sha, now
from acquire_quarry_overviews import acquire_plans
import rasterio
from rasterio.warp import transform


def main():
    original = DATA / 'imagery_overviews/NIIGYO_APR_CONTEXT/released_aerial_RGB_3857.tif'
    # Reviewed physical benched surface within the original released overview.
    with rasterio.open(original) as src:
        east, north = src.transform * (735.5, 666.5)
    lon, lat = transform('EPSG:3857', 'EPSG:4326', [east], [north])
    historic = next(row for row in read(HERE / 'quarry_aerial_overview_protocol.json')['planned_cases'] if row['id'] == 'NT06_SAIHOJI')
    one, two = historic['numeric_as_wgs84_search_hint'], historic['tokyo_datum_hypothesis_wgs84']
    saihoji = [(one[i]+two[i])/2 for i in range(2)]
    definitions, layers = read(DATA / 'sources/layers_20240102noto.txt'), {}
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
    specs = [('NIIGYO_R2_JAN05', lon[0], lat[0], '20240102_noto_wazimanaka_0105do', ['2024-01-05', '2024-01-05']),
             ('NIIGYO_R2_APR_CONTEXT', lon[0], lat[0], '20240102noto_0405_0426do', ['2024-04-05', '2024-04-26']),
             ('SAIHOJI_R2_JAN02', *saihoji, '20240102noto_suzu_0102do', ['2024-01-02', '2024-01-02']),
             ('SAIHOJI_R2_APR_CONTEXT', *saihoji, '20240102noto_0405_0426do', ['2024-04-05', '2024-04-26'])]
    plans, zoom = [], 17
    for identifier, longitude, latitude, layer_id, dates in specs:
        x = int((longitude+180)/360*2**zoom)
        y = int((1-math.asinh(math.tan(math.radians(latitude)))/math.pi)/2*2**zoom)
        layer = layers[layer_id]
        plans.append({'id': identifier, 'name': identifier, 'numeric_as_wgs84_search_hint': [longitude, latitude],
                      'source_coordinate_datum_confirmed': False, 'mine_survey_position_confirmed': False,
                      'search_location_type': 'affine-tied visible benched feature' if identifier.startswith('NIIGYO') else 'midpoint of unresolved original/Tokyo-datum historic quarry hints',
                      'layer_id': layer_id, 'layer_title': layer['title'], 'provider_url_template': layer['url'],
                      'capture_date_interval': dates, 'zoom': zoom,
                      'tiles': [[a,b] for b in range(y-2,y+3) for a in range(x-2,x+3)]})
    protocol = {'registered_at': now(), 'planned_cases': plans, 'max_aerial_tile_requests': 100,
                'max_download_bytes': 32*1024**2, 'summary_filename': 'r2_source_localization.json',
                'reason': 'Initial refined center was east of the physical quarry; use original affine/pixel tie, and bracket the historical datum for the second quarry lead',
                'niigyo_original_image_sha256': sha(original), 'niigyo_source_pixel_xy': [735.5,666.5],
                'saihoji_original_historic_hint': historic,
                'all_prior_protocols_preserved': True, 'no_model_outputs_viewed': True,
                'april_context_is_not_january_truth': True, 'training_allowed': False, 'model_predictions': 0}
    path = HERE / 'r2_source_localization_protocol.json'
    if path.exists():
        old = read(path)
        if old['planned_cases'] != plans:
            raise ValueError('Frozen source localization changed')
        protocol = old
    else:
        write(path, protocol)
    print({'niigyo_affine_search_center': [lon[0],lat[0]], 'saihoji_search_center': saihoji}, flush=True)
    acquire_plans(protocol)


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    main()
