"""Align the quarry with the final January photography used by the optical source."""
import sys
from screen_noto import DATA, HERE, now, read, write
from acquire_quarry_overviews import acquire_plans


def main():
    path = HERE / 'january_final_epoch_protocol.json'
    if not path.exists():
        plan = dict(read(HERE / 'r2_source_localization_protocol.json')['planned_cases'][0])
        definitions = read(DATA / 'sources/layers_20240102noto.txt')
        found = []
        def walk(value):
            if isinstance(value, dict):
                if value.get('id') == '20240102noto_wazimanaka_0111do':
                    found.append(value)
                for item in value.values():
                    walk(item)
            elif isinstance(value, list):
                for item in value:
                    walk(item)
        walk(definitions)
        if len(found) != 1:
            raise ValueError('Ambiguous official January 11 definition')
        layer = found[0]
        plan.update(id='NIIGYO_R2_JAN11', name='NIIGYO_R2_JAN11', layer_id=layer['id'],
                    layer_title=layer['title'], provider_url_template=layer['url'],
                    capture_date_interval=['2024-01-11', '2024-01-11'])
        write(path, {'registered_at': now(), 'planned_cases': [plan],
              'max_aerial_tile_requests': 25, 'max_download_bytes': 8 * 1024**2,
              'summary_filename': 'january_final_epoch_acquisition.json',
              'purpose': 'January 15 optical inventory uses January 2/5/11 photos; avoid April truth substitution',
              'training_allowed': False, 'model_predictions': 0})
    acquire_plans(read(path))


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    main()
