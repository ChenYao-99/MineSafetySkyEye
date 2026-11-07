"""Select pre-event annual released photography by fixed newest-first years."""
import json
import sys
from screen_noto import DATA, HERE, now, read, write
from acquire_quarry_overviews import acquire_plans


def main():
    protocol_path = HERE / 'pre_event_context_protocol.json'
    base = read(HERE / 'r2_source_localization_protocol.json')['planned_cases'][0]
    years = [2022, 2020, 2016, 2010]
    plans = []
    for year in years:
        plan = dict(base)
        plan.update(id=f'NIIGYO_PRE_FY{year}_PROBE', name=f'Pre-event annual photo FY{year}',
                    layer_id=f'nendophoto{year}', layer_title=f'Annual photograph fiscal year {year}',
                    provider_url_template=f'https://maps.gsi.go.jp/xyz/nendophoto{year}/{{z}}/{{x}}/{{y}}.png',
                    capture_date_interval=[f'{year}-04-01', f'{year+1}-03-31'],
                    capture_date_precision='fiscal-year interval; exact day not established',
                    tiles=[base['tiles'][len(base['tiles']) // 2]])
        plans.append(plan)
    protocol = {'registered_at': now(), 'planned_cases': plans,
                'max_aerial_tile_requests': 4, 'max_download_bytes': 4 * 1024**2,
                'selection_rule': 'newest fixed candidate year with returned tile; stop at first success',
                'maximum_followup_mosaic_tiles': 25, 'maximum_followup_bytes': 8 * 1024**2,
                'official_schema_source': 'https://maps.gsi.go.jp/development/index.html',
                'purpose': 'exclude pre-existing excavation as event failure; not safety labels',
                'model_predictions': 0, 'training_allowed': False,
                'annual_layer_not_exact_sensor_date': True}
    if protocol_path.exists():
        protocol = read(protocol_path)
    else:
        write(protocol_path, protocol)
    selected = None
    for plan in protocol['planned_cases']:
        acquire_plans({**protocol, 'planned_cases': [plan], 'summary_filename': plan['id'] + '.json'})
        result = read(DATA / 'imagery_overviews' / plan['id'] / 'qualification.json')
        if result['valid_source_pixel_fraction'] > .95:
            selected = plan
            break
    if selected:
        plan = {**selected, 'id': selected['id'].replace('_PROBE', ''), 'tiles': base['tiles']}
        expanded = {**protocol, 'planned_cases': [plan],
                    'max_aerial_tile_requests': protocol['maximum_followup_mosaic_tiles'],
                    'max_download_bytes': protocol['maximum_followup_bytes'],
                    'summary_filename': 'pre_event_context_acquisition.json'}
        followup = HERE / 'pre_event_context_selected.json'
        if not followup.exists():
            write(followup, expanded)
        acquire_plans(read(followup))
    write(DATA / 'qualification/pre_event_context_selection.json',
          {'completed_at': now(), 'selected': selected, 'model_predictions': 0,
           'pre_event_is_not_verified_safe_background': True})


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    main()
