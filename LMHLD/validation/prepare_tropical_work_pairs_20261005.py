"""Register official-footprint optical windows and a whole-region work holdout."""
from pathlib import Path
from datetime import datetime, timezone, timedelta
import json
import sys

ROOT = Path('D:/LMHLD')
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'cache/research_dependencies'))
from runtime_paths import configure_runtime_paths
configure_runtime_paths()
from model_registry import file_sha256
from shapely.geometry import shape, Point, box
from rasterio.warp import transform_geom, transform_bounds

SOURCE = ROOT / 'datasets/tropical_work_sources_20261005'
OUT = ROOT / 'datasets/tropical_work_pairs_20261005'
DEFINITIONS = {
    'AU_ANDOOM_SLATS2324': ('ANDOOM', 6000, 'Andoom_Weipa_complex', 'auxiliary_development_train'),
    'AU_BAUXITEHILLS_SLATS2324': ('BAUXITE HILLS 1', 5000, 'BauxiteHills', 'auxiliary_development_train'),
    'AU_MTCARBINE_SLATS2324': ('MOUNT CARBINE', 3500, 'MtCarbine', 'auxiliary_development_train'),
    'AU_CAPEFLATTERY_SLATS2324': ('CAPE FLATTERY', 5000, 'CapeFlattery_mining_region', 'development_normal_holdout'),
}


def load(path):
    return json.loads(path.read_text(encoding='utf-8'))


def write_frozen(path, value, ignore_timestamp=False):
    if path.exists():
        old = load(path)
        assert all(old.get(key) == val for key, val in value.items()
                   if not (ignore_timestamp and key == 'registered_at')), str(path)
    else:
        path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def run():
    OUT.mkdir(exist_ok=True)
    source = load(SOURCE / 'change_original.geojson')
    identities = load(SOURCE / 'mine_identity_original.geojson')
    sites = {}
    for case, (name, radius, group, role) in DEFINITIONS.items():
        identity = next(f for f in identities['features'] if f['properties']['occur_name'] == name)
        center = identity['geometry']['coordinates']
        crs = 'EPSG:32754' if center[0] < 144 else 'EPSG:32755'
        point = shape(transform_geom('EPSG:4326', crs, identity['geometry']))
        selected = []
        for feature in source['features']:
            geom = shape(transform_geom('EPSG:4326', crs, feature['geometry']))
            if geom.distance(point) <= radius:
                selected.append(geom)
        assert selected, case
        bounds = [min(g.bounds[0] for g in selected)-1000, min(g.bounds[1] for g in selected)-1000,
                  max(g.bounds[2] for g in selected)+1000, max(g.bounds[3] for g in selected)+1000]
        assert max(bounds[2]-bounds[0], bounds[3]-bounds[1]) < 15000
        bbox = list(transform_bounds(crs, 'EPSG:4326', *bounds))
        # Preserve every published mining polygon intersecting this complete window.
        features = [f for f in source['features'] if shape(f['geometry']).intersects(box(*bbox))]
        subset = {'type': 'FeatureCollection', 'features': features}
        folder = OUT / case
        folder.mkdir(exist_ok=True)
        target = folder / 'official_mining_change_subset.geojson'
        write_frozen(target, subset)
        sites[case] = {
            'site_id': case, 'mine_group': group, 'role': role, 'center_wgs84': center,
            'bbox_wgs84': bbox, 'crs': crs, 'commodity': identity['properties']['main_commodity'],
            'coordinate_source': 'official mine point;all source clearing polygons within fixed radius;complete extent plus1000m margin',
            'identity_source': 'https://spatial-gis.information.qld.gov.au/arcgis/rest/services/GeoscientificInformation/MiningResources/MapServer/14',
            'identity_object_id': identity['properties']['objectid'],
            'historical_identity_date_current_ms': identity['properties']['date_current'],
            'historical_point_not_current_operation_or_exact_mine_boundary': True,
            'original_change_object_ids': [f['properties']['objectid'] for f in features],
            'source_subset_sha256': file_sha256(target), 'original_source_sha256': file_sha256(SOURCE / 'change_original.geojson'),
            'source_radius_m': radius, 'not_a_new_blind_mine': True,
            'normal_or_stability_labels_available': False,
            'operator_or_report_general_context_not_pixel_truth': True,
            'related_Weipa_operation_training_only_not_independent_Amrun_holdout': name == 'ANDOOM',
            'scope': 'source-supported annual expected mining-work appearance;no hidden stability or exact daily accidentabsence assertion',
        }
    plan = {
        'registered_at': datetime.now(timezone(timedelta(hours=8))).isoformat(), 'sites': sites,
        'source_discovery_sha256': file_sha256(SOURCE / 'acquisition.json'),
        'source_nominal_period': 'August2023 toAugust2024',
        'scene_interval': '2023-07-01/2024-09-30',
        'catalogue_intervals': ['2023-07-01/2023-07-31', '2024-09-01/2024-09-30'],
        'scene_cutoffs': {'pre_latest': '2023-07-31', 'post_earliest': '2024-09-01'},
        'max_native_window_dimension': 1600,
        'acquisition_rule': 'latest fullcover July2023 and earliest fullcover September2024 with scenecloud<20%;no model or derivedlabelsize selection',
        'reference_rule': 'unchanged40m government polygon interior;common observed actual vegetation loss;confidence.25;unknown elsewhere;alignmentgateandoriginalvisualreview required',
        'training_sites': [case for case, row in sites.items() if row['role'] == 'auxiliary_development_train'],
        'normal_holdout_sites': [case for case, row in sites.items() if row['role'] == 'development_normal_holdout'],
        'whole_region_holdout_not_accident_blindset_or_verified_safety': True,
        'reserved_geography_and_Morenci_training_excluded': True,
        'classifier_predictions_at_registration': 0, 'pixel_labels_accepted': False,
        'max_round_acquisition_budget_MiB': 250,
    }
    write_frozen(OUT / 'site_plan.json', plan, ignore_timestamp=True)
    for case, row in sites.items():
        print(case, row['role'], 'official source polygons', len(row['original_change_object_ids']), flush=True)


if __name__ == '__main__':
    run()
