"""Source-only candidate footprints within an analyst search window, not mine labels."""
import json
import sqlite3
import sys
from screen_noto import DATA, HERE, read, write, now, sha
sys.path.insert(0, str(HERE.parent / 'phase7_20261015/vendor'))
from shapely.geometry import shape, mapping, box
from rasterio.warp import transform_geom
import rasterio
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from join_references import gpkg_geometry
from overlay_source_references import rings


def main():
    root = DATA / 'imagery_overviews/NIIGYO_R2_JAN05'
    image_path = root / 'released_aerial_RGB_3857.tif'
    with rasterio.open(image_path) as src:
        rgb = src.read().transpose(1, 2, 0)
        grid, bounds = src.transform, src.bounds
    # Includes the visible quarry and peripheral slopes; never legal boundary.
    x0, y1 = grid * (200, 1080)
    x1, y0 = grid * (820, 400)
    search = shape(transform_geom('EPSG:3857', 'EPSG:6675', mapping(box(x0, y1, x1, y0))))
    rows = []
    for path in sorted((DATA / 'sources').glob('*_hokai_all.geojson')):
        for index, feature in enumerate(read(path)['features']):
            original = shape(transform_geom('EPSG:4326', 'EPSG:6675', feature['geometry']))
            if original.is_valid and original.intersects(search):
                rows.append(('optical', f'{path.stem}:{index}', None, original,
                             {'source_file': str(path), 'source_sha256': sha(path), 'feature_index': index,
                              'original_properties': feature.get('properties')}))
    with sqlite3.connect((DATA / 'sources/20240101noto_landslidepolygon.gpkg').as_uri() + '?mode=ro', uri=True) as con:
        west, south, east, north = search.bounds
        query = 'select p.fid,p.code,p.geom from landslidepolygon p join rtree_landslidepolygon_geom r on p.fid=r.id where r.maxx>=? and r.minx<=? and r.maxy>=? and r.miny<=?'
        for fid, code, blob in con.execute(query, [west, east, south, north]):
            original = gpkg_geometry(blob)
            if original.is_valid and original.intersects(search):
                rows.append(('dem', str(fid), code, original, {'original_fid': fid, 'original_code': code}))
    optical = [r[3] for r in rows if r[0] == 'optical']
    features, report = [], []
    fig, axes = plt.subplots(1, 2, figsize=(14, 8), constrained_layout=True)
    for axis in axes:
        axis.imshow(rgb, extent=[bounds.left, bounds.right, bounds.bottom, bounds.top])
        axis.set_xlim(x0, x1)
        axis.set_ylim(y1, y0)
        axis.set_axis_off()
    for family, identifier, code, native, provenance in rows:
        props = {'family': family, 'id': identifier, 'code': code, 'original_area_m2': native.area,
                 'eroded_10m_area_m2': native.buffer(-10).area,
                 'eroded_20m_area_m2': native.buffer(-20).area,
                 'max_optical_overlap_fraction': max((native.intersection(o).area / native.area for o in optical), default=0),
                 'search_window_is_not_quarry_boundary': True,
                 'mine_accident_label_admitted': False, 'training_allowed': False, **provenance}
        report.append(props)
        features.append({'type': 'Feature', 'geometry': transform_geom('EPSG:6675', 'EPSG:4326', mapping(native)), 'properties': props})
        geom = shape(transform_geom('EPSG:6675', 'EPSG:3857', mapping(native)))
        axis = axes[0 if family == 'optical' else 1]
        for ring in rings(geom):
            axis.plot(*ring.xy, color='yellow' if family == 'optical' else 'cyan', linewidth=.8)
        cx, cy = geom.representative_point().coords[0]
        axis.text(cx, cy, identifier.split(':')[-1] + (f' c{code}' if code else ''),
                  color='black', fontsize=8, bbox={'facecolor': 'white', 'alpha': .7, 'pad': .3})
    axes[0].set_title('Original optical components (IDs are original list indices)')
    axes[1].set_title('Original DEM components (fid and classification code)')
    fig.suptitle('Niigyo quarry search window | 2024-01-05 RGB | source-only qualification\nWindow is not a legal mine or accident boundary; no model output')
    out = DATA / 'qualification'
    fig.savefig(out / 'niigyo_source_candidates.png', dpi=170)
    plt.close(fig)
    write(out / 'niigyo_source_candidates.geojson', {'type': 'FeatureCollection', 'features': features})
    write(out / 'niigyo_source_candidates.json', {'completed_at': now(),
          'analyst_search_pixel_window': [200, 400, 820, 1080], 'original_image_sha256': sha(image_path),
          'rows': report, 'source_only_geometry_selection': True, 'new_model_predictions': 0})
    print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    main()
