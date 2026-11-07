"""Show original reference geometry on released imagery, with date limits visible."""
import json
import sqlite3
import sys
from pathlib import Path

from screen_noto import DATA, HERE, read, write, now
sys.path.insert(0, str(HERE.parent / 'phase7_20261015/vendor'))
from shapely.geometry import shape, mapping, box
from rasterio.warp import transform_geom
import rasterio
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from join_references import gpkg_geometry


def rings(geom):
    if geom.geom_type == 'Polygon':
        yield geom.exterior
    elif geom.geom_type == 'MultiPolygon':
        for polygon in geom.geoms:
            yield polygon.exterior


def main(identifiers=None):
    optical = []
    for path in sorted((DATA / 'sources').glob('*_hokai_all.geojson')):
        for index, feature in enumerate(read(path)['features']):
            geom = shape(transform_geom('EPSG:4326', 'EPSG:3857', feature['geometry']))
            if geom.is_valid:
                optical.append((path.name, index, geom))
    records = []
    connection = sqlite3.connect((DATA / 'sources/20240101noto_landslidepolygon.gpkg').as_uri() + '?mode=ro', uri=True)
    for identifier in identifiers or ['NIIGYO_R2_JAN05', 'NIIGYO_R2_JAN11', 'NIIGYO_R2_APR_CONTEXT']:
        root = DATA / 'imagery_overviews' / identifier
        with rasterio.open(root / 'released_aerial_RGB_3857.tif') as src:
            rgb = src.read().transpose(1, 2, 0)
            bounds, grid = src.bounds, src.transform
        frame = box(*bounds)
        local_optical = [(name, index, geom) for name, index, geom in optical if geom.intersects(frame)]
        native_frame = shape(transform_geom('EPSG:3857', 'EPSG:6675', mapping(frame)))
        west, south, east, north = native_frame.bounds
        dem = []
        query = 'select p.fid,p.code,p.geom from landslidepolygon p join rtree_landslidepolygon_geom r on p.fid=r.id where r.maxx>=? and r.minx<=? and r.maxy>=? and r.miny<=?'
        for fid, code, blob in connection.execute(query, [west, east, south, north]):
            original = gpkg_geometry(blob)
            if original.is_valid:
                geom = shape(transform_geom('EPSG:6675', 'EPSG:3857', mapping(original)))
                if geom.intersects(frame):
                    dem.append((fid, code, geom))
        plan = read(root / 'qualification.json')['plan']
        fig, axes = plt.subplots(1, 2, figsize=(14, 7), constrained_layout=True)
        extent = [bounds.left, bounds.right, bounds.bottom, bounds.top]
        for axis in axes:
            axis.imshow(rgb, extent=extent)
            axis.set_xlim(bounds.left, bounds.right)
            axis.set_ylim(bounds.bottom, bounds.top)
            axis.set_axis_off()
        for name, index, geom in local_optical:
            for ring in rings(geom):
                axes[0].plot(*ring.xy, color='yellow', linewidth=.7)
        for fid, code, geom in dem:
            for ring in rings(geom):
                axes[1].plot(*ring.xy, color='cyan', linewidth=.7)
        axes[0].set_title(f'Original January optical interpretation: {len(local_optical)} components')
        axes[1].set_title(f'Original DEM interpretation: {len(dem)} components')
        dates = plan.get('capture_date_interval')
        date_label = (' to '.join(dates) if dates else 'January 2024') + ': source QA'
        if 'APR_CONTEXT' in identifier:
            date_label += '; locating context only'
        fig.suptitle(identifier + '\n' + date_label + '\nApril context is not January pixel truth; no model output', fontsize=12,
                     fontfamily='DejaVu Sans')
        output = root / 'reference_overlay.png'
        fig.savefig(output, dpi=160)
        plt.close(fig)
        record = {'id': identifier, 'optical_reference_components_in_frame': len(local_optical),
                  'dem_reference_components_in_frame': len(dem), 'overlay': str(output),
                  'geographic_bounds_wgs84': list(shape(transform_geom('EPSG:3857', 'EPSG:4326', mapping(frame))).bounds),
                  'original_source_geometries_modified': False, 'model_predictions': 0,
                  'capture_date_interval': dates,
                  'components_are_not_independent_accidents_or_mines': True}
        write(root / 'reference_overlay.json', record)
        records.append(record)
        print(json.dumps(record, ensure_ascii=False), flush=True)
    connection.close()
    write(DATA / 'qualification/overview_reference_alignment.json', {'completed_at': now(), 'records': records})


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    main(sys.argv[1:] or None)
