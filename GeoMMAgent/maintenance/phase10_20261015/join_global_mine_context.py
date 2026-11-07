"""New mining footprint provider × original official event polygons, source-only."""
from __future__ import annotations
import io
import json
import sqlite3
import struct
import sys
from zipfile import ZipFile
from pathlib import Path

from qualify_mine_context import DATA, OLD, HERE, read, write, sha, now
sys.path.insert(0, str(HERE.parent / "phase7_20261015/vendor"))
sys.path.insert(0, str(HERE.parent / "phase7_20261015"))
import shapefile
from shapely.geometry import shape, box, mapping
from shapely.wkb import loads
from shapely.strtree import STRtree
from rasterio.warp import transform_geom
from rasterio.crs import CRS
from screen_hualien import protected_overlap


def gpkg_polygon(blob):
    if blob[:2] != b"GP" or blob[2] != 0:
        raise ValueError("Unexpected GeoPackage header")
    flags = blob[3]
    byteorder = '<' if flags & 1 else '>'
    if struct.unpack(byteorder + 'i', blob[4:8])[0] != 4326:
        raise ValueError("Unexpected source geometry CRS")
    envelope_sizes = {0: 0, 1: 32, 2: 48, 3: 48, 4: 64}
    offset = 8 + envelope_sizes[(flags >> 1) & 7]
    geom = loads(blob[offset:])
    if geom.geom_type != "Polygon" or not geom.is_valid or geom.is_empty:
        raise ValueError("Invalid original mine geometry")
    return geom


def main():
    plan = read(HERE / "global_mine_discovery_protocol.json")
    path = DATA / "sources/global_mining_polygons_v2.gpkg"
    receipt = read(DATA / "receipts/global_mining_polygons_v2.gpkg.json")
    if sha(path) != receipt["sha256"]:
        raise ValueError("Original mining footprint data changed")
    previous = read(HERE.parent / "phase9_20261015/protocol.json")
    historic = read(OLD.parent / "public_reference_screen_20261015/qualification/historical_quarry_footprints.geojson")
    old_quarries = [shape(row["geometry"]) for row in historic["features"]]
    mines = []
    bbox = plan["discovery_region_wgs84"]
    with sqlite3.connect(path.as_uri() + "?mode=ro", uri=True) as connection:
        query = ('select p.fid,p.geom,p.ISO3_CODE,p.COUNTRY_NAME,p.AREA from mining_polygons p '
                 'join rtree_mining_polygons_geom r on p.fid=r.id '
                 'where r.maxx>=? and r.minx<=? and r.maxy>=? and r.miny<=? order by p.fid')
        for fid, blob, iso, country, area in connection.execute(query, [bbox[0], bbox[2], bbox[1], bbox[3]]):
            geom = gpkg_polygon(blob)
            reasons = []
            for item in previous["previously_observed_windows"]:
                if geom.intersects(box(*item["bounds"])):
                    reasons.append("previous_observed_window:" + item["case_id"])
            if any(geom.intersects(row) for row in old_quarries):
                reasons.append("previous_qualified_mine_footprint")
            for case in protected_overlap(geom):
                reasons.append("protected_geography:" + case)
            native = shape(transform_geom("EPSG:4326", "EPSG:3826", mapping(geom)))
            mines.append({"fid": fid, "geographic": geom, "native": native, "search": native.buffer(100),
                          "iso3": iso, "country": country, "source_area_km2": area,
                          "prior_exclusions": reasons})
    # Conservative spatial grouping only, not a statement of legal mine identity.
    parents = list(range(len(mines)))
    def find(i):
        while parents[i] != i:
            parents[i] = parents[parents[i]]
            i = parents[i]
        return i
    for i, mine in enumerate(mines):
        for j in range(i):
            if mine["native"].distance(mines[j]["native"]) <= 1000:
                parents[find(i)] = find(j)
    groups = {}
    for i, mine in enumerate(mines):
        groups.setdefault(find(i), []).append(mine)
    for rows in groups.values():
        group_id = f"MAUS2022_CLUSTER_{min(row['fid'] for row in rows)}"
        exclusions = sorted({value for row in rows for value in row["prior_exclusions"]})
        for row in rows:
            row.update(conservative_group_id=group_id, cluster_exclusions=exclusions)
    write(DATA / "qualification/taiwan_global_mine_footprints.geojson", {"type": "FeatureCollection", "features": [
        {"type": "Feature", "geometry": mapping(mine["geographic"]), "properties": {
            key: value for key, value in mine.items() if key not in ["native", "search", "geographic"]}}
        for mine in mines], "source_doi": plan["source_doi"], "licence": plan["licence"],
        "original_source_sha256": receipt["sha256"], "role": "mining land use only; not actual pit face, permit or failure truth"})
    tree = STRtree([mine["search"] for mine in mines])
    features, sources = [], []
    for year in plan["source_years"]:
        archive_path = OLD / "sources" / f"official_{year}_inventory.zip"
        source_receipt = read(OLD / "receipts" / (archive_path.name + ".json"))
        if sha(archive_path) != source_receipt["sha256"]:
            raise ValueError("Original official event archive changed")
        count = nearby = priorities = 0
        with ZipFile(archive_path) as archive:
            base = next(name[:-4] for name in archive.namelist() if name.lower().endswith(".shp"))
            crs = CRS.from_wkt(archive.read(base + ".prj").decode("utf-8-sig"))
            reader = shapefile.Reader(shp=io.BytesIO(archive.read(base + ".shp")),
                        shx=io.BytesIO(archive.read(base + ".shx")), dbf=io.BytesIO(archive.read(base + ".dbf")),
                        encoding=archive.read(base + ".cpg").decode("ascii").strip())
            parameters = crs.to_dict()
            expected = {"proj": "tmerc", "lat_0": 0, "lon_0": 121, "k": .9999,
                        "x_0": 250000, "y_0": 0, "ellps": "GRS80", "units": "m"}
            if any(parameters.get(key) != value for key, value in expected.items()):
                raise ValueError("Projection parameters changed; needs source review")
            legacy = year in [2020, 2022]
            for index, record in enumerate(reader.iterShapeRecords()):
                count += 1
                candidates = [mines[int(i)] for i in tree.query(box(*record.shape.bbox))]
                if not candidates:
                    continue
                native = shape(record.shape.__geo_interface__)
                if not native.is_valid or native.is_empty:
                    continue
                matches = []
                for mine in candidates:
                    if not native.intersects(mine["search"]):
                        continue
                    area = native.intersection(mine["native"]).area
                    fraction = area / native.area
                    priority = (fraction >= .5 and native.area >= 30000 and
                                native.buffer(-20).area >= 10000 and not mine["cluster_exclusions"])
                    matches.append({"mine_fid": mine["fid"], "group": mine["conservative_group_id"],
                                    "overlap_m2": area, "overlap_fraction": fraction,
                                    "distance_m": native.distance(mine["native"]),
                                    "prior_exclusions": mine["cluster_exclusions"], "source_priority": priority})
                if matches:
                    nearby += 1
                    priorities += int(any(row["source_priority"] for row in matches))
                    features.append({"type": "Feature", "geometry": transform_geom(crs, "EPSG:4326", mapping(native)),
                        "properties": {"source_year": year, "source_row_zero_based": index,
                            "original_properties": record.record.as_dict(), "original_area_m2": native.area,
                            "core_20m_area_m2": native.buffer(-20).area, "legacy_datum_verification_pending": legacy,
                            "matches": matches, "training_allowed": False, "accident_truth_admitted": False}})
        sources.append({"year": year, "original_objects": count, "mine_or_100m_nearby_objects": nearby,
                        "fresh_large_priority_objects": priorities})
    write(DATA / "qualification/global_inventory_relations.geojson", {"type": "FeatureCollection", "features": features})
    summary = {"completed_at": now(), "original_global_mine_objects": 44929,
               "discovery_region_mine_polygons": len(mines), "conservative_spatial_groups": len(groups),
               "grouping_distance_m": 1000, "grouping_is_not_legal_mine_identity": True,
               "sources": sources, "source_priority_records": [feature["properties"] for feature in features
                   if any(row["source_priority"] for row in feature["properties"]["matches"])],
               "new_training": 0, "new_predictions": 0, "mine_accident_labels_admitted": 0}
    write(DATA / "qualification/global_mine_discovery.json", summary)
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
