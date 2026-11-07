"""Acquire explicitly registered public research sources, with hashes and provenance."""
import argparse
import hashlib
import json
import pathlib
import sys
import urllib.request
from datetime import datetime

ROOT = pathlib.Path(__file__).resolve().parents[1]
DATA_HOME = ROOT / "experiments"
OUT = DATA_HOME / "mining_validation_20261001" / "sources"
SOURCES = {
    "utah_inventory_bingham": {
        "url": "https://services1.arcgis.com/99lidPhWCzftIe9K/ArcGIS/rest/services/LandslideInventoryPolygons/FeatureServer/0/query?geometry=-112.17%2C40.49%2C-112.11%2C40.56&geometryType=esriGeometryEnvelope&inSR=4326&outSR=4326&spatialRel=esriSpatialRelIntersects&outFields=*&f=geojson",
        "filename": "utah_inventory_near_bingham.geojson", "type": "metadata",
        "source_page": "https://gis.utah.gov/products/sgid/geoscience/landslide-inventory-polygons/",
        "use": "check whether the inventory actually contains the 2013 mine event; never assume a matching bbox is the event",
    },
    "las_cruces_metadata": {
        "url": "https://zenodo.org/api/records/4704955",
        "filename": "las_cruces_repository_metadata.json", "type": "metadata",
        "source_page": "https://zenodo.org/records/4704955", "use": "verify reuse license and actual attached files",
    },
    "bingham_post_20130502": {
        "url": "https://assets.science.nasa.gov/dynamicimage/assets/science/esd/eo/images/imagerecords/81000/81364/binghamcanyon_ali_2013122_lrg.jpg?crop=faces%2Cfocalpoint&fit=clip&h=1234&w=1032",
        "filename": "bingham_post_20130502.jpg", "type": "image",
        "source_page": "https://science.nasa.gov/earth/earth-observatory/sizing-up-the-landslide-at-bingham-canyon-mine-81364/",
        "event_date": "2013-04-10", "acquisition_date": "2013-05-02",
        "credit": "NASA Earth Observatory / Jesse Allen and Robert Simmon / NASA EO-1 ALI team",
        "use": "independent historical mine-landslide validation candidate, not training",
    },
    "bingham_pre_20110720": {
        "url": "https://assets.science.nasa.gov/dynamicimage/assets/science/esd/eo/images/imagerecords/81000/81364/bingham_aer_2011201_lrg.jpg?crop=faces%2Cfocalpoint&fit=clip&h=12344&w=10322",
        "filename": "bingham_pre_20110720.jpg", "type": "image",
        "source_page": "https://science.nasa.gov/earth/earth-observatory/sizing-up-the-landslide-at-bingham-canyon-mine-81364/",
        "event_date": "2013-04-10", "acquisition_date": "2011-07-20",
        "credit": "USDA National Agriculture Imagery Program (NAIP), via NASA Earth Observatory",
        "use": "context/pre-event reference; different sensor/resolution, not pixel-aligned change truth",
    },
    "hainan_paper": {
        "url": "https://mdpi-res.com/d_attachment/remotesensing/remotesensing-18-01774/article_deploy/remotesensing-18-01774.pdf",
        "filename": "hainan_openpit_segmentation_paper.pdf", "type": "paper_candidate",
        "source_page": "https://www.mdpi.com/2072-4292/18/11/1774",
        "use": "check actual title, data access and published reference masks before any use",
    },
    "las_cruces_paper": {
        "url": "https://zenodo.org/records/4704955/files/Remote%20analysis%20of%20an%20open-pit%20slope%20failure.pdf?download=1",
        "filename": "las_cruces_remote_analysis.pdf", "type": "paper_candidate",
        "source_page": "https://zenodo.org/records/4704955",
        "event_date": "2019-01", "use": "check availability of imagery and published boundaries; not a raw dataset",
    },
}


def acquire(keys):
    OUT.mkdir(parents=True, exist_ok=True)
    manifest_path = OUT / "acquisition_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    failures = []
    for key in keys:
        spec = dict(SOURCES[key])
        target = OUT / spec["filename"]
        try:
            if not target.exists():
                request = urllib.request.Request(spec["url"], headers={"User-Agent": "GeoMMAgent-Research/1.0"})
                with urllib.request.urlopen(request, timeout=45) as response:
                    limit = 120 * 1024 * 1024 if spec["type"].startswith("paper") else 40 * 1024 * 1024
                    data = response.read(limit + 1)
                    spec["resolved_url"] = response.geturl()
                    spec["content_type"] = response.headers.get("Content-Type")
                if len(data) > limit:
                    raise ValueError("registered asset exceeds its size limit")
                if spec["type"] == "image" and not data.startswith(b"\xff\xd8"):
                    raise ValueError("response is not a JPEG image")
                if spec["type"].startswith("paper") and not data.startswith(b"%PDF"):
                    raise ValueError("response is not a PDF")
                target.write_bytes(data)
            data = target.read_bytes()
            spec.update(status="downloaded", bytes=len(data), sha256=hashlib.sha256(data).hexdigest(),
                        downloaded_at=datetime.now().isoformat(timespec="seconds"))
            print(f"{key}: acquired {len(data)} bytes", flush=True)
        except Exception as exc:
            spec.update(status="unavailable", error=f"{type(exc).__name__}: {exc}")
            failures.append(key)
            print(f"{key}: {spec['error']}", flush=True)
        manifest[key] = spec
        manifest_path.write_bytes((json.dumps(manifest, ensure_ascii=False, indent=2) + "\n").encode())
    return failures


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("--keys", nargs="+", choices=list(SOURCES), default=list(SOURCES))
    args = parser.parse_args()
    sys.exit(1 if acquire(args.keys) else 0)
