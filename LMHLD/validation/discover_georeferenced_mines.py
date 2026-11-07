# -*- coding: utf-8 -*-
"""Acquire small metadata and inspect remote ZIP indexes before bulk downloads."""
import argparse
import hashlib
import json
import struct
import sys
import time
import urllib.parse
import urllib.request
import zlib
from datetime import datetime
from pathlib import Path
import requests

HTTP = requests.Session()

ROOT = Path(__file__).resolve().parents[1] / "datasets/mine_benchmark_20261002"
DOWNLOADS = ROOT / "downloads"
SOURCES = {
    "dlr_metadata": ("https://zenodo.org/api/records/17007637", "json"),
    "dlr_readme": ("https://zenodo.org/records/17007637/files/README.txt?download=1", "txt"),
    "dlr_description": ("https://zenodo.org/records/17007637/files/S1S2_dataset.pdf?download=1", "pdf"),
    "dlr_paper": ("https://gfzpublic.gfz.de/rest/items/item_5035903_1/component/file_5035910/content", "pdf"),
    "dlr_paper_library": ("https://elib.dlr.de/214209/1/Landslide%20mapping%20with%20deep%20learning%20the%20role%20of%20pre-%20post-event%20SAR%20features%20and%20multi-sensor%20data%20fusion.pdf", "pdf"),
    "xinjing_paper": ("https://mdpi-res.com/d_attachment/remotesensing/remotesensing-17-00019/article_deploy/remotesensing-17-00019.pdf", "pdf"),
    "amyntaio_metadata": ("https://zenodo.org/api/records/821031", "json"),
}


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")


def request(url, headers=None):
    return urllib.request.Request(url, headers={"User-Agent": "GeoMMAgent-Research/1.0", **(headers or {})})


def small(key):
    url, ext = SOURCES[key]
    DOWNLOADS.mkdir(parents=True, exist_ok=True)
    path = DOWNLOADS/f"{key}.{ext}"
    ledger_path = ROOT/"acquisition_manifest.json"
    ledger = json.loads(ledger_path.read_text(encoding="utf-8")) if ledger_path.exists() else {}
    entry = {"url": url, "requested_at": datetime.now().isoformat(timespec="seconds")}
    try:
        if not path.exists():
            with HTTP.get(url, headers={"User-Agent":"GeoMMAgent-Research/1.0"}, stream=True, timeout=(20,45)) as response:
                response.raise_for_status()
                data = response.raw.read(50*1024*1024+1)
                entry.update(resolved_url=response.url, content_type=response.headers.get("Content-Type"))
            if len(data)>50*1024*1024:
                raise ValueError("Small-source size cap exceeded")
            if ext=="pdf" and not data.startswith(b"%PDF"):
                raise ValueError("Response is not PDF")
            if ext=="json":
                json.loads(data)
            path.write_bytes(data)
        data = path.read_bytes()
        entry.update(status="downloaded", file=str(path), bytes=len(data), sha256=hashlib.sha256(data).hexdigest())
        print(f"{key}: {len(data)} bytes", flush=True)
    except Exception as exc:
        entry.update(status="unavailable", error=f"{type(exc).__name__}: {exc}")
        print(f"{key}: {entry['error']}", flush=True)
    ledger[key] = entry
    write(ledger_path, ledger)
    return entry


def byte_range(url, start, end):
    for attempt in range(4):
        try:
            with HTTP.get(url, headers={"User-Agent":"GeoMMAgent-Research/1.0", "Range":f"bytes={start}-{end}", "Accept-Encoding":"identity"},
                          stream=True, timeout=(20,35)) as response:
                if response.status_code!=206:
                    raise ValueError(f"Server ignored Range ({response.status_code}); refuse implicit full download")
                if not response.headers.get("Content-Range", "").startswith(f"bytes {start}-{end}/"):
                    raise ValueError("Response range differs from requested range")
                data = response.raw.read(end-start+2)
                if len(data)!=end-start+1:
                    raise OSError("Incomplete range response")
                return data
        except (urllib.error.URLError, OSError, requests.exceptions.RequestException) as exc:
            if attempt==3:
                raise
            print(f"Transient range fetch failure; retry {attempt+1}/3: {type(exc).__name__}",flush=True)
            time.sleep(1+attempt)


def zip_index():
    metadata = json.loads((DOWNLOADS/"dlr_metadata.json").read_text(encoding="utf-8"))
    file = next(f for f in metadata["files"] if f["key"].endswith(".zip"))
    url, size = file["links"]["self"], int(file["size"])
    start = max(0, size-1024*1024)
    tail = byte_range(url, start, size-1)
    at = tail.rfind(b"PK\x05\x06")
    if at<0:
        raise ValueError("ZIP end record not found")
    _, disk, cd_disk, n_disk, n_total, cd_size, cd_start, comment = struct.unpack_from("<4s4H2IH",tail,at)
    if n_total==65535 or cd_start==0xffffffff or cd_size==0xffffffff:
        loc = tail[at-20:at]
        sig, loc_disk, offset, disks = struct.unpack("<4sIQI",loc)
        if sig!=b"PK\x06\x07" or disks!=1:
            raise ValueError("Unsupported ZIP64 layout")
        record = byte_range(url,offset,offset+55)
        sig, length, made, need, disk, cd_disk, n_disk, n_total, cd_size, cd_start = struct.unpack("<4sQ2H2I4Q",record)
    if cd_size>20*1024*1024:
        raise ValueError("Central directory is unexpectedly large")
    central = (tail[cd_start-start:cd_start-start+cd_size] if cd_start>=start
               else byte_range(url,cd_start,cd_start+cd_size-1))
    rows, offset = [], 0
    while offset<len(central):
        header = struct.unpack_from("<4s6H3I5H2I",central,offset)
        if header[0]!=b"PK\x01\x02":
            raise ValueError("Invalid central-directory record")
        flags, method, crc, comp, raw = header[3], header[4], header[7], header[8], header[9]
        nlen, elen, clen, local = header[10], header[11], header[12], header[16]
        name = central[offset+46:offset+46+nlen].decode("utf-8" if flags&2048 else "cp437")
        extra = central[offset+46+nlen:offset+46+nlen+elen]
        cursor = 0
        while cursor+4<=len(extra):
            kind, length = struct.unpack_from("<HH",extra,cursor)
            if kind==1:
                cursor2 = cursor+4
                if raw==0xffffffff:
                    raw=struct.unpack_from("<Q",extra,cursor2)[0]; cursor2+=8
                if comp==0xffffffff:
                    comp=struct.unpack_from("<Q",extra,cursor2)[0]; cursor2+=8
                if local==0xffffffff:
                    local=struct.unpack_from("<Q",extra,cursor2)[0]
            cursor+=4+length
        rows.append({"name":name,"compressed_bytes":comp,"raw_bytes":raw,"method":method,"crc32":crc,"flags":flags,"local_offset":local})
        offset+=46+nlen+elen+clen
    if len(rows)!=n_total:
        raise ValueError("Central-directory item count mismatch")
    result={"source":url,"archive_size":size,"checksum":file["checksum"],"license":metadata["metadata"].get("license"),"entries":rows,
            "index_download_bytes":len(tail)+max(0,cd_size-len(tail)),"created_at":datetime.now().isoformat(timespec="seconds")}
    write(ROOT/"remote_archive_index.json",result)
    print(f"ZIP {size/1e9:.2f}GB; indexed {len(rows)} entries using ~1MB tail",flush=True)
    events=sorted({row["name"].split("/original_scenes/")[1].split("/")[0]
                   for row in rows if "/original_scenes/" in row["name"]})
    print("Event IDs: "+", ".join(events),flush=True)
    for row in rows:
        if "/vector_data/" in row["name"] and not row["name"].endswith("/"):
            print(f"{row['name']} ({row['raw_bytes']/1e6:.2f} MB)",flush=True)


def extract_members(names):
    index=json.loads((ROOT/"remote_archive_index.json").read_text(encoding="utf-8"))
    output=ROOT/"raw_sources/dlr"
    output.mkdir(parents=True,exist_ok=True)
    ledger_path=ROOT/"selected_archive_members.json"
    ledger=json.loads(ledger_path.read_text(encoding="utf-8")) if ledger_path.exists() else {}
    for requested in names:
        matches=[r for r in index["entries"] if r["name"].endswith(requested) and not r["name"].endswith("/")]
        if len(matches)!=1:
            raise ValueError(f"Member name is missing/ambiguous: {requested}")
        row=matches[0]
        if row["raw_bytes"]>80*1024*1024 or row["compressed_bytes"]>80*1024*1024:
            raise ValueError("Single-member 80MB cap exceeded")
        if row["flags"]&1:
            raise ValueError("Encrypted ZIP members are unsupported")
        target=(output/row["name"]).resolve()
        if not target.is_relative_to(output.resolve()):
            raise ValueError("ZIP member escapes intended D: directory")
        if not target.exists():
            at=row["local_offset"]
            header=byte_range(index["source"],at,at+29)
            fields=struct.unpack("<4s5H3I2H",header)
            if fields[0]!=b"PK\x03\x04" or fields[3]!=row["method"]:
                raise ValueError("Local ZIP header mismatch")
            start=at+30+fields[9]+fields[10]
            parts=[]
            for lo in range(start,start+row["compressed_bytes"],2*1024*1024):
                hi=min(lo+2*1024*1024,start+row["compressed_bytes"])-1
                parts.append(byte_range(index["source"],lo,hi))
            payload=b"".join(parts)
            if row["method"]==8:
                decoder=zlib.decompressobj(-15)
                data=decoder.decompress(payload,row["raw_bytes"]+1)
                if not decoder.eof:
                    raise ValueError("Incomplete or oversized deflate member")
            elif row["method"]==0:
                data=payload
            else:
                raise ValueError(f"Unsupported ZIP method: {row['method']}")
            if len(data)!=row["raw_bytes"] or zlib.crc32(data)&0xffffffff!=row["crc32"]:
                raise ValueError("Extracted ZIP member size/CRC mismatch")
            target.parent.mkdir(parents=True,exist_ok=True)
            target.write_bytes(data)
        data=target.read_bytes()
        if len(data)!=row["raw_bytes"] or zlib.crc32(data)&0xffffffff!=row["crc32"]:
            raise ValueError("Existing extracted member changed")
        ledger[row["name"]]={"path":str(target),"bytes":len(data),"compressed_transfer_bytes":row["compressed_bytes"],
            "sha256":hashlib.sha256(data).hexdigest(),"crc32":row["crc32"],"source":index["source"],
            "acquired_at":datetime.now().isoformat(timespec="seconds"),"license":index["license"]}
        write(ledger_path,ledger)
        print(f"Acquired {requested}: {len(data)} bytes",flush=True)


def acquire_event_rasters(events):
    index=json.loads((ROOT/"remote_archive_index.json").read_text(encoding="utf-8"))
    members=[]
    for eid in events:
        for row in index["entries"]:
            if f"/original_scenes/{eid}/" not in row["name"]:
                continue
            if row["name"].endswith(("__B02.tif","__B03.tif","__B04.tif","__B08.tif",
                                      "__MASK.tif","__DEM.tif","__SLOPE.tif","__files_meta.csv")):
                members.append(row["name"])
    if not members:
        raise ValueError("No selected event raster members")
    print(f"Selected {len(members)} original raster/metadata files for {events}",flush=True)
    extract_members(members)


if __name__=="__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    parser=argparse.ArgumentParser()
    parser.add_argument("--keys",nargs="+",choices=list(SOURCES))
    parser.add_argument("--zip-index",action="store_true")
    parser.add_argument("--members",nargs="+")
    parser.add_argument("--events",nargs="+")
    args=parser.parse_args()
    for key in args.keys or []:
        small(key)
    if args.zip_index:
        zip_index()
    if args.members:
        extract_members(args.members)
    if args.events:
        acquire_event_rasters(args.events)
