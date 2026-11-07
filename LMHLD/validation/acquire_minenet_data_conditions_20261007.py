"""Acquire bounded original MineNetCD members; never equate change with accident.

Uses the publisher's HTTP byte ranges and ZIP directory/CRC, avoiding a full3.5GB
download. Records that whole-archive MD5 is supplied, not locally verified.
"""
from pathlib import Path, PurePosixPath
import argparse
import hashlib
import json
import requests
import struct
import sys
import zlib

ROOT = Path('D:/LMHLD/datasets/data_conditions_20261007')


def acquire(entry, directory, session, budget):
    target = ROOT / 'MineNetCD' / PurePosixPath(entry['name'])
    assert target.resolve().is_relative_to(ROOT.resolve())
    assert entry['method'] in (0, 8) and entry['uncompressed'] < 128 * 1024**2
    if target.exists():
        existing = target.read_bytes()
        if len(existing) == entry['uncompressed'] and zlib.crc32(existing) == entry['crc32']:
            return target, hashlib.sha256(existing).hexdigest(), 0
        raise RuntimeError('Previously saved member has changed: ' + entry['name'])
    offset = entry['offset']
    span = 30 + len(entry['name'].encode('utf-8')) + 1024 + entry['compressed']
    if span > budget:
        raise RuntimeError('Acquisition would exceed registered transfer budget')
    response = session.get(directory['source'], headers={'Range': f'bytes={offset}-{offset+span-1}'}, timeout=(10, 45), stream=True)
    try:
        assert response.status_code == 206 and response.headers.get('Content-Range', '').startswith(f'bytes {offset}-')
        data = bytearray()
        for chunk in response.iter_content(1024 * 1024):
            data.extend(chunk)
            assert len(data) <= span
    finally:
        response.close()
    header = struct.unpack('<4s5H3L2H', data[:30])
    assert header[0] == b'PK\x03\x04' and header[3] == entry['method']
    name_length, extra_length = header[-2:]
    assert data[30:30+name_length].decode('utf-8') == entry['name']
    start = 30 + name_length + extra_length
    compressed = data[start:start+entry['compressed']]
    assert len(compressed) == entry['compressed']
    raw = zlib.decompress(compressed, -15) if entry['method'] == 8 else bytes(compressed)
    assert len(raw) == entry['uncompressed'] and zlib.crc32(raw) == entry['crc32']
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(raw)
    return target, hashlib.sha256(raw).hexdigest(), len(data)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sites', nargs='*', default=[])
    args = parser.parse_args()
    directory = json.loads((ROOT / 'MineNetCD_zip_directory.json').read_text(encoding='utf-8'))
    entries = [row for row in directory['entries'] if row['name'].endswith(('.csv', '.xlsx'))]
    for site in args.sites:
        members = [row for row in directory['entries'] if row['name'].startswith(f'MCD/{site}/') and row['uncompressed']]
        assert len(members) == 3, site
        entries.extend(members)
    # No data-driven site selection or model call; selected sites registered before pixels.
    protocol = {'source_DOI': '10.14278/rodare.3251', 'source_url': directory['source'],
        'site_ids': args.sites, 'transfer_budget_bytes': 512 * 1024**2,
        'scope': 'author mining-change auxiliary data;not accident or verified normal-safety labels',
        'whole_archive_MD5_locally_verified': False, 'per_member_ZIP_CRC32_and_SHA256_verified': True,
        'new_model_predictions_or_training': 0}
    protocol_path = ROOT / ('MineNetCD_subset_protocol.json' if args.sites else 'MineNetCD_metadata_protocol.json')
    if protocol_path.exists():
        assert json.loads(protocol_path.read_text(encoding='utf-8')) == protocol
    else:
        protocol_path.write_text(json.dumps(protocol, ensure_ascii=False, indent=2), encoding='utf-8')
    used = 0
    rows = []
    with requests.Session() as session:
        for entry in entries:
            target, sha, count = acquire(entry, directory, session, protocol['transfer_budget_bytes']-used)
            used += count
            rows.append({'member': entry['name'], 'path': str(target), 'sha256': sha,
                'ZIP_CRC32': entry['crc32'], 'bytes': entry['uncompressed']})
            print('VERIFIED', entry['name'], entry['uncompressed'], flush=True)
    receipt = {'passed': True, 'bytes_transferred': used, 'protocol': protocol, 'files': rows}
    (ROOT / ('MineNetCD_subset_acquisition.json' if args.sites else 'MineNetCD_metadata_acquisition.json')).write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2), encoding='utf-8')


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    main()
