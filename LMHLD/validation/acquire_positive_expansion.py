"""Acquire public mine-event sources on D:, with an auditable download ledger."""
import argparse
import hashlib
import json
import re
import sys
import urllib.parse
import urllib.request
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "experiments" / "positive_expansion_20261001" / "sources"
PAGES = {
    "collolar_portal": "https://unosat.org/products/2117",
    "hpakant_ntu": "https://scholars.lib.ntu.edu.tw/entities/publication/3b97a144-33f6-475a-8e03-0bf5b80fb3b8",
    "hpakant_author": "https://www.earth.sinica.edu.tw/en/research/fields/detail/48",
    "hpakant_publisher": "https://www.sciencedirect.com/science/article/pii/S0924271621001489",
}


def acquire(key, url, kind="html"):
    OUT.mkdir(parents=True, exist_ok=True)
    target = OUT / f"{key}.{kind}"
    ledger_path = OUT / "acquisition_manifest.json"
    ledger = json.loads(ledger_path.read_text(encoding="utf-8")) if ledger_path.exists() else {}
    entry = {"url": url, "requested_at": datetime.now().isoformat(timespec="seconds")}
    try:
        if not target.exists():
            request = urllib.request.Request(url, headers={"User-Agent": "GeoMMAgent-Research/1.0"})
            with urllib.request.urlopen(request, timeout=40) as response:
                data = response.read(80*1024*1024+1)
                entry.update(resolved_url=response.geturl(), content_type=response.headers.get("Content-Type"))
            if len(data) > 80*1024*1024:
                raise ValueError("Asset exceeds the registered 80 MB limit")
            if kind == "pdf" and not data.startswith(b"%PDF"):
                raise ValueError("Not a PDF; do not store access-denied HTML as data")
            target.write_bytes(data)
        data = target.read_bytes()
        entry.update(status="downloaded", bytes=len(data), sha256=hashlib.sha256(data).hexdigest(), file=str(target))
        print(f"{key}: {len(data)} bytes", flush=True)
        if kind == "html":
            source = data.decode("utf-8", errors="replace")
            links = re.findall(r'(?:href|src)=[\"\']([^\"\']+)', source)
            links += re.findall(r'https?[^\s\"<>]+?\.pdf[^\s\"<>]*', source)
            for link in sorted(set(links)):
                if any(s in link.lower() for s in ("pdf", "download", "bitstream", "2117", "collolar", "hpakant")):
                    print(urllib.parse.urljoin(url, link), flush=True)
    except Exception as exc:
        entry.update(status="unavailable", error=f"{type(exc).__name__}: {exc}")
        print(f"{key}: {entry['error']}", flush=True)
    ledger[key] = entry
    ledger_path.write_text(json.dumps(ledger, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    return entry["status"] == "downloaded"


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("--pages", nargs="+", choices=list(PAGES))
    parser.add_argument("--key")
    parser.add_argument("--url")
    parser.add_argument("--kind", choices=["pdf", "html", "jpg", "json"], default="pdf")
    args = parser.parse_args()
    if args.pages:
        for key in args.pages:
            acquire(key, PAGES[key])
    elif args.key and args.url:
        sys.exit(0 if acquire(args.key, args.url, args.kind) else 1)
    else:
        parser.error("Choose --pages or --key/--url")
