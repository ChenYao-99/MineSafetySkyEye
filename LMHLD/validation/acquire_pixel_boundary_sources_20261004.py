"""Fetch bounded public source files to D:, without accessing any model."""
from pathlib import Path
from datetime import datetime
import concurrent.futures, hashlib, json, sys, urllib.request

ROOT = Path('D:/LMHLD')
sys.path.insert(0, str(ROOT))
from runtime_paths import configure_runtime_paths
configure_runtime_paths()
OUT = ROOT/'datasets/pixel_boundary_sources_20261004'
SOURCES = {
    'wang2026_mining_deformation.pdf': 'https://research.utwente.nl/files/529777815/1-s2.0-S1569843226001330-main.pdf',
    'dmre_mhsi2020_2021.pdf': 'https://www.dmre.gov.za/Portals/0/Resource%20Center/Mine%20Health%20and%20Safety%20Inspectorate%20Annual%20Reports/2020-2021%20MHSI%20Annual%20Report.pdf?ver=2023-10-13-094805-180',
    'ibge_archive_target_index.json': 'https://web.archive.org/cdx/search/cdx?url=www.ibge.gov.br/np_download/novoportal/extras/Municipios_localidades_Afetados_MG*&output=json&filter=statuscode:200&collapse=urlkey',
    'ibge_geoftp_environment_index.html': 'https://geoftp.ibge.gov.br/informacoes_ambientais/',
}

def fetch(item):
    name, url = item
    dest = OUT/name
    row = {'url': url, 'path': str(dest), 'source_role': 'public evidence only; no model inputs or accepted labels yet'}
    try:
        if dest.exists():
            data = dest.read_bytes()
        else:
            with urllib.request.urlopen(urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'}), timeout=35) as resp:
                data = resp.read(45*1024*1024+1)
                row['http_status'] = resp.status
            if len(data)>45*1024*1024: raise ValueError('Source exceeds45MiB budget')
            if name.endswith('.pdf') and not data.startswith(b'%PDF'): raise ValueError('Response is not an actual PDF')
            if name.endswith('.json'): json.loads(data)
            dest.write_bytes(data)
        row.update(obtained=True, bytes=len(data), sha256=hashlib.sha256(data).hexdigest())
    except Exception as exc:
        row.update(obtained=False, error=str(exc))
    return row

if __name__=='__main__':
    sys.stdout.reconfigure(encoding='utf-8'); OUT.mkdir(parents=True, exist_ok=True)
    with concurrent.futures.ThreadPoolExecutor(4) as pool: rows=list(pool.map(fetch, SOURCES.items()))
    value={'created_at': datetime.now().isoformat(timespec='seconds'), 'datasets_drive':'D', 'model_predictions_run':0, 'sources':rows}
    (OUT/'acquisition.json').write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(rows, ensure_ascii=False))
