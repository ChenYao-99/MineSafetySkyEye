"""Inspect publicly served publisher/agency attachments; never bypass authentication."""
from pathlib import Path
from datetime import datetime
import concurrent.futures, hashlib, json, sys, requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
ROOT=Path('D:/LMHLD');sys.path.insert(0,str(ROOT))
from runtime_paths import configure_runtime_paths
configure_runtime_paths()
OUT=ROOT/'datasets/pixel_boundary_sources_20261004'
SOURCES={
 'wang2026_fig5.jpg':'https://ars.els-cdn.com/content/image/1-s2.0-S1569843226001330-gr5_lrg.jpg',
 'wang2026_supplement.pdf':'https://ars.els-cdn.com/content/image/1-s2.0-S1569843226001330-mmc1.pdf',
 'copler2025_field_paper.pdf':'https://gfzpublic.gfz.de/rest/items/item_5035234_2/component/file_5035235/content',
 'dmre_mhsi2020_2021.pdf':'https://www.dmre.gov.za/Portals/0/Resource%20Center/Mine%20Health%20and%20Safety%20Inspectorate%20Annual%20Reports/2020-2021%20MHSI%20Annual%20Report.pdf?ver=2023-10-13-094805-180',
 'ibge_archive_target_index.json':'https://web.archive.org/cdx/search/cdx?url=www.ibge.gov.br/np_download/novoportal/extras/Municipios_localidades_Afetados_MG*&output=json&filter=statuscode:200&collapse=urlkey',
 'ibge_2019_public_release.html':'https://agenciadenoticias.ibge.gov.br/agencia-noticias/2012-agencia-de-noticias/noticias/23808-novos-dados-geoespaciais-mostram-area-atingida-pelo-rompimento-da-barragem',
}
def fetch(item):
 name,url=item;dest=OUT/name;row={'url':url,'path':str(dest)}
 try:
  if dest.exists():data=dest.read_bytes()
  else:
   s=requests.Session();s.mount('https://',HTTPAdapter(max_retries=Retry(total=1,backoff_factor=.5,status_forcelist=(429,500,502,503,504))))
   with s.get(url,headers={'User-Agent':'Mozilla/5.0'},timeout=(10,25),stream=True) as r:
    row['http_status']=r.status_code;r.raise_for_status();data=bytearray()
    for block in r.iter_content(1<<18):
     data.extend(block)
     if len(data)>45*1024*1024:raise ValueError('Public source exceeds45MiB')
   data=bytes(data)
   if name.endswith('.pdf') and not data.startswith(b'%PDF'):raise ValueError('Not an actual PDF')
   if name.endswith('.jpg') and not data.startswith(b'\xff\xd8'):raise ValueError('Not an actual JPEG')
   if name.endswith('.json'):json.loads(data)
   dest.write_bytes(data)
  row.update(obtained=True,bytes=len(data),sha256=hashlib.sha256(data).hexdigest())
 except Exception as exc:row.update(obtained=False,error=str(exc))
 print(json.dumps(row,ensure_ascii=False),flush=True);return row
if __name__=='__main__':
 sys.stdout.reconfigure(encoding='utf-8');OUT.mkdir(parents=True,exist_ok=True)
 with concurrent.futures.ThreadPoolExecutor(4) as pool:rows=list(pool.map(fetch,SOURCES.items()))
 (OUT/'attachment_acquisition_v2.json').write_text(json.dumps({'created_at':datetime.now().isoformat(timespec='seconds'),'model_predictions':0,'sources':rows},ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
