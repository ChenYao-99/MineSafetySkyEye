"""Bounded public evidence acquisition, all files/dependencies/tmp remain D:."""
from pathlib import Path
from datetime import datetime
import concurrent.futures,json,sys,requests,hashlib
ROOT=Path('D:/LMHLD');sys.path.insert(0,str(ROOT))
from runtime_paths import configure_runtime_paths
configure_runtime_paths()
OUT=ROOT/'datasets/maxclosure_sources_20261005'
SOURCES={
 'berkeley_mining_failure_thesis.pdf':'https://escholarship.org/content/qt6vx4j2q3/qt6vx4j2q3.pdf',
 'dlr_dataset_description.pdf':'https://zenodo.org/records/17007637/files/S1S2_dataset.pdf?download=1',
 'cl_before_reference.jpg':'https://blogs.agu.org/landslideblog/files/2021/11/21_11-ministro-hales-1.jpg',
 'cl_after_reference.jpg':'https://blogs.agu.org/landslideblog/files/2021/11/21_11-ministro-hales-2.jpg'}
def fetch(item):
 name,url=item;path=OUT/name;row={'url':url,'path':str(path),'role':'source evidence only;no model input or automatic ground truth'}
 try:
  if path.exists():data=path.read_bytes()
  else:
   with requests.get(url,headers={'User-Agent':'Mozilla/5.0'},timeout=(10,30),stream=True) as r:
    row['http_status']=r.status_code;r.raise_for_status();data=bytearray()
    for block in r.iter_content(1<<18):
     data.extend(block)
     if len(data)>45*1024*1024:raise ValueError('Public attachment exceeds45MiB budget')
   data=bytes(data)
   if name.endswith('.pdf') and not data.startswith(b'%PDF'):raise ValueError('Not an actual PDF')
   if name.endswith('.jpg') and not data.startswith(b'\xff\xd8'):raise ValueError('Not an actual JPG')
   path.write_bytes(data)
  row.update(obtained=True,bytes=len(data),sha256=hashlib.sha256(data).hexdigest())
 except Exception as e:row.update(obtained=False,error=str(e))
 print(json.dumps(row,ensure_ascii=False),flush=True);return row
if __name__=='__main__':
 sys.stdout.reconfigure(encoding='utf-8');OUT.mkdir(exist_ok=True)
 with concurrent.futures.ThreadPoolExecutor(4) as pool:rows=list(pool.map(fetch,SOURCES.items()))
 (OUT/'acquisition.json').write_text(json.dumps({'created_at':datetime.now().isoformat(timespec='seconds'),'source_role':'read-only evidence;model0','sources':rows},ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
