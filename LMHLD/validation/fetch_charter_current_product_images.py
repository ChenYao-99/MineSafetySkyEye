"""Download exact current official product images discovered in the public page."""
from pathlib import Path
from datetime import datetime
import json,sys,requests,concurrent.futures,hashlib
ROOT=Path('D:/LMHLD');sys.path.insert(0,str(ROOT))
from runtime_paths import configure_runtime_paths
configure_runtime_paths()
OUT=ROOT/'datasets/maxclosure_sources_20261005/jagersfontein'
URLS={f'charter_product_{i}.img':'https://disasterscharter.org/cos-api/api/file/public/article-image/'+str(i) for i in (28498583,28498588,28499138,28499143)}
def fetch(item):
 name,url=item;p=OUT/name;row={'url':url,'path':str(p)}
 try:
  if p.exists():data=p.read_bytes()
  else:
   r=requests.get(url,timeout=(10,30));r.raise_for_status();data=r.content
   if len(data)>15*1024*1024 or not(data.startswith(b'\xff\xd8') or data.startswith(b'\x89PNG')):raise ValueError('Not a bounded actual map image')
   p.write_bytes(data)
  from PIL import Image
  with Image.open(p) as im:row['dimensions']=list(im.size)
  row.update(obtained=True,bytes=len(data),sha256=hashlib.sha256(data).hexdigest())
 except Exception as e:row.update(obtained=False,error=str(e))
 print(json.dumps(row),flush=True);return row
if __name__=='__main__':
 sys.stdout.reconfigure(encoding='utf-8')
 with concurrent.futures.ThreadPoolExecutor(4) as pool:rows=list(pool.map(fetch,URLS.items()))
 (OUT/'current_official_products.json').write_text(json.dumps({'created_at':datetime.now().isoformat(timespec='seconds'),'sources':rows,'map_is_not_automatically_original_survey_GIS':True,'classifiers_run':0},indent=2)+'\n',encoding='utf-8')
