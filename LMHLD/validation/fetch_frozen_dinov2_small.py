"""Download one bounded official-compatible feature backbone on D:, with pinned hashes."""
from pathlib import Path
from datetime import datetime
import json,sys,hashlib,requests
ROOT=Path('D:/LMHLD');sys.path.insert(0,str(ROOT))
from runtime_paths import configure_runtime_paths
configure_runtime_paths()
OUT=ROOT/'cache/foundation_features/dinov2_small';OUT.mkdir(parents=True,exist_ok=True)
REPO='timm/vit_small_patch14_dinov2.lvd142m'
def run():
 manifest=OUT/'acquisition.json'
 if manifest.exists():
  v=json.loads(manifest.read_text())
  for name,digest in v['assets'].items():
   if hashlib.sha256((OUT/name).read_bytes()).hexdigest()!=digest:raise ValueError('Frozen backbone asset changed')
  print('FrozenDINOsmall exists;no redownload');return
 r=requests.get('https://huggingface.co/api/models/'+REPO,timeout=(10,30));r.raise_for_status();meta=r.json();revision=meta['sha']
 assets={}
 for name in ('config.json','model.safetensors'):
  url='https://huggingface.co/'+REPO+'/resolve/'+revision+'/'+name;path=OUT/name;partial=OUT/(name+'.download')
  with requests.get(url,timeout=(10,30),stream=True) as response:
   response.raise_for_status();total=0;expected=None
   for h in [*response.history,response]:
    value=h.headers.get('x-linked-etag','').strip('"')
    if len(value)==64:expected=value
   with partial.open('wb') as f:
    for block in response.iter_content(1<<20):
     total+=len(block)
     if total>110*1024*1024:raise ValueError('Small backbone exceeds110MiB filebudget')
     f.write(block)
  digest=hashlib.sha256(partial.read_bytes()).hexdigest()
  if expected and digest!=expected:raise ValueError('OfficialLFS digest mismatch')
  if name.endswith('.json'):json.loads(partial.read_bytes())
  partial.replace(path);assets[name]=digest;print(name,total,'D:',flush=True)
 v={'created_at':datetime.now().isoformat(timespec='seconds'),'repository':REPO,'revision':revision,'license':'apache-2.0','model_role':'frozen generic photograph representation;not mining accident-trained or field-qualified',
   'source_model_card':'https://huggingface.co/'+REPO,'original_authors':'https://github.com/facebookresearch/dinov2','assets':assets,'mine_detector_or_reserved_predictions':0,'source_generic_pretraining_exposure_not_universally_audited':True}
 manifest.write_text(json.dumps(v,indent=2)+'\n',encoding='utf-8');print('Pinnedsmall backbone saved;not productionpromotion')
if __name__=='__main__':sys.stdout.reconfigure(encoding='utf-8');run()
