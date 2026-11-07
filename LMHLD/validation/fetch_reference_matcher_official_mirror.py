"""Official Kornia-hosted copy of the generic matching weights; D: only."""
from pathlib import Path
from datetime import datetime
import hashlib,json,sys,requests
ROOT=Path('D:/LMHLD');sys.path.insert(0,str(ROOT))
from runtime_paths import configure_runtime_paths
configure_runtime_paths()
import torch
OUT=ROOT/'cache/reference_registration';OUT.mkdir(exist_ok=True)
DEST=OUT/'loftr_outdoor.ckpt';URL='https://huggingface.co/kornia/loftr/resolve/main/loftr_outdoor.ckpt'
if DEST.exists():raise SystemExit('Matcher already present; do not download again')
partial=DEST.with_suffix('.download');total=0
with requests.get(URL,timeout=(10,30),stream=True) as r:
 r.raise_for_status();revision=None;expected=None
 for resp in [*r.history,r]:
  revision=resp.headers.get('x-repo-commit') or revision
  value=resp.headers.get('x-linked-etag','').strip('"')
  if len(value)==64:expected=value
 with partial.open('wb') as f:
  for block in r.iter_content(1<<20):
   total+=len(block)
   if total>100*1024*1024:raise ValueError('Matcher exceeds100MiB')
   f.write(block)
digest=hashlib.sha256(partial.read_bytes()).hexdigest()
if expected and digest!=expected:raise ValueError('Official LFS digest mismatch')
data=torch.load(partial,map_location='cpu',weights_only=True)
if not isinstance(data.get('state_dict'),dict):raise ValueError('Unexpected official checkpoint')
partial.replace(DEST)
v={'created_at':datetime.now().isoformat(timespec='seconds'),'url':URL,'official_repository':'kornia/loftr','revision':revision,'sha256':digest,'official_LFS_sha256':expected,'bytes':total,'original_primary_CMP_endpoint_TLS_failed':True,'weights_only_load_verified':True,'scope':'generic reference georegistration only; never accident segmentation or production registry'}
(OUT/'acquisition.json').write_text(json.dumps(v,indent=2)+'\n',encoding='utf-8')
sys.stdout.reconfigure(encoding='utf-8');print(json.dumps(v))
