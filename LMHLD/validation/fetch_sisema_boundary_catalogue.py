from pathlib import Path
from datetime import datetime
import hashlib,json,sys,urllib.request
OUT=Path('D:/LMHLD/datasets/operational_boundary_sources_20261004/brumadinho')
URL='https://visualizador.idesisema.meioambiente.mg.gov.br/api/layers'
with urllib.request.urlopen(URL,timeout=30) as r:data=r.read(6*1024*1024+1)
if len(data)>6*1024*1024:raise ValueError('Layer catalogue too large')
parsed=json.loads(data);path=OUT/'sisema_layer_catalogue.json';path.write_bytes(data)
rows=[row for row in parsed['data'] if any(word in json.dumps(row,ensure_ascii=False).lower() for word in ('brumadinho','fundão','fundao'))]
manifest={'url':URL,'downloaded_at':datetime.now().isoformat(timespec='seconds'),'sha256':hashlib.sha256(data).hexdigest(),
          'role':'independent official source discovery; no prediction or labels','layers':rows}
(OUT/'sisema_relevant_layers.json').write_bytes((json.dumps(manifest,ensure_ascii=False,indent=2)+'\n').encode('utf-8'))
sys.stdout.reconfigure(encoding='utf-8');print(json.dumps(rows,ensure_ascii=False))
