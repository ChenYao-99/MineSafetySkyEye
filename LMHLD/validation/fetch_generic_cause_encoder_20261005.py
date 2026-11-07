from pathlib import Path
from datetime import datetime,timezone,timedelta
import json,sys
ROOT=Path('D:/LMHLD');sys.path.insert(0,str(ROOT))
from runtime_paths import configure_runtime_paths
configure_runtime_paths()
import torch
from model_registry import file_sha256
OUT=ROOT/'cache/torch/cause_encoder';OUT.mkdir(parents=True,exist_ok=True)
URL='https://download.pytorch.org/models/resnet18-f37072fd.pth'
if __name__=='__main__':
    torch.hub.load_state_dict_from_url(URL,model_dir=str(OUT),map_location='cpu',check_hash=True,progress=False,weights_only=True)
    p=OUT/'resnet18-f37072fd.pth'
    record={'acquired_at':datetime.now(timezone(timedelta(hours=8))).isoformat(),'source':URL,
            'sha256':file_sha256(p),'scope':'generic ImageNet ResNet18,not mine-supervised initialization',
            'known_mine_labels_from_project_consumed':False,'generic_pretraining_content_universally_audited':False}
    (OUT/'acquisition.json').write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8')
    print('Generic official encoder cached onD:',record['sha256'])
