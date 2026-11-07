"""Freeze current service artifacts only; this is not a model promotion or field certificate."""
import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import detector_service as service
from model_registry import file_sha256


def freeze():
    root = Path(service.MODELS_ROOT).resolve()
    output = ROOT/'model_registry.json'
    if output.exists():
        raise FileExistsError('Registry already frozen; review explicit changes instead of replacing it')
    entries = {}
    def add(key,path,protocol,status):
        path = Path(path).resolve()
        if not path.is_file():
            return
        if not path.is_relative_to(root):
            raise ValueError('Model artifact escaped D model root')
        entries[key] = {'path':path.relative_to(root).as_posix(),'sha256':file_sha256(path),
                        'protocol':protocol,'validation_status':status}
    for tag,(path,patch,norm,scale) in service.UPLOAD_MODELS.items():
        add('upload:'+tag,path,{'patch_size':patch,'threshold':service.UPLOAD_THRESHOLDS[tag],
                              'patch_norm':norm,'input_scale':scale},
            'development_recall_insufficient' if tag=='mine' else
            'public_reference_regression_only' if tag=='rgb_guarded' else 'legacy_experimental')
    for region,(path,patch) in service.WEIGHTS.items():
        add('region:'+region,path,{'patch_size':patch,'threshold':.5,'patch_norm':None,'input_scale':1.0},'source_dataset_only')
    add('component:fusion32',service.FUSION32_WEIGHTS,{'patch_size':32,'threshold':.5,'patch_norm':None,'input_scale':1.0},'source_dataset_only')
    add('component:nir_synth',service.NIR_SYNTH_WEIGHTS,{'input':'BGR/minmax/255','output':'synthetic_NIR'},'experimental_synthetic_not_measured')
    if entries['upload:mine']['sha256'] != '0d8d8a0540419d4409951ed91e69499fac206b68303f3feec08a32f0841cebef':
        raise ValueError('Default weight changed unexpectedly')
    record = {'schema_version':1,'frozen_at':datetime.now().isoformat(timespec='seconds'),
              'scope':'existing service artifacts; no newly trained candidate promoted',
              'field_safety_validated':False,'artifacts':entries}
    with output.open('x',encoding='utf-8',newline='\n') as stream:
        json.dump(record,stream,ensure_ascii=False,indent=2)
        stream.write('\n')
    print(json.dumps({'registry':str(output),'artifacts':len(entries),'default_changed':False}))


if __name__=='__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    freeze()
