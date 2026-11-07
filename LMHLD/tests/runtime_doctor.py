"""Machine-readable local deployment preflight; no external API/weights/dataset downloads."""
import json,sys,platform,os
from pathlib import Path
from importlib.metadata import version,PackageNotFoundError
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import torch
import detector_service as service
from model_registry import file_sha256


def run():
    records=[]
    for tag,entry in service.UPLOAD_MODELS.items():
        path,patch,norm,scale=entry
        try:
            artifact=service.model_registry.verify('upload:'+tag,path,{'patch_size':patch,'threshold':service.UPLOAD_THRESHOLDS[tag],'patch_norm':norm,'input_scale':scale})
            records.append({'model_tag':tag,'available':True,'weights_sha256':artifact['weights_sha256'],'validation_status':artifact['validation_status']})
        except Exception as exc:
            records.append({'model_tag':tag,'available':False,'reason':str(exc)})
    packages={}
    for name in ('torch','torchvision','segmentation-models-pytorch','numpy','opencv-python','Flask','flask-cors','Pillow','rasterio'):
        try:packages[name]=version(name)
        except PackageNotFoundError:packages[name]=None
    caches={k:os.environ.get(k) for k in ('TORCH_HOME','HF_HOME','TMP','TEMP')}
    record={'python':platform.python_version(),'platform':platform.platform(),'executable':sys.executable,
            'packages':packages,'cuda_available':torch.cuda.is_available(),'torch_cuda_build':torch.version.cuda,
            'GPU':torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            'model_artifacts':records,'default_ready':next(r for r in records if r['model_tag']=='mine')['available'],
            'source_fingerprints':{name:file_sha256(ROOT/name) for name in ('detector.py','detector_service.py','raster_upload.py','upload_utils.py','model_registry.py','input_quality.py')},
            'configured_cache_locations':caches,'external_API_checked':False,'deployment_accuracy_or_field_safety_certified':False,
            'weights_not_automatically_promoted':True}
    output=ROOT/'experiments/runtime_integrity_20261004';output.mkdir(exist_ok=True)
    (output/'runtime_environment.json').write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'default_ready':record['default_ready'],'available_upload_models':sum(r['available'] for r in records),'GPU':record['GPU'],'downloads':0}))
    if not record['default_ready']:raise SystemExit(1)


if __name__=='__main__':sys.stdout.reconfigure(encoding='utf-8');run()
