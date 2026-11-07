"""Detector runtime source identity; independent of model output or runtime data."""
from pathlib import Path
import hashlib,json


def runtime_fingerprint(root,role='detector'):
    root=Path(root)
    names=['detector_service.py','spatial_input_contract.py','local_access.py','server_runtime.py','runtime_identity.py','detector.py','model_registry.py','raster_upload.py',
           'upload_utils.py','input_quality.py','protected_inputs.py','mine_work_context.py','geo_geometry.py',
           'geo_tools.py','knowledge_base.py','knowledge_sources.json','knowledge_chunks.json','model_registry.json','runtime_paths.py','rgb_adapter.py','data_paths.py']
    hashes={name:hashlib.sha256((root/name).read_bytes()).hexdigest() if (root/name).is_file() else 'missing' for name in names}
    return hashlib.sha256(json.dumps(hashes,sort_keys=True,separators=(',',':')).encode('utf-8')).hexdigest()


if __name__ == '__main__':
    print(runtime_fingerprint(Path(__file__).resolve().parent))
