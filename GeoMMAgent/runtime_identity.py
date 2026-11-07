"""Version only non-sensitive runtime sources and catalogs, never credentials/data."""
from pathlib import Path
import hashlib,json


def runtime_fingerprint(root,role):
    root=Path(root)
    if role=='backend':
        names=['backend.py','review_packet.py','manual_annotations.py','detector_supervision.py','local_access.py','server_runtime.py','runtime_identity.py','case_archive.py','comparison_reporting.py','evidence_reporting.py',
               'review_handoff.py','report_grounding.py','evidence_integrity.py','report_workflow.py','http_contract.py','project_runtime.py']
        folders=['configs','coordinator','exec_agents','toolkit','geomm_mcp']
    else:
        names=['detector_service.py','spatial_input_contract.py','local_access.py','server_runtime.py','runtime_identity.py','detector.py','model_registry.py','raster_upload.py',
               'upload_utils.py','input_quality.py','protected_inputs.py','mine_work_context.py','geo_geometry.py',
               'geo_tools.py','knowledge_base.py','knowledge_sources.json','knowledge_chunks.json','model_registry.json','runtime_paths.py','rgb_adapter.py','data_paths.py']
        folders=[]
    files=[root/name for name in names]
    for folder in folders:
        files.extend(p for p in (root/folder).rglob('*') if p.is_file() and p.suffix in ('.py','.yaml','.yml','.json') and '__pycache__' not in p.parts)
    hashes={str(p.relative_to(root)).replace('\\','/'):hashlib.sha256(p.read_bytes()).hexdigest() if p.is_file() else 'missing' for p in files}
    return hashlib.sha256(json.dumps(hashes,sort_keys=True,separators=(',',':')).encode('utf-8')).hexdigest()


if __name__=='__main__':
    import sys
    print(runtime_fingerprint(Path(__file__).resolve().parent,sys.argv[1]))
