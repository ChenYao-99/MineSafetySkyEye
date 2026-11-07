"""Read-only startup checks in each service's own Python environment."""
from pathlib import Path
import argparse,json,sys,os
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from project_runtime import configure_runtime_paths
configure_runtime_paths()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--role',choices=['agent','detector'],required=True)
    parser.add_argument('--source',required=True);args=parser.parse_args();source=Path(args.source).resolve()
    checks={};details={};sys.path.insert(0,str(source))
    import importlib.metadata
    for name in ['flask','waitress','numpy']:
        try:details[name]=importlib.metadata.version(name);checks[name]=bool(details[name])
        except importlib.metadata.PackageNotFoundError:checks[name]=False
    checks['D_drive']=source.drive.upper()=='D:'
    if args.role=='agent':
        from dotenv import load_dotenv
        load_dotenv(source/'.env')
        checks['source']=all((source/name).is_file() for name in ['backend.py','case_archive.py','detector_supervision.py','local_access.py','server_runtime.py','runtime_identity.py','矿安天眼_前端演示(后端版).html'])
        checks['frontend_mirrors']=not (source/'kat_demo.html').exists() or (source/'kat_demo.html').read_bytes()==(source/'矿安天眼_前端演示(后端版).html').read_bytes()
        from case_archive import CaseArchive
        path=Path(os.environ.get('KAT_CASE_ARCHIVE_PATH',str(source/'data/case_archive/cases.sqlite3'))).resolve()
        checks['archive_D_drive']=path.drive.upper()=='D:'
        if path.exists():
            # Opening a private copy avoids any schema/journal mutation of the live DB.
            import sqlite3,tempfile
            with tempfile.TemporaryDirectory(dir=os.environ['TEMP']) as folder:
                copy=Path(folder)/'archive.sqlite3'
                from contextlib import closing
                with closing(sqlite3.connect(path.as_uri()+'?mode=ro',uri=True)) as db:
                    target=sqlite3.connect(copy)
                    try:db.backup(target)
                    finally:target.close()
                archive=CaseArchive(copy);archive.list_cases(limit=1)
                with archive.connection() as db:
                    checks['archive_integrity']=db.execute('PRAGMA integrity_check').fetchone()[0]=='ok' and not db.execute('PRAGMA foreign_key_check').fetchall()
        else:checks['archive_integrity']=True
    else:
        import torch
        checks['cuda_available']=torch.cuda.is_available();details['torch']=torch.__version__;details['cuda_build']=torch.version.cuda
        from model_registry import ModelRegistry
        models=Path(os.environ.get('KAT_MODELS_ROOT',str(source/'models_prior')))
        registry=ModelRegistry(os.environ.get('KAT_MODEL_REGISTRY',str(source/'model_registry.json')),models)
        details['default_model']=registry.verify('upload:mine',models/'real_ft_T1b_ours_seed42/best_model.pth',
            {'patch_size':32,'threshold':.4,'patch_norm':'minmax','input_scale':1.0})
        checks['default_model']=True
        import math
        reserved_path=source/'datasets/mine_benchmark_20261002/mine_case_registry.json'
        primary=json.loads(reserved_path.read_text(encoding='utf-8'))
        expected=['CL0002','KG0001','TR_COP2024','US_BC2021']
        checks['reserve_registry']=all(primary['cases'][key]['role']=='reserved_final_test' and len(primary['cases'][key]['protected_geographic_extent_wgs84'])==4
            and all(math.isfinite(v) for v in primary['cases'][key]['protected_geographic_extent_wgs84']) for key in expected)
        auxiliary_path=source/'datasets/input_reserve_registry.json'
        if auxiliary_path.exists():
            auxiliary=json.loads(auxiliary_path.read_text(encoding='utf-8'))
            checks['auxiliary_reserve_registry']=auxiliary['schema_version']==1 and all(case['role']=='reserved_final_test'
                and case['input_sha256_values'] and len(case['protected_geographic_extent_wgs84'])==4 for case in auxiliary['cases'].values())
        details['optional_source_dataset_configured']=bool(os.environ.get('KAT_DATA_ROOT'))
    result={'passed':all(checks.values()),'role':args.role,'source':str(source),'python_prefix':sys.prefix,'checks':checks,'details':details,
        'scope':'runtime and default artifact readiness;not model accuracy or field qualification'}
    print(json.dumps(result,ensure_ascii=False));return 0 if result['passed'] else 1


if __name__=='__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    try:sys.exit(main())
    except Exception as error:
        print(json.dumps({'passed':False,'error_type':type(error).__name__,'error':'Deployment preflight failed;check source,default artifacts,dependencies and archive integrity'}));sys.exit(1)
