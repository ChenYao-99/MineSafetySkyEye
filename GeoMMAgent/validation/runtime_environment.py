"""Capture/check installed dependency versions without reading credentials or network."""
from pathlib import Path
import argparse,hashlib,json,platform,sys,struct
from importlib import metadata
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from project_runtime import configure_runtime_paths
configure_runtime_paths()


def installed():
    packages={};local_sources=[]
    for distribution in metadata.distributions():
        name=distribution.metadata.get('Name')
        if not name:continue
        name=name.lower().replace('_','-');packages[name]=distribution.version
        # Record source-install presence without storing URL/path/user/password values.
        if distribution.read_text('direct_url.json'):local_sources.append(name)
    return {'python_version':platform.python_version(),'platform':platform.system(),'architecture':platform.machine() or str(struct.calcsize('P')*8)+'bit',
            'pointer_bits':struct.calcsize('P')*8,
            'packages':dict(sorted(packages.items())),'source_installed_packages':sorted(set(local_sources))}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['capture','check']);parser.add_argument('environment',choices=['GeoMMA','lmhld']);args=parser.parse_args()
    directory=ROOT/'deployment';directory.mkdir(exist_ok=True)
    manifest=directory/(args.environment+'.environment.json');requirements=directory/('requirements.'+args.environment+'.lock.txt')
    current=installed()
    if args.mode=='capture':
        manifest.write_text(json.dumps(current,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
        requirements.write_text('# Captured working environment;use the matching Python/CUDA build and local source packages.\n'+
                                '\n'.join(name+'=='+version for name,version in current['packages'].items())+'\n',encoding='utf-8')
        print(json.dumps({'captured':True,'environment':args.environment,'packages':len(current['packages']),
                         'python_version':current['python_version'],'manifest':str(manifest)},ensure_ascii=False))
    else:
        recorded=json.loads(manifest.read_text(encoding='utf-8'))
        differences={name:{'expected':version,'actual':current['packages'].get(name)} for name,version in recorded['packages'].items() if current['packages'].get(name)!=version}
        system_match=all(recorded[key]==current[key] for key in ['python_version','platform','architecture'])
        result={'passed':not differences and system_match,'environment':args.environment,'system_match':system_match,
                'package_mismatches':differences,'extra_packages':sorted(set(current['packages'])-set(recorded['packages'])),
                'requirements_sha256':hashlib.sha256(requirements.read_bytes()).hexdigest(),
                'scope':'current environment comparison;not a clean-machine installation acceptance'}
        out=ROOT/'cache/archive_workflow_acceptance_20261006';out.mkdir(exist_ok=True)
        (out/(args.environment+'_environment_check.json')).write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
        print(json.dumps({'passed':result['passed'],'environment':args.environment,'mismatch_count':len(differences),'system_match':system_match}))
        if not result['passed']:raise SystemExit(1)


if __name__=='__main__':main()
