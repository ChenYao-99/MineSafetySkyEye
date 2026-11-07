"""Copy runtime source only; reference immutable model/data assets explicitly."""
from pathlib import Path
from datetime import datetime,timezone
import ast,hashlib,json,shutil,sys,os,stat
ROOT=Path('D:/GeoMMAgent');ML=Path('D:/LMHLD');OUT=ROOT/'cache/deployment_soak_20261006'


def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def copy_runtime(source,destination,entries):
    destination.mkdir(parents=True,exist_ok=True);pending=list(entries);done=set()
    while pending:
        name=pending.pop()
        if name in done:continue
        done.add(name);candidate=source/name
        if candidate.is_dir():
            for p in candidate.rglob('*'):
                if p.is_file() and p.suffix in ('.py','.yaml','.yml','.json','.md','.txt') and '__pycache__' not in p.parts:
                    relative=p.relative_to(source);target=destination/relative;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,target)
                    if p.suffix=='.py':pending.extend(imports(source,p))
        elif candidate.is_file():
            target=destination/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(candidate,target)
            if candidate.suffix=='.py':pending.extend(imports(source,candidate))


def imports(root,path):
    try:tree=ast.parse(path.read_text(encoding='utf-8-sig'))
    except (SyntaxError,UnicodeError):return []
    result=[]
    for node in ast.walk(tree):
        names=[a.name for a in node.names] if isinstance(node,ast.Import) else [node.module] if isinstance(node,ast.ImportFrom) and not node.level else []
        for name in names:
            if not name:continue
            first=name.split('.')[0]
            if (root/(first+'.py')).is_file():result.append(first+'.py')
            elif (root/first/'__init__.py').is_file():result.append(first)
    return result


def main():
    release=OUT/'release';agent=release/'GeoMMAgent';detector=release/'LMHLD'
    copy_runtime(ROOT,agent,['backend.py','coordinator','configs','exec_agents','toolkit','geomm_mcp','project_runtime.py',
                             'case_archive.py','review_handoff.py','comparison_reporting.py','http_contract.py',
                             'evidence_reporting.py','report_grounding.py'])
    copy_runtime(ML,detector,['detector_service.py','knowledge_base.py','geo_tools.py','geo_geometry.py'])
    for name in ['knowledge_sources.json','knowledge_chunks.json','model_registry.json']:shutil.copy2(ML/name,detector/name)
    for name in ['LICENSE','NOTICE']:
        if (ROOT/name).exists():shutil.copy2(ROOT/name,agent/name)
    shutil.copy2(ROOT/'矿安天眼_前端演示(后端版).html',agent/'矿安天眼_前端演示(后端版).html')
    token=Path('D:/Miniconda3/Lib/site-packages/dashscope/resources/qwen.tiktoken')
    (agent/'resources').mkdir(exist_ok=True);shutil.copy2(token,agent/'resources/qwen.tiktoken')
    owned_files=[]
    for folder,dirs,names in os.walk(release,followlinks=False):
        dirs[:]=[name for name in dirs if not os.lstat(Path(folder)/name).st_file_attributes & stat.FILE_ATTRIBUTE_REPARSE_POINT and name!='__pycache__']
        owned_files.extend(Path(folder)/name for name in names if not name.endswith('.pyc'))
    manifest={'created_at':datetime.now(timezone.utc).isoformat(),'source_root':str(release),
        'new_environments':{env:str(OUT/'envs'/env) for env in ['GeoMMA','lmhld']},
        'source_files':{str(p.relative_to(release)):sha(p) for p in owned_files},
        'reference_model_assets':str(ML/'models_prior'),'reference_data_assets':str(ML/'datasets'),
        'reference_bindings_are_existing_project_assets_not_new_labels':True,
        'credentials_or_original_environment_site_packages_copied':False,
        'qualification_scope':'fresh virtual environments and independent source copy on same Windows machine;fixed project assets mounted read-only by convention;not another-machine offline installation'}
    (OUT/'release_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print('Prepared isolated source release:',len(manifest['source_files']),'files;no credentials/cache/training history copied')


if __name__=='__main__':main()
