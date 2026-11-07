"""Read-only dependency/provenance audit before user-authorized source cleanup."""
from pathlib import Path
from collections import defaultdict
import ast,json,os,re,sys,subprocess,hashlib
ROOTS=[Path('D:/GeoMMAgent'),Path('D:/LMHLD')]
OUT=ROOTS[1]/'cache/source_cleanup_20261005';OUT.mkdir(parents=True,exist_ok=True)
CODE={'.py','.js','.mjs','.ps1','.sh','.bat'}
TEXT=CODE|{'.md','.txt','.json','.yaml','.yml','.html'}
OMIT={'.git','.idea','.aws','.agents','.codex','node_modules','research_dependencies','browser','chrome-win','chromium','site-packages','__pycache__'}
HASH_PATTERN=re.compile(r'(?<![0-9a-fA-F])[0-9a-fA-F]{64}(?![0-9a-fA-F])')
FILE_PATTERN=re.compile(r'(?<![\w-])([\w.-]+\.(?:py|js|mjs|ps1|sh|bat|log|png|json|txt))(?![\w-])')
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
    return h.hexdigest()
def inside(p,root):return p.resolve().is_relative_to(root.resolve())
def walk(root):
    for base,dirs,names in os.walk(root,followlinks=False):
        dirs[:]=[d for d in dirs if d not in OMIT and not Path(base,d).is_symlink()]
        for name in names:
            p=Path(base,name)
            if not p.is_symlink():yield p
def changed_tracked(root):
    result=subprocess.run(['git','-C',str(root),'status','--porcelain=v1','-z','--untracked-files=no'],capture_output=True)
    if result.returncode:raise ValueError('Cannot establish tracked change protection')
    entries=result.stdout.decode('utf-8').split('\0');protected=set();i=0
    while i<len(entries):
        e=entries[i];i+=1
        if not e:continue
        protected.add((root/e[3:]).resolve())
        if e[0] in 'RC' and i<len(entries):protected.add((root/entries[i]).resolve());i+=1
    return protected
def run():
    codes={};texts={};instructions=[];counts={}
    for root in ROOTS:
        n=0
        for p in walk(root):
            if p.name=='AGENTS.md':instructions.append(str(p))
            relative=p.relative_to(root);n+=1
            if p.name.startswith('.env'):continue
            # Runtime/download caches contain third-party programs, not our source.
            top=relative.parts[0]
            if p.suffix in CODE and top not in ('cache','datasets','experiments','models_prior','artifact_runtime','assets','uploads','temp_images','yolo_out','results'):
                codes[p.resolve()]={'root':root,'relative':relative,'text':p.read_text(encoding='utf-8-sig',errors='replace')}
            if p.suffix in TEXT and p.stat().st_size<=4*1024**2:
                if top=='cache' and not any(k in relative.as_posix() for k in ('cleanup','foundation_features','reference_registration','competition_readiness','source_cleanup')):continue
                if top=='artifact_runtime':continue
                texts[p.resolve()]=p.read_text(encoding='utf-8-sig',errors='replace')
        counts[str(root)]=n
    by_name=defaultdict(set);by_module=defaultdict(set)
    for p,row in codes.items():
        by_name[p.name].add(p)
        if p.suffix=='.py':
            parts=list(row['relative'].with_suffix('').parts)
            if parts[-1]=='__init__':parts.pop()
            by_module['.'.join(parts)].add(p)
    digest_references=defaultdict(set);file_refs=defaultdict(set);imports=defaultdict(set);parse_errors=[]
    for path,content in texts.items():
        # Only JSON is treated as a frozen-digest authority; docs are references.
        # Cleanup inventories describe removals; they do not require those files
        # to remain at their old paths. Dataset/study manifests do bind sources.
        authority=any(path.is_relative_to(root/sub) for root in ROOTS for sub in ('datasets','experiments','models_prior','validation'))
        if path.suffix=='.json' and authority:
            for digest in HASH_PATTERN.findall(content):digest_references[digest.lower()].add(path)
        for name in FILE_PATTERN.findall(content):
            for p in by_name.get(name,()):
                if p!=path:file_refs[p].add(path)
    for p,row in codes.items():
        if p.suffix!='.py':continue
        try:tree=ast.parse(row['text'],filename=str(p))
        except SyntaxError as e:parse_errors.append({'path':str(p),'line':e.lineno});continue
        package='.'.join(row['relative'].parent.parts)
        for node in ast.walk(tree):
            names=[]
            if isinstance(node,ast.Import):names=[a.name for a in node.names]
            elif isinstance(node,ast.ImportFrom):
                base=node.module or ''
                if node.level:
                    parts=package.split('.') if package else [];parts=parts[:len(parts)-node.level+1]
                    base='.'.join(parts+([base] if base else []))
                names=[base]+[base+'.'+a.name for a in node.names]
            elif isinstance(node,ast.Call) and isinstance(node.func,(ast.Name,ast.Attribute)):
                name=node.func.id if isinstance(node.func,ast.Name) else node.func.attr
                if name in ('import_module','__import__') and node.args and isinstance(node.args[0],ast.Constant) and isinstance(node.args[0].value,str):names=[node.args[0].value]
            for name in names:
                candidates=set(by_module.get(name,()))
                for length in range(1,len(name.split('.'))):candidates.update(by_module.get('.'.join(name.split('.')[:length]),()))
                # Existing scripts also put validation directly on sys.path.
                candidates.update(by_name.get(name.split('.')[0]+'.py',()))
                for target in candidates:
                    if target!=p:imports[target].add(p)
    changed=set().union(*(changed_tracked(root) for root in ROOTS))
    rows=[]
    for p,row in codes.items():
        digest=sha(p);protected=[]
        if p in changed:protected.append('tracked user modification')
        if digest in digest_references:protected.append('frozen/provenance JSON SHA256')
        if imports[p]:protected.append('imported local module')
        if row['relative'].parts[0] in ('tests',):protected.append('test entry;discovered dynamically')
        if p.name.startswith('test_'):protected.append('test entry')
        try:doc=ast.get_docstring(ast.parse(row['text'])) if p.suffix=='.py' else None
        except SyntaxError:doc=None
        rows.append({'path':str(p),'project':str(row['root']),'relative':str(row['relative']),'bytes':p.stat().st_size,'sha256':digest,
                     'protected_reasons':protected,'imported_by':[str(x) for x in sorted(imports[p])],
                     'referenced_by':[str(x) for x in sorted(file_refs[p])],
                     'docstring':(doc or '')[:300]})
    candidates=[r for r in rows if not r['protected_reasons']]
    result={'schema_version':1,'scope':'project source only;all raw datasets,current weights,metadata/results and thirdparty code retained',
            'roots':[str(r) for r in ROOTS],'instructions_found':instructions,'source_files':len(rows),'text_files_scanned':len(texts),
            'protected_source_count':len(rows)-len(candidates),'candidate_count_not_yet_approved_for_deletion':len(candidates),
            'parse_errors':parse_errors,'source_rows':rows,'json_frozen_source_sha_references':{d:[str(p) for p in sorted(paths)] for d,paths in digest_references.items() if any(r['sha256']==d for r in rows)}}
    (OUT/'audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    summary={'sources':len(rows),'protected':len(rows)-len(candidates),'possible_candidates':len(candidates),'instruction_files':instructions,
             'candidates':[{'path':r['path'],'bytes':r['bytes'],'references':len(r['referenced_by']),'description':r['docstring'][:130]} for r in candidates]}
    print(json.dumps(summary,ensure_ascii=False,indent=2))
if __name__=='__main__':sys.stdout.reconfigure(encoding='utf-8');run()
