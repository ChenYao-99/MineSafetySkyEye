"""Read-only inventory for cleanup. Never scans credential/Git metadata contents."""
import ast
import json
import os
import stat
from collections import defaultdict
from pathlib import Path

ROOTS=[Path('D:/GeoMMAgent'),Path('D:/LMHLD')]
SKIP={'.git','.agents','.codex','.aws','.ssh'}
HERE=Path(__file__).resolve().parent


def main():
    records=[];errors=[];sources=[];links=[]
    for root in ROOTS:
        for base,dirs,files in os.walk(root,onerror=lambda exc:errors.append(str(exc))):
            retained=[]
            for name in dirs:
                if name in SKIP:continue
                path=Path(base)/name
                if path.lstat().st_file_attributes & stat.FILE_ATTRIBUTE_REPARSE_POINT:
                    links.append({'path':str(path),'target':str(path.resolve())})
                else:retained.append(name)
            dirs[:]=retained
            for name in files:
                path=Path(base)/name
                try:
                    metadata=path.lstat()
                    if metadata.st_file_attributes & stat.FILE_ATTRIBUTE_REPARSE_POINT:
                        links.append({'path':str(path),'target':str(path.resolve())})
                        continue
                    relative=path.relative_to(root)
                    records.append({'root':str(root),'path':str(path),'relative':str(relative),
                         'bytes':metadata.st_size,'suffix':path.suffix.lower(),'mtime_ns':metadata.st_mtime_ns})
                    if path.suffix=='.py' and not any(part in {'vendor','site-packages','node_modules','artifact_runtime'} for part in relative.parts):
                        sources.append(path)
                except OSError as exc:errors.append(str(exc))
    aggregates=defaultdict(lambda:{'files':0,'bytes':0})
    for record in records:
        relative=Path(record['relative'])
        key=(record['root'],relative.parts[0] if len(relative.parts)>1 else 'ROOT_FILES')
        aggregates[key]['files']+=1;aggregates[key]['bytes']+=record['bytes']
    imports=[]
    for path in sources:
        try:
            tree=ast.parse(path.read_text(encoding='utf-8-sig'))
            names=[]
            for node in ast.walk(tree):
                if isinstance(node,ast.Import):names.extend(alias.name for alias in node.names)
                elif isinstance(node,ast.ImportFrom) and node.module:names.append(node.module)
            imports.append({'path':str(path),'imports':sorted(set(names))})
        except (OSError,SyntaxError,UnicodeError) as exc:
            errors.append(str(path)+': '+str(exc))
    result={'roots':list(map(str,ROOTS)),'records':records,'imports':imports,'errors':errors,'skipped_links':links}
    (HERE/'inventory_20261016.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    summary=[{'root':key[0],'name':key[1],**value} for key,value in aggregates.items()]
    print(json.dumps({'totals':sorted(summary,key=lambda row:-row['bytes']),
                     'source_files':len(sources),'errors':errors[:20]},ensure_ascii=True),flush=True)


if __name__=='__main__':main()
