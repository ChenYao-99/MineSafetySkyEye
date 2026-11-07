"""Restore selected exact historical files; never overwrite current working code.

  python validation/restore_retired_source.py --list
  python validation/restore_retired_source.py --file LMHLD/validation/make_normal_focal_runner.py
  python validation/restore_retired_source.py --all
"""
from pathlib import Path
import argparse,json,sys,zipfile,hashlib
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from model_registry import file_sha256
BASE=ROOT/'cache/source_cleanup_20261005'
def run():
    parser=argparse.ArgumentParser();mode=parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--list',action='store_true');mode.add_argument('--file',action='append');mode.add_argument('--all',action='store_true')
    args=parser.parse_args();plan=json.loads((BASE/'plan.json').read_text(encoding='utf-8-sig'))
    archive=Path(plan['archive']['path']).resolve()
    if archive.parent!=BASE.resolve() or file_sha256(archive)!=plan['archive']['sha256']:raise ValueError('Recovery archive changed')
    rows={r['archive_member']:r for r in plan['archived_files_to_remove']}
    if args.list:
        print('\n'.join(rows));return
    wanted=list(rows) if args.all else args.file
    if any(name not in rows for name in wanted):raise ValueError('Only explicitly recorded archive members can be restored')
    with zipfile.ZipFile(archive) as z:
        for name in wanted:
            row=rows[name];p=Path(row['path']).resolve();root=Path(row['root']).resolve()
            if p.drive.upper()!='D:' or root.drive.upper()!='D:':raise ValueError('Restore files must stay on D:')
            if root not in [Path('D:/GeoMMAgent').resolve(),ROOT.resolve()] or not p.is_relative_to(root):raise ValueError('Restore target escaped authorised workspaces')
            current=p.parent
            while current!=root.parent:
                if current.exists() and current.stat().st_file_attributes & 0x400:raise ValueError('Restore path contains a reparse point')
                current=current.parent
            content=z.read(name)
            if hashlib.sha256(content).hexdigest()!=row['sha256']:raise ValueError('Archived file checksum failed')
            if p.exists():
                if file_sha256(p)!=row['sha256']:raise ValueError('Refusing to overwrite edited current file: '+str(p))
                print('Already identical:',name);continue
            p.parent.mkdir(parents=True,exist_ok=True)
            with p.open('xb') as stream:stream.write(content)
            print('Restored:',name)
if __name__=='__main__':sys.stdout.reconfigure(encoding='utf-8');run()
