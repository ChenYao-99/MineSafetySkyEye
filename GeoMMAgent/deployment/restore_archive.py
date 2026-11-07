"""Restore to a new archive path; never replace a running or existing archive."""
from pathlib import Path
import argparse,json,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from project_runtime import configure_runtime_paths
configure_runtime_paths()
from archive_recovery import restore_archive


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',required=True);parser.add_argument('--destination',required=True)
    args=parser.parse_args()
    result=restore_archive(args.source,args.destination)
    print(json.dumps({'passed':True,**result,'activation':'Stop project services,then set KAT_CASE_ARCHIVE_PATH to the new destination and start normally'},ensure_ascii=False))


if __name__=='__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    try:main()
    except Exception as error:
        print(json.dumps({'passed':False,'error_type':type(error).__name__,'error':'Restore was not accepted;check backup integrity,D: free space and an unused destination. Existing archives were not replaced.'}));sys.exit(1)
